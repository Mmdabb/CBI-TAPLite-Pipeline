from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..reconstruction import reconstruct_episode_speed, smoothstep_anchor_curve
from .settings import DashboardSettings


def _mae(observed: np.ndarray, modeled: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(modeled)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(observed[valid] - modeled[valid])))


def _first_positive(row: pd.Series, fields: tuple[str, ...], fallback: float) -> float:
    for field in fields:
        value = pd.to_numeric(row.get(field), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return float(fallback)


def _taplite_speed_profile(
    row: pd.Series,
    time_minutes: np.ndarray,
) -> np.ndarray:
    try:
        payload = json.loads(str(row.get("assignment_speed_profile_json", "")))
        profile_time = np.asarray(payload["time_minutes"], dtype=float)
        profile_speed = np.asarray(payload["speed_mph"], dtype=float)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return np.full(len(time_minutes), np.nan)
    valid = np.isfinite(profile_time) & np.isfinite(profile_speed)
    if not valid.any():
        return np.full(len(time_minutes), np.nan)
    profile_time = profile_time[valid]
    profile_speed = profile_speed[valid]
    order = np.argsort(profile_time, kind="stable")
    return np.interp(
        time_minutes,
        profile_time[order],
        profile_speed[order],
    )


def reconstruction_curves(
    row: pd.Series,
    time_minutes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    free_flow = _first_positive(
        row,
        (
            "freeflow_speed_mph",
            "assignment_free_speed_mph",
            "free_flow_speed_model_mph",
        ),
        65.0,
    )
    cutoff = _first_positive(
        row,
        ("threshold_used", "assignment_cutoff_speed_mph"),
        0.70 * free_flow,
    )
    length = float(
        row["network_length_mi"]
        if pd.notna(row.get("network_length_mi"))
        else row["length_mi"]
    )
    discharge = _first_positive(
        row,
        (
            "mu_obs_vphpl",
            "assignment_mu_vphpl",
            "per_lane_hourly_capacity",
            "capacity_vphpl",
        ),
        1800.0,
    )
    window_start = float(np.min(time_minutes))
    window_end = float(np.max(time_minutes))
    if bool(row.get("has_accepted_episode", False)):
        try:
            windows = json.loads(
                str(row.get("accepted_episode_windows_json", "[]"))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            windows = []
        if not windows:
            windows = [row.to_dict()]
        anchors = [(window_start, free_flow), (window_end, free_flow)]
        for episode in windows:
            anchors.extend(
                [
                    (
                        float(episode["t0_hour"]) * 60.0,
                        float(episode["threshold_used"]),
                    ),
                    (
                        float(episode["t3_hour"]) * 60.0,
                        float(episode["threshold_used"]),
                    ),
                ]
            )
        cbi = smoothstep_anchor_curve(time_minutes, anchors)
        for episode in windows:
            episode_t0 = float(episode["t0_hour"])
            episode_t3 = float(episode["t3_hour"])
            mask = (
                (time_minutes >= episode_t0 * 60.0)
                & (time_minutes <= episode_t3 * 60.0)
            )
            if not mask.any():
                continue
            episode_discharge = _first_positive(
                pd.Series(episode),
                ("mu_obs_vphpl", "per_lane_hourly_capacity"),
                discharge,
            )
            episode_length = _first_positive(
                pd.Series(episode),
                ("length_mi",),
                length,
            )
            cbi[mask] = reconstruct_episode_speed(
                time_minutes[mask],
                t0_hour=episode_t0,
                t2_hour=float(episode["t2_hour"]),
                t3_hour=episode_t3,
                minimum_speed_mph=float(episode["min_speed_mph"]),
                cutoff_mph=float(episode["threshold_used"]),
                free_flow_mph=free_flow,
                length_mi=episode_length,
                discharge_vphpl=episode_discharge,
                window_start_minute=episode_t0 * 60.0,
                window_end_minute=episode_t3 * 60.0,
                left_anchor_mph=float(episode["threshold_used"]),
                right_anchor_mph=float(episode["threshold_used"]),
            )
    elif bool(row.get("has_assignment_boundaries", False)):
        cbi = smoothstep_anchor_curve(
            time_minutes,
            [
                (window_start, free_flow),
                (float(row["projected_t0_hour"]) * 60.0, cutoff),
                (float(row["projected_t2_hour"]) * 60.0, float(row["vt2_C_vol"])),
                (float(row["projected_t3_hour"]) * 60.0, cutoff),
                (window_end, free_flow),
            ],
        )
    else:
        cbi = np.full(len(time_minutes), free_flow)
    projected = np.full(len(time_minutes), np.nan)
    if (
        row["projection_status"] == "ready"
        and bool(row.get("has_assignment_speed_profile", False))
    ):
        projected = _taplite_speed_profile(row, time_minutes)
    elif row["projection_status"] == "ready":
        projection_discharge = _first_positive(
            row,
            ("assignment_mu_vphpl",),
            discharge,
        )
        projection_free_flow = _first_positive(
            row,
            ("assignment_free_speed_mph",),
            free_flow,
        )
        projection_cutoff = _first_positive(
            row,
            ("assignment_cutoff_speed_mph",),
            cutoff,
        )
        projected = reconstruct_episode_speed(
            time_minutes,
            t0_hour=float(row["projected_t0_hour"]),
            t2_hour=float(row["projected_t2_hour"]),
            t3_hour=float(row["projected_t3_hour"]),
            minimum_speed_mph=float(row["vt2_C_vol"]),
            cutoff_mph=projection_cutoff,
            free_flow_mph=projection_free_flow,
            length_mi=length,
            discharge_vphpl=projection_discharge,
            window_start_minute=window_start,
            window_end_minute=window_end,
        )
    return cbi, projected


def build_speed_metrics(
    projection: pd.DataFrame,
    heatmap: pd.DataFrame,
    settings: DashboardSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heat = heatmap.copy()
    heat["tmc_code"] = (
        heat["tmc_code"].astype("string").str.strip().str.upper()
    )
    heat_lookup = {
        str(tmc): group.set_index("time_slot_min")["avg_speed_mph"]
        for tmc, group in heat.groupby("tmc_code")
    }
    rows: list[dict[str, object]] = []
    for item in projection.itertuples(index=False):
        row = pd.Series(item._asdict())
        period_start, period_end = settings.periods[str(row["period"])]
        observed_series = heat_lookup.get(str(row["tmc_code"]))
        if observed_series is None:
            continue
        time = np.array(
            sorted(
                float(value)
                for value in observed_series.index
                if period_start <= float(value) < period_end
            ),
            dtype=float,
        )
        if not len(time):
            continue
        observed = (
            pd.to_numeric(observed_series.reindex(time), errors="coerce")
            .to_numpy(dtype=float)
        )
        if np.isfinite(observed).sum() < 8:
            continue
        cbi, projected = reconstruction_curves(row, time)
        rows.append(
            {
                "corridor": row["corridor"],
                "period": row["period"],
                "net_link_id": row["net_link_id"],
                "tmc_code": row["tmc_code"],
                "projection_status": row["projection_status"],
                "speed_mae_cbi_mph": _mae(observed, cbi),
                "speed_mae_assignment_projection_mph": _mae(
                    observed, projected
                ),
            }
        )
    link_metrics = pd.DataFrame(rows)

    summary_rows: list[dict[str, object]] = []
    ready = projection[projection["projection_status"].eq("ready")].copy()
    for period in [*settings.periods, "ALL"]:
        group = ready if period == "ALL" else ready[ready["period"].eq(period)]
        speeds = (
            link_metrics
            if period == "ALL"
            else link_metrics[link_metrics["period"].eq(period)]
        )
        summary_rows.append(
            {
                "period": period,
                "n_ready_link_periods": int(len(group)),
                "P_A_mean": group["P_A"].mean(),
                "P_C_mean": group["P_C_vol"].mean(),
                "MAE_P_hr": (group["P_C_vol"] - group["P_A"]).abs().mean(),
                "MAE_DC": (
                    group["dc_dta_vol"] - group["DC_obs"]
                ).abs().mean(),
                "MAE_speed_CBI_mph": speeds["speed_mae_cbi_mph"].mean()
                if not speeds.empty
                else np.nan,
                "MAE_speed_assignment_projection_mph": speeds[
                    "speed_mae_assignment_projection_mph"
                ].mean()
                if not speeds.empty
                else np.nan,
            }
        )
    return link_metrics, pd.DataFrame(summary_rows)
