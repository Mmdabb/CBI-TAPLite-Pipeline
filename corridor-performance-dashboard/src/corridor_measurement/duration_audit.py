"""TMC-aligned congestion-duration and D/C reconciliation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def threshold_duration_hours(
    frame: pd.DataFrame,
    *,
    speed_column: str,
    threshold_column: str,
    interval_minutes: int,
) -> float:
    """Count valid intervals whose speed is at or below the threshold."""

    speed = pd.to_numeric(frame[speed_column], errors="coerce")
    threshold = pd.to_numeric(frame[threshold_column], errors="coerce")
    valid = speed.notna() & threshold.notna()
    if not valid.any():
        return np.nan
    return float((speed[valid] <= threshold[valid]).sum() * interval_minutes / 60.0)


def selected_episode_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Collapse non-overlapping selected CBI episodes to one TMC-period row."""

    columns = [
        "corridor",
        "tmc_code",
        "period",
        "cbi_selected_episode_count",
        "cbi_selected_episode_p_hours",
        "cbi_selected_episode_doc",
        "cbi_selected_episode_demand_veh_per_lane",
        "cbi_capacity_reference_hours",
        "cbi_capacity_volume_veh_per_lane",
    ]
    if episodes.empty:
        return pd.DataFrame(columns=columns)
    frame = episodes.copy()
    selected = frame.get("reconstruction_selected", False)
    if not isinstance(selected, pd.Series):
        selected = pd.Series(selected, index=frame.index)
    selected = selected.astype("string").str.lower().isin(("true", "1", "yes"))
    frame = frame[selected].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["period"] = frame["period"].astype("string").str.upper()
    for column in (
        "P_hr",
        "demand_capacity_ratio",
        "episode_demand",
        "capacity_reference_hours",
        "capacity_volume_veh_per_lane",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(["corridor", "tmc_code", "period"], dropna=False)
    result = grouped.agg(
        cbi_selected_episode_count=("P_hr", "size"),
        cbi_selected_episode_p_hours=("P_hr", "sum"),
        cbi_selected_episode_doc=("demand_capacity_ratio", "max"),
        cbi_selected_episode_demand_veh_per_lane=("episode_demand", "sum"),
        cbi_capacity_reference_hours=("capacity_reference_hours", "max"),
        cbi_capacity_volume_veh_per_lane=("capacity_volume_veh_per_lane", "max"),
    ).reset_index()
    return result[columns]


def build_tmc_period_audit(
    measurement: pd.DataFrame,
    cbi_profiles: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    periods: Mapping[str, Mapping[str, object]],
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Build one duration and D/C comparison row per TMC and period."""

    measurement = measurement.copy()
    measurement["period"] = measurement["period"].astype("string").str.upper()
    measurement["tmc_code"] = measurement["tmc_code"].astype("string")
    cbi_profiles = cbi_profiles.copy()
    cbi_profiles["tmc_code"] = cbi_profiles["tmc_code"].astype("string")
    joined = measurement.merge(
        cbi_profiles[
            ["corridor", "tmc_code", "t_min", "speed_qvdf_model", "congestion_threshold_mph"]
        ].drop_duplicates(["corridor", "tmc_code", "t_min"]),
        on=["corridor", "tmc_code", "t_min"],
        how="left",
        validate="many_to_one",
    )
    joined["common_threshold_mph"] = pd.to_numeric(
        joined["cbi_tmc_congestion_threshold_mph"], errors="coerce"
    ).combine_first(pd.to_numeric(joined["congestion_threshold_mph"], errors="coerce"))

    rows: list[dict[str, object]] = []
    keys = ["corridor", "tmc_code", "period"]
    for (corridor, tmc_code, period), frame in joined.groupby(keys, sort=True):
        definition = periods[str(period).lower()]
        period_hours = (
            float(definition["end_min"]) - float(definition["start_min"])
        ) / 60.0
        first = frame.sort_values("t_min").iloc[0]
        observed_p = threshold_duration_hours(
            frame,
            speed_column="observed_tmc_speed_mph",
            threshold_column="common_threshold_mph",
            interval_minutes=interval_minutes,
        )
        qvdf_p = threshold_duration_hours(
            frame,
            speed_column="speed_qvdf_model",
            threshold_column="common_threshold_mph",
            interval_minutes=interval_minutes,
        )
        taplite_p = threshold_duration_hours(
            frame,
            speed_column="model_tmc_speed_mph",
            threshold_column="common_threshold_mph",
            interval_minutes=interval_minutes,
        )
        rows.append(
            {
                "corridor": corridor,
                "tmc_code": tmc_code,
                "period": period,
                "direction": first.get("direction"),
                "road_order": first.get("road_order"),
                "period_hours": period_hours,
                "common_threshold_mph": first.get("common_threshold_mph"),
                "observed_same_threshold_p_hours": observed_p,
                "cbi_qvdf_same_threshold_p_hours": qvdf_p,
                "taplite_same_threshold_p_hours": taplite_p,
                "taplite_kernel_p_hours": first.get("taplite_period_p_hours"),
                "taplite_kernel_doc": first.get("taplite_period_doc"),
                "taplite_period_volume": first.get("taplite_period_volume"),
                "gmns_link_count": first.get("gmns_link_count"),
                "cbi_qvdf_minus_taplite_profile_p_hours": qvdf_p - taplite_p,
                "cbi_qvdf_vs_taplite_profile_abs_error_hours": abs(qvdf_p - taplite_p),
            }
        )
    result = pd.DataFrame(rows)
    episode_summary = selected_episode_summary(episodes)
    result = result.merge(episode_summary, on=keys, how="left", validate="one_to_one")
    for column in (
        "taplite_kernel_p_hours",
        "taplite_kernel_doc",
        "cbi_selected_episode_p_hours",
        "cbi_selected_episode_doc",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["cbi_selected_p_minus_qvdf_profile_p_hours"] = (
        result["cbi_selected_episode_p_hours"] - result["cbi_qvdf_same_threshold_p_hours"]
    )
    result["taplite_kernel_p_minus_profile_p_hours"] = (
        result["taplite_kernel_p_hours"] - result["taplite_same_threshold_p_hours"]
    )
    result["cbi_doc_minus_taplite_doc"] = (
        result["cbi_selected_episode_doc"] - result["taplite_kernel_doc"]
    )
    result["same_profile_duration_within_15_min"] = (
        result["cbi_qvdf_vs_taplite_profile_abs_error_hours"] <= interval_minutes / 60.0
    )
    return result


def summarize_duration_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-threshold duration agreement by corridor and period."""

    def summarize(group: pd.DataFrame) -> pd.Series:
        duration_error = pd.to_numeric(
            group["cbi_qvdf_vs_taplite_profile_abs_error_hours"], errors="coerce"
        )
        doc_difference = pd.to_numeric(group["cbi_doc_minus_taplite_doc"], errors="coerce")
        paired = duration_error.notna() & doc_difference.notna()
        correlation = (
            duration_error[paired].corr(doc_difference[paired].abs())
            if paired.sum() >= 2
            else np.nan
        )
        return pd.Series(
            {
                "tmc_period_count": len(group),
                "mean_abs_profile_duration_error_hours": duration_error.mean(),
                "median_abs_profile_duration_error_hours": duration_error.median(),
                "max_abs_profile_duration_error_hours": duration_error.max(),
                "within_15_min_share": group["same_profile_duration_within_15_min"].mean(),
                "mean_cbi_selected_doc": pd.to_numeric(
                    group["cbi_selected_episode_doc"], errors="coerce"
                ).mean(),
                "mean_taplite_doc": pd.to_numeric(
                    group["taplite_kernel_doc"], errors="coerce"
                ).mean(),
                "abs_duration_error_vs_abs_doc_difference_correlation": correlation,
            }
        )

    return audit.groupby(["corridor", "period"], sort=True).apply(summarize).reset_index()
