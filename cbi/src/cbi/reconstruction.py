from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineSettings
from .episodes import detector_config
from .fundamental_diagram import inverse_s3_flow
from .state_transition import detection_speed_series


def predict_duration_hours(demand_capacity_ratio: float, parameters: dict) -> float:
    return float(
        parameters["f_d"] * max(float(demand_capacity_ratio), 0.0) ** parameters["n"]
    )


def predict_minimum_speed(
    cutoff_mph: float,
    duration_hours: float,
    parameters: dict,
) -> float:
    return float(
        np.clip(
            cutoff_mph
            / (
                1.0
                + parameters["f_p"]
                * max(float(duration_hours), 1e-6) ** parameters["s"]
            ),
            1.0,
            cutoff_mph,
        )
    )


def predicted_bounds_about_t2(
    t0_observed_hour: float,
    t2_observed_hour: float,
    t3_observed_hour: float,
    predicted_duration_hour: float,
) -> tuple[float, float, float]:
    """Keep detected t2 fixed and preserve the detected left/right asymmetry."""

    observed_duration = max(t3_observed_hour - t0_observed_hour, 1e-6)
    left_share = np.clip(
        (t2_observed_hour - t0_observed_hour) / observed_duration, 0.05, 0.95
    )
    duration = max(float(predicted_duration_hour), 1e-3)
    return (
        float(t2_observed_hour - left_share * duration),
        float(t2_observed_hour),
        float(t2_observed_hour + (1.0 - left_share) * duration),
    )


def _smoothstep(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def qvdf_queue_shape(
    values: np.ndarray,
    peak_fraction: float,
) -> np.ndarray:
    """Return the asymmetric, unit-peak QVDF queue shape.

    ``values`` is normalized over the complete ``t0``--``t3`` episode and
    ``peak_fraction`` locates the detected ``t2`` within that interval.  The
    small interior clipping only protects the two limiting cases; the returned
    curve is explicitly zero at both episode boundaries.
    """

    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    rho = float(np.clip(peak_fraction, 1e-6, 1.0 - 1e-6))
    shape = np.zeros_like(x)
    interior = (x > 0.0) & (x < 1.0)
    if interior.any():
        xi = x[interior]
        shape[interior] = np.power(xi / rho, 4.0 * rho) * np.power(
            (1.0 - xi) / (1.0 - rho),
            4.0 * (1.0 - rho),
        )
    return np.clip(shape, 0.0, 1.0)


def select_nonoverlapping_episodes(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all compatible accepted episodes and audit overlap exclusions.

    Candidate priority is the approved deterministic rule: lowest minimum
    speed, then longest duration, then earliest t2.  Higher-priority episodes
    are accepted first; a lower-priority candidate is excluded only when its
    open interval overlaps one already accepted for the same link and period.
    """

    if episodes.empty:
        empty = episodes.copy()
        return empty, empty.assign(overlap_resolution_reason=pd.Series(dtype=str))

    candidates = episodes.copy()
    for column in ("t0_hour", "t2_hour", "t3_hour", "min_speed_mph", "P_hr"):
        candidates[column] = pd.to_numeric(candidates.get(column), errors="coerce")
    if "P_hr" not in episodes:
        candidates["P_hr"] = candidates["t3_hour"] - candidates["t0_hour"]
    candidates["_source_order"] = np.arange(len(candidates))
    candidates = candidates.sort_values(
        ["min_speed_mph", "P_hr", "t2_hour", "_source_order"],
        ascending=[True, False, True, True],
        kind="mergesort",
        na_position="last",
    )

    accepted_indices: list[object] = []
    dropped_rows: list[pd.Series] = []
    accepted_intervals: list[tuple[float, float, object]] = []
    for index, row in candidates.iterrows():
        t0 = float(row["t0_hour"])
        t3 = float(row["t3_hour"])
        conflict = next(
            (
                accepted_index
                for accepted_t0, accepted_t3, accepted_index in accepted_intervals
                if t0 < accepted_t3 and t3 > accepted_t0
            ),
            None,
        )
        if conflict is None and np.isfinite(t0) and np.isfinite(t3) and t3 > t0:
            accepted_indices.append(index)
            accepted_intervals.append((t0, t3, index))
        else:
            excluded = row.copy()
            excluded["overlap_resolution_reason"] = (
                f"overlaps_higher_priority_episode:{conflict}"
                if conflict is not None
                else "invalid_episode_bounds"
            )
            dropped_rows.append(excluded)

    accepted = (
        candidates.loc[accepted_indices]
        .sort_values(["t0_hour", "t3_hour", "_source_order"], kind="mergesort")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )
    dropped = pd.DataFrame(dropped_rows)
    if dropped.empty:
        dropped = candidates.iloc[0:0].copy()
        dropped["overlap_resolution_reason"] = pd.Series(dtype=str)
    dropped = dropped.drop(columns="_source_order", errors="ignore").reset_index(drop=True)
    return accepted, dropped


def reconstruction_episode_selection_audit(episodes: pd.DataFrame) -> pd.DataFrame:
    """Label accepted episodes used or excluded by reconstruction overlap logic."""

    if episodes.empty:
        result = episodes.copy()
        result["reconstruction_selected"] = pd.Series(dtype=bool)
        result["reconstruction_selection_reason"] = pd.Series(dtype=str)
        return result
    accepted_only = episodes[
        episodes.get(
            "is_clean_valid_episode",
            pd.Series(False, index=episodes.index),
        ).fillna(False)
    ].copy()
    if accepted_only.empty:
        accepted_only["reconstruction_selected"] = pd.Series(dtype=bool)
        accepted_only["reconstruction_selection_reason"] = pd.Series(dtype=str)
        return accepted_only
    rows: list[pd.DataFrame] = []
    group_columns = [
        column for column in ("link_id", "period") if column in accepted_only
    ]
    groups = (
        accepted_only.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [((), accepted_only)]
    )
    for _, group in groups:
        selected, dropped = select_nonoverlapping_episodes(group)
        selected["reconstruction_selected"] = True
        selected["reconstruction_selection_reason"] = (
            "accepted_nonoverlapping_episode"
        )
        if not dropped.empty:
            dropped["reconstruction_selected"] = False
            dropped["reconstruction_selection_reason"] = dropped[
                "overlap_resolution_reason"
            ]
        rows.extend([selected, dropped])
    return pd.concat(rows, ignore_index=True, sort=False).drop(
        columns="overlap_resolution_reason", errors="ignore"
    )


def detection_smoothed_speed(
    speed: np.ndarray | pd.Series,
    settings: PipelineSettings,
) -> np.ndarray:
    """Use the detector's sole smoothing implementation for displayed/model input."""

    values = np.asarray(speed, dtype=float)
    frame = pd.DataFrame({"speed_mph": values})
    return detection_speed_series(frame, detector_config(settings))


def smoothstep_anchor_curve(
    time_minutes: np.ndarray,
    anchors: list[tuple[float, float]],
) -> np.ndarray:
    """Interpolate minute/speed anchors with a monotone smoothstep per segment."""

    time = np.asarray(time_minutes, dtype=float)
    finite = [
        (float(timestamp), float(speed))
        for timestamp, speed in anchors
        if np.isfinite(timestamp) and np.isfinite(speed)
    ]
    if not finite:
        return np.full(len(time), np.nan)
    unique: dict[float, float] = {}
    for timestamp, speed in sorted(finite):
        unique[timestamp] = speed
    points = sorted(unique.items())
    model = np.interp(
        time,
        [point[0] for point in points],
        [point[1] for point in points],
    )
    for (start_t, start_v), (end_t, end_v) in zip(points[:-1], points[1:]):
        segment = (time >= start_t) & (time <= end_t)
        if segment.any() and end_t > start_t:
            model[segment] = start_v + (end_v - start_v) * _smoothstep(
                (time[segment] - start_t) / (end_t - start_t)
            )
    return model


def reconstruct_episode_speed(
    time_minutes: np.ndarray,
    *,
    t0_hour: float,
    t2_hour: float,
    t3_hour: float,
    minimum_speed_mph: float,
    cutoff_mph: float,
    free_flow_mph: float,
    length_mi: float,
    discharge_vphpl: float,
    window_start_minute: float | None = None,
    window_end_minute: float | None = None,
    left_anchor_mph: float | None = None,
    right_anchor_mph: float | None = None,
) -> np.ndarray:
    """Asymmetric QVDF queue-to-speed reconstruction peaked at detected t2.

    Inside ``t0``--``t3`` the queue uses the complete-episode QVDF shape.  The
    regions before ``t0`` and after ``t3`` retain smoothstep shoulders between
    their outer anchors and the congestion cutoff.
    """

    minute = np.asarray(time_minutes, dtype=float)
    hour = minute / 60.0
    t2 = float(np.clip(t2_hour, t0_hour + 1e-6, t3_hour - 1e-6))
    cutoff = float(cutoff_mph)
    free_flow = float(max(free_flow_mph, cutoff))
    vt2 = float(np.clip(minimum_speed_mph, 1.0, cutoff))
    length = float(max(length_mi, 1e-6))
    mu = float(max(discharge_vphpl, 1e-6))
    running_time_at_cutoff = length / max(cutoff, 1e-6)
    maximum_queue = max(
        mu * (length / max(vt2, 1e-6) - running_time_at_cutoff), 0.0
    )

    queue = np.zeros(len(hour), dtype=float)
    inside = (hour >= t0_hour) & (hour <= t3_hour)
    duration = max(float(t3_hour) - float(t0_hour), 1e-6)
    rho = (t2 - float(t0_hour)) / duration
    if inside.any():
        x = (hour[inside] - float(t0_hour)) / duration
        queue[inside] = maximum_queue * qvdf_queue_shape(x, rho)

    model = np.full(len(hour), free_flow, dtype=float)
    model[inside] = length / (
        queue[inside] / mu + running_time_at_cutoff
    )

    start_minute = (
        float(np.nanmin(minute))
        if window_start_minute is None
        else float(window_start_minute)
    )
    end_minute = (
        float(np.nanmax(minute)) if window_end_minute is None else float(window_end_minute)
    )
    left_anchor = float(
        free_flow if left_anchor_mph is None else np.clip(left_anchor_mph, 1.0, free_flow)
    )
    right_anchor = float(
        free_flow
        if right_anchor_mph is None
        else np.clip(right_anchor_mph, 1.0, free_flow)
    )
    before = (minute >= start_minute) & (hour < t0_hour)
    if before.any():
        fraction = (minute[before] - start_minute) / max(
            t0_hour * 60.0 - start_minute, 1e-6
        )
        model[before] = left_anchor + (cutoff - left_anchor) * _smoothstep(fraction)
    after = (hour > t3_hour) & (minute <= end_minute)
    if after.any():
        fraction = (minute[after] - t3_hour * 60.0) / max(
            end_minute - t3_hour * 60.0, 1e-6
        )
        model[after] = cutoff + (right_anchor - cutoff) * _smoothstep(fraction)
    return np.clip(model, 1.0, free_flow)


def reconstruct_daily_profile(
    profile: pd.DataFrame,
    episodes: pd.DataFrame,
    parameter_lookup: dict[tuple[int, str], dict[str, object]],
    settings: PipelineSettings,
) -> np.ndarray:
    """Build one full-day curve from anchors and every accepted episode.

    Smoothstep interpolation supplies the uncongested shoulders and all gaps
    between anchors.  Each accepted, non-overlapping episode then replaces its
    own ``t0``--``t3`` segment with the QVDF queue-to-speed reconstruction.
    """

    ordered = profile.sort_values("t_min")
    time = ordered["t_min"].to_numpy(dtype=float)
    observed = pd.to_numeric(ordered["speed_mph"], errors="coerce").to_numpy(dtype=float)
    link_id = int(ordered["link_id"].iloc[0])
    free_flow = float(ordered["corridor_freeflow_speed_mph"].iloc[0])
    anchors: list[tuple[float, float]] = [
        (float(time[0]), float(min(observed[0], free_flow)))
    ]
    if episodes.empty or "link_id" not in episodes.columns:
        link_episodes = episodes.iloc[0:0].copy()
    else:
        accepted_mask = (
            episodes["is_clean_valid_episode"].fillna(False)
            if "is_clean_valid_episode" in episodes
            else pd.Series(False, index=episodes.index)
        )
        link_episodes = episodes[episodes["link_id"].eq(link_id) & accepted_mask].copy()
    selected_episodes: list[pd.Series] = []
    for period, (start, end) in settings.periods.items():
        mask = (time >= start) & (time < end)
        if not mask.any():
            continue
        period_times = time[mask]
        period_speeds = observed[mask]
        start_anchor = float(period_times[0])
        end_anchor = float(period_times[-1])
        anchors.append((start_anchor, float(min(period_speeds[0], free_flow))))
        candidates = (
            link_episodes[link_episodes["period"].eq(period)]
            if not link_episodes.empty and "period" in link_episodes
            else link_episodes.iloc[0:0].copy()
        )
        selected, _ = select_nonoverlapping_episodes(candidates)
        for _, episode in selected.iterrows():
            t0 = max(float(episode["t0_hour"]), start / 60.0)
            t2 = float(episode["t2_hour"])
            t3 = min(float(episode["t3_hour"]), end / 60.0)
            if t0 < t2 < t3:
                cutoff = float(episode["threshold_used"])
                anchors.extend([(t0 * 60.0, cutoff), (t3 * 60.0, cutoff)])
                selected_episodes.append(episode)
        anchors.append((end_anchor, float(min(period_speeds[-1], free_flow))))
    anchors.append((float(time[-1]), float(min(observed[-1], free_flow))))

    unique: dict[int, tuple[float, float]] = {}
    for timestamp, speed in sorted(anchors):
        unique[int(round(timestamp))] = (float(timestamp), float(speed))
    points = [unique[key] for key in sorted(unique)]
    model = smoothstep_anchor_curve(time, points)
    for episode in selected_episodes:
        t0 = float(episode["t0_hour"])
        t2 = float(episode["t2_hour"])
        t3 = float(episode["t3_hour"])
        segment = (time >= t0 * 60.0) & (time <= t3 * 60.0)
        if not segment.any():
            continue
        mu = float(pd.to_numeric(episode.get("mu_obs_vphpl"), errors="coerce"))
        if not np.isfinite(mu) or mu <= 0:
            mu = float(pd.to_numeric(episode.get("per_lane_hourly_capacity"), errors="coerce"))
        if not np.isfinite(mu) or mu <= 0:
            mu = 1800.0
        length = float(pd.to_numeric(episode.get("length_mi"), errors="coerce"))
        if not np.isfinite(length) or length <= 0:
            length = float(
                pd.to_numeric(ordered.get("length_mi"), errors="coerce").median()
            )
        if not np.isfinite(length) or length <= 0:
            length = 0.5
        model[segment] = reconstruct_episode_speed(
            time[segment],
            t0_hour=t0,
            t2_hour=t2,
            t3_hour=t3,
            minimum_speed_mph=float(episode["min_speed_mph"]),
            cutoff_mph=float(episode["threshold_used"]),
            free_flow_mph=free_flow,
            length_mi=length,
            discharge_vphpl=mu,
            window_start_minute=t0 * 60.0,
            window_end_minute=t3 * 60.0,
            left_anchor_mph=float(episode["threshold_used"]),
            right_anchor_mph=float(episode["threshold_used"]),
        )
    return np.clip(model, 1.0, free_flow)


def conserve_flow_from_speed(
    speed_mph: np.ndarray,
    *,
    vf_mph: float,
    kc_vpmpl: float,
    s3_m: float,
    capacity_vphpl: float,
    demand_veh_per_lane: float,
    interval_hours: float,
    max_iterations: int = 80,
) -> tuple[np.ndarray, float, float]:
    """Closest inverse-S3 flow satisfying cumulative demand exactly."""

    base = inverse_s3_flow(
        speed_mph, vf_mph, kc_vpmpl, s3_m, capacity_vphpl
    )
    if len(base) == 0:
        return base, 0.0, 0.0
    target_sum = float(demand_veh_per_lane) / max(interval_hours, 1e-9)
    lower, upper = -float(capacity_vphpl), float(capacity_vphpl)
    multiplier = 0.0
    for _ in range(max_iterations):
        multiplier = 0.5 * (lower + upper)
        total = float(
            np.clip(base + multiplier, 0.0, capacity_vphpl).sum()
        )
        if abs(total - target_sum) <= 1e-6 * max(target_sum, 1.0):
            break
        if total < target_sum:
            lower = multiplier
        else:
            upper = multiplier
    adjusted = np.clip(base + multiplier, 0.0, capacity_vphpl)
    return (
        adjusted,
        float(multiplier),
        float(adjusted.sum() * interval_hours),
    )
