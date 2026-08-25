"""Refreshed state-transition congestion episode detector core.

This module contains detection only. Loading/QC belongs to ``preprocessing`` and
``qc``; physical/MAD/Huber filtering belongs to ``outliers``; calibration
belongs to ``calibration``. There is no standalone fallback workflow here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .config import WORKFLOW_PERIODS, period_duration_hours

TIME_PERIODS = WORKFLOW_PERIODS


@dataclass
class StateTransitionConfig:
    """Settings used by the refreshed state-transition detector."""

    min_duration_min: float = 15.0
    merge_gap_min: float = 10.0
    default_interval_minutes: float = 5.0
    freeflow_percentile: float = 85.0
    freeflow_min_mph: float = 35.0
    freeflow_max_mph: float = 85.0
    use_stage0_corridor_freeflow: bool = True
    default_per_lane_capacity_vphpl: float = 2000.0
    periods: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(WORKFLOW_PERIODS)
    )

    uncongested_max_duration_min: float = 30.0
    mild_max_duration_min: float = 60.0
    recurring_min_duration_min: float = 60.0
    severe_min_duration_min: float = 120.0
    severe_min_speed_ratio: float = 0.50
    mild_speed_ratio_lo: float = 0.70
    event_z_threshold: float = 2.5

    min_discharge_duration_min: float = 30.0
    min_discharge_positive_dq_share: float = 0.60
    allow_net_increasing_discharge: bool = True
    min_discharge_speed_gain_mph: float = 0.0
    require_congested_speed_for_discharge: bool = True
    t0_onset_search_intervals: int = 2
    allow_pre_qc_t0_speed_fallback: bool = True
    t3_recovery_search_intervals: int = 2
    allow_pre_qc_t3_speed_fallback: bool = True
    boundary_speed_max_factor: float = 1.25
    boundary_speed_max_mph: float = 90.0

    free_threshold_ratio: float = 0.85
    congested_threshold_ratio: float = 0.70
    breakdown_drop_mph: float = 10.0
    breakdown_window_intervals: int = 2
    recovery_rise_mph: float = 8.0
    max_noncongested_gap_intervals: int = 2
    allow_rolling_mean_extension: bool = True
    rolling_mean_extension_intervals: int = 6
    rolling_mean_extension_ratio: float = 1.0
    use_detection_smoothing: bool = True
    detection_smoothing_method: str = "savgol"
    detection_smoothing_window_intervals: int = 5
    detection_smoothing_polyorder: int = 2


def _period_for_timestamp(
    timestamp: object,
    periods: dict[str, tuple[int, int]] | None = None,
) -> str:
    stamp = pd.Timestamp(timestamp)
    minute = stamp.hour * 60 + stamp.minute + stamp.second / 60.0
    definitions = TIME_PERIODS if periods is None else periods
    for tag, (start_minute, end_minute) in definitions.items():
        if start_minute < end_minute:
            if start_minute <= minute < end_minute:
                return tag
        elif minute >= start_minute or minute < end_minute:
            return tag
    return "NT"


def episode_time_period_tag(
    t0_timestamp,
    t3_timestamp,
    interval_minutes: float = 5.0,
) -> str:
    """Return semicolon-separated period tags touched by an episode."""

    t0 = pd.Timestamp(t0_timestamp)
    t3 = pd.Timestamp(t3_timestamp)
    if pd.isna(t0) or pd.isna(t3):
        return ""
    if t3 < t0:
        t3 += pd.Timedelta(days=1)
    step = pd.Timedelta(minutes=max(float(interval_minutes), 1e-6))
    tags: list[str] = []
    current = t0
    while current <= t3:
        tag = _period_for_timestamp(current)
        if not tags or tags[-1] != tag:
            tags.append(tag)
        current += step
    end_tag = _period_for_timestamp(t3)
    if not tags or tags[-1] != end_tag:
        tags.append(end_tag)
    return ";".join(tags)


def estimate_interval_minutes(
    entity_df: pd.DataFrame,
    default: float = 5.0,
) -> float:
    diffs = (
        entity_df["timestamp"]
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(diffs.median()) if len(diffs) else float(default)


def detection_speed_series(
    entity_df: pd.DataFrame,
    config: StateTransitionConfig,
) -> np.ndarray:
    """Return the speed used only for detection and state classification."""

    raw = entity_df["speed_mph"].to_numpy(dtype=float)
    if not config.use_detection_smoothing:
        return raw.copy()
    filled = (
        pd.Series(raw)
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    if np.isfinite(filled).sum() < 3:
        return raw.copy()

    method = str(config.detection_smoothing_method).lower()
    if method == "rolling_mean":
        window = max(1, int(config.detection_smoothing_window_intervals))
        smoothed = (
            pd.Series(filled)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
    elif method == "savgol":
        from scipy.signal import savgol_filter

        window = max(3, int(config.detection_smoothing_window_intervals))
        if window % 2 == 0:
            window += 1
        n = len(filled)
        if window > n:
            window = n if n % 2 else n - 1
        polyorder = max(1, int(config.detection_smoothing_polyorder))
        if window <= polyorder:
            window = polyorder + 2
            if window % 2 == 0:
                window += 1
        if window > n or window < 3:
            return raw.copy()
        polyorder = min(polyorder, window - 1)
        smoothed = savgol_filter(
            filled,
            window_length=window,
            polyorder=polyorder,
            mode="interp",
        )
    else:
        raise ValueError(
            "Unknown detection_smoothing_method: "
            f"{config.detection_smoothing_method}"
        )
    smoothed = np.asarray(smoothed, dtype=float)
    smoothed[~np.isfinite(raw)] = np.nan
    return smoothed


def estimate_freeflow_speed(
    entity_df: pd.DataFrame,
    config: StateTransitionConfig,
    speed_override=None,
) -> float:
    if (
        config.use_stage0_corridor_freeflow
        and "corridor_freeflow_speed_mph" in entity_df
    ):
        freeflow = pd.to_numeric(
            entity_df["corridor_freeflow_speed_mph"], errors="coerce"
        ).to_numpy(dtype=float)
        freeflow = freeflow[np.isfinite(freeflow) & (freeflow > 0)]
        if len(freeflow):
            return float(
                np.clip(
                    np.nanmedian(freeflow),
                    config.freeflow_min_mph,
                    config.freeflow_max_mph,
                )
            )
    speed = (
        np.asarray(speed_override, dtype=float)
        if speed_override is not None
        else entity_df["speed_mph"].to_numpy(dtype=float)
    )
    valid = speed[np.isfinite(speed)]
    if len(valid) == 0:
        return float("nan")
    return float(
        np.clip(
            np.nanpercentile(valid, config.freeflow_percentile),
            config.freeflow_min_mph,
            config.freeflow_max_mph,
        )
    )


def freeflow_source_for_entity(
    entity_df: pd.DataFrame,
    config: StateTransitionConfig,
) -> str:
    if (
        config.use_stage0_corridor_freeflow
        and "corridor_freeflow_speed_mph" in entity_df
    ):
        freeflow = pd.to_numeric(
            entity_df["corridor_freeflow_speed_mph"], errors="coerce"
        ).to_numpy(dtype=float)
        if np.any(np.isfinite(freeflow) & (freeflow > 0)):
            return "stage0_corridor"
    return "percentile_fallback"


def classify_regime(
    duration_min: float,
    min_speed_mph: float,
    congested_threshold: float,
    z_score: float = 0.0,
    config: StateTransitionConfig | None = None,
) -> str:
    """Classify an episode for calibration preparation."""

    config = config or StateTransitionConfig()
    if not (
        np.isfinite(duration_min)
        and np.isfinite(min_speed_mph)
        and np.isfinite(congested_threshold)
    ):
        return "uncongested"
    if (
        duration_min < config.uncongested_max_duration_min
        or min_speed_mph >= congested_threshold
    ):
        return "uncongested"
    if abs(float(z_score)) > config.event_z_threshold:
        return "event"
    if (
        duration_min >= config.severe_min_duration_min
        and min_speed_mph
        < config.severe_min_speed_ratio * congested_threshold
    ):
        return "severe"
    if duration_min >= config.recurring_min_duration_min:
        return "recurring"
    return "mild"


def _is_physical_boundary_speed(
    value: float,
    config: StateTransitionConfig,
    freeflow_speed_mph: float,
) -> bool:
    upper = float(config.boundary_speed_max_mph)
    if np.isfinite(freeflow_speed_mph) and freeflow_speed_mph > 0.0:
        upper = min(
            upper,
            float(config.boundary_speed_max_factor)
            * float(freeflow_speed_mph),
        )
    return bool(
        np.isfinite(value)
        and 0.0 <= float(value) <= upper
    )


def _directional_speed_fallback(
    clean_speed: np.ndarray,
    pre_qc_speed: np.ndarray,
    *,
    boundary_idx: int,
    direction: Literal["backward", "forward"],
    search_intervals: int,
    freeflow_speed_mph: float,
    config: StateTransitionConfig,
    allow_pre_qc_fallback: bool,
    search_candidate_is_valid: Callable[[float, int], bool] | None = None,
    pre_qc_candidate_is_valid: Callable[[float, int], bool] | None = None,
) -> dict[str, object]:
    """Recover one speed without assigning any episode-boundary meaning.

    The original cleaned value is retained when physical.  Otherwise the
    helper searches cleaned values in the requested direction and finally may
    use the pre-QC value at the original index.  It reports a candidate index,
    but the caller alone decides whether that index changes a T0, T3, or any
    future boundary.
    """

    clean = np.asarray(clean_speed, dtype=float)
    pre_qc = np.asarray(pre_qc_speed, dtype=float)
    if not len(clean):
        raise ValueError("clean_speed must contain at least one value")
    if len(pre_qc) != len(clean):
        raise ValueError("clean_speed and pre_qc_speed must have equal length")
    if direction not in {"backward", "forward"}:
        raise ValueError("direction must be 'backward' or 'forward'")

    original_idx = min(max(int(boundary_idx), 0), len(clean) - 1)
    original_post_qc = clean[original_idx]
    original_pre_qc = pre_qc[original_idx]

    if _is_physical_boundary_speed(
        original_post_qc, config, freeflow_speed_mph
    ):
        return {
            "resolved_idx": original_idx,
            "speed_mph": float(original_post_qc),
            "source": "post_qc",
            "shift_intervals": 0,
            "original_speed_post_qc_mph": float(original_post_qc),
            "original_speed_pre_qc_mph": (
                float(original_pre_qc)
                if np.isfinite(original_pre_qc)
                else np.nan
            ),
        }

    step = -1 if direction == "backward" else 1
    max_search = max(0, int(search_intervals))
    for offset in range(1, max_search + 1):
        candidate_idx = original_idx + step * offset
        if candidate_idx < 0 or candidate_idx >= len(clean):
            break
        candidate_speed = clean[candidate_idx]
        if not _is_physical_boundary_speed(
            candidate_speed, config, freeflow_speed_mph
        ):
            continue
        if (
            search_candidate_is_valid is not None
            and not search_candidate_is_valid(
                float(candidate_speed), candidate_idx
            )
        ):
            continue
        return {
            "resolved_idx": candidate_idx,
            "speed_mph": float(candidate_speed),
            "source": "shifted_post_qc",
            "shift_intervals": candidate_idx - original_idx,
            "original_speed_post_qc_mph": np.nan,
            "original_speed_pre_qc_mph": (
                float(original_pre_qc)
                if np.isfinite(original_pre_qc)
                else np.nan
            ),
        }

    if (
        allow_pre_qc_fallback
        and _is_physical_boundary_speed(
            original_pre_qc, config, freeflow_speed_mph
        )
        and (
            pre_qc_candidate_is_valid is None
            or pre_qc_candidate_is_valid(
                float(original_pre_qc), original_idx
            )
        )
    ):
        return {
            "resolved_idx": original_idx,
            "speed_mph": float(original_pre_qc),
            "source": "pre_qc_fallback",
            "shift_intervals": 0,
            "original_speed_post_qc_mph": np.nan,
            "original_speed_pre_qc_mph": float(original_pre_qc),
        }

    return {
        "resolved_idx": original_idx,
        "speed_mph": np.nan,
        "source": "missing",
        "shift_intervals": 0,
        "original_speed_post_qc_mph": np.nan,
        "original_speed_pre_qc_mph": (
            float(original_pre_qc)
            if np.isfinite(original_pre_qc)
            else np.nan
        ),
    }


def _speed_arrays(entity_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.to_numeric(
        entity_df["speed_mph"], errors="coerce"
    ).to_numpy(dtype=float)
    pre_qc_column = (
        entity_df["speed_mph_pre_qc"]
        if "speed_mph_pre_qc" in entity_df
        else entity_df["speed_mph"]
    )
    pre_qc = pd.to_numeric(
        pre_qc_column, errors="coerce"
    ).to_numpy(dtype=float)
    return clean, pre_qc


def _resolve_t0_boundary(
    entity_df: pd.DataFrame,
    *,
    t0_idx: int,
    congested_threshold: float,
    freeflow_speed_mph: float,
    config: StateTransitionConfig,
) -> dict[str, object]:
    """Resolve the onset speed and let this T0 section move T0 backward."""

    clean, pre_qc = _speed_arrays(entity_df)
    fallback = _directional_speed_fallback(
        clean,
        pre_qc,
        boundary_idx=t0_idx,
        direction="backward",
        search_intervals=config.t0_onset_search_intervals,
        freeflow_speed_mph=freeflow_speed_mph,
        config=config,
        allow_pre_qc_fallback=config.allow_pre_qc_t0_speed_fallback,
        search_candidate_is_valid=lambda value, _index: (
            value >= congested_threshold
        ),
    )
    return {
        "t0_idx": fallback["resolved_idx"],
        "t0_boundary_speed_mph": fallback["speed_mph"],
        "t0_boundary_speed_source": fallback["source"],
        "t0_boundary_shift_intervals": fallback["shift_intervals"],
        "original_t0_speed_post_qc_mph": fallback[
            "original_speed_post_qc_mph"
        ],
        "original_t0_speed_pre_qc_mph": fallback[
            "original_speed_pre_qc_mph"
        ],
    }


def _resolve_t3_boundary(
    entity_df: pd.DataFrame,
    *,
    t2_idx: int,
    t3_idx: int,
    congested_threshold: float,
    freeflow_speed_mph: float,
    config: StateTransitionConfig,
) -> dict[str, object]:
    """Resolve the recovery speed and let this T3 section move T3 forward."""

    clean, pre_qc = _speed_arrays(entity_df)
    t2_speed = (
        clean[int(t2_idx)]
        if 0 <= int(t2_idx) < len(clean)
        else np.nan
    )

    def recovered(value: float, _index: int) -> bool:
        return bool(
            np.isfinite(t2_speed)
            and value >= congested_threshold
            and value - t2_speed >= config.min_discharge_speed_gain_mph
        )

    def usable_delta_v(value: float, _index: int) -> bool:
        return bool(
            np.isfinite(t2_speed)
            and value - t2_speed >= config.min_discharge_speed_gain_mph
        )

    fallback = _directional_speed_fallback(
        clean,
        pre_qc,
        boundary_idx=t3_idx,
        direction="forward",
        search_intervals=config.t3_recovery_search_intervals,
        freeflow_speed_mph=freeflow_speed_mph,
        config=config,
        allow_pre_qc_fallback=config.allow_pre_qc_t3_speed_fallback,
        search_candidate_is_valid=recovered,
        pre_qc_candidate_is_valid=usable_delta_v,
    )
    return {
        "t3_idx": fallback["resolved_idx"],
        "t3_boundary_speed_mph": fallback["speed_mph"],
        "t3_boundary_speed_source": fallback["source"],
        "t3_boundary_shift_intervals": fallback["shift_intervals"],
        "original_t3_speed_post_qc_mph": fallback[
            "original_speed_post_qc_mph"
        ],
        "original_t3_speed_pre_qc_mph": fallback[
            "original_speed_pre_qc_mph"
        ],
    }


def identify_discharge_window(
    speed: np.ndarray,
    flow: np.ndarray,
    t2_idx: int,
    t3_idx: int,
    congested_threshold: float,
    interval_minutes: float,
    config: StateTransitionConfig,
    boundary_speed_mph: float | None = None,
) -> dict[str, object]:
    """Measure the post-``t2`` congested discharge window."""

    invalid = {
        "discharge_start_idx": np.nan,
        "discharge_end_idx": np.nan,
        "discharge_duration_min": 0.0,
        "mu_obs_vphpl": np.nan,
        "discharge_positive_dq_share": np.nan,
        "discharge_net_dq": np.nan,
        "discharge_speed_gain_mph": np.nan,
        "discharge_trend_ok": False,
        "discharge_speed_recovery_ok": False,
        "discharge_window_valid": False,
    }
    start = int(t2_idx) + 1
    end = int(t3_idx)
    if end < start or start < 0:
        return invalid

    speed = np.asarray(speed, dtype=float)
    flow = np.asarray(flow, dtype=float)
    candidate = np.arange(start, min(end, len(speed) - 1) + 1)
    if not len(candidate):
        return invalid
    finite_speed = np.isfinite(speed[candidate])
    keep = (
        finite_speed & (speed[candidate] < congested_threshold)
        if config.require_congested_speed_for_discharge
        else finite_speed
    )
    kept = candidate[keep]
    if not len(kept):
        return invalid

    duration = float(len(kept) * interval_minutes)
    sampled_flow = flow[kept] if len(flow) else np.array([], dtype=float)
    finite_flow = sampled_flow[np.isfinite(sampled_flow)]
    mu_obs = (
        float(np.nanmedian(sampled_flow)) if len(finite_flow) else np.nan
    )
    if len(finite_flow) >= 2:
        changes = np.diff(finite_flow)
        changes = changes[np.isfinite(changes)]
        positive_share = (
            float(np.mean(changes >= 0.0)) if len(changes) else np.nan
        )
        net_change = float(finite_flow[-1] - finite_flow[0])
    else:
        positive_share = np.nan
        net_change = np.nan

    start_speed = speed[int(t2_idx)] if 0 <= int(t2_idx) < len(speed) else np.nan
    end_speed = (
        float(boundary_speed_mph)
        if boundary_speed_mph is not None
        and np.isfinite(boundary_speed_mph)
        else (
            speed[int(t3_idx)]
            if 0 <= int(t3_idx) < len(speed)
            else np.nan
        )
    )
    speed_gain = (
        float(end_speed - start_speed)
        if np.isfinite(start_speed) and np.isfinite(end_speed)
        else np.nan
    )
    trend_ok = (
        np.isfinite(positive_share)
        and positive_share >= config.min_discharge_positive_dq_share
    ) or (
        config.allow_net_increasing_discharge
        and np.isfinite(net_change)
        and net_change >= 0.0
    )
    recovery_ok = bool(
        np.isfinite(speed_gain)
        and speed_gain >= config.min_discharge_speed_gain_mph
    )
    valid = bool(
        duration >= config.min_discharge_duration_min
        and trend_ok
        and recovery_ok
        and np.isfinite(mu_obs)
    )
    return {
        "discharge_start_idx": int(kept[0]),
        "discharge_end_idx": int(kept[-1]),
        "discharge_duration_min": duration,
        "mu_obs_vphpl": mu_obs,
        "discharge_positive_dq_share": positive_share,
        "discharge_net_dq": net_change,
        "discharge_speed_gain_mph": speed_gain,
        "discharge_trend_ok": bool(trend_ok),
        "discharge_speed_recovery_ok": recovery_ok,
        "discharge_window_valid": valid,
    }


def merge_episode_bounds(
    bounds: list[tuple[int, int]],
    gap_intervals: int,
) -> list[tuple[int, int]]:
    if not bounds:
        return []
    merged = [sorted(bounds)[0]]
    for start, end in sorted(bounds)[1:]:
        last_start, last_end = merged[-1]
        if start - last_end - 1 <= gap_intervals:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _group_key(entity_df: pd.DataFrame) -> tuple[str, str, str]:
    return (
        str(entity_df["aggregation_mode"].iloc[0]),
        str(entity_df["entity_id"].iloc[0]),
        str(entity_df["date"].iloc[0]),
    )


def classify_state_transition_states(
    entity_df: pd.DataFrame,
    config: StateTransitionConfig,
) -> tuple[np.ndarray, float, float, float]:
    """Classify free, breakdown, congested, severe, and recovery states."""

    speed = detection_speed_series(entity_df, config)
    freeflow = estimate_freeflow_speed(
        entity_df,
        config,
        speed_override=speed,
    )
    congested_threshold = config.congested_threshold_ratio * freeflow
    severe_threshold = 0.5 * congested_threshold
    free_threshold = config.free_threshold_ratio * freeflow
    window = max(1, int(config.breakdown_window_intervals))

    states = np.full(len(speed), "normal", dtype=object)
    states[speed >= free_threshold] = "free"
    states[speed <= congested_threshold] = "congested"
    states[speed <= severe_threshold] = "severe_congestion"
    for index in range(window, len(speed)):
        if not (
            np.isfinite(speed[index])
            and np.isfinite(speed[index - window])
        ):
            continue
        if (
            speed[index - window] - speed[index]
            >= config.breakdown_drop_mph
            and speed[index] > congested_threshold
        ):
            states[index] = "breakdown"
        elif (
            speed[index] - speed[index - window]
            >= config.recovery_rise_mph
            and speed[index] > congested_threshold
        ):
            states[index] = "recovery"
    for index in range(len(speed) - 1):
        if (
            np.isfinite(speed[index])
            and np.isfinite(speed[index + 1])
            and speed[index] > congested_threshold
            and speed[index + 1] <= congested_threshold
        ):
            states[index] = "breakdown"
    return states, freeflow, congested_threshold, free_threshold


def _episode_from_bounds(
    entity_df: pd.DataFrame,
    start: int,
    end: int,
    threshold_used: float,
    freeflow_speed_mph: float,
    interval_minutes: float,
    config: StateTransitionConfig,
    notes: str,
) -> dict[str, object] | None:
    n = len(entity_df)
    start = max(0, int(start))
    end = min(n - 1, int(end))
    if n == 0 or end < start:
        return None
    speed = entity_df["speed_mph"].to_numpy(dtype=float)
    original_start = start
    original_end = end
    segment = speed[original_start : original_end + 1]
    if not np.isfinite(segment).any():
        return None

    # T2 remains the cleaned minimum of the originally detected episode.  A
    # later T0/T3 boundary recovery must not change which point defines T2.
    t2_idx = original_start + int(np.nanargmin(segment))
    t0_boundary = _resolve_t0_boundary(
        entity_df,
        t0_idx=original_start,
        congested_threshold=threshold_used,
        freeflow_speed_mph=freeflow_speed_mph,
        config=config,
    )
    start = int(t0_boundary["t0_idx"])
    t3_boundary = _resolve_t3_boundary(
        entity_df,
        t2_idx=t2_idx,
        t3_idx=original_end,
        congested_threshold=threshold_used,
        freeflow_speed_mph=freeflow_speed_mph,
        config=config,
    )
    end = int(t3_boundary["t3_idx"])
    segment = speed[start : end + 1]
    timestamps = entity_df["timestamp"].reset_index(drop=True)
    minimum = float(np.nanmin(segment))
    duration = float((end - start + 1) * interval_minutes)
    severity = float(
        np.nansum(np.maximum(0.0, threshold_used - segment))
        * interval_minutes
    )
    states, _, _, _ = classify_state_transition_states(entity_df, config)
    severe = np.flatnonzero(
        states[start : end + 1] == "severe_congestion"
    )
    severe_duration = (
        0.0
        if not len(severe)
        else float((severe[-1] - severe[0] + 1) * interval_minutes)
    )

    flow = (
        entity_df["flow"].to_numpy(dtype=float)
        if "flow" in entity_df
        else np.full(n, np.nan)
    )
    flow_segment = flow[start : end + 1]
    episode_demand = (
        np.nan
        if np.isnan(flow_segment).all()
        else float(np.nansum(flow_segment * interval_minutes / 60.0))
    )
    discharge = identify_discharge_window(
        speed,
        flow,
        t2_idx,
        end,
        threshold_used,
        interval_minutes,
        config,
        boundary_speed_mph=float(
            t3_boundary["t3_boundary_speed_mph"]
        ),
    )
    capacity = (
        entity_df["per_lane_hourly_capacity"].to_numpy(dtype=float)
        if "per_lane_hourly_capacity" in entity_df
        else np.full(n, np.nan)
    )
    capacity_segment = capacity[start : end + 1]
    per_lane_capacity = (
        np.nan
        if np.isnan(capacity_segment).all()
        else float(np.nanmedian(capacity_segment))
    )
    calibration_period = _period_for_timestamp(
        timestamps.iloc[t2_idx], config.periods
    )
    capacity_reference_hours = period_duration_hours(
        calibration_period, config.periods
    )
    capacity_volume = (
        per_lane_capacity * capacity_reference_hours
        if np.isfinite(per_lane_capacity)
        else np.nan
    )
    demand_capacity_ratio = (
        np.nan
        if (
            not np.isfinite(episode_demand)
            or not np.isfinite(per_lane_capacity)
            or per_lane_capacity <= 0
        )
        else float(episode_demand / capacity_volume)
    )
    regime = classify_regime(
        duration,
        minimum,
        threshold_used,
        config=config,
    )
    is_valid_for_mu = bool(
        duration >= config.uncongested_max_duration_min
        and minimum < threshold_used
        and discharge["discharge_window_valid"]
        and np.isfinite(discharge["mu_obs_vphpl"])
    )
    return {
        "method": "state_transition",
        "aggregation_mode": str(entity_df["aggregation_mode"].iloc[0]),
        "entity_id": str(entity_df["entity_id"].iloc[0]),
        "date": str(entity_df["date"].iloc[0]),
        "time_period_tag": episode_time_period_tag(
            timestamps.iloc[start],
            timestamps.iloc[end],
            interval_minutes,
        ),
        "calibration_period": calibration_period,
        "regime_label": regime,
        "is_valid_for_mu": is_valid_for_mu,
        "t0": str(entity_df["time_of_day"].iloc[start]),
        "t2": str(entity_df["time_of_day"].iloc[t2_idx]),
        "t3": str(entity_df["time_of_day"].iloc[end]),
        "original_t0": str(
            entity_df["time_of_day"].iloc[original_start]
        ),
        "original_t3": str(
            entity_df["time_of_day"].iloc[original_end]
        ),
        "t0_timestamp": str(timestamps.iloc[start]),
        "t2_timestamp": str(timestamps.iloc[t2_idx]),
        "t3_timestamp": str(timestamps.iloc[end]),
        "original_t0_timestamp": str(timestamps.iloc[original_start]),
        "original_t3_timestamp": str(timestamps.iloc[original_end]),
        "duration_min": duration,
        "duration_mad_zscore": np.nan,
        "severe_congestion_duration_min": severe_duration,
        "min_speed_mph": minimum,
        "mean_speed_mph": float(np.nanmean(segment)),
        "freeflow_speed_mph": float(freeflow_speed_mph),
        "freeflow_source": freeflow_source_for_entity(entity_df, config),
        "threshold_used": float(threshold_used),
        "speed_drop_mph": float(freeflow_speed_mph - minimum),
        "t0_boundary_speed_mph": t0_boundary["t0_boundary_speed_mph"],
        "t0_boundary_speed_source": t0_boundary[
            "t0_boundary_speed_source"
        ],
        "t0_boundary_shift_intervals": t0_boundary[
            "t0_boundary_shift_intervals"
        ],
        "original_t0_speed_post_qc_mph": t0_boundary[
            "original_t0_speed_post_qc_mph"
        ],
        "original_t0_speed_pre_qc_mph": t0_boundary[
            "original_t0_speed_pre_qc_mph"
        ],
        "t3_boundary_speed_mph": t3_boundary["t3_boundary_speed_mph"],
        "t3_boundary_speed_source": t3_boundary[
            "t3_boundary_speed_source"
        ],
        "t3_boundary_shift_intervals": t3_boundary[
            "t3_boundary_shift_intervals"
        ],
        "original_t3_speed_post_qc_mph": t3_boundary[
            "original_t3_speed_post_qc_mph"
        ],
        "original_t3_speed_pre_qc_mph": t3_boundary[
            "original_t3_speed_pre_qc_mph"
        ],
        "severity": severity,
        "episode_demand": episode_demand,
        "capacity_reference_hours": capacity_reference_hours,
        "capacity_volume_veh_per_lane": capacity_volume,
        "demand_capacity_basis": "episode_demand_over_period_capacity",
        **{
            key: discharge[key]
            for key in (
                "discharge_start_idx",
                "discharge_end_idx",
                "discharge_duration_min",
                "discharge_window_valid",
                "mu_obs_vphpl",
                "discharge_positive_dq_share",
                "discharge_net_dq",
                "discharge_speed_gain_mph",
                "discharge_trend_ok",
                "discharge_speed_recovery_ok",
            )
        },
        "per_lane_hourly_capacity": per_lane_capacity,
        "demand_capacity_ratio": demand_capacity_ratio,
        "demand_capacity_ratio_mad_zscore": np.nan,
        "num_intervals": int(end - start + 1),
        "measured_outlier_flag": False,
        "measured_outlier_reasons": "",
        "notes": notes,
        "_start_idx": start,
        "_end_idx": end,
        "_original_start_idx": original_start,
        "_original_end_idx": original_end,
        "_group_key": _group_key(entity_df),
    }


def _episodes_from_bounds(
    entity_df: pd.DataFrame,
    bounds: list[tuple[int, int]],
    threshold_used: float,
    freeflow_speed_mph: float,
    interval_minutes: float,
    config: StateTransitionConfig,
    notes: str,
) -> list[dict[str, object]]:
    gap = int(round(config.merge_gap_min / max(interval_minutes, 1e-6)))
    minimum = int(
        np.ceil(config.min_duration_min / max(interval_minutes, 1e-6))
    )
    episodes: list[dict[str, object]] = []
    for start, end in merge_episode_bounds(bounds, gap):
        if end - start + 1 < minimum:
            continue
        episode = _episode_from_bounds(
            entity_df,
            start,
            end,
            threshold_used,
            freeflow_speed_mph,
            interval_minutes,
            config,
            notes,
        )
        if episode:
            episodes.append(episode)
    return episodes


def _next_speed_below_threshold(
    speed: np.ndarray,
    index: int,
    threshold: float,
) -> bool:
    next_index = index + 1
    return bool(
        next_index < len(speed)
        and np.isfinite(speed[next_index])
        and speed[next_index] < threshold
    )


def _allow_rolling_mean_extension(
    speed: np.ndarray,
    start: int,
    current_index: int,
    congested_threshold: float,
    config: StateTransitionConfig,
) -> bool:
    if not config.allow_rolling_mean_extension:
        return False
    lookback = max(1, int(config.rolling_mean_extension_intervals))
    recent_start = max(start, current_index - lookback + 1)
    recent = speed[recent_start : current_index + 1]
    recent = recent[np.isfinite(recent)]
    return bool(
        len(recent)
        and float(np.mean(recent))
        <= config.rolling_mean_extension_ratio * congested_threshold
    )


def detect_state_transition(
    entity_df: pd.DataFrame,
    config: StateTransitionConfig,
) -> list[dict[str, object]]:
    """Detect congestion episodes using explicit traffic-state transitions."""

    detection_speed = detection_speed_series(entity_df, config)
    interval = estimate_interval_minutes(
        entity_df,
        config.default_interval_minutes,
    )
    states, freeflow, congested_threshold, _ = (
        classify_state_transition_states(entity_df, config)
    )
    bounds: list[tuple[int, int]] = []
    in_episode = False
    seen_entry = False
    start = 0
    last_supported = 0
    noncongested_gap = 0
    active_states = {
        "breakdown",
        "congested",
        "severe_congestion",
        "recovery",
    }
    entry_states = {"breakdown", "congested", "severe_congestion"}
    for index, state in enumerate(states):
        if not in_episode:
            if state in entry_states:
                seen_entry = True
            if seen_entry and _next_speed_below_threshold(
                detection_speed,
                index,
                congested_threshold,
            ):
                in_episode = True
                start = index
                last_supported = index
                noncongested_gap = 0
                seen_entry = False
            continue

        if state in active_states:
            last_supported = index
            noncongested_gap = 0
            continue
        noncongested_gap += 1
        if noncongested_gap <= config.max_noncongested_gap_intervals:
            last_supported = index
            continue
        if _allow_rolling_mean_extension(
            detection_speed,
            start,
            index,
            congested_threshold,
            config,
        ):
            last_supported = index
            continue

        exit_start = max(start, index - noncongested_gap + 1)
        bounds.append(
            (
                start,
                min(max(start, exit_start), len(detection_speed) - 1),
            )
        )
        in_episode = False
        seen_entry = state in entry_states
        last_supported = 0
        noncongested_gap = 0

    if in_episode:
        bounds.append((start, last_supported))

    notes = (
        "detector=refreshed_state_transition;"
        f"interval_minutes={interval};"
        f"free_ratio={config.free_threshold_ratio};"
        f"congested_ratio={config.congested_threshold_ratio};"
        f"breakdown_drop_mph={config.breakdown_drop_mph};"
        f"recovery_rise_mph={config.recovery_rise_mph};"
        f"max_gap_intervals={config.max_noncongested_gap_intervals};"
        f"smoothing={config.detection_smoothing_method};"
        "boundary_rule=next_below_congested_start;"
        "t3_rule=explicit_state_machine_exit;"
        "outlier_filter=external_refreshed_physical_mad_huber"
    )
    return _episodes_from_bounds(
        entity_df,
        bounds,
        congested_threshold,
        freeflow,
        interval,
        config,
        notes,
    )
