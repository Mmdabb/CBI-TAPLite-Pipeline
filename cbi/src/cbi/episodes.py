from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import PipelineSettings, WORKFLOW_PERIODS
from .outliers import apply_outlier_screen
from .state_transition import StateTransitionConfig, detect_state_transition


LOGGER = logging.getLogger("cbi")

EPISODE_CANDIDATE_COLUMNS = [
    "episode_id",
    "detector",
    "aggregation_mode",
    "sensor_uid",
    "tmc_code",
    "corridor",
    "link_id",
    "network_link_id",
    "network_from_node_id",
    "network_to_node_id",
    "network_link_type",
    "network_mapping_status",
    "direction",
    "road_order",
    "length_mi",
    "lanes",
    "lanes_source",
    "capacity_source",
    "reference_speed_source",
    "date",
    "weekday",
    "period",
    "t0_hour",
    "t2_hour",
    "t3_hour",
    "original_t0_hour",
    "original_t3_hour",
    "P_hr",
    "duration_min",
    "threshold_used",
    "freeflow_speed_mph",
    "min_speed_mph",
    "mean_speed_mph",
    "episode_demand",
    "qdf",
    "plf",
    "qdf_episode_demand",
    "qdf_period_demand",
    "qdf_clip_start_hour",
    "qdf_clip_end_hour",
    "qdf_period_start_hour",
    "qdf_period_end_hour",
    "qdf_period_duration_hours",
    "qdf_integration_rule",
    "capacity_reference_hours",
    "capacity_volume_veh_per_lane",
    "demand_capacity_ratio",
    "demand_capacity_basis",
    "flow_synthetic",
    "demand_is_proxy",
    "per_lane_hourly_capacity",
    "mu_obs_vphpl",
    "discharge_duration_min",
    "discharge_window_valid",
    "t0_boundary_speed_mph",
    "t0_boundary_speed_source",
    "t0_boundary_shift_intervals",
    "original_t0_speed_post_qc_mph",
    "original_t0_speed_pre_qc_mph",
    "t3_boundary_speed_mph",
    "t3_boundary_speed_source",
    "t3_boundary_shift_intervals",
    "original_t3_speed_post_qc_mph",
    "original_t3_speed_pre_qc_mph",
    "is_valid_for_mu",
    "magnitude",
    "severity",
    "regime_label",
]

FILTER_SUMMARY_COLUMNS = [
    "n_hard_flags",
    "n_soft_flags",
    "measured_outlier_flag",
    "measured_outlier_reasons",
    "is_clean_valid_episode",
]


def _odd_interval_count(target_minutes: float, interval_minutes: float, minimum: int = 3) -> int:
    count = max(minimum, int(round(target_minutes / max(interval_minutes, 1e-6))))
    return count if count % 2 else count + 1


def detector_config(settings: PipelineSettings) -> StateTransitionConfig:
    """Adapt the refreshed five-minute detector settings to the input interval."""

    interval = float(settings.interval_minutes)
    return StateTransitionConfig(
        min_duration_min=settings.minimum_episode_minutes,
        merge_gap_min=settings.merge_gap_minutes,
        default_interval_minutes=interval,
        periods={**WORKFLOW_PERIODS, **settings.periods},
        min_discharge_duration_min=settings.minimum_discharge_minutes,
        min_discharge_positive_dq_share=0.60,
        allow_net_increasing_discharge=True,
        min_discharge_speed_gain_mph=0.0,
        require_congested_speed_for_discharge=True,
        t0_onset_search_intervals=2,
        allow_pre_qc_t0_speed_fallback=True,
        t3_recovery_search_intervals=2,
        allow_pre_qc_t3_speed_fallback=True,
        boundary_speed_max_factor=1.25,
        boundary_speed_max_mph=90.0,
        use_stage0_corridor_freeflow=True,
        free_threshold_ratio=0.85,
        congested_threshold_ratio=settings.cutoff_ratio,
        breakdown_drop_mph=10.0,
        breakdown_window_intervals=max(1, int(round(10.0 / interval))),
        recovery_rise_mph=10.0,
        max_noncongested_gap_intervals=max(1, int(round(15.0 / interval))),
        allow_rolling_mean_extension=True,
        rolling_mean_extension_intervals=max(1, int(round(30.0 / interval))),
        rolling_mean_extension_ratio=1.0,
        use_detection_smoothing=True,
        detection_smoothing_method="savgol",
        detection_smoothing_window_intervals=_odd_interval_count(25.0, interval),
        detection_smoothing_polyorder=2,
    )


def _entity_frame(group: pd.DataFrame, average_weekday: bool) -> pd.DataFrame:
    frame = group.sort_values("datetime").reset_index(drop=True)
    speed_column = (
        "speed_mph_clean_repaired"
        if "speed_mph_clean_repaired" in frame
        else "speed_mph"
    )
    timestamp = pd.to_datetime(frame["datetime"])
    return pd.DataFrame(
        {
            "entity_id": frame["sensor_uid"].astype(str),
            "timestamp": timestamp,
            "date": "" if average_weekday else timestamp.dt.date.astype(str),
            "time_of_day": timestamp.dt.strftime("%H:%M:%S"),
            "speed_mph": pd.to_numeric(frame[speed_column], errors="coerce"),
            "speed_mph_pre_qc": pd.to_numeric(
                (
                    frame["speed_mph_raw"]
                    if "speed_mph_raw" in frame
                    else frame[speed_column]
                ),
                errors="coerce",
            ),
            "corridor_freeflow_speed_mph": pd.to_numeric(
                frame["corridor_freeflow_speed_mph"], errors="coerce"
            ),
            "flow": pd.to_numeric(frame["flow_vph"], errors="coerce"),
            "per_lane_hourly_capacity": pd.to_numeric(
                frame["fd_capacity_vphpl"], errors="coerce"
            ),
            "aggregation_mode": (
                "multiday_average" if average_weekday else "daily"
            ),
        }
    )


def _minute_of_day(timestamp: object) -> float:
    stamp = pd.Timestamp(timestamp)
    return stamp.hour * 60.0 + stamp.minute + stamp.second / 60.0


def _period_window(
    timestamp: object,
    period: str,
    settings: PipelineSettings,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the concrete half-open NVTA period containing ``timestamp``."""

    stamp = pd.Timestamp(timestamp)
    start_minute, end_minute = settings.periods[period]
    midnight = stamp.normalize()
    start = midnight + pd.Timedelta(minutes=start_minute)
    end = midnight + pd.Timedelta(minutes=end_minute)
    if start_minute >= end_minute:
        if stamp < end:
            start -= pd.Timedelta(days=1)
        else:
            end += pd.Timedelta(days=1)
    return start, end


def _integrated_flow(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_minutes: float,
) -> float:
    """Integrate interval flow over the exact overlap with ``[start, end)``."""

    if end <= start or frame.empty:
        return 0.0
    timestamps = pd.to_datetime(frame["datetime"])
    bin_end = timestamps + pd.Timedelta(minutes=float(interval_minutes))
    overlap_start = timestamps.where(timestamps >= start, start)
    overlap_end = bin_end.where(bin_end <= end, end)
    overlap_hours = (
        (overlap_end - overlap_start).dt.total_seconds().clip(lower=0.0)
        / 3600.0
    )
    flow = pd.to_numeric(frame["flow_vph"], errors="coerce")
    return float(
        np.nansum(flow.to_numpy(dtype=float) * overlap_hours.to_numpy(dtype=float))
    )


def period_clipped_qdf(
    frame: pd.DataFrame,
    *,
    period: str,
    t0_timestamp: object,
    t2_timestamp: object,
    t3_timestamp: object,
    settings: PipelineSettings,
) -> dict[str, object]:
    """Calculate QDF/PLF from only the episode share inside T2's period."""

    period_start, period_end = _period_window(t2_timestamp, period, settings)
    episode_start = max(pd.Timestamp(t0_timestamp), period_start)
    episode_end = min(pd.Timestamp(t3_timestamp), period_end)
    period_demand = _integrated_flow(
        frame, period_start, period_end, settings.interval_minutes
    )
    clipped_episode_demand = _integrated_flow(
        frame, episode_start, episode_end, settings.interval_minutes
    )
    qdf = (
        clipped_episode_demand / period_demand
        if period_demand > 0.0 and np.isfinite(clipped_episode_demand)
        else np.nan
    )
    duration_hours = (period_end - period_start).total_seconds() / 3600.0
    plf = (
        1.0 / (qdf * duration_hours)
        if np.isfinite(qdf) and qdf > 0.0 and duration_hours > 0.0
        else np.nan
    )
    return {
        "qdf": float(qdf) if np.isfinite(qdf) else np.nan,
        "plf": float(plf) if np.isfinite(plf) else np.nan,
        "qdf_episode_demand": clipped_episode_demand,
        "qdf_period_demand": period_demand,
        "qdf_clip_start_hour": _minute_of_day(episode_start) / 60.0,
        "qdf_clip_end_hour": _minute_of_day(episode_end) / 60.0,
        "qdf_period_start_hour": _minute_of_day(period_start) / 60.0,
        "qdf_period_end_hour": _minute_of_day(period_end) / 60.0,
        "qdf_period_duration_hours": duration_hours,
        "qdf_integration_rule": "episode_period_overlap_over_period_total",
    }


def detect_episode_candidates(
    observations: pd.DataFrame,
    settings: PipelineSettings,
    *,
    average_weekday: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Return every state-transition episode before any episode screening."""

    logger = logger or LOGGER
    work = observations.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if not average_weekday:
        work = work[work["datetime"].dt.dayofweek < 5].copy()
    cfg = detector_config(settings)
    rows: list[dict[str, object]] = []
    grouping = ["sensor_uid"] if average_weekday else ["sensor_uid", "date"]
    grouper: str | list[str] = grouping[0] if len(grouping) == 1 else grouping
    for _, group in work.groupby(grouper, sort=False):
        frame = _entity_frame(group, average_weekday)
        detected = detect_state_transition(frame, cfg)
        if not detected:
            continue
        metadata = group.iloc[0]
        for item in detected:
            t0_min = _minute_of_day(item["t0_timestamp"])
            t2_min = _minute_of_day(item["t2_timestamp"])
            t3_min = _minute_of_day(item["t3_timestamp"])
            original_t0_min = _minute_of_day(
                item.get("original_t0_timestamp", item["t0_timestamp"])
            )
            original_t3_min = _minute_of_day(
                item.get("original_t3_timestamp", item["t3_timestamp"])
            )
            period = settings.period_for_minute(t2_min)
            if period is None:
                continue
            qdf_metrics = period_clipped_qdf(
                group,
                period=period,
                t0_timestamp=item["t0_timestamp"],
                t2_timestamp=item["t2_timestamp"],
                t3_timestamp=item["t3_timestamp"],
                settings=settings,
            )
            item.update(
                {
                    "detector": "state_transition",
                    "sensor_uid": str(metadata["sensor_uid"]),
                    "link_id": int(metadata["link_id"]),
                    "tmc_code": str(metadata.get("tmc_code", metadata["link_id"])),
                    "network_link_id": metadata.get("network_link_id", np.nan),
                    "network_from_node_id": metadata.get(
                        "network_from_node_id", np.nan
                    ),
                    "network_to_node_id": metadata.get(
                        "network_to_node_id", np.nan
                    ),
                    "network_link_type": metadata.get(
                        "network_link_type", np.nan
                    ),
                    "network_mapping_status": str(
                        metadata.get("network_mapping_status", "unmapped")
                    ),
                    "corridor": str(metadata["corridor"]),
                    "direction": str(metadata.get("direction", "")),
                    "road_order": metadata.get("road_order", metadata["link_id"]),
                    "length_mi": float(metadata.get("length_mi", 0.5)),
                    "lanes": float(metadata.get("lanes", 1.0)),
                    "lanes_source": str(
                        metadata.get("lanes_source", "")
                    ),
                    "capacity_source": str(
                        metadata.get("capacity_source", "")
                    ),
                    "reference_speed_source": str(
                        metadata.get("reference_speed_source", "")
                    ),
                    "flow_synthetic": bool(metadata.get("flow_synthetic", True)),
                    "demand_is_proxy": bool(
                        metadata.get("flow_synthetic", True)
                    ),
                    "source_format": str(metadata.get("source_format", "")),
                    "date": "Weekday" if average_weekday else str(metadata["date"]),
                    "weekday": 0
                    if average_weekday
                    else int(pd.Timestamp(metadata["datetime"]).weekday()),
                    "calibration_period": period,
                    "period": period,
                    "t0_hour": t0_min / 60.0,
                    "t2_hour": t2_min / 60.0,
                    "t3_hour": t3_min / 60.0,
                    "original_t0_hour": original_t0_min / 60.0,
                    "original_t3_hour": original_t3_min / 60.0,
                    "P_hr": float(item["duration_min"]) / 60.0,
                    **qdf_metrics,
                    "m": (
                        (t2_min - t0_min) / float(item["duration_min"])
                        if float(item["duration_min"]) > 0
                        else 0.5
                    ),
                    "magnitude": (
                        float(item["threshold_used"])
                        / max(float(item["min_speed_mph"]), 1e-6)
                        - 1.0
                    ),
                }
            )
            rows.append(item)
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        logger.warning("No state-transition episodes detected")
        return episodes
    episodes["episode_id"] = [
        f"state_transition__{sensor}__{date}__{index:04d}"
        for index, (sensor, date) in enumerate(
            zip(episodes["sensor_uid"], episodes["date"]), start=1
        )
    ]
    logger.info(
        "State-transition detection: %s pre-filter episode candidates",
        len(episodes),
    )
    return episodes


def screen_episode_candidates(
    candidates: pd.DataFrame,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply the sole physical/MAD/Huber screen to detected candidates."""

    logger = logger or LOGGER
    if candidates.empty:
        return candidates.copy()
    scored = apply_outlier_screen(candidates)
    logger.info(
        "Episode screening: %s candidates; %s clean valid; %s outliers",
        len(scored),
        int(scored["is_clean_valid_episode"].sum()),
        int(scored["measured_outlier_flag"].sum()),
    )
    return scored


def detect_episodes(
    observations: pd.DataFrame,
    settings: PipelineSettings,
    *,
    average_weekday: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compatibility entry point for detection followed by the authoritative screen."""

    candidates = detect_episode_candidates(
        observations,
        settings,
        average_weekday=average_weekday,
        logger=logger,
    )
    return screen_episode_candidates(candidates, logger=logger)


def episode_candidate_audit(candidates: pd.DataFrame) -> pd.DataFrame:
    """Stable pre-filter output containing every detected TMC episode."""

    columns = EPISODE_CANDIDATE_COLUMNS
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in candidates]
    out = candidates[available].copy()
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out[columns]


def episode_filter_audit(episodes: pd.DataFrame) -> pd.DataFrame:
    """Self-contained product for every candidate and its filtering decision.

    ``episode_id`` remains a one-to-one key with the pre-filter candidate
    table. The audit repeats the complete candidate payload, adds every
    individual filter flag, and concludes with the aggregate decision fields.
    """

    if episodes.empty:
        return pd.DataFrame(
            columns=EPISODE_CANDIDATE_COLUMNS + FILTER_SUMMARY_COLUMNS
        )
    flag_columns = sorted(
        column
        for column in episodes.columns
        if column.startswith("flag_")
    )
    columns = (
        EPISODE_CANDIDATE_COLUMNS
        + flag_columns
        + FILTER_SUMMARY_COLUMNS
    )
    out = episodes.copy()
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out[columns].copy()
