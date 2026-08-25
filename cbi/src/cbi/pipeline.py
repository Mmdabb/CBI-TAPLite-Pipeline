from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import calibrate_episodes
from .config import CorridorSpec, PipelineSettings, period_duration_hours
from .episodes import (
    detect_episode_candidates,
    episode_candidate_audit,
    episode_filter_audit,
    screen_episode_candidates,
)
from .fundamental_diagram import (
    attach_fd_context,
    calibrate_measured_fd,
    prepare_synthetic_flow,
)
from .outputs import (
    CONSERVED_FLOW_COLUMNS,
    HANDOFF_COLUMNS,
    attach_link_identifiers,
    build_handoff,
    calibration_quality,
    format_episode_table,
    quality_gates,
    stage0_table,
    table6_congestion_stats,
    table7_calibrated,
    table8_gamma,
    timeseries_quality,
)
from .reconstruction import reconstruction_episode_selection_audit
from .output_contract import create_step_directories, step_dir
from .preprocessing import (
    build_average_weekday,
    load_corridor,
    qkv_audit,
)
from .qc import run_qc


_FIGURE_LOCK = threading.Lock()


def _logger(
    output_dir: Path,
    log_name: str = "run.log",
    *,
    console_output: bool = True,
) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cbi.{output_dir.resolve()}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(
        output_dir / log_name, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    if console_output:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    return logger


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    run_id: str,
) -> None:
    """Write a CSV with its producing run identifier embedded in every row."""

    out = frame.copy()
    if "run_id" in out:
        out = out.drop(columns=["run_id"])
    out.insert(0, "run_id", run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _qc_window(interval_minutes: int) -> int:
    count = max(3, int(round(55.0 / max(interval_minutes, 1))))
    return count if count % 2 else count + 1


def _write_current_format_outputs(
    output_dir: Path,
    *,
    spec: CorridorSpec,
    settings: PipelineSettings,
    observations: pd.DataFrame,
    average: pd.DataFrame,
    fd: pd.DataFrame,
    episodes: pd.DataFrame,
    average_episodes: pd.DataFrame,
    selected_applied: pd.DataFrame,
    source_is_average: bool,
    run_id: str,
) -> dict[str, int]:
    paths = create_step_directories(output_dir)
    stage0 = stage0_table(fd, observations, settings)
    average_formatted = format_episode_table(
        average_episodes,
        average,
        fd,
        settings,
        average_weekday=True,
    )
    daily_formatted = format_episode_table(
        episodes,
        observations,
        fd,
        settings,
        average_weekday=source_is_average,
    )
    table6 = table6_congestion_stats(
        daily_formatted,
        episodes,
        settings,
        data_basis=(
            "average_weekday" if source_is_average else "daily"
        ),
    )
    table7 = table7_calibrated(episodes, selected_applied)
    table8 = table8_gamma(
        episodes, selected_applied, fd, observations
    )
    calibration = calibration_quality(
        episodes, selected_applied, settings
    )
    time_quality = timeseries_quality(observations, average, settings)
    gates = quality_gates(calibration, time_quality)
    handoff, parameters, accounting = build_handoff(
        average,
        average_episodes,
        selected_applied,
        fd,
        observations,
        settings,
    )
    table6 = attach_link_identifiers(table6, observations)
    table7 = attach_link_identifiers(table7, observations)
    table8 = attach_link_identifiers(table8, observations)
    time_quality = attach_link_identifiers(time_quality, observations)

    link_reference = (
        observations.groupby("link_id", as_index=False)
        .agg(
            sensor_uid=("sensor_uid", "first"),
            tmc_code=("tmc_code", "first"),
            corridor=("corridor", "first"),
            direction=("direction", "first"),
            road_order=("road_order", "first"),
            length_mi=("length_mi", "first"),
            lanes=("lanes", "first"),
            lanes_source=("lanes_source", "first"),
            capacity_prior_vphpl=("capacity_prior_vphpl", "first"),
            capacity_source=("capacity_source", "first"),
            reference_speed_mph=("reference_speed_mph", "first"),
            reference_speed_source=("reference_speed_source", "first"),
            network_link_id=("network_link_id", "first"),
            network_from_node_id=("network_from_node_id", "first"),
            network_to_node_id=("network_to_node_id", "first"),
            network_path_link_count=("network_path_link_count", "first"),
            network_match_distance_ft=("network_match_distance_ft", "first"),
            network_bearing_diff_deg=("network_bearing_diff_deg", "first"),
            network_match_score=("network_match_score", "first"),
            network_match_available_weight=("network_match_available_weight", "first"),
            network_geometry_overlap_score=("network_geometry_overlap_score", "first"),
            network_road_name_agreement_score=("network_road_name_agreement_score", "first"),
            network_direction_compatibility_score=("network_direction_compatibility_score", "first"),
            network_functional_class_compatibility_score=(
                "network_functional_class_compatibility_score", "first"
            ),
            network_relative_position_score=("network_relative_position_score", "first"),
            network_observation_quality_score=("network_observation_quality_score", "first"),
            network_length_compatibility_score=("network_length_compatibility_score", "first"),
            network_link_tmc_rank=("network_link_tmc_rank", "first"),
            network_tmc_link_rank=("network_tmc_link_rank", "first"),
            network_node_pair_tmc_rank=("network_node_pair_tmc_rank", "first"),
            network_selected_for_node_pair_lookup=(
                "network_selected_for_node_pair_lookup", "first"
            ),
            network_mapping_status=("network_mapping_status", "first"),
        )
        .sort_values("link_id")
    )
    profile_columns = [
        "link_id",
        "sensor_uid",
        "tmc_code",
        "corridor",
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_mapping_status",
        "direction",
        "road_order",
        "length_mi",
        "lanes",
        "lanes_source",
        "capacity_prior_vphpl",
        "capacity_source",
        "reference_speed_mph",
        "reference_speed_source",
        "corridor_freeflow_speed_mph",
        "fd_capacity_vphpl",
        "fd_vc_mph",
        "t_min",
        "speed_mph",
        "speed_mph_raw",
        "flow_vph",
        "n_days",
        "flow_synthetic",
    ]
    profile = average[
        [column for column in profile_columns if column in average]
    ].rename(
        columns={
            "speed_mph": "avg_weekday_speed_mph",
            "speed_mph_raw": "avg_weekday_speed_mph_pre_qc",
            "flow_vph": "avg_weekday_flow_veh_per_hr_lane",
            "flow_synthetic": "demand_is_proxy",
            "corridor_freeflow_speed_mph": (
                "free_flow_speed_model_mph"
            ),
            "fd_capacity_vphpl": "capacity_vphpl",
            "fd_vc_mph": "speed_at_capacity_mph",
        }
    )

    _write_csv(
        link_reference,
        paths["input_qc"] / "link_reference.csv",
        run_id=run_id,
    )
    _write_csv(
        stage0,
        paths["fundamental_diagram"] / "link_fd_context.csv",
        run_id=run_id,
    )
    _write_csv(
        profile,
        paths["profiles"] / "average_weekday_profile.csv",
        run_id=run_id,
    )
    _write_csv(
        handoff[HANDOFF_COLUMNS],
        paths["handoff"] / "average_weekday_time_dependent.csv",
        run_id=run_id,
    )
    _write_csv(
        handoff[CONSERVED_FLOW_COLUMNS],
        paths["handoff"] / "qvdf_conserved_flow.csv",
        run_id=run_id,
    )
    _write_csv(
        parameters,
        paths["handoff"] / "link_period_qvdf_parameters.csv",
        run_id=run_id,
    )
    _write_csv(
        accounting,
        paths["handoff"] / "corridor_accounting.csv",
        run_id=run_id,
    )
    _write_csv(
        calibration,
        paths["quality"] / "qvdf_validation_by_period.csv",
        run_id=run_id,
    )
    _write_csv(
        time_quality[
        [
            "link_id",
            "n_weekdays",
            "n_time_bins",
            "expected_time_bins",
            "full_day_complete",
            "smooth_vs_raw_R2",
            "smooth_vs_raw_RMSE",
        ]
        + [
            column
            for column in (
                "sensor_uid",
                "tmc_code",
                "network_link_id",
                "network_from_node_id",
                "network_to_node_id",
                "network_mapping_status",
            )
            if column in time_quality
        ]
        ],
        paths["quality"] / "profile_smoothing_quality_by_link.csv",
        run_id=run_id,
    )
    _write_csv(
        time_quality[
        [
            "link_id",
            "n_weekdays",
            "day2day_RMSE_mph",
            "day2day_R2",
            "day2day_CV",
        ]
        + [
            column
            for column in (
                "sensor_uid",
                "tmc_code",
                "network_link_id",
                "network_from_node_id",
                "network_to_node_id",
                "network_mapping_status",
            )
            if column in time_quality
        ]
        ],
        paths["quality"] / "daily_speed_variability_by_link.csv",
        run_id=run_id,
    )
    _write_csv(
        gates,
        paths["quality"] / "quality_gates.csv",
        run_id=run_id,
    )
    _write_csv(
        average_formatted,
        paths["tables"] / "average_weekday_episode_period_summary.csv",
        run_id=run_id,
    )
    _write_csv(
        table6,
        paths["tables"] / "congestion_frequency_by_link_period.csv",
        run_id=run_id,
    )
    _write_csv(
        table7,
        paths["tables"] / "qvdf_link_period_validation.csv",
        run_id=run_id,
    )
    _write_csv(
        table8,
        paths["tables"] / "qvdf_derived_gamma_by_weekday.csv",
        run_id=run_id,
    )

    return {
        "stage0_rows": len(stage0),
        "formatted_episode_rows": len(average_formatted),
        "clean_episode_rows": int(
            episodes["is_clean_valid_episode"].sum()
        )
        if not episodes.empty
        else 0,
        "table7_rows": len(table7),
        "handoff_rows": len(handoff),
    }


@dataclass
class EpisodeDetectionInputs:
    input_audit: dict[str, object]
    raw_qc_summary: dict[str, object]
    average_qc_summary: dict[str, object]
    modeled: pd.DataFrame
    fd: pd.DataFrame
    average: pd.DataFrame
    calibration_observations: pd.DataFrame
    episode_candidates: pd.DataFrame
    average_episode_candidates: pd.DataFrame
    source_is_average: bool


def prepare_episode_detection_candidates(
    spec: CorridorSpec,
    settings: PipelineSettings,
    logger: logging.Logger,
) -> EpisodeDetectionInputs:
    """Run the one authoritative preprocessing/QC/FD/detection data flow."""

    raw, input_audit = load_corridor(spec, settings, logger)
    raw_qc, raw_qc_summary = run_qc(
        raw,
        hampel_window=_qc_window(settings.interval_minutes),
        dataset_kind="raw"
        if spec.source == "inrix_folder"
        else "multiday_average",
    )
    raw_qc["speed_mph"] = pd.to_numeric(
        raw_qc["speed_mph_clean_repaired"], errors="coerce"
    )
    if spec.data_mode == "speed_only":
        modeled, fd = prepare_synthetic_flow(
            raw_qc,
            capacity_vphpl=None,
            default_free_flow_mph=spec.free_flow_mph,
        )
    else:
        fd = calibrate_measured_fd(raw_qc)
        modeled = attach_fd_context(raw_qc, fd)
        modeled["density_vpm"] = modeled["flow_vph"] / modeled[
            "speed_mph"
        ].where(modeled["speed_mph"] > 1.0)
    modeled = attach_fd_context(modeled, fd)
    logger.info(
        "Preprocessing/QC: rows=%s, links=%s, repaired=%s, pass=%.2f%%",
        f"{len(modeled):,}",
        modeled["sensor_uid"].nunique(),
        int(raw_qc_summary["n_isolated_repaired"]),
        100.0 * float(raw_qc_summary["qc_pass_repaired_rate"]),
    )

    average = build_average_weekday(modeled)
    average_qc, average_qc_summary = run_qc(
        average,
        hampel_window=_qc_window(settings.interval_minutes),
        dataset_kind="multiday_average",
    )
    average_qc["speed_mph"] = pd.to_numeric(
        average_qc["speed_mph_clean_repaired"], errors="coerce"
    )
    average_qc = attach_fd_context(average_qc, fd)
    average_qc["density_vpm"] = average_qc["flow_vph"] / average_qc[
        "speed_mph"
    ].where(average_qc["speed_mph"] > 1.0)

    source_is_average = spec.source == "avgweekday_csv"
    calibration_observations = average_qc if source_is_average else modeled
    episode_candidates = detect_episode_candidates(
        calibration_observations,
        settings,
        average_weekday=source_is_average,
        logger=logger,
    )
    if source_is_average:
        average_episode_candidates = episode_candidates.copy()
    else:
        average_episode_candidates = detect_episode_candidates(
            average_qc, settings, average_weekday=True, logger=logger
        )
    return EpisodeDetectionInputs(
        input_audit=input_audit,
        raw_qc_summary=raw_qc_summary,
        average_qc_summary=average_qc_summary,
        modeled=modeled,
        fd=fd,
        average=average_qc,
        calibration_observations=calibration_observations,
        episode_candidates=episode_candidates,
        average_episode_candidates=average_episode_candidates,
        source_is_average=source_is_average,
    )


def _accepted_episode_candidate_audit(
    screened: pd.DataFrame,
) -> pd.DataFrame:
    if screened.empty:
        return episode_candidate_audit(screened)
    return episode_candidate_audit(
        screened[screened["is_clean_valid_episode"].fillna(False)]
    )


def run_episode_detection_only(
    spec: CorridorSpec,
    output_dir: Path,
    settings: PipelineSettings | None = None,
    *,
    console_output: bool = True,
) -> dict[str, object]:
    """Refresh detected episodes, screening audits, and accepted-only handoffs."""

    settings = settings or PipelineSettings()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Episode staging output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = create_step_directories(output_dir)
    run_id = f"episode-refresh-{uuid.uuid4().hex}"
    logger = _logger(
        paths["metadata"],
        "episode_detection_refresh.log",
        console_output=console_output,
    )
    started = time.perf_counter()
    logger.info(
        "Episode-only refresh: %s | periods=%s",
        spec.key,
        settings.periods,
    )
    prepared = prepare_episode_detection_candidates(spec, settings, logger)
    daily_screened = screen_episode_candidates(
        prepared.episode_candidates,
        logger=logger,
    )
    average_screened = screen_episode_candidates(
        prepared.average_episode_candidates,
        logger=logger,
    )
    daily_pre_filter = episode_candidate_audit(prepared.episode_candidates)
    average_pre_filter = episode_candidate_audit(
        prepared.average_episode_candidates
    )
    daily_accepted = _accepted_episode_candidate_audit(daily_screened)
    average_accepted = _accepted_episode_candidate_audit(average_screened)
    _write_csv(
        daily_pre_filter,
        paths["episode_detection"] / "daily_episode_candidates.csv",
        run_id=run_id,
    )
    _write_csv(
        average_pre_filter,
        paths["episode_detection"]
        / "average_weekday_episode_candidates.csv",
        run_id=run_id,
    )
    _write_csv(
        episode_filter_audit(daily_screened),
        paths["episode_filtering"] / "daily_episode_filter_audit.csv",
        run_id=run_id,
    )
    _write_csv(
        episode_filter_audit(average_screened),
        paths["episode_filtering"]
        / "average_weekday_episode_filter_audit.csv",
        run_id=run_id,
    )
    _write_csv(
        daily_accepted,
        paths["episode_filtering"] / "daily_episodes_accepted.csv",
        run_id=run_id,
    )
    _write_csv(
        average_accepted,
        paths["episode_filtering"]
        / "average_weekday_episodes_accepted.csv",
        run_id=run_id,
    )
    manifest = {
        "status": "PASS",
        "run_id": run_id,
        "corridor": spec.key,
        "mode": "episode_detection_only",
        "periods_minutes": settings.periods,
        "daily_candidates_pre_filter": len(daily_pre_filter),
        "daily_episodes_accepted": len(daily_accepted),
        "daily_episodes_rejected": len(daily_pre_filter) - len(daily_accepted),
        "average_weekday_candidates_pre_filter": len(average_pre_filter),
        "average_weekday_episodes_accepted": len(average_accepted),
        "average_weekday_episodes_rejected": (
            len(average_pre_filter) - len(average_accepted)
        ),
        "input_audit": prepared.input_audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json(
        paths["metadata"] / "episode_detection_refresh.json",
        manifest,
    )
    logger.info(
        "Episode-only refresh complete: daily=%s/%s accepted, "
        "average_weekday=%s/%s accepted",
        len(daily_accepted),
        len(daily_pre_filter),
        len(average_accepted),
        len(average_pre_filter),
    )
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()
    return manifest


def run_corridor(
    spec: CorridorSpec,
    output_root: Path,
    settings: PipelineSettings | None = None,
    *,
    generate_figures: bool = True,
) -> dict[str, object]:
    """Run the single integrated flow from source observations to 3_CBI outputs."""

    settings = settings or PipelineSettings()
    output_dir = Path(output_root).resolve() / spec.key
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output already exists for {spec.key}: {output_dir}. "
            "Use a new output root or remove the intentional prior run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = create_step_directories(output_dir)
    run_id = f"corridor-{uuid.uuid4().hex}"
    logger = _logger(paths["metadata"])
    started = time.perf_counter()
    logger.info("================ %s [%s] ================", spec.name, spec.key)
    logger.info(
        "Core: refreshed QC -> state-transition episodes -> physical/MAD/Huber filter "
        "-> robust bounded QVDF calibration"
    )

    prepared = prepare_episode_detection_candidates(spec, settings, logger)
    input_audit = prepared.input_audit
    raw_qc_summary = prepared.raw_qc_summary
    average_qc_summary = prepared.average_qc_summary
    modeled = prepared.modeled
    fd = prepared.fd
    average_qc = prepared.average
    calibration_observations = prepared.calibration_observations
    episode_candidates = prepared.episode_candidates
    average_episode_candidates = prepared.average_episode_candidates
    source_is_average = prepared.source_is_average
    episodes = screen_episode_candidates(episode_candidates, logger=logger)
    primary_basis = "average_weekday" if source_is_average else "daily"
    all_calibration, applied = calibrate_episodes(
        episodes,
        settings,
        data_basis=primary_basis,
    )
    logger.info(
        "Calibration: candidate fits=%s, applied link-period fits=%s",
        len(all_calibration),
        len(applied),
    )

    if source_is_average:
        average_episodes = episodes.copy()
        average_calibration = all_calibration.copy()
        average_applied = applied.copy()
    else:
        average_episodes = screen_episode_candidates(
            average_episode_candidates, logger=logger
        )
        average_calibration, average_applied = calibrate_episodes(
            average_episodes,
            settings,
            data_basis="average_weekday",
        )
    # One authoritative QVDF parameter set: daily evidence wins whenever raw
    # daily data exist. The average-weekday branch remains an audit product,
    # not a competing final calibration.
    selected_applied = applied.copy()
    counts = _write_current_format_outputs(
        output_dir,
        spec=spec,
        settings=settings,
        observations=calibration_observations,
        average=average_qc,
        fd=fd,
        episodes=episodes,
        average_episodes=average_episodes,
        selected_applied=selected_applied,
        source_is_average=source_is_average,
        run_id=run_id,
    )

    _write_csv(
        episode_candidate_audit(episode_candidates),
        paths["episode_detection"] / "daily_episode_candidates.csv",
        run_id=run_id,
    )
    _write_csv(
        episode_candidate_audit(average_episode_candidates),
        paths["episode_detection"]
        / "average_weekday_episode_candidates.csv",
        run_id=run_id,
    )
    _write_csv(
        episode_filter_audit(episodes),
        paths["episode_filtering"] / "daily_episode_filter_audit.csv",
        run_id=run_id,
    )
    _write_csv(
        episode_filter_audit(average_episodes),
        paths["episode_filtering"]
        / "average_weekday_episode_filter_audit.csv",
        run_id=run_id,
    )
    _write_csv(
        _accepted_episode_candidate_audit(episodes),
        paths["episode_filtering"] / "daily_episodes_accepted.csv",
        run_id=run_id,
    )
    _write_csv(
        _accepted_episode_candidate_audit(average_episodes),
        paths["episode_filtering"]
        / "average_weekday_episodes_accepted.csv",
        run_id=run_id,
    )
    _write_csv(
        reconstruction_episode_selection_audit(average_episodes),
        paths["handoff"]
        / "average_weekday_reconstruction_episode_selection.csv",
        run_id=run_id,
    )
    _write_csv(
        all_calibration,
        paths["calibration"] / "primary_qvdf_fit_catalog.csv",
        run_id=run_id,
    )
    if not source_is_average:
        _write_csv(
            average_calibration,
            paths["calibration"]
            / "average_weekday_diagnostic_fit_catalog.csv",
            run_id=run_id,
        )
    selected_public = attach_link_identifiers(
        selected_applied, calibration_observations
    )
    _write_csv(
        selected_public,
        paths["calibration"] / "qvdf_selected_parameters.csv",
        run_id=run_id,
    )
    _json(
        paths["input_qc"] / "input_audit.json",
        {"run_id": run_id, **input_audit},
    )
    _json(
        paths["input_qc"] / "qc_summary.json",
        {
            "run_id": run_id,
            "source": raw_qc_summary,
            "average_weekday": average_qc_summary,
        },
    )
    _json(
        paths["quality"] / "qkv_consistency_audit.json",
        {
            "run_id": run_id,
            "modeled": qkv_audit(modeled, "post_qc_fd_context"),
            "average_weekday": qkv_audit(
                average_qc, "average_weekday_qkv_rebuilt"
            ),
        },
    )

    figure_count = 0
    if generate_figures:
        from .figures import generate_corridor_figures

        with _FIGURE_LOCK:
            figure_count = generate_corridor_figures(
                spec=spec,
                settings=settings,
                output_dir=output_dir,
                observations=modeled,
                average=average_qc,
                fd=fd,
                episodes=episodes,
                applied=selected_applied,
                average_episodes=average_episodes,
                average_applied=selected_applied,
            )
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "PASS",
        "run_id": run_id,
        "corridor": spec.key,
        "source": spec.source,
        "data_mode": spec.data_mode,
        "authoritative_calibration_basis": primary_basis,
        "average_weekday_calibration_role": (
            "authoritative"
            if source_is_average
            else "diagnostic_only"
        ),
        "output_contract": "numbered-step-folders-v1",
        "average_weekday_profile_window": "00:00-24:00",
        "handoff_window": (
            f"{settings.wide_window[0] // 60:02d}:"
            f"{settings.wide_window[0] % 60:02d}-"
            f"{settings.wide_window[1] // 60:02d}:"
            f"{settings.wide_window[1] % 60:02d}"
        ),
        "core_episode_detector": "state_transition",
        "demand_capacity_basis": (
            "period_demand_over_hourly_capacity_times_period_duration"
        ),
        "period_duration_hours": {
            period: period_duration_hours(period, settings.periods)
            for period in settings.periods
        },
        "input_audit": input_audit,
        "qkv_post_qc": qkv_audit(modeled, "post_qc_fd_context"),
        "qkv_average_weekday": qkv_audit(
            average_qc, "average_weekday_qkv_rebuilt"
        ),
        "raw_qc_pass_rate": raw_qc_summary["qc_pass_repaired_rate"],
        "episodes_detected": len(episodes),
        "episode_candidates_pre_filter": len(episode_candidates),
        "average_weekday_candidates_pre_filter": len(
            average_episode_candidates
        ),
        "episodes_clean": counts["clean_episode_rows"],
        "calibration_rows": len(applied),
        "average_weekday_calibration_rows": len(average_applied),
        "figures": figure_count,
        "elapsed_seconds": elapsed,
        **counts,
    }
    _json(paths["metadata"] / "run_manifest.json", manifest)
    logger.info(
        "DONE: %s | clean episodes=%s, calibrations=%s, handoff rows=%s, figures=%s, %.1fs",
        output_dir,
        manifest["episodes_clean"],
        manifest["calibration_rows"],
        manifest["handoff_rows"],
        figure_count,
        elapsed,
    )
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()
    return manifest
