"""End-to-end corridor profile construction and comparison."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd

from .cube_qvdf import (
    CUBE_VOLUME_COLUMNS,
    TAPLITE_QVDF_KERNEL_COMMIT,
    TAPLITE_QVDF_KERNEL_URL,
    load_cube_qvdf_profiles,
)
from .metrics import (
    congestion_episodes,
    congestion_fit_metrics,
    speed_profile_metrics,
    weighted_harmonic_mean,
)
from .runtime import recommend_workers
from .volume_vmt_vht import (
    CUBE_SPEED_COLUMNS,
    build_corridor_period_comparison,
    build_scatter_metrics,
    create_scatter_figures,
    load_period_link_comparison,
)


CBI_PROFILE_COLUMNS = [
    "tmc_code",
    "direction",
    "road_order",
    "length_mi",
    "t_min",
    "avg_weekday_speed_mph",
    "speed_at_capacity_mph",
    "free_flow_speed_model_mph",
]
DASHBOARD_LINK_REFERENCE_COLUMNS = [
    "tmc_code",
    "direction",
    "road_order",
    "length_mi",
    "network_link_id",
    "network_from_node_id",
    "network_to_node_id",
    "network_mapping_status",
]
MAPPING_COLUMNS = [
    "tmc",
    "road",
    "direction",
    "road_order",
    "sequence",
    "link_id",
    "from_node_id",
    "to_node_id",
    "length_mi",
    "facility_class",
]
ROUTE_SUMMARY_COLUMNS = [
    "tmc",
    "route_link_count",
    "confidence",
    "status",
]


@dataclass(frozen=True)
class ScopedPaths:
    workspace_root: Path
    codebase_root: Path
    cbi_corridors_dir: Path
    mapmatching_run_dir: Path
    taplite_assignment_dir: Path
    results_root: Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_scoped_path(workspace_root: Path, value: str) -> Path:
    """Resolve a configured path and prove that it remains in the workspace."""
    candidate = (workspace_root / value).resolve()
    workspace = workspace_root.resolve()
    if not _is_relative_to(candidate, workspace):
        raise ValueError(f"Configured path escapes the permitted workspace: {value}")
    return candidate


def load_settings(
    config_path: Path,
    *,
    cbi_corridors_dir: Path | None = None,
    mapmatching_run_dir: Path | None = None,
    taplite_assignment_dir: Path | None = None,
    results_root: Path | None = None,
) -> Tuple[MutableMapping[str, object], ScopedPaths]:
    codebase_root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    workspace_root = config_path.parent.resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        settings: MutableMapping[str, object] = json.load(handle)

    selected_tmc_count = int(settings.get("selected_tmc_count_per_corridor", 5))
    if selected_tmc_count < 1:
        raise ValueError("selected_tmc_count_per_corridor must be positive.")
    error_heatmap_max = float(settings.get("heatmap_error_max_mph", 40.0))
    if not np.isfinite(error_heatmap_max) or error_heatmap_max <= 0:
        raise ValueError("heatmap_error_max_mph must be a positive finite value.")
    scatter_figure_dpi = int(settings.get("scatter_figure_dpi", 120))
    if scatter_figure_dpi < 1:
        raise ValueError("scatter_figure_dpi must be positive.")
    settings["scatter_figure_dpi"] = scatter_figure_dpi
    worker_fraction = float(settings.get("worker_fraction", 0.50))
    if not 0.0 < worker_fraction <= 1.0:
        raise ValueError("worker_fraction must be greater than 0 and at most 1.")
    worker_sample_seconds = float(settings.get("worker_sample_seconds", 0.25))
    if worker_sample_seconds <= 0:
        raise ValueError("worker_sample_seconds must be positive.")
    configured_cube_columns = settings.get("cube_volume_columns", CUBE_VOLUME_COLUMNS)
    if not isinstance(configured_cube_columns, Mapping):
        raise ValueError("cube_volume_columns must be a period-to-column mapping.")
    missing_cube_periods = sorted(
        set(settings.get("periods", {})).difference(configured_cube_columns)
    )
    if missing_cube_periods:
        raise ValueError(
            "cube_volume_columns is missing periods: "
            + ", ".join(missing_cube_periods)
        )
    settings["cube_volume_columns"] = {
        str(period): str(column)
        for period, column in configured_cube_columns.items()
    }

    def configured(explicit: Path | None, key: str) -> Path:
        value = explicit if explicit is not None else settings.get(key)
        if value in (None, ""):
            raise ValueError(
                f"{key} is required as a CLI argument or configuration value; "
                "automatic latest-run discovery is disabled."
            )
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else config_path.parent / candidate).resolve()

    cbi_corridors_dir = configured(cbi_corridors_dir, "cbi_corridors_dir")
    mapmatching_run_dir = configured(mapmatching_run_dir, "mapmatching_run_dir")
    taplite_assignment_dir = configured(taplite_assignment_dir, "taplite_assignment_dir")
    results_root = configured(results_root, "results_root")

    paths = ScopedPaths(
        workspace_root=workspace_root,
        codebase_root=codebase_root,
        cbi_corridors_dir=cbi_corridors_dir.resolve(),
        mapmatching_run_dir=mapmatching_run_dir.resolve(),
        taplite_assignment_dir=taplite_assignment_dir.resolve(),
        results_root=results_root.resolve(),
    )
    for path in (
        paths.cbi_corridors_dir,
        paths.mapmatching_run_dir,
        paths.taplite_assignment_dir,
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Required input directory does not exist: {path}")
    return settings, paths


def _required_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def load_cbi_profiles(cbi_corridors_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for corridor_dir in sorted(cbi_corridors_dir.iterdir(), key=lambda path: path.name):
        if not corridor_dir.is_dir():
            continue
        source = corridor_dir / "03-profiles" / "average_weekday_profile.csv"
        if not source.is_file():
            continue
        header = pd.read_csv(source, nrows=0)
        _required_columns(header, CBI_PROFILE_COLUMNS, source)
        frame = pd.read_csv(
            source,
            usecols=CBI_PROFILE_COLUMNS,
            dtype={"tmc_code": "string", "direction": "string"},
        )
        frame.insert(0, "corridor", corridor_dir.name)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No CBI average-weekday profiles found under {cbi_corridors_dir}")

    profiles = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "road_order",
        "length_mi",
        "t_min",
        "avg_weekday_speed_mph",
        "speed_at_capacity_mph",
        "free_flow_speed_model_mph",
    ]
    for column in numeric_columns:
        profiles[column] = pd.to_numeric(profiles[column], errors="coerce")
    profiles["tmc_code"] = profiles["tmc_code"].str.strip()
    profiles = profiles.dropna(subset=["corridor", "tmc_code", "t_min"])
    profiles["t_min"] = profiles["t_min"].astype(int)

    duplicate_keys = profiles.duplicated(["corridor", "tmc_code", "t_min"])
    if duplicate_keys.any():
        raise ValueError(
            "CBI average-weekday profiles contain duplicate corridor/TMC/time rows."
        )
    return profiles


def cbi_tmc_reference(profiles: pd.DataFrame) -> pd.DataFrame:
    reference_columns = [
        "corridor",
        "tmc_code",
        "direction",
        "road_order",
        "length_mi",
        "speed_at_capacity_mph",
        "free_flow_speed_model_mph",
    ]
    reference = profiles[reference_columns].drop_duplicates()
    counts = reference.groupby(["corridor", "tmc_code"]).size()
    if (counts > 1).any():
        # Threshold/free-flow fields should be constant across the average weekday.
        reference = (
            reference.sort_values(["corridor", "tmc_code"])
            .groupby(["corridor", "tmc_code"], as_index=False)
            .agg(
                direction=("direction", "first"),
                road_order=("road_order", "first"),
                length_mi=("length_mi", "first"),
                speed_at_capacity_mph=("speed_at_capacity_mph", "median"),
                free_flow_speed_model_mph=("free_flow_speed_model_mph", "median"),
            )
        )
    return reference


def load_dashboard_tmc_reference(
    cbi_corridors_dir: Path, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Use the exact membership source consumed by the integrated dashboard."""
    frames: List[pd.DataFrame] = []
    for corridor_dir in sorted(cbi_corridors_dir.iterdir(), key=lambda path: path.name):
        if not corridor_dir.is_dir():
            continue
        source = corridor_dir / "01-input-and-qc" / "link_reference.csv"
        if not source.is_file():
            continue
        header = pd.read_csv(source, nrows=0)
        _required_columns(header, DASHBOARD_LINK_REFERENCE_COLUMNS, source)
        frame = pd.read_csv(
            source,
            usecols=DASHBOARD_LINK_REFERENCE_COLUMNS,
            dtype={
                "tmc_code": "string",
                "direction": "string",
                "network_link_id": "string",
                "network_from_node_id": "string",
                "network_to_node_id": "string",
                "network_mapping_status": "string",
            },
        )
        frame.insert(0, "corridor", corridor_dir.name)
        frames.append(frame)
    if not frames:
        raise ValueError(
            f"No integrated-dashboard link_reference.csv inputs found under "
            f"{cbi_corridors_dir}"
        )
    reference = pd.concat(frames, ignore_index=True)
    reference["tmc_code"] = reference["tmc_code"].str.strip()
    for column in ("road_order", "length_mi"):
        reference[column] = pd.to_numeric(reference[column], errors="coerce")
    if reference.duplicated(["corridor", "tmc_code"]).any():
        raise ValueError(
            "Dashboard link_reference contains duplicate corridor/TMC membership rows."
        )

    profile_context = cbi_tmc_reference(profiles)[
        [
            "corridor",
            "tmc_code",
            "speed_at_capacity_mph",
            "free_flow_speed_model_mph",
        ]
    ]
    reference = reference.merge(
        profile_context,
        on=["corridor", "tmc_code"],
        how="left",
        validate="one_to_one",
    )
    profile_membership = set(
        map(tuple, profiles[["corridor", "tmc_code"]].drop_duplicates().to_numpy())
    )
    dashboard_membership = set(
        map(tuple, reference[["corridor", "tmc_code"]].to_numpy())
    )
    extra_profiles = sorted(profile_membership - dashboard_membership)
    if extra_profiles:
        raise ValueError(
            "Observed profiles contain TMCs absent from the integrated-dashboard "
            f"link_reference membership; examples={extra_profiles[:5]}"
        )
    return reference


def build_membership_audit(
    profiles: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    dashboard = (
        reference.groupby("corridor", as_index=False)
        .agg(
            dashboard_link_reference_tmc_count=("tmc_code", "nunique"),
            dashboard_primary_network_link_count=("network_link_id", "nunique"),
            minimum_road_order=("road_order", "min"),
            maximum_road_order=("road_order", "max"),
        )
    )
    observed = (
        profiles.groupby("corridor", as_index=False)["tmc_code"]
        .nunique()
        .rename(columns={"tmc_code": "observed_profile_tmc_count"})
    )
    audit = dashboard.merge(observed, on="corridor", how="outer", validate="one_to_one")
    audit["membership_counts_match"] = audit[
        "dashboard_link_reference_tmc_count"
    ].eq(audit["observed_profile_tmc_count"])
    return audit.sort_values("corridor")


def load_period_mapping(
    mapping_dir: Path,
    *,
    canonical_node_pair_mapping: Path,
    strict_qa_only: bool,
    strict_qa_statuses: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapping_source = mapping_dir / "full_tmc_to_link.csv"
    summary_source = mapping_dir / "full_route_match_summary.csv"
    for source in (mapping_source, summary_source):
        if not source.is_file():
            raise FileNotFoundError(f"Required map-matching file not found: {source}")

    mapping = pd.read_csv(
        mapping_source,
        usecols=MAPPING_COLUMNS,
        dtype={
            "tmc": "string",
            "road": "string",
            "direction": "string",
            "link_id": "string",
            "from_node_id": "string",
            "to_node_id": "string",
        },
    )
    summary = pd.read_csv(
        summary_source,
        usecols=ROUTE_SUMMARY_COLUMNS,
        dtype={"tmc": "string", "status": "string"},
    )
    for frame in (mapping, summary):
        frame["tmc"] = frame["tmc"].str.strip()
    for column in ("road_order", "sequence", "length_mi"):
        mapping[column] = pd.to_numeric(mapping[column], errors="coerce")
    canonical_source = Path(canonical_node_pair_mapping)
    if not canonical_source.is_file():
        raise FileNotFoundError(
            "Canonical node-pair TMC mapping is required for corridor "
            f"measurement: {canonical_source}"
        )
    canonical_header = set(pd.read_csv(canonical_source, nrows=0).columns)
    canonical_required = {
        "tmc",
        "from_node_id",
        "to_node_id",
        "selected_for_node_pair_lookup",
    }
    missing_canonical = sorted(canonical_required - canonical_header)
    if missing_canonical:
        raise ValueError(
            f"{canonical_source} is missing canonical columns: "
            + ", ".join(missing_canonical)
        )
    canonical = pd.read_csv(
        canonical_source,
        usecols=sorted(canonical_required),
        dtype={
            "tmc": "string",
            "from_node_id": "string",
            "to_node_id": "string",
        },
        low_memory=False,
    )
    canonical["tmc"] = canonical["tmc"].str.strip()
    canonical["selected_for_node_pair_lookup"] = (
        canonical["selected_for_node_pair_lookup"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    canonical = canonical.loc[canonical["selected_for_node_pair_lookup"]].copy()
    canonical = canonical[["tmc", "from_node_id", "to_node_id"]].drop_duplicates()
    if canonical.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError("Canonical node-pair mapping contains duplicate winners")
    for column in ("from_node_id", "to_node_id"):
        mapping[column] = mapping[column].astype("string").str.strip()
        canonical[column] = canonical[column].astype("string").str.strip()
    mapping = mapping.merge(
        canonical.assign(canonical_node_pair_winner=True),
        on=["tmc", "from_node_id", "to_node_id"],
        how="inner",
        validate="many_to_one",
    )
    summary["route_link_count"] = pd.to_numeric(
        summary["route_link_count"], errors="coerce"
    )
    summary["confidence"] = pd.to_numeric(summary["confidence"], errors="coerce")

    mapping = mapping.merge(
        summary[["tmc", "route_link_count", "confidence", "status"]],
        on="tmc",
        how="left",
        validate="many_to_one",
    )
    if strict_qa_only:
        allowed = {str(value) for value in strict_qa_statuses}
        mapping = mapping[mapping["status"].isin(allowed)].copy()
    return mapping, summary


def load_general_purpose_tmc_codes(
    mapmatching_run_dir: Path,
    mapping_products: Mapping[str, object],
) -> tuple[set[str], dict[str, int]]:
    """Return TMCs classified exclusively as general-purpose for visualization.

    The period-specific map-matching products are authoritative. A TMC is
    eligible only when every nonblank ``facility_class`` value is ``gp`` and
    no row is blank or classified differently. This deliberately prevents
    managed/express observations from leaking into an otherwise GP corridor.
    """

    classifications: dict[str, set[str]] = {}
    row_count = 0
    for period, product_name in mapping_products.items():
        source = Path(mapmatching_run_dir) / str(product_name) / "full_tmc_to_link.csv"
        if not source.is_file():
            raise FileNotFoundError(
                f"Required {str(period).upper()} mapping file not found: {source}"
            )
        header = pd.read_csv(source, nrows=0)
        _required_columns(header, ("tmc", "facility_class"), source)
        frame = pd.read_csv(
            source,
            usecols=["tmc", "facility_class"],
            dtype={"tmc": "string", "facility_class": "string"},
            low_memory=False,
        )
        frame["tmc"] = frame["tmc"].str.strip()
        frame["facility_class"] = frame["facility_class"].str.strip().str.lower()
        frame = frame.loc[frame["tmc"].notna() & frame["tmc"].ne("")]
        row_count += int(len(frame))
        for tmc_code, values in frame.groupby("tmc", sort=False)["facility_class"]:
            normalized = {
                "unclassified" if pd.isna(value) or not str(value).strip() else str(value)
                for value in values
            }
            classifications.setdefault(str(tmc_code), set()).update(normalized)

    eligible = {
        tmc_code for tmc_code, values in classifications.items() if values == {"gp"}
    }
    managed = {
        tmc_code for tmc_code, values in classifications.items() if "managed" in values
    }
    unclassified = {
        tmc_code
        for tmc_code, values in classifications.items()
        if "unclassified" in values
    }
    conflicts = {
        tmc_code for tmc_code, values in classifications.items() if len(values) > 1
    }
    return eligible, {
        "mapping_rows_checked": row_count,
        "unique_tmc_count": len(classifications),
        "general_purpose_tmc_count": len(eligible),
        "managed_tmc_count": len(managed),
        "unclassified_tmc_count": len(unclassified),
        "conflicting_tmc_count": len(conflicts),
    }


def _minute_from_speed_column(column: str) -> int:
    prefix = "spd_mph_"
    if not column.startswith(prefix):
        raise ValueError(f"Not a TAPlite time-dependent speed column: {column}")
    hour_text, minute_text = column[len(prefix) :].split(":")
    return int(hour_text) * 60 + int(minute_text)


def load_link_performance(source: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not source.is_file():
        raise FileNotFoundError(f"TAPlite link-performance file not found: {source}")
    header = pd.read_csv(source, nrows=0)
    speed_columns = [
        column for column in header.columns if column.startswith("spd_mph_")
    ]
    if not speed_columns:
        raise ValueError(f"No time-dependent speed columns found in {source}")
    period_metric_columns = ["volume", "doc", "P"]
    _required_columns(header, period_metric_columns, source)
    columns = ["link_id"] + period_metric_columns + speed_columns
    if "iteration_no" in header.columns:
        columns.insert(1, "iteration_no")
    frame = pd.read_csv(source, usecols=columns, dtype={"link_id": "string"})
    frame["link_id"] = frame["link_id"].str.strip()
    if "iteration_no" in frame.columns:
        frame["iteration_no"] = pd.to_numeric(frame["iteration_no"], errors="coerce")
        frame = frame.sort_values("iteration_no").drop_duplicates("link_id", keep="last")
        frame = frame.drop(columns="iteration_no")
    elif frame["link_id"].duplicated().any():
        raise ValueError(f"Duplicate link_id rows found in {source}")
    for column in period_metric_columns + speed_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("link_id", verify_integrity=True)
    minute_lookup = {column: _minute_from_speed_column(column) for column in speed_columns}
    return frame, minute_lookup


def _aggregate_observed_profile(
    period_profiles: pd.DataFrame,
    eligible_tmcs: pd.DataFrame,
    *,
    cutoff_ratio: float,
    minimum_length_coverage: float,
) -> pd.DataFrame:
    if period_profiles.empty or eligible_tmcs.empty:
        return pd.DataFrame(
            columns=[
                "corridor",
                "t_min",
                "observed_speed_mph",
                "cbi_congestion_threshold_mph",
                "observed_length_coverage",
                "observed_valid_tmc_count",
            ]
        )
    eligible = eligible_tmcs[["corridor", "tmc_code"]].drop_duplicates()
    data = period_profiles.merge(
        eligible, on=["corridor", "tmc_code"], how="inner", validate="many_to_one"
    ).copy()
    fallback = data["free_flow_speed_model_mph"] * cutoff_ratio
    data["threshold_mph"] = data["speed_at_capacity_mph"].where(
        data["speed_at_capacity_mph"].gt(0), fallback
    )

    rows: List[Dict[str, float]] = []
    for (corridor, t_min), group in data.groupby(
        ["corridor", "t_min"], sort=True, observed=True
    ):
        weights = pd.to_numeric(group["length_mi"], errors="coerce").to_numpy(float)
        speeds = pd.to_numeric(
            group["avg_weekday_speed_mph"], errors="coerce"
        ).to_numpy(float)
        thresholds = pd.to_numeric(group["threshold_mph"], errors="coerce").to_numpy(
            float
        )
        positive_weights = np.isfinite(weights) & (weights > 0)
        speed_valid = positive_weights & np.isfinite(speeds) & (speeds > 0)
        total_length = float(weights[positive_weights].sum())
        valid_length = float(weights[speed_valid].sum())
        coverage = valid_length / total_length if total_length > 0 else float("nan")
        speed = weighted_harmonic_mean(speeds, weights)
        threshold = weighted_harmonic_mean(thresholds, weights)
        if not np.isfinite(coverage) or coverage < minimum_length_coverage:
            speed = float("nan")
        rows.append(
            {
                "corridor": corridor,
                "t_min": int(t_min),
                "observed_speed_mph": speed,
                "cbi_congestion_threshold_mph": threshold,
                "observed_length_coverage": coverage,
                "observed_valid_tmc_count": int(speed_valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def _collapse_corridor_links(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(
            columns=[
                "corridor",
                "link_id",
                "from_node_id",
                "to_node_id",
                "length_mi",
                "corridor_road_order",
                "route_sequence",
                "tmc_codes",
                "qa_statuses",
                "performance_available",
                "cube_profile_available",
                "taplite_period_volume",
                "cube_period_volume",
                "taplite_zero_cube_positive",
            ]
        )

    def joined(values: pd.Series) -> str:
        return ";".join(sorted({str(value) for value in values.dropna()}))

    collapsed = (
        mapping.sort_values(
            ["corridor", "cbi_road_order", "sequence", "tmc", "link_id"],
            kind="stable",
        )
        .groupby(["corridor", "link_id"], as_index=False, sort=False)
        .agg(
            from_node_id=("from_node_id", "first"),
            to_node_id=("to_node_id", "first"),
            length_mi=("length_mi", "max"),
            corridor_road_order=("cbi_road_order", "min"),
            route_sequence=("sequence", "min"),
            tmc_codes=("tmc", joined),
            qa_statuses=("status", joined),
            performance_available=("performance_available", "max"),
            cube_profile_available=("cube_profile_available", "max"),
            taplite_period_volume=("taplite_period_volume", "first"),
            cube_period_volume=("cube_period_volume", "first"),
            taplite_zero_cube_positive=("taplite_zero_cube_positive", "max"),
        )
    )
    collapsed["corridor_link_order"] = (
        collapsed.sort_values(
            ["corridor", "corridor_road_order", "route_sequence", "link_id"],
            kind="stable",
        )
        .groupby("corridor")
        .cumcount()
        .add(1)
        .astype(int)
    )
    return collapsed


def _aggregate_model_profile(
    links: pd.DataFrame,
    performance: pd.DataFrame,
    minute_lookup: Mapping[str, int],
    *,
    comparison_interval_minutes: int,
    source_interval_minutes: int,
    minimum_length_coverage: float,
) -> pd.DataFrame:
    output_columns = [
        "corridor",
        "t_min",
        "model_speed_mph",
        "model_length_coverage",
        "model_source_sample_count",
    ]
    if links.empty:
        return pd.DataFrame(columns=output_columns)

    expected_source_samples = comparison_interval_minutes // source_interval_minutes
    if (
        expected_source_samples < 1
        or comparison_interval_minutes % source_interval_minutes != 0
    ):
        raise ValueError(
            "comparison_interval_minutes must be a positive multiple of "
            "model_source_interval_minutes."
        )

    speed_columns = list(minute_lookup)
    rows: List[Dict[str, float]] = []
    for corridor, corridor_links in links.groupby("corridor", sort=True):
        corridor_links = corridor_links.copy()
        corridor_links["length_mi"] = pd.to_numeric(
            corridor_links["length_mi"], errors="coerce"
        )
        link_ids = corridor_links["link_id"].astype("string")
        available_ids = [link_id for link_id in link_ids if link_id in performance.index]
        if not available_ids:
            continue
        values = performance.reindex(link_ids)[speed_columns]
        weights = corridor_links["length_mi"].to_numpy(float)
        total_weight = float(weights[np.isfinite(weights) & (weights > 0)].sum())
        for column in speed_columns:
            speeds = pd.to_numeric(values[column], errors="coerce").to_numpy(float)
            valid = (
                np.isfinite(weights)
                & (weights > 0)
                & np.isfinite(speeds)
                & (speeds > 0)
            )
            valid_weight = float(weights[valid].sum())
            coverage = valid_weight / total_weight if total_weight > 0 else float("nan")
            speed = weighted_harmonic_mean(speeds, weights)
            if not np.isfinite(coverage) or coverage < minimum_length_coverage:
                speed = float("nan")
            rows.append(
                {
                    "corridor": corridor,
                    "source_t_min": int(minute_lookup[column]),
                    "model_speed_mph": speed,
                    "model_length_coverage": coverage,
                }
            )
    if not rows:
        return pd.DataFrame(columns=output_columns)

    source_profile = pd.DataFrame(rows)
    source_profile["t_min"] = (
        source_profile["source_t_min"] // comparison_interval_minutes
    ) * comparison_interval_minutes
    aligned = (
        source_profile.groupby(["corridor", "t_min"], as_index=False)
        .agg(
            model_speed_mph=("model_speed_mph", "mean"),
            model_length_coverage=("model_length_coverage", "min"),
            model_source_sample_count=("model_speed_mph", "count"),
        )
        .sort_values(["corridor", "t_min"])
    )
    incomplete = aligned["model_source_sample_count"] < expected_source_samples
    aligned.loc[incomplete, "model_speed_mph"] = np.nan
    return aligned


def _aggregate_tmc_model_profile(
    mapping: pd.DataFrame,
    performance: pd.DataFrame,
    minute_lookup: Mapping[str, int],
    *,
    comparison_interval_minutes: int,
    source_interval_minutes: int,
    minimum_length_coverage: float,
) -> pd.DataFrame:
    """Build one model speed profile for each mapped TMC path."""
    output_columns = [
        "corridor",
        "tmc_code",
        "t_min",
        "model_tmc_speed_mph",
        "model_tmc_length_coverage",
        "model_source_sample_count",
        "taplite_period_volume",
        "taplite_period_doc",
        "taplite_period_p_hours",
        "gmns_link_count",
        "gmns_link_ids",
    ]
    if mapping.empty:
        return pd.DataFrame(columns=output_columns)
    expected_source_samples = comparison_interval_minutes // source_interval_minutes
    speed_columns = list(minute_lookup)
    rows: List[Dict[str, float]] = []
    for (corridor, tmc_code), tmc_links in mapping.groupby(
        ["corridor", "tmc"], sort=True
    ):
        links = (
            tmc_links.sort_values(["sequence", "link_id"], kind="stable")
            .drop_duplicates("link_id")
            .copy()
        )
        links["length_mi"] = pd.to_numeric(links["length_mi"], errors="coerce")
        link_ids = links["link_id"].astype("string")
        values = performance.reindex(link_ids)[speed_columns]
        weights = links["length_mi"].to_numpy(float)
        total_weight = float(weights[np.isfinite(weights) & (weights > 0)].sum())
        period_values = performance.reindex(link_ids)[["volume", "doc", "P"]]

        def weighted_arithmetic(column: str) -> float:
            metric = pd.to_numeric(period_values[column], errors="coerce").to_numpy(
                float
            )
            valid_metric = (
                np.isfinite(weights)
                & (weights > 0)
                & np.isfinite(metric)
            )
            if not valid_metric.any():
                return float("nan")
            return float(np.average(metric[valid_metric], weights=weights[valid_metric]))

        period_volume = weighted_arithmetic("volume")
        period_doc = weighted_arithmetic("doc")
        period_p = weighted_arithmetic("P")
        gmns_link_ids = ";".join(str(value) for value in link_ids)
        for column in speed_columns:
            speeds = pd.to_numeric(values[column], errors="coerce").to_numpy(float)
            valid = (
                np.isfinite(weights)
                & (weights > 0)
                & np.isfinite(speeds)
                & (speeds > 0)
            )
            valid_weight = float(weights[valid].sum())
            coverage = valid_weight / total_weight if total_weight > 0 else float("nan")
            speed = weighted_harmonic_mean(speeds, weights)
            if not np.isfinite(coverage) or coverage < minimum_length_coverage:
                speed = float("nan")
            rows.append(
                {
                    "corridor": corridor,
                    "tmc_code": tmc_code,
                    "source_t_min": int(minute_lookup[column]),
                    "model_tmc_speed_mph": speed,
                    "model_tmc_length_coverage": coverage,
                    "taplite_period_volume": period_volume,
                    "taplite_period_doc": period_doc,
                    "taplite_period_p_hours": period_p,
                    "gmns_link_count": int(len(link_ids)),
                    "gmns_link_ids": gmns_link_ids,
                }
            )
    if not rows:
        return pd.DataFrame(columns=output_columns)
    source_profile = pd.DataFrame(rows)
    source_profile["t_min"] = (
        source_profile["source_t_min"] // comparison_interval_minutes
    ) * comparison_interval_minutes
    aligned = (
        source_profile.groupby(["corridor", "tmc_code", "t_min"], as_index=False)
        .agg(
            model_tmc_speed_mph=("model_tmc_speed_mph", "mean"),
            model_tmc_length_coverage=("model_tmc_length_coverage", "min"),
            model_source_sample_count=("model_tmc_speed_mph", "count"),
            taplite_period_volume=("taplite_period_volume", "first"),
            taplite_period_doc=("taplite_period_doc", "first"),
            taplite_period_p_hours=("taplite_period_p_hours", "first"),
            gmns_link_count=("gmns_link_count", "first"),
            gmns_link_ids=("gmns_link_ids", "first"),
        )
        .sort_values(["corridor", "tmc_code", "t_min"])
    )
    incomplete = aligned["model_source_sample_count"] < expected_source_samples
    aligned.loc[incomplete, "model_tmc_speed_mph"] = np.nan
    return aligned


def _build_period_tmc_profiles(
    period_profiles: pd.DataFrame,
    expanded_mapping: pd.DataFrame,
    tmc_coverage: pd.DataFrame,
    performance: pd.DataFrame,
    minute_lookup: Mapping[str, int],
    cube_performance: pd.DataFrame,
    cube_minute_lookup: Mapping[str, int],
    *,
    period: str,
    settings: Mapping[str, object],
) -> pd.DataFrame:
    """Align every observed TMC with its mapped GMNS-path model profile."""
    base = period_profiles[
        [
            "corridor",
            "tmc_code",
            "direction",
            "road_order",
            "length_mi",
            "t_min",
            "avg_weekday_speed_mph",
            "speed_at_capacity_mph",
            "free_flow_speed_model_mph",
        ]
    ].copy()
    base = base.rename(columns={"avg_weekday_speed_mph": "observed_tmc_speed_mph"})
    base["cbi_tmc_congestion_threshold_mph"] = base[
        "speed_at_capacity_mph"
    ].where(
        base["speed_at_capacity_mph"].gt(0),
        base["free_flow_speed_model_mph"]
        * float(settings["congestion_cutoff_ratio"]),
    )
    coverage = tmc_coverage[
        [
            "corridor",
            "tmc_code",
            "tmc_model_length_coverage",
            "eligible_for_comparison",
            "qa_status",
        ]
    ].drop_duplicates(["corridor", "tmc_code"])
    base = base.merge(
        coverage,
        on=["corridor", "tmc_code"],
        how="left",
        validate="many_to_one",
    )
    base["eligible_for_comparison"] = base["eligible_for_comparison"].fillna(False)
    eligible_mapping = expanded_mapping[
        expanded_mapping["eligible_for_comparison"].fillna(False)
    ]
    model = _aggregate_tmc_model_profile(
        eligible_mapping,
        performance,
        minute_lookup,
        comparison_interval_minutes=int(settings["comparison_interval_minutes"]),
        source_interval_minutes=int(settings["model_source_interval_minutes"]),
        minimum_length_coverage=float(
            settings["minimum_corridor_model_length_coverage"]
        ),
    )
    cube = _aggregate_tmc_model_profile(
        eligible_mapping,
        cube_performance,
        cube_minute_lookup,
        comparison_interval_minutes=int(settings["comparison_interval_minutes"]),
        source_interval_minutes=int(settings["model_source_interval_minutes"]),
        minimum_length_coverage=float(
            settings["minimum_corridor_model_length_coverage"]
        ),
    ).rename(
        columns={
            "model_tmc_speed_mph": "cube_qvdf_tmc_speed_mph",
            "model_tmc_length_coverage": "cube_qvdf_tmc_length_coverage",
            "model_source_sample_count": "cube_qvdf_source_sample_count",
            "taplite_period_volume": "cube_period_volume",
            "taplite_period_doc": "cube_period_doc",
            "taplite_period_p_hours": "cube_period_p_hours",
            "gmns_link_count": "cube_gmns_link_count",
            "gmns_link_ids": "cube_gmns_link_ids",
        }
    )
    result = base.merge(
        model,
        on=["corridor", "tmc_code", "t_min"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        cube,
        on=["corridor", "tmc_code", "t_min"],
        how="left",
        validate="one_to_one",
    )
    result.insert(1, "period", period.upper())
    result["clock_time"] = result["t_min"].map(_format_clock)
    result["tmc_speed_error_mph"] = (
        result["model_tmc_speed_mph"] - result["observed_tmc_speed_mph"]
    )
    result["cube_tmc_speed_error_mph"] = (
        result["cube_qvdf_tmc_speed_mph"]
        - result["observed_tmc_speed_mph"]
    )
    result["taplite_cube_tmc_speed_error_mph"] = (
        result["model_tmc_speed_mph"] - result["cube_qvdf_tmc_speed_mph"]
    )
    return result.sort_values(["corridor", "road_order", "tmc_code", "t_min"])


def _format_clock(t_min: float) -> str:
    if not np.isfinite(t_min):
        return ""
    minute = int(t_min)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _complete_period_corridor_grid(
    aligned: pd.DataFrame,
    tmc_reference: pd.DataFrame,
    *,
    start_min: int,
    end_min: int,
    interval_minutes: int,
) -> pd.DataFrame:
    """Retain explicit blank rows when a corridor has no eligible period links."""

    corridors = sorted(tmc_reference["corridor"].dropna().astype(str).unique())
    grid = pd.MultiIndex.from_product(
        [corridors, range(start_min, end_min, interval_minutes)],
        names=["corridor", "t_min"],
    ).to_frame(index=False)
    return grid.merge(
        aligned,
        on=["corridor", "t_min"],
        how="left",
        validate="one_to_one",
    )


def build_period_comparison(
    *,
    period: str,
    period_settings: Mapping[str, object],
    settings: Mapping[str, object],
    profiles: pd.DataFrame,
    tmc_reference: pd.DataFrame,
    mapping: pd.DataFrame,
    route_summary: pd.DataFrame,
    performance: pd.DataFrame,
    minute_lookup: Mapping[str, int],
    cube_performance: pd.DataFrame,
    cube_minute_lookup: Mapping[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_min = int(period_settings["start_min"])
    end_min = int(period_settings["end_min"])
    period_profiles = profiles[
        profiles["t_min"].ge(start_min) & profiles["t_min"].lt(end_min)
    ].copy()

    expanded = mapping.merge(
        tmc_reference.rename(
            columns={
                "tmc_code": "tmc",
                "direction": "cbi_direction",
                "road_order": "cbi_road_order",
                "length_mi": "cbi_tmc_length_mi",
            }
        ),
        on="tmc",
        how="inner",
        validate="many_to_many",
    )
    expanded["performance_available"] = expanded["link_id"].isin(performance.index)
    expanded["cube_profile_available"] = expanded["link_id"].isin(
        cube_performance.index
    )
    expanded["taplite_period_volume"] = expanded["link_id"].map(
        performance["volume"]
    )
    expanded["cube_period_volume"] = expanded["link_id"].map(
        cube_performance["volume"]
    )
    expanded["taplite_zero_volume"] = (
        pd.to_numeric(expanded["taplite_period_volume"], errors="coerce")
        .fillna(0.0)
        .le(0.0)
    )
    expanded["cube_positive_volume"] = (
        pd.to_numeric(expanded["cube_period_volume"], errors="coerce")
        .fillna(0.0)
        .gt(0.0)
    )
    expanded["taplite_zero_cube_positive"] = (
        expanded["taplite_zero_volume"] & expanded["cube_positive_volume"]
    )
    expanded["length_mi"] = pd.to_numeric(expanded["length_mi"], errors="coerce")
    expanded["valid_mapping_length_mi"] = expanded["length_mi"].where(
        expanded["length_mi"].gt(0), 0.0
    )
    expanded["available_mapping_length_mi"] = expanded[
        "valid_mapping_length_mi"
    ].where(expanded["performance_available"], 0.0)

    tmc_coverage = (
        expanded.groupby(["corridor", "tmc"], as_index=False)
        .agg(
            mapped_model_length_mi=("valid_mapping_length_mi", "sum"),
            available_model_length_mi=("available_mapping_length_mi", "sum"),
            mapped_link_count=("link_id", "nunique"),
            qa_status=("status", "first"),
        )
        .rename(columns={"tmc": "tmc_code"})
    )
    denominator = tmc_coverage["mapped_model_length_mi"].replace(0, np.nan)
    tmc_coverage["tmc_model_length_coverage"] = (
        tmc_coverage["available_model_length_mi"] / denominator
    )
    minimum_tmc_coverage = float(settings["minimum_tmc_model_length_coverage"])
    tmc_coverage["eligible_for_comparison"] = (
        tmc_coverage["tmc_model_length_coverage"] >= minimum_tmc_coverage
    )

    expanded = expanded.merge(
        tmc_coverage[
            ["corridor", "tmc_code", "tmc_model_length_coverage", "eligible_for_comparison"]
        ].rename(columns={"tmc_code": "tmc"}),
        on=["corridor", "tmc"],
        how="left",
        validate="many_to_one",
    )
    mapping_inventory = expanded[
        [
            "corridor",
            "tmc",
            "cbi_direction",
            "cbi_road_order",
            "sequence",
            "link_id",
            "from_node_id",
            "to_node_id",
            "length_mi",
            "status",
            "confidence",
            "performance_available",
            "cube_profile_available",
            "taplite_period_volume",
            "cube_period_volume",
            "taplite_zero_volume",
            "cube_positive_volume",
            "taplite_zero_cube_positive",
            "tmc_model_length_coverage",
            "eligible_for_comparison",
        ]
    ].copy()
    mapping_inventory.insert(1, "period", period.upper())
    mapping_inventory = mapping_inventory.rename(
        columns={
            "tmc": "tmc_code",
            "cbi_direction": "direction",
            "cbi_road_order": "road_order",
            "sequence": "route_sequence",
            "status": "qa_status",
        }
    ).sort_values(
        ["corridor", "road_order", "tmc_code", "route_sequence", "link_id"],
        kind="stable",
    )

    eligible_rows = expanded[expanded["eligible_for_comparison"].fillna(False)].copy()
    corridor_links = _collapse_corridor_links(eligible_rows)
    corridor_links.insert(1, "period", period.upper())

    observed = _aggregate_observed_profile(
        period_profiles,
        tmc_coverage[tmc_coverage["eligible_for_comparison"]],
        cutoff_ratio=float(settings["congestion_cutoff_ratio"]),
        minimum_length_coverage=float(
            settings["minimum_corridor_observed_length_coverage"]
        ),
    )
    modeled = _aggregate_model_profile(
        corridor_links,
        performance,
        minute_lookup,
        comparison_interval_minutes=int(settings["comparison_interval_minutes"]),
        source_interval_minutes=int(settings["model_source_interval_minutes"]),
        minimum_length_coverage=float(
            settings["minimum_corridor_model_length_coverage"]
        ),
    )
    cube_modeled = _aggregate_model_profile(
        corridor_links,
        cube_performance,
        cube_minute_lookup,
        comparison_interval_minutes=int(settings["comparison_interval_minutes"]),
        source_interval_minutes=int(settings["model_source_interval_minutes"]),
        minimum_length_coverage=float(
            settings["minimum_corridor_model_length_coverage"]
        ),
    ).rename(
        columns={
            "model_speed_mph": "cube_qvdf_speed_mph",
            "model_length_coverage": "cube_qvdf_length_coverage",
            "model_source_sample_count": "cube_qvdf_source_sample_count",
        }
    )
    aligned = observed.merge(
        modeled, on=["corridor", "t_min"], how="outer", validate="one_to_one"
    )
    aligned = aligned.merge(
        cube_modeled,
        on=["corridor", "t_min"],
        how="outer",
        validate="one_to_one",
    )
    aligned = _complete_period_corridor_grid(
        aligned,
        tmc_reference,
        start_min=start_min,
        end_min=end_min,
        interval_minutes=int(settings["comparison_interval_minutes"]),
    )
    aligned.insert(1, "period", period.upper())
    aligned["clock_time"] = aligned["t_min"].map(_format_clock)
    both_valid = aligned[
        ["observed_speed_mph", "model_speed_mph", "cbi_congestion_threshold_mph"]
    ].notna()
    aligned["speed_error_mph"] = (
        aligned["model_speed_mph"] - aligned["observed_speed_mph"]
    )
    aligned["cube_speed_error_mph"] = (
        aligned["cube_qvdf_speed_mph"] - aligned["observed_speed_mph"]
    )
    aligned["taplite_cube_speed_error_mph"] = (
        aligned["model_speed_mph"] - aligned["cube_qvdf_speed_mph"]
    )
    denominator_speed = aligned["observed_speed_mph"].where(
        aligned["observed_speed_mph"].abs()
        >= float(settings["mape_minimum_observed_speed_mph"])
    )
    aligned["absolute_percentage_error_pct"] = (
        aligned["speed_error_mph"].abs() / denominator_speed.abs() * 100.0
    )
    aligned["observed_congested"] = (
        aligned["observed_speed_mph"] < aligned["cbi_congestion_threshold_mph"]
    ).where(both_valid["observed_speed_mph"] & both_valid["cbi_congestion_threshold_mph"])
    aligned["model_congested"] = (
        aligned["model_speed_mph"] < aligned["cbi_congestion_threshold_mph"]
    ).where(both_valid["model_speed_mph"] & both_valid["cbi_congestion_threshold_mph"])
    cube_valid = aligned[
        ["cube_qvdf_speed_mph", "cbi_congestion_threshold_mph"]
    ].notna()
    aligned["cube_congested"] = (
        aligned["cube_qvdf_speed_mph"]
        < aligned["cbi_congestion_threshold_mph"]
    ).where(
        cube_valid["cube_qvdf_speed_mph"]
        & cube_valid["cbi_congestion_threshold_mph"]
    )
    aligned = aligned.sort_values(["corridor", "t_min"])

    all_tmc_counts = (
        tmc_reference.groupby("corridor")["tmc_code"]
        .nunique()
        .rename("cbi_tmc_count")
    )
    mapped_tmc_counts = (
        tmc_coverage.groupby("corridor")["tmc_code"]
        .nunique()
        .rename("mapped_tmc_count")
    )
    eligible_tmc_counts = (
        tmc_coverage[tmc_coverage["eligible_for_comparison"]]
        .groupby("corridor")["tmc_code"]
        .nunique()
        .rename("eligible_tmc_count")
    )
    matched_status_counts = (
        tmc_coverage[tmc_coverage["qa_status"].eq("matched")]
        .groupby("corridor")["tmc_code"]
        .nunique()
        .rename("qa_matched_tmc_count")
    )
    link_counts = (
        corridor_links.groupby("corridor")["link_id"]
        .nunique()
        .rename("gmns_link_count_used")
    )
    link_lengths = (
        corridor_links.groupby("corridor")["length_mi"]
        .sum(min_count=1)
        .rename("gmns_corridor_length_mi")
    )
    cube_link_counts = (
        corridor_links[corridor_links["cube_profile_available"].fillna(False)]
        .groupby("corridor")["link_id"]
        .nunique()
        .rename("cube_qvdf_link_count_available")
    )
    zero_assignment_cube_positive_counts = (
        corridor_links[
            corridor_links["taplite_zero_cube_positive"].fillna(False)
        ]
        .groupby("corridor")["link_id"]
        .nunique()
        .rename("taplite_zero_cube_positive_link_count")
    )
    audit = (
        pd.DataFrame(index=sorted(tmc_reference["corridor"].unique()))
        .join(all_tmc_counts)
        .join(mapped_tmc_counts)
        .join(eligible_tmc_counts)
        .join(matched_status_counts)
        .join(link_counts)
        .join(link_lengths)
        .join(cube_link_counts)
        .join(zero_assignment_cube_positive_counts)
        .fillna(
            {
                "mapped_tmc_count": 0,
                "eligible_tmc_count": 0,
                "qa_matched_tmc_count": 0,
                "gmns_link_count_used": 0,
                "cube_qvdf_link_count_available": 0,
                "taplite_zero_cube_positive_link_count": 0,
            }
        )
        .reset_index()
        .rename(columns={"index": "corridor"})
    )
    audit.insert(1, "period", period.upper())
    audit["tmc_mapping_coverage_pct"] = (
        audit["mapped_tmc_count"] / audit["cbi_tmc_count"] * 100.0
    )
    audit["tmc_comparison_coverage_pct"] = (
        audit["eligible_tmc_count"] / audit["cbi_tmc_count"] * 100.0
    )
    audit["qa_matched_share_of_mapped_pct"] = (
        audit["qa_matched_tmc_count"]
        / audit["mapped_tmc_count"].replace(0, np.nan)
        * 100.0
    )
    tmc_profiles = _build_period_tmc_profiles(
        period_profiles,
        expanded,
        tmc_coverage,
        performance,
        minute_lookup,
        cube_performance,
        cube_minute_lookup,
        period=period,
        settings=settings,
    )
    return aligned, audit, mapping_inventory, corridor_links, tmc_profiles


def _metrics_for_frame(
    frame: pd.DataFrame,
    *,
    settings: Mapping[str, object],
) -> Dict[str, float]:
    result = speed_profile_metrics(
        frame["observed_speed_mph"],
        frame["model_speed_mph"],
        mape_minimum_observed_speed_mph=float(
            settings["mape_minimum_observed_speed_mph"]
        ),
    )
    cube_metrics = speed_profile_metrics(
        frame["observed_speed_mph"],
        frame["cube_qvdf_speed_mph"],
        mape_minimum_observed_speed_mph=float(
            settings["mape_minimum_observed_speed_mph"]
        ),
    )
    result.update(
        {
            f"cube_vs_observed_{name}": value
            for name, value in cube_metrics.items()
        }
    )
    taplite_cube_metrics = speed_profile_metrics(
        frame["cube_qvdf_speed_mph"],
        frame["model_speed_mph"],
        mape_minimum_observed_speed_mph=float(
            settings["mape_minimum_observed_speed_mph"]
        ),
    )
    result.update(
        {
            f"taplite_vs_cube_{name}": value
            for name, value in taplite_cube_metrics.items()
        }
    )
    result.update(
        congestion_fit_metrics(
            frame, interval_minutes=int(settings["comparison_interval_minutes"])
        )
    )
    cube_congestion_frame = frame.copy()
    cube_congestion_frame["model_congested"] = cube_congestion_frame[
        "cube_congested"
    ]
    cube_congestion_metrics = congestion_fit_metrics(
        cube_congestion_frame,
        interval_minutes=int(settings["comparison_interval_minutes"]),
    )
    result.update(
        {
            f"cube_vs_observed_{name}": value
            for name, value in cube_congestion_metrics.items()
        }
    )
    result["mean_observed_speed_mph"] = pd.to_numeric(
        frame["observed_speed_mph"], errors="coerce"
    ).mean()
    result["mean_model_speed_mph"] = pd.to_numeric(
        frame["model_speed_mph"], errors="coerce"
    ).mean()
    result["mean_cube_qvdf_speed_mph"] = pd.to_numeric(
        frame["cube_qvdf_speed_mph"], errors="coerce"
    ).mean()
    result["mean_observed_length_coverage_pct"] = (
        pd.to_numeric(frame["observed_length_coverage"], errors="coerce").mean()
        * 100.0
    )
    result["mean_model_length_coverage_pct"] = (
        pd.to_numeric(frame["model_length_coverage"], errors="coerce").mean() * 100.0
    )
    result["mean_cube_qvdf_length_coverage_pct"] = (
        pd.to_numeric(
            frame["cube_qvdf_length_coverage"], errors="coerce"
        ).mean()
        * 100.0
    )
    return result


def build_metrics_tables(
    profiles: pd.DataFrame,
    aligned: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    settings: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corridors = sorted(profiles["corridor"].unique())
    periods = list(settings["periods"].keys())
    expected_interval_count = int(
        sum(
            (
                int(period_settings["end_min"])
                - int(period_settings["start_min"])
            )
            / int(settings["comparison_interval_minutes"])
            for period_settings in settings["periods"].values()
        )
    )
    daily_rows: List[Dict[str, float]] = []
    period_rows: List[Dict[str, float]] = []

    for corridor in corridors:
        frame = aligned[aligned["corridor"].eq(corridor)].copy()
        row: Dict[str, float] = {"corridor": corridor}
        if frame.empty:
            row.update(
                _metrics_for_frame(
                    pd.DataFrame(
                        columns=[
                            "observed_speed_mph",
                            "model_speed_mph",
                            "cube_qvdf_speed_mph",
                            "observed_congested",
                            "model_congested",
                            "cube_congested",
                            "observed_length_coverage",
                            "model_length_coverage",
                            "cube_qvdf_length_coverage",
                            "t_min",
                        ]
                    ),
                    settings=settings,
                )
            )
        else:
            row.update(_metrics_for_frame(frame, settings=settings))
        row["periods_with_matched_intervals"] = int(
            frame.dropna(subset=["observed_speed_mph", "model_speed_mph"])[
                "period"
            ].nunique()
        )
        row["periods_with_cube_matched_intervals"] = int(
            frame.dropna(
                subset=["observed_speed_mph", "cube_qvdf_speed_mph"]
            )["period"].nunique()
        )
        row["expected_interval_count"] = expected_interval_count
        row["matched_interval_coverage_pct"] = (
            row["matched_interval_count"] / expected_interval_count * 100.0
        )
        daily_rows.append(row)

        for period in periods:
            period_frame = frame[frame["period"].eq(period.upper())]
            period_row: Dict[str, float] = {
                "corridor": corridor,
                "period": period.upper(),
            }
            period_row.update(_metrics_for_frame(period_frame, settings=settings))
            period_rows.append(period_row)

    daily = pd.DataFrame(daily_rows)
    period_table = pd.DataFrame(period_rows)

    period_table = period_table.merge(
        audit, on=["corridor", "period"], how="left", validate="one_to_one"
    )
    audit_daily = (
        audit.groupby("corridor", as_index=False)
        .agg(
            minimum_tmc_mapping_coverage_pct=("tmc_mapping_coverage_pct", "min"),
            minimum_tmc_comparison_coverage_pct=(
                "tmc_comparison_coverage_pct",
                "min",
            ),
            minimum_qa_matched_share_of_mapped_pct=(
                "qa_matched_share_of_mapped_pct",
                "min",
            ),
            minimum_gmns_link_count_used=("gmns_link_count_used", "min"),
            maximum_gmns_link_count_used=("gmns_link_count_used", "max"),
            minimum_cube_qvdf_link_count_available=(
                "cube_qvdf_link_count_available",
                "min",
            ),
            maximum_taplite_zero_cube_positive_link_count=(
                "taplite_zero_cube_positive_link_count",
                "max",
            ),
        )
    )
    daily = daily.merge(audit_daily, on="corridor", how="left", validate="one_to_one")
    daily["result_status"] = np.select(
        [
            daily["matched_interval_count"].eq(0),
            daily["periods_with_matched_intervals"].lt(len(periods)),
            daily["matched_interval_count"].lt(expected_interval_count),
        ],
        [
            "no_aligned_intervals",
            "partial_period_coverage",
            "partial_interval_coverage",
        ],
        default="complete",
    )

    valid_speed = daily["matched_interval_count"].gt(0)
    valid_duration_mape = daily["observed_congestion_duration_min"].gt(0)
    valid_cube_speed = daily["cube_vs_observed_matched_interval_count"].gt(0)
    valid_taplite_cube_speed = daily["taplite_vs_cube_matched_interval_count"].gt(0)
    overall = pd.DataFrame(
        [
            {
                "corridor_count": len(daily),
                "corridors_with_speed_results": int(valid_speed.sum()),
                "corridors_with_all_periods": int(
                    daily["periods_with_matched_intervals"].eq(len(periods)).sum()
                ),
                "corridors_with_complete_interval_coverage": int(
                    daily["matched_interval_count"].eq(expected_interval_count).sum()
                ),
                "corridor_mean_speed_mae_mph": daily.loc[
                    valid_speed, "mae_mph"
                ].mean(),
                "corridor_mean_speed_mape_pct": daily.loc[
                    valid_speed, "mape_pct"
                ].mean(),
                "corridor_mean_speed_rmse_mph": daily.loc[
                    valid_speed, "rmse_mph"
                ].mean(),
                "corridor_mean_cube_vs_observed_speed_mae_mph": daily.loc[
                    valid_cube_speed, "cube_vs_observed_mae_mph"
                ].mean(),
                "corridor_mean_cube_vs_observed_speed_mape_pct": daily.loc[
                    valid_cube_speed, "cube_vs_observed_mape_pct"
                ].mean(),
                "corridor_mean_taplite_vs_cube_speed_mae_mph": daily.loc[
                    valid_taplite_cube_speed, "taplite_vs_cube_mae_mph"
                ].mean(),
                "corridor_mean_taplite_vs_cube_speed_mape_pct": daily.loc[
                    valid_taplite_cube_speed, "taplite_vs_cube_mape_pct"
                ].mean(),
                "congestion_duration_mae_min": daily.loc[
                    valid_speed, "congestion_duration_absolute_error_min"
                ].mean(),
                "congestion_duration_mape_pct": daily.loc[
                    valid_duration_mape, "congestion_duration_ape_pct"
                ].mean(),
                "cube_congestion_duration_mae_min": daily.loc[
                    valid_cube_speed,
                    "cube_vs_observed_congestion_duration_absolute_error_min",
                ].mean(),
                "corridors_with_observed_congestion": int(valid_duration_mape.sum()),
                "mean_congestion_iou_pct": daily.loc[
                    daily["congestion_union_min"].gt(0), "congestion_iou_pct"
                ].mean(),
                "mean_tmc_comparison_coverage_pct": daily[
                    "minimum_tmc_comparison_coverage_pct"
                ].mean(),
            }
        ]
    )
    return daily, period_table, overall


def build_episode_table(
    aligned: pd.DataFrame, *, interval_minutes: int
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for corridor, frame in aligned.groupby("corridor", sort=True):
        for source, flag_column in (
            ("CBI_OBSERVED", "observed_congested"),
            ("TAPLITE_MODEL", "model_congested"),
            ("CUBE_QVDF", "cube_congested"),
        ):
            episodes = congestion_episodes(
                frame, flag_column, interval_minutes=interval_minutes
            )
            for index, episode in enumerate(episodes, start=1):
                rows.append(
                    {
                        "corridor": corridor,
                        "source": source,
                        "episode_number": index,
                        **episode,
                        "start_time": _format_clock(float(episode["start_min"])),
                        "end_time": _format_clock(float(episode["end_min"])),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "corridor",
            "source",
            "episode_number",
            "start_min",
            "end_min",
            "duration_min",
            "start_time",
            "end_time",
        ],
    )


def validate_results(
    profiles: pd.DataFrame,
    aligned: pd.DataFrame,
    tmc_profiles: pd.DataFrame,
    daily: pd.DataFrame,
    period_table: pd.DataFrame,
    *,
    settings: Mapping[str, object],
) -> Dict[str, object]:
    """Fail the run if temporal coverage or metric accounting is inconsistent."""
    corridors = sorted(profiles["corridor"].unique())
    interval_minutes = int(settings["comparison_interval_minutes"])
    expected_minutes = sorted(
        minute
        for period_settings in settings["periods"].values()
        for minute in range(
            int(period_settings["start_min"]),
            int(period_settings["end_min"]),
            interval_minutes,
        )
    )
    if aligned.duplicated(["corridor", "t_min"]).any():
        raise ValueError("Aligned profiles contain duplicate corridor/time rows.")
    if tmc_profiles.duplicated(["corridor", "tmc_code", "t_min"]).any():
        raise ValueError("TMC profiles contain duplicate corridor/TMC/time rows.")
    if sorted(aligned["corridor"].unique()) != corridors:
        raise ValueError("Aligned profile corridor set does not match the CBI corridor set.")
    for corridor, group in aligned.groupby("corridor", sort=False):
        actual_minutes = sorted(group["t_min"].dropna().astype(int).unique())
        if actual_minutes != expected_minutes:
            raise ValueError(
                f"{corridor} does not contain the complete configured time grid."
            )
    if len(daily) != len(corridors):
        raise ValueError("Daily metric row count does not equal corridor count.")
    expected_period_rows = len(corridors) * len(settings["periods"])
    if len(period_table) != expected_period_rows:
        raise ValueError("Period metric row count is inconsistent.")

    for column in (
        "observed_speed_mph",
        "model_speed_mph",
        "cube_qvdf_speed_mph",
        "cbi_congestion_threshold_mph",
    ):
        values = pd.to_numeric(aligned[column], errors="coerce").dropna()
        if (values <= 0).any() or (values > 150).any():
            raise ValueError(f"{column} contains an implausible value.")

    matched_from_profiles = int(
        aligned[["observed_speed_mph", "model_speed_mph"]].notna().all(axis=1).sum()
    )
    matched_from_metrics = int(daily["matched_interval_count"].sum())
    if matched_from_profiles != matched_from_metrics:
        raise ValueError("Matched interval counts do not reconcile.")
    cube_matched_from_profiles = int(
        aligned[["observed_speed_mph", "cube_qvdf_speed_mph"]]
        .notna()
        .all(axis=1)
        .sum()
    )
    cube_matched_from_metrics = int(
        daily["cube_vs_observed_matched_interval_count"].sum()
    )
    if cube_matched_from_profiles != cube_matched_from_metrics:
        raise ValueError("Cube matched interval counts do not reconcile.")
    return {
        "status": "passed",
        "corridor_count": len(corridors),
        "expected_intervals_per_corridor": len(expected_minutes),
        "aligned_profile_row_count": len(aligned),
        "matched_profile_row_count": matched_from_profiles,
        "cube_matched_profile_row_count": cube_matched_from_profiles,
        "tmc_profile_row_count": len(tmc_profiles),
        "duplicate_corridor_time_row_count": 0,
        "duplicate_corridor_tmc_time_row_count": 0,
        "speed_range_check": "passed_0_to_150_mph",
        "metric_reconciliation": "passed",
    }


def _markdown_value(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(
    output_dir: Path,
    daily: pd.DataFrame,
    overall: pd.DataFrame,
    *,
    settings: Mapping[str, object],
) -> None:
    summary = overall.iloc[0]
    lines = [
        "# TAPlite, Cube-QVDF, and CBI corridor profile measurement",
        "",
        "## Outcome",
        "",
        f"- Corridors evaluated: {int(summary['corridor_count'])}",
        f"- Corridors with aligned speed results: "
        f"{int(summary['corridors_with_speed_results'])}",
        f"- Corridors with all AM/MD/PM periods: "
        f"{int(summary['corridors_with_all_periods'])}",
        f"- Corridors with all expected 15-minute observations: "
        f"{int(summary['corridors_with_complete_interval_coverage'])}",
        f"- Mean corridor MAE: "
        f"{_markdown_value(summary['corridor_mean_speed_mae_mph'])} mph",
        f"- Mean corridor MAPE: "
        f"{_markdown_value(summary['corridor_mean_speed_mape_pct'])}%",
        f"- Mean Cube-QVDF versus CBI corridor MAE: "
        f"{_markdown_value(summary['corridor_mean_cube_vs_observed_speed_mae_mph'])} mph",
        f"- Mean Cube-QVDF versus CBI corridor MAPE: "
        f"{_markdown_value(summary['corridor_mean_cube_vs_observed_speed_mape_pct'])}%",
        f"- Mean TAPlite versus Cube-QVDF corridor MAE: "
        f"{_markdown_value(summary['corridor_mean_taplite_vs_cube_speed_mae_mph'])} mph",
        f"- Congestion-duration MAE: "
        f"{_markdown_value(summary['congestion_duration_mae_min'])} minutes",
        f"- Congestion-duration MAPE among corridors with observed congestion: "
        f"{_markdown_value(summary['congestion_duration_mape_pct'])}%",
        f"- Cube-QVDF congestion-duration MAE: "
        f"{_markdown_value(summary['cube_congestion_duration_mae_min'])} minutes",
        "",
        "## Method",
        "",
        "- AM, MD, and PM TAPlite `link_performance.csv` speed columns are joined "
        "into a 06:00–19:00 model day.",
        "- The same GMNS links are independently reconstructed with the Cube "
        "`I4AMVOL`, `I4MDVOL`, and `I4PMVOL` period volumes and the QVDF "
        "parameters in each period's `link.csv`.",
        "- Cube-volume profiles reproduce TAPlite's C++ `Link_QueueVDF` equations; "
        "the exact kernel commit and source URL are recorded in the run manifest "
        "and link-period audit.",
        "- Each period uses its own existing TMC-to-GMNS route/path mapping.",
        "- Corridor membership and TMC order come from each CBI corridor's "
        "`01-input-and-qc/link_reference.csv`, the same source used by the "
        "integrated dashboard builder.",
        "- Physical GMNS links are ordered by CBI corridor/TMC order and route "
        "sequence; duplicate physical links are counted once in the corridor.",
        "- Model and observed corridor speeds are distance-weighted harmonic means "
        "(total distance divided by total travel time).",
        "- TAPlite's 5-minute model speeds are averaged onto CBI's 15-minute grid.",
        "- Only TMCs with at least "
        f"{float(settings['minimum_tmc_model_length_coverage']) * 100:.0f}% "
        "mapped-model length coverage are included in both profiles.",
        "- Both observed and modeled congestion use the same CBI threshold: the "
        "distance-weighted CBI speed-at-capacity value, based on the configured "
        f"{float(settings['congestion_cutoff_ratio']) * 100:.0f}% "
        "free-flow cutoff.",
        f"- MAPE excludes observed speeds below "
        f"{float(settings['mape_minimum_observed_speed_mph']):.1f} mph.",
        "",
        "## Corridor-by-corridor results",
        "",
        "| Corridor | Status | Intervals | TAPlite-CBI MAE mph | TAPlite-CBI MAPE % | "
        "Cube-CBI MAE mph | Cube-CBI MAPE % | TAPlite-Cube MAE mph | "
        "Obs congestion min | TAPlite congestion min | Cube congestion min | "
        "TAPlite duration abs. error min | Cube duration abs. error min | "
        "TAPlite IoU % | Cube IoU % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    columns = [
        "corridor",
        "result_status",
        "matched_interval_count",
        "mae_mph",
        "mape_pct",
        "cube_vs_observed_mae_mph",
        "cube_vs_observed_mape_pct",
        "taplite_vs_cube_mae_mph",
        "observed_congestion_duration_min",
        "model_congestion_duration_min",
        "cube_vs_observed_model_congestion_duration_min",
        "congestion_duration_absolute_error_min",
        "cube_vs_observed_congestion_duration_absolute_error_min",
        "congestion_iou_pct",
        "cube_vs_observed_congestion_iou_pct",
    ]
    for row in daily.sort_values("corridor")[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `01-corridor-results/`: full-day/period metrics, aligned corridor profiles, and this report.",
            "- `02-tmc-results/`: complete TMC profiles plus the five behavior-diverse figure selections (or all TMCs when fewer than five exist).",
            "- `03-congestion-results/`: contiguous observed, TAPlite, and Cube-QVDF congestion episodes.",
            "- `04-network-mapping/`: ordered TMC-to-GMNS tables, de-duplicated corridor links, and the Cube-QVDF link-period reconstruction audit.",
            "- `05-quality-assurance/`: mapping and corridor-membership audits.",
            "- `06-figures/`: selected three-source profiles, three-panel speed heatmaps, three pairwise absolute-error heatmaps, and their index.",
            "- `07-run-metadata/`: exact settings, inputs, validation, and row counts.",
            "- `08-volume-vmt-vht-comparison/`: link- and corridor-level Cube-versus-TAPlite Volume, VMT, and VHT scatter data and figures.",
            "",
        ]
    )
    report_path = output_dir / "01-corridor-results" / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _relative_string(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def run_measurement(
    *,
    config_path: Path,
    output_dir: Path,
    cbi_corridors_dir: Path,
    mapmatching_run_dir: Path,
    taplite_assignment_dir: Path,
    workers: int | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    settings, paths = load_settings(
        config_path,
        cbi_corridors_dir=cbi_corridors_dir,
        mapmatching_run_dir=mapmatching_run_dir,
        taplite_assignment_dir=taplite_assignment_dir,
        results_root=output_dir,
    )
    if output_dir.exists() and any(
        item.name not in {"logs", "qa", "normalized-inputs"} for item in output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Result directory is not empty; use a new output directory: {output_dir}"
        )

    profiles = load_cbi_profiles(paths.cbi_corridors_dir)
    tmc_reference = load_dashboard_tmc_reference(paths.cbi_corridors_dir, profiles)
    membership_audit = build_membership_audit(profiles, tmc_reference)
    mapping_products = settings["mapping_products"]
    periods = settings["periods"]
    canonical_node_pair_mapping = (
        paths.cbi_corridors_dir.parent
        / "shared"
        / "network-mapping"
        / "canonical_node_pair_tmc.csv"
    ).resolve()
    if not canonical_node_pair_mapping.is_file():
        raise FileNotFoundError(
            "The selected CBI run does not contain its frozen canonical "
            f"node-pair mapping: {canonical_node_pair_mapping}"
        )

    aligned_frames: List[pd.DataFrame] = []
    audit_frames: List[pd.DataFrame] = []
    inventory_frames: List[pd.DataFrame] = []
    corridor_link_frames: List[pd.DataFrame] = []
    tmc_profile_frames: List[pd.DataFrame] = []
    cube_link_audit_frames: List[pd.DataFrame] = []
    link_scatter_frames: List[pd.DataFrame] = []
    input_files: List[Path] = [canonical_node_pair_mapping]

    for period, period_settings in periods.items():
        product_name = str(mapping_products[period])
        mapping_dir = (paths.mapmatching_run_dir / product_name).resolve()
        if not _is_relative_to(mapping_dir, paths.mapmatching_run_dir):
            raise ValueError(f"Mapping product escapes configured run: {product_name}")
        mapping, route_summary = load_period_mapping(
            mapping_dir,
            canonical_node_pair_mapping=canonical_node_pair_mapping,
            strict_qa_only=bool(settings["strict_qa_only"]),
            strict_qa_statuses=settings["strict_qa_statuses"],
        )
        performance_source = (
            paths.taplite_assignment_dir / period / "link_performance.csv"
        ).resolve()
        performance, minute_lookup = load_link_performance(performance_source)
        link_source = (paths.taplite_assignment_dir / period / "link.csv").resolve()
        cube_volume_column = str(settings["cube_volume_columns"][period])
        link_scatter_frames.append(
            load_period_link_comparison(
                performance_source,
                link_source,
                period=period,
                cube_volume_column=cube_volume_column,
                cube_speed_column=CUBE_SPEED_COLUMNS.get(period),
            )
        )
        cube_performance, cube_minute_lookup, cube_link_audit = (
            load_cube_qvdf_profiles(
                link_source,
                cube_volume_column=cube_volume_column,
                period_start_min=int(period_settings["start_min"]),
                period_end_min=int(period_settings["end_min"]),
                interval_minutes=int(settings["model_source_interval_minutes"]),
                link_ids=mapping["link_id"].dropna().astype(str).unique(),
            )
        )
        expected_minutes = set(
            range(
                int(period_settings["start_min"]),
                int(period_settings["end_min"]),
                int(settings["model_source_interval_minutes"]),
            )
        )
        actual_minutes = set(minute_lookup.values())
        if actual_minutes != expected_minutes:
            raise ValueError(
                f"{period.upper()} TAPlite speed columns do not exactly cover "
                f"the configured period. Missing={sorted(expected_minutes - actual_minutes)}, "
                f"extra={sorted(actual_minutes - expected_minutes)}"
            )
        cube_actual_minutes = set(cube_minute_lookup.values())
        if cube_actual_minutes != expected_minutes:
            raise ValueError(
                f"{period.upper()} Cube QVDF speed columns do not exactly cover "
                f"the configured period. Missing="
                f"{sorted(expected_minutes - cube_actual_minutes)}, "
                f"extra={sorted(cube_actual_minutes - expected_minutes)}"
            )

        mapped_reference = mapping.merge(
            tmc_reference[["corridor", "tmc_code"]].rename(
                columns={"tmc_code": "tmc"}
            ),
            on="tmc",
            how="left",
            validate="many_to_many",
        )

        def joined(values: pd.Series) -> str:
            return ";".join(sorted({str(value) for value in values.dropna()}))

        mapped_link_reference = (
            mapped_reference.groupby("link_id", as_index=False)
            .agg(
                corridors=("corridor", joined),
                tmc_codes=("tmc", joined),
                from_node_id=("from_node_id", "first"),
                to_node_id=("to_node_id", "first"),
                mapped_length_mi=("length_mi", "max"),
            )
        )
        cube_period_audit = mapped_link_reference.merge(
            cube_link_audit,
            on="link_id",
            how="left",
            validate="one_to_one",
        )
        taplite_period_values = performance[["volume", "doc", "P"]].rename(
            columns={
                "volume": "taplite_period_volume",
                "doc": "taplite_period_doc",
                "P": "taplite_period_p_hours",
            }
        )
        cube_period_audit = cube_period_audit.merge(
            taplite_period_values.reset_index(),
            on="link_id",
            how="left",
            validate="one_to_one",
        )
        cube_period_audit.insert(0, "period", period.upper())
        cube_period_audit["taplite_zero_volume"] = (
            pd.to_numeric(
                cube_period_audit["taplite_period_volume"], errors="coerce"
            )
            .fillna(0.0)
            .le(0.0)
        )
        cube_period_audit["cube_positive_volume"] = (
            pd.to_numeric(
                cube_period_audit["cube_period_volume"], errors="coerce"
            )
            .fillna(0.0)
            .gt(0.0)
        )
        cube_period_audit["taplite_zero_cube_positive"] = (
            cube_period_audit["taplite_zero_volume"]
            & cube_period_audit["cube_positive_volume"]
        )

        aligned, audit, inventory, corridor_links, tmc_profiles = build_period_comparison(
            period=period,
            period_settings=period_settings,
            settings=settings,
            profiles=profiles,
            tmc_reference=tmc_reference,
            mapping=mapping,
            route_summary=route_summary,
            performance=performance,
            minute_lookup=minute_lookup,
            cube_performance=cube_performance,
            cube_minute_lookup=cube_minute_lookup,
        )
        aligned_frames.append(aligned)
        audit_frames.append(audit)
        inventory_frames.append(inventory)
        corridor_link_frames.append(corridor_links)
        tmc_profile_frames.append(tmc_profiles)
        cube_link_audit_frames.append(cube_period_audit)
        input_files.extend(
            [
                mapping_dir / "full_tmc_to_link.csv",
                mapping_dir / "full_route_match_summary.csv",
                performance_source,
                link_source,
            ]
        )

    aligned_all = pd.concat(aligned_frames, ignore_index=True)
    audit_all = pd.concat(audit_frames, ignore_index=True)
    inventory_all = pd.concat(inventory_frames, ignore_index=True)
    corridor_links_all = pd.concat(corridor_link_frames, ignore_index=True)
    tmc_profiles_all = pd.concat(tmc_profile_frames, ignore_index=True)
    cube_link_audit_all = pd.concat(cube_link_audit_frames, ignore_index=True)
    link_scatter_all = pd.concat(link_scatter_frames, ignore_index=True)
    corridor_scatter_all = build_corridor_period_comparison(
        link_scatter_all,
        corridor_links_all,
    )
    scatter_metrics = build_scatter_metrics(
        link_scatter_all,
        corridor_scatter_all,
    )
    daily, period_table, overall = build_metrics_tables(
        profiles, aligned_all, audit_all, settings=settings
    )
    episodes = build_episode_table(
        aligned_all, interval_minutes=int(settings["comparison_interval_minutes"])
    )
    validation = validate_results(
        profiles,
        aligned_all,
        tmc_profiles_all,
        daily,
        period_table,
        settings=settings,
    )
    general_purpose_tmcs, visualization_scope = load_general_purpose_tmc_codes(
        paths.mapmatching_run_dir,
        mapping_products,
    )
    visualization_profiles = tmc_profiles_all.loc[
        tmc_profiles_all["tmc_code"].astype("string").str.strip().isin(
            general_purpose_tmcs
        )
    ].copy()
    visualization_scope.update(
        {
            "source_profile_tmc_count": int(tmc_profiles_all["tmc_code"].nunique()),
            "visualized_tmc_count": int(
                visualization_profiles["tmc_code"].nunique()
            ),
            "excluded_profile_tmc_count": int(
                tmc_profiles_all["tmc_code"].nunique()
                - visualization_profiles["tmc_code"].nunique()
            ),
            "visualized_corridor_count": int(
                visualization_profiles["corridor"].nunique()
            ),
        }
    )
    figure_task_count = int(visualization_profiles["corridor"].nunique())
    worker_plan = recommend_workers(
        figure_task_count,
        target_fraction=float(settings.get("worker_fraction", 0.50)),
        sample_seconds=float(settings.get("worker_sample_seconds", 0.25)),
        explicit_workers=workers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(settings.get("write_figures", True)):
        from .figures import create_corridor_figures

        (
            figure_manifest,
            selected_tmc_profiles,
            selected_tmc_period_metrics,
        ) = create_corridor_figures(
            visualization_profiles,
            output_dir,
            settings=settings,
            workers=worker_plan.workers,
        )
        scatter_manifest = create_scatter_figures(
            link_scatter_all,
            corridor_scatter_all,
            scatter_metrics,
            output_dir,
            figure_dpi=int(settings.get("scatter_figure_dpi", 120)),
        )
    else:
        figure_manifest = pd.DataFrame()
        selected_tmc_profiles = pd.DataFrame()
        selected_tmc_period_metrics = pd.DataFrame()
        scatter_manifest = pd.DataFrame()
    outputs = {
        "01-corridor-results/corridor_metrics.csv": daily,
        "01-corridor-results/corridor_period_metrics.csv": period_table,
        "01-corridor-results/daily_corridor_profiles.csv": aligned_all,
        "01-corridor-results/overall_metrics.csv": overall,
        "02-tmc-results/tmc_daily_profiles.csv": tmc_profiles_all,
        "02-tmc-results/selected_tmc_profiles.csv": selected_tmc_profiles,
        "02-tmc-results/selected_tmc_period_metrics.csv": selected_tmc_period_metrics,
        "03-congestion-results/congestion_episodes.csv": episodes,
        "04-network-mapping/corridor_tmc_gmns_link_mapping.csv": inventory_all,
        "04-network-mapping/corridor_gmns_links_used.csv": corridor_links_all,
        "04-network-mapping/cube_qvdf_link_period_audit.csv": cube_link_audit_all,
        "05-quality-assurance/corridor_mapping_audit.csv": audit_all,
        "05-quality-assurance/corridor_membership_audit.csv": membership_audit,
        "06-figures/figure_manifest.csv": figure_manifest,
        "08-volume-vmt-vht-comparison/data/link_period_comparison.csv": link_scatter_all,
        "08-volume-vmt-vht-comparison/data/corridor_period_comparison.csv": corridor_scatter_all,
        "08-volume-vmt-vht-comparison/data/scatter_metrics.csv": scatter_metrics,
        "08-volume-vmt-vht-comparison/scatter_manifest.csv": scatter_manifest,
    }
    for relative_path, frame in outputs.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, float_format="%.6f")

    write_report(output_dir, daily, overall, settings=settings)
    profile_sources = [
        source
        for corridor_dir in sorted(
            paths.cbi_corridors_dir.iterdir(), key=lambda path: path.name
        )
        for source in (
            corridor_dir / "03-profiles" / "average_weekday_profile.csv",
            corridor_dir / "01-input-and-qc" / "link_reference.csv",
        )
        if source.is_file()
    ]
    manifest = {
        "run_id": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": "0.6.1",
        "worker_plan": worker_plan.to_dict(),
        "workspace_scope": _relative_string(paths.codebase_root.parent, paths.workspace_root),
        "settings": settings,
        "data_columns": {
            "taplite_time_dependent_speed": "spd_mph_HH:MM",
            "taplite_period_volume": "volume",
            "taplite_period_demand_capacity_ratio": "doc",
            "taplite_period_congestion_duration_hours": "P",
            "cube_period_volume": settings["cube_volume_columns"],
            "cube_time_dependent_speed": "C++ Link_QueueVDF reconstruction",
            "cube_period_speed_for_vht": CUBE_SPEED_COLUMNS,
            "cube_vmt": "cube_period_volume * vdf_length_mi",
            "cube_vht": "cube_period_volume * vdf_length_mi / cube_period_speed_mph",
            "taplite_volume": "link_performance.csv:volume",
            "taplite_vmt": "taplite_volume * vdf_length_mi",
            "taplite_vht": "taplite_volume * link_performance.csv:travel_time / 60",
            "taplite_recorded_vmt_vht": "audit_only_due_to_per-mode_accumulation",
            "dashboard_corridor_membership": "01-input-and-qc/link_reference.csv:tmc_code",
        },
        "visualization_scope": {
            "classification_source": "full_tmc_to_link.csv:facility_class",
            "included_class": "gp",
            "policy": "include only TMCs classified exclusively as gp across mapping products",
            **visualization_scope,
        },
        "cube_qvdf_provenance": {
            "taplite4mpo_kernel_commit": TAPLITE_QVDF_KERNEL_COMMIT,
            "kernel_function": "Link_QueueVDF",
            "source_url": TAPLITE_QVDF_KERNEL_URL,
        },
        "inputs": [
            _relative_string(path, paths.workspace_root)
            for path in profile_sources + input_files
        ],
        "row_counts": {
            filename: int(len(frame)) for filename, frame in outputs.items()
        },
        "validation": validation,
    }
    manifest_path = output_dir / "07-run-metadata" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
