from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CorridorSpec, PipelineSettings
from .network_mapping import load_canonical_mapping


LOGGER = logging.getLogger("cbi")
_MODEL_LINK_ATTRIBUTES_CACHE: dict[Path, pd.DataFrame] = {}

CANONICAL_REQUIRED_COLUMNS = {
    "sensor_uid",
    "tmc_code",
    "link_id",
    "network_link_id",
    "network_from_node_id",
    "network_to_node_id",
    "network_link_type",
    "network_mapping_status",
    "datetime",
    "speed_mph",
    "corridor",
    "direction",
    "road_order",
    "length_mi",
    "lanes",
    "lanes_source",
    "reference_speed_mph",
    "reference_speed_source",
    "capacity_prior_vphpl",
    "capacity_source",
    "has_volume",
    "flow_synthetic",
    "source_format",
}


def _latest_tmc_metadata(path: Path) -> tuple[pd.DataFrame, int]:
    """Keep the newest record per TMC while retaining the original link order."""

    metadata = pd.read_csv(path, encoding="utf-8-sig")
    metadata["tmc"] = metadata["tmc"].astype(str)
    metadata["_original_order"] = np.arange(len(metadata))
    first_order = metadata.groupby("tmc", sort=False)["_original_order"].min()
    before = len(metadata)
    if "active_start_date" in metadata:
        metadata["_active_start"] = pd.to_datetime(
            metadata["active_start_date"], errors="coerce", utc=True
        )
        metadata = metadata.sort_values(["tmc", "_active_start", "_original_order"])
    metadata = metadata.drop_duplicates("tmc", keep="last")
    metadata["_corridor_order"] = metadata["tmc"].map(first_order)
    metadata = metadata.sort_values("_corridor_order").reset_index(drop=True)
    metadata["link_id"] = np.arange(1, len(metadata) + 1)
    return metadata, before - len(metadata)


def _model_link_attributes(path: Path | None) -> pd.DataFrame:
    columns = [
        "tmc",
        "lanes",
        "capacity",
        "network_free_speed_mph",
        "network_link_type",
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_path_link_count",
        "network_match_distance_ft",
        "network_bearing_diff_deg",
        "network_match_score",
        "network_match_available_weight",
        "network_geometry_overlap_score",
        "network_road_name_agreement_score",
        "network_direction_compatibility_score",
        "network_functional_class_compatibility_score",
        "network_relative_position_score",
        "network_observation_quality_score",
        "network_length_compatibility_score",
        "network_link_tmc_rank",
        "network_tmc_link_rank",
        "network_node_pair_tmc_rank",
        "network_selected_for_node_pair_lookup",
    ]
    if path is None or not path.is_file():
        return pd.DataFrame(columns=columns)
    resolved = path.resolve()
    cached = _MODEL_LINK_ATTRIBUTES_CACHE.get(resolved)
    if cached is not None:
        return cached
    model = load_canonical_mapping(resolved)
    link_count = (
        model.groupby("tmc")["link_id"]
        .nunique(dropna=True)
        .rename("network_path_link_count")
    )
    # A TMC may touch several links and a link may touch several TMCs.  Only
    # node-pair winners are eligible for modeled CBI pairing so the TMC used
    # for calibration, evaluation, and visualization cannot diverge.  When a
    # winning TMC owns several node pairs, retain its highest-ranked winning
    # link as the one-row CBI reference while the network dictionaries still
    # retain every node-pair win.
    attributes = model[
        model["selected_for_node_pair_lookup"].fillna(False)
    ].copy()
    attributes = attributes.sort_values(
        [
            "tmc",
            "tmc_link_rank",
            "composite_match_score",
            "distance_to_tmc_ft",
            "first_map_occurrence",
            "link_id",
        ],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
        na_position="last",
    ).drop_duplicates("tmc", keep="first")
    attributes = attributes.merge(link_count, on="tmc", how="left")
    attributes = attributes.rename(
        columns={
            "free_speed": "network_free_speed_mph",
            "link_type": "network_link_type",
            "link_id": "network_link_id",
            "from_node_id": "network_from_node_id",
            "to_node_id": "network_to_node_id",
            "distance_to_tmc_ft": "network_match_distance_ft",
            "bearing_diff_deg": "network_bearing_diff_deg",
            "composite_match_score": "network_match_score",
            "composite_available_weight": "network_match_available_weight",
            "geometry_overlap_score": "network_geometry_overlap_score",
            "road_name_agreement_score": "network_road_name_agreement_score",
            "direction_compatibility_score": "network_direction_compatibility_score",
            "functional_class_compatibility_score": (
                "network_functional_class_compatibility_score"
            ),
            "relative_position_score": "network_relative_position_score",
            "observation_quality_score": "network_observation_quality_score",
            "length_compatibility_score": "network_length_compatibility_score",
            "link_tmc_rank": "network_link_tmc_rank",
            "tmc_link_rank": "network_tmc_link_rank",
            "node_pair_tmc_rank": "network_node_pair_tmc_rank",
            "selected_for_node_pair_lookup": (
                "network_selected_for_node_pair_lookup"
            ),
        }
    )
    for column in columns:
        if column not in attributes:
            attributes[column] = np.nan
    attributes = attributes[columns]
    _MODEL_LINK_ATTRIBUTES_CACHE[resolved] = attributes
    return attributes


def prime_model_link_attributes(
    path: Path,
    attributes: pd.DataFrame,
) -> None:
    """Seed a worker with the parent's reduced model-link lookup."""

    _MODEL_LINK_ATTRIBUTES_CACHE[Path(path).resolve()] = attributes


def load_inrix_folder(
    spec: CorridorSpec,
    settings: PipelineSettings,
    logger: logging.Logger | None = None,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one 3_CBI-style INRIX corridor into the canonical internal schema."""

    logger = logger or LOGGER
    folder = spec.path
    readings_path = folder / "Readings.csv"
    metadata_path = folder / "TMC_Identification.csv"
    if not readings_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{spec.key}: missing Readings.csv or TMC_Identification.csv")

    metadata, duplicates_removed = _latest_tmc_metadata(metadata_path)
    header = pd.read_csv(readings_path, nrows=0).columns.tolist()
    required = {"tmc_code", "measurement_tstamp", "speed"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{spec.key}: Readings.csv is missing {missing}")
    desired = [
        "tmc_code",
        "measurement_tstamp",
        "speed",
        "reference_speed",
        "confidence_score",
        "cvalue",
    ]
    usecols = [column for column in desired if column in header]
    confidence_available = "confidence_score" in header

    pieces: list[pd.DataFrame] = []
    rows_read = 0
    confidence_rows_dropped = 0
    for chunk in pd.read_csv(readings_path, usecols=usecols, chunksize=chunksize):
        rows_read += len(chunk)
        chunk["tmc_code"] = chunk["tmc_code"].astype(str)
        chunk["datetime"] = pd.to_datetime(chunk["measurement_tstamp"], errors="coerce")
        chunk["speed"] = pd.to_numeric(chunk["speed"], errors="coerce")
        chunk = chunk.dropna(subset=["tmc_code", "datetime", "speed"])
        off_grid = (
            chunk["datetime"].dt.minute.mod(15).ne(0)
            | chunk["datetime"].dt.second.ne(0)
            | chunk["datetime"].dt.microsecond.ne(0)
        )
        if off_grid.any():
            example = chunk.loc[off_grid, "measurement_tstamp"].iloc[0]
            raise ValueError(
                f"{spec.key}: Readings.csv contains non-15-minute RITIS "
                f"timestamps (example: {example!r}). Rebuild this corridor "
                "from a validated 15-minute observed-speed source."
            )
        if confidence_available:
            confidence = pd.to_numeric(chunk["confidence_score"], errors="coerce")
            keep = confidence >= settings.confidence_min
            confidence_rows_dropped += int((~keep).sum())
            chunk = chunk[keep]
        chunk["datetime"] = chunk["datetime"].dt.floor(f"{settings.interval_minutes}min")
        aggregation = {"speed": "median"}
        if "reference_speed" in chunk:
            chunk["reference_speed"] = pd.to_numeric(
                chunk["reference_speed"], errors="coerce"
            )
            aggregation["reference_speed"] = "median"
        pieces.append(
            chunk.groupby(["tmc_code", "datetime"], as_index=False).agg(aggregation)
        )
    if not pieces:
        raise ValueError(f"{spec.key}: no usable speed observations")

    data = pd.concat(pieces, ignore_index=True)
    aggregation = {"speed": "median"}
    if "reference_speed" in data:
        aggregation["reference_speed"] = "median"
    data = data.groupby(["tmc_code", "datetime"], as_index=False).agg(aggregation)
    data = data.dropna(subset=["speed"])

    metadata_columns = {
        "tmc": "tmc_code",
        "miles": "length_mi",
    }
    metadata = metadata.rename(columns=metadata_columns)
    keep_metadata = [
        column
        for column in (
            "tmc_code",
            "link_id",
            "road",
            "direction",
            "length_mi",
            "road_order",
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
        )
        if column in metadata
    ]
    data = data.merge(metadata[keep_metadata], on="tmc_code", how="inner")

    model = _model_link_attributes(spec.model_link_map).rename(columns={"tmc": "tmc_code"})
    if not model.empty:
        # CBI calibration uses the same frozen node-pair-winning TMC
        # population as network resources, model evaluation, and dashboards.
        # A non-winning observed TMC must not be calibrated as an unmapped
        # surrogate and then disappear from downstream evaluation.
        data = data.merge(model, on="tmc_code", how="inner")
        if data.empty:
            raise ValueError(
                f"{spec.key} has no observations for a frozen node-pair-winning TMC"
            )
    if "lanes" not in data:
        data["lanes"] = np.nan
    if "capacity" not in data:
        data["capacity"] = np.nan
    if "reference_speed" not in data:
        data["reference_speed"] = np.nan
    for column in (
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_path_link_count",
        "network_match_distance_ft",
        "network_bearing_diff_deg",
        "network_free_speed_mph",
        "network_link_type",
    ):
        if column not in data:
            data[column] = np.nan

    lanes_numeric = pd.to_numeric(data["lanes"], errors="coerce")
    data["lanes_source"] = np.where(
        lanes_numeric.notna(), "mapped_network", "default_one_lane"
    )
    data["lanes"] = lanes_numeric.fillna(1.0).clip(lower=1.0)
    capacity_numeric = pd.to_numeric(data["capacity"], errors="coerce")
    data["capacity_source"] = np.where(
        capacity_numeric.notna(), "mapped_network", "corridor_default"
    )
    data["capacity_prior_vphpl"] = (
        capacity_numeric
        .fillna(float(spec.capacity_vphpl))
        .clip(lower=1.0)
    )
    reference_speed = pd.to_numeric(data["reference_speed"], errors="coerce")
    network_speed = pd.to_numeric(
        data["network_free_speed_mph"], errors="coerce"
    )
    data["reference_speed_mph"] = (
        reference_speed.fillna(network_speed).fillna(float(spec.free_flow_mph))
    )
    data["reference_speed_source"] = np.select(
        [reference_speed.notna(), network_speed.notna()],
        ["inrix_reference_speed", "mapped_network_free_speed"],
        default="corridor_default",
    )
    data["speed_mph"] = pd.to_numeric(data["speed"], errors="coerce")
    data["sensor_uid"] = spec.key + "::" + data["tmc_code"].astype(str)
    data["corridor"] = spec.key
    data["network_mapping_status"] = np.where(
        pd.to_numeric(data["network_link_id"], errors="coerce").notna(),
        "mapped_primary_link",
        "unmapped",
    )
    for column in (
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_path_link_count",
    ):
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).astype("Int64")
    if "direction" not in data:
        data["direction"] = ""
    if "road_order" not in data:
        data["road_order"] = data["link_id"]
    data["length_mi"] = pd.to_numeric(data.get("length_mi"), errors="coerce").fillna(0.5)
    data["has_volume"] = False
    data["flow_synthetic"] = True
    data["source_format"] = "inrix_tmc"
    data["flow_vph"] = np.nan
    data["density_vpm"] = np.nan
    data["weekday"] = data["datetime"].dt.weekday
    data["date"] = data["datetime"].dt.date.astype(str)
    data["t_min"] = data["datetime"].dt.hour * 60 + data["datetime"].dt.minute

    audit = {
        "rows_read": int(rows_read),
        "rows_after_resample": int(len(data)),
        "sensors": int(data["sensor_uid"].nunique()),
        "metadata_duplicates_removed": int(duplicates_removed),
        "confidence_score_available": bool(confidence_available),
        "confidence_rows_dropped": int(confidence_rows_dropped),
        "model_map_matches": int(data.loc[data["capacity"].notna(), "sensor_uid"].nunique()),
    }
    logger.info(
        "Loaded %s: %s rows, %s links; confidence filter=%s; model-map matches=%s",
        spec.key,
        f"{len(data):,}",
        data["sensor_uid"].nunique(),
        confidence_available,
        audit["model_map_matches"],
    )
    validate_canonical(data)
    return data.sort_values(["sensor_uid", "datetime"]).reset_index(drop=True), audit


def load_average_weekday(
    spec: CorridorSpec,
    settings: PipelineSettings,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load an NVTA bundled average-weekday CSV into the same canonical schema."""

    logger = logger or LOGGER
    data = pd.read_csv(spec.path)
    required = {
        "link_id",
        "t_min",
        "avg_weekday_speed_mph",
        "length_mi",
        "lanes",
        "avg_weekday_flow_veh_per_hr_lane",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{spec.key}: average-weekday input is missing {missing}")
    data["link_id"] = pd.to_numeric(data["link_id"], errors="raise").astype(int)
    data["t_min"] = pd.to_numeric(data["t_min"], errors="raise").astype(int)
    data["datetime"] = pd.Timestamp("2000-01-03") + pd.to_timedelta(data["t_min"], unit="m")
    data["speed_mph"] = pd.to_numeric(data["avg_weekday_speed_mph"], errors="coerce")
    data["supplied_flow_vphpl"] = pd.to_numeric(
        data["avg_weekday_flow_veh_per_hr_lane"], errors="coerce"
    )
    data["flow_vph"] = (
        data["supplied_flow_vphpl"] if spec.data_mode == "measured" else np.nan
    )
    data["density_vpm"] = np.where(
        pd.to_numeric(data["flow_vph"], errors="coerce").notna()
        & data["speed_mph"].gt(1.0),
        data["flow_vph"] / data["speed_mph"],
        np.nan,
    )
    data["sensor_uid"] = spec.key + "::" + data["link_id"].astype(str)
    data["tmc_code"] = data["link_id"].astype(str)
    data["network_link_id"] = pd.Series(
        pd.NA, index=data.index, dtype="Int64"
    )
    data["network_from_node_id"] = pd.Series(
        pd.NA, index=data.index, dtype="Int64"
    )
    data["network_to_node_id"] = pd.Series(
        pd.NA, index=data.index, dtype="Int64"
    )
    data["network_path_link_count"] = pd.Series(
        pd.NA, index=data.index, dtype="Int64"
    )
    data["network_match_distance_ft"] = np.nan
    data["network_bearing_diff_deg"] = np.nan
    data["network_free_speed_mph"] = np.nan
    data["network_link_type"] = np.nan
    data["network_mapping_status"] = "unmapped"
    data["corridor"] = spec.key
    data["direction"] = spec.key.rsplit("_", 1)[-1]
    data["road_order"] = data["link_id"]
    data["reference_speed_mph"] = float(spec.free_flow_mph)
    data["reference_speed_source"] = "corridor_default"
    data["capacity_prior_vphpl"] = float(spec.capacity_vphpl)
    data["capacity_source"] = "corridor_default"
    data["lanes_source"] = "provided_input"
    data["has_volume"] = spec.data_mode == "measured"
    data["flow_synthetic"] = spec.data_mode != "measured"
    data["source_format"] = "avgweekday_csv"
    data["weekday"] = 0
    data["date"] = "Weekday"
    audit = {
        "rows_read": int(len(data)),
        "rows_after_resample": int(len(data)),
        "sensors": int(data["sensor_uid"].nunique()),
        "metadata_duplicates_removed": 0,
        "confidence_score_available": False,
        "confidence_rows_dropped": 0,
        "model_map_matches": 0,
    }
    logger.info(
        "Loaded %s average weekday: %s links x %s time bins",
        spec.key,
        data["link_id"].nunique(),
        data["t_min"].nunique(),
    )
    validate_canonical(data)
    return data.sort_values(["sensor_uid", "datetime"]).reset_index(drop=True), audit


def load_corridor(
    spec: CorridorSpec,
    settings: PipelineSettings,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if spec.source == "inrix_folder":
        return load_inrix_folder(spec, settings, logger)
    if spec.source == "avgweekday_csv":
        return load_average_weekday(spec, settings, logger)
    raise ValueError(f"Unsupported source: {spec.source}")


def validate_canonical(data: pd.DataFrame) -> None:
    missing = sorted(CANONICAL_REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Canonical preprocessing output is missing {missing}")
    if data.empty:
        raise ValueError("Canonical preprocessing output is empty")
    if data["sensor_uid"].isna().any() or data["datetime"].isna().any():
        raise ValueError("Canonical preprocessing output has missing sensor/time identifiers")


def build_average_weekday(qc: pd.DataFrame) -> pd.DataFrame:
    """Build the one-profile-per-link product from repaired weekday speeds."""

    work = qc.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if not work["date"].astype(str).eq("Weekday").all():
        work = work[work["datetime"].dt.weekday < 5].copy()
    work["t_min"] = work["datetime"].dt.hour * 60 + work["datetime"].dt.minute
    work["model_speed_mph"] = pd.to_numeric(
        work["speed_mph_clean_repaired"], errors="coerce"
    )
    aggregation = {
        "speed_mph": ("model_speed_mph", "mean"),
        "flow_vph": ("flow_vph", "mean"),
        "n_days": ("date", "nunique"),
    }
    for column in (
        "link_id",
        "tmc_code",
        "corridor",
        "direction",
        "road_order",
        "length_mi",
        "lanes",
        "lanes_source",
        "reference_speed_mph",
        "reference_speed_source",
        "capacity_prior_vphpl",
        "capacity_source",
        "has_volume",
        "flow_synthetic",
        "source_format",
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_path_link_count",
        "network_match_distance_ft",
        "network_bearing_diff_deg",
        "network_free_speed_mph",
        "network_link_type",
        "network_mapping_status",
        "corridor_freeflow_speed_mph",
        "fd_capacity_vphpl",
        "fd_vc_mph",
    ):
        if column in work:
            aggregation[column] = (column, "first")
    average = work.groupby(["sensor_uid", "t_min"], as_index=False).agg(**aggregation)
    average["datetime"] = pd.Timestamp("2000-01-03") + pd.to_timedelta(
        average["t_min"], unit="m"
    )
    average["date"] = "Weekday"
    average["weekday"] = 0
    average["density_vpm"] = average["flow_vph"] / average["speed_mph"].where(
        average["speed_mph"] > 1.0
    )
    return average.sort_values(["sensor_uid", "t_min"]).reset_index(drop=True)


def qkv_audit(data: pd.DataFrame, stage: str) -> dict[str, object]:
    q = pd.to_numeric(data["flow_vph"], errors="coerce").to_numpy(dtype=float)
    k = pd.to_numeric(data["density_vpm"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(data["speed_mph"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(q) & np.isfinite(k) & np.isfinite(v) & (np.abs(q) > 1e-6)
    relative = np.abs(q[valid] - k[valid] * v[valid]) / np.maximum(
        np.abs(q[valid]), 1.0
    )
    return {
        "stage": stage,
        "n_checked": int(valid.sum()),
        "median_relative_error": float(np.median(relative)) if len(relative) else np.nan,
        "p95_relative_error": (
            float(np.percentile(relative, 95)) if len(relative) else np.nan
        ),
        "share_within_1pct": float(np.mean(relative <= 0.01)) if len(relative) else np.nan,
    }
