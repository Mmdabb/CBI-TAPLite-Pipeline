"""Derive complete-period GMNS link volumes from CBI observed speeds."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


PROFILE_COLUMNS = [
    "corridor",
    "tmc_code",
    "t_min",
    "avg_weekday_speed_mph",
]
FD_COLUMNS = [
    "tmc_code",
    "capacity_vphpl",
    "speed_at_capacity_mph",
    "critical_density_veh_per_mile_lane",
    "free_flow_speed_model_mph",
    "s3_shape_m",
]
MAPPING_COLUMNS = [
    "corridor",
    "period",
    "tmc_code",
    "link_id",
    "eligible_for_comparison",
]


def _required_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    source: Path | str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes", "y"})
    )


def inverse_s3_flow_from_speed(
    speed_mph: np.ndarray | pd.Series,
    *,
    free_flow_speed_mph: np.ndarray | pd.Series,
    critical_density_vpmpl: np.ndarray | pd.Series,
    shape_m: np.ndarray | pd.Series,
    capacity_vphpl: np.ndarray | pd.Series,
) -> np.ndarray:
    """Reproduce CBI's inverse-S3 synthetic per-lane flow calculation."""

    speed = np.asarray(speed_mph, dtype=float)
    free_flow = np.asarray(free_flow_speed_mph, dtype=float)
    critical_density = np.asarray(critical_density_vpmpl, dtype=float)
    shape = np.asarray(shape_m, dtype=float)
    capacity = np.asarray(capacity_vphpl, dtype=float)
    output = np.full(speed.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(speed)
        & (speed >= 0)
        & np.isfinite(free_flow)
        & (free_flow > 0)
        & np.isfinite(critical_density)
        & (critical_density > 0)
        & np.isfinite(shape)
        & (shape > 0)
        & np.isfinite(capacity)
        & (capacity > 0)
    )
    clipped = np.minimum(speed[valid], 0.99 * free_flow[valid])
    ratio = np.maximum(
        free_flow[valid] / np.maximum(clipped, 1.0),
        1.00001,
    )
    density = critical_density[valid] * np.power(
        np.maximum(np.power(ratio, shape[valid] / 2.0) - 1.0, 1e-8),
        1.0 / shape[valid],
    )
    output[valid] = np.minimum(clipped * density, capacity[valid])
    return output


def load_cbi_observed_profiles(cbi_corridors_dir: Path) -> pd.DataFrame:
    """Load full-day average speeds and the matching inverse-S3 contexts."""

    rows: List[pd.DataFrame] = []
    for corridor_dir in sorted(cbi_corridors_dir.iterdir()):
        if not corridor_dir.is_dir() or corridor_dir.name.startswith("_"):
            continue
        profile_source = (
            corridor_dir / "03-profiles" / "average_weekday_profile.csv"
        )
        fd_source = (
            corridor_dir
            / "02-fundamental-diagram"
            / "link_fd_context.csv"
        )
        if not profile_source.is_file() or not fd_source.is_file():
            continue
        profile = pd.read_csv(
            profile_source,
            usecols=lambda column: column in PROFILE_COLUMNS,
            dtype={"tmc_code": "string"},
        )
        _required_columns(profile, PROFILE_COLUMNS, profile_source)
        profile["corridor"] = corridor_dir.name
        context = pd.read_csv(
            fd_source,
            usecols=lambda column: column in FD_COLUMNS,
            dtype={"tmc_code": "string"},
        )
        _required_columns(context, FD_COLUMNS, fd_source)
        if context["tmc_code"].duplicated().any():
            raise ValueError(f"Duplicate TMC contexts in {fd_source}")
        rows.append(
            profile.merge(
                context,
                on="tmc_code",
                how="left",
                validate="many_to_one",
            )
        )
    if not rows:
        raise FileNotFoundError(
            f"No CBI average-weekday profiles found under {cbi_corridors_dir}"
        )
    profiles = pd.concat(rows, ignore_index=True)
    for column in (
        "t_min",
        "avg_weekday_speed_mph",
        "capacity_vphpl",
        "speed_at_capacity_mph",
        "critical_density_veh_per_mile_lane",
        "free_flow_speed_model_mph",
        "s3_shape_m",
    ):
        profiles[column] = pd.to_numeric(profiles[column], errors="coerce")
    profiles["observed_derived_flow_vphpl"] = inverse_s3_flow_from_speed(
        profiles["avg_weekday_speed_mph"],
        free_flow_speed_mph=profiles["free_flow_speed_model_mph"],
        critical_density_vpmpl=profiles[
            "critical_density_veh_per_mile_lane"
        ],
        shape_m=profiles["s3_shape_m"],
        capacity_vphpl=profiles["capacity_vphpl"],
    )
    profiles["observed_volume_method"] = (
        "cbi_average_weekday_speed_inverse_s3"
    )
    return profiles


def _link_lane_lookup(performance: pd.DataFrame) -> pd.DataFrame:
    _required_columns(performance, ["link_id"], "link_performance")
    current = performance.copy()
    current["_link_key"] = current["link_id"].astype("string").str.strip()
    if "iteration_no" in current:
        current["_iteration"] = pd.to_numeric(
            current["iteration_no"], errors="coerce"
        )
        current = current.sort_values("_iteration").drop_duplicates(
            "_link_key", keep="last"
        )
    elif current["_link_key"].duplicated().any():
        raise ValueError("link_performance has duplicate link IDs")
    lanes = pd.Series(np.nan, index=current.index, dtype=float)
    if {"link_capacity", "lane_capacity"}.issubset(current.columns):
        link_capacity = pd.to_numeric(
            current["link_capacity"], errors="coerce"
        )
        lane_capacity = pd.to_numeric(
            current["lane_capacity"], errors="coerce"
        )
        lanes = (link_capacity / lane_capacity).where(lane_capacity > 0)
    if "lanes" in current:
        fallback = pd.to_numeric(current["lanes"], errors="coerce")
        lanes = lanes.where(lanes.gt(0), fallback)
    current["gmns_lanes"] = lanes
    return current[["_link_key", "gmns_lanes"]]


def derive_period_link_profiles(
    profiles: pd.DataFrame,
    mapping: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    period: str,
    start_min: int,
    end_min: int,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Map complete-period CBI synthetic flows to physical GMNS links."""

    _required_columns(mapping, MAPPING_COLUMNS, "corridor mapping")
    selected_mapping = mapping.copy()
    selected_mapping["period"] = (
        selected_mapping["period"].astype("string").str.upper()
    )
    selected_mapping = selected_mapping[
        selected_mapping["period"].eq(period.upper())
        & _as_bool(selected_mapping["eligible_for_comparison"])
    ][["corridor", "tmc_code", "link_id"]].drop_duplicates()
    selected_mapping["tmc_code"] = (
        selected_mapping["tmc_code"].astype("string").str.strip()
    )
    selected_mapping["_link_key"] = (
        selected_mapping["link_id"].astype("string").str.strip()
    )

    selected_profiles = profiles[
        profiles["t_min"].ge(start_min)
        & profiles["t_min"].lt(end_min)
        & profiles["t_min"].mod(interval_minutes).eq(0)
    ].copy()
    joined = selected_mapping.merge(
        selected_profiles,
        on=["corridor", "tmc_code"],
        how="inner",
        validate="many_to_many",
    )
    joined = joined[
        joined["observed_derived_flow_vphpl"].notna()
        & joined["avg_weekday_speed_mph"].notna()
    ]
    lane_lookup = _link_lane_lookup(performance)
    rows: List[Dict[str, object]] = []
    for (link_key, minute), group in joined.groupby(
        ["_link_key", "t_min"], sort=True
    ):
        tmc_codes = sorted(group["tmc_code"].dropna().astype(str).unique())
        corridors = sorted(group["corridor"].dropna().astype(str).unique())
        rows.append(
            {
                "period": period.upper(),
                "link_id": link_key,
                "t_min": int(minute),
                "time_of_day": f"{int(minute) // 60:02d}:{int(minute) % 60:02d}",
                "observed_speed_mph": float(
                    pd.to_numeric(
                        group["avg_weekday_speed_mph"], errors="coerce"
                    ).median()
                ),
                "observed_derived_flow_vphpl": float(
                    pd.to_numeric(
                        group["observed_derived_flow_vphpl"],
                        errors="coerce",
                    ).median()
                ),
                "tmc_candidate_count": len(tmc_codes),
                "tmc_codes": ";".join(tmc_codes),
                "corridors": ";".join(corridors),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.merge(
        lane_lookup,
        left_on="link_id",
        right_on="_link_key",
        how="left",
        validate="many_to_one",
    ).drop(columns="_link_key")
    result["observed_derived_flow_vph"] = (
        result["observed_derived_flow_vphpl"] * result["gmns_lanes"]
    )
    result["observed_derived_interval_volume"] = (
        result["observed_derived_flow_vph"]
        * float(interval_minutes)
        / 60.0
    )
    result["profile_interval_minutes"] = int(interval_minutes)
    result["observed_volume_method"] = (
        "median_tmc_inverse_s3_flow_times_gmns_lanes"
    )
    return result.sort_values(["link_id", "t_min"], kind="stable")


def enrich_link_performance(
    performance: pd.DataFrame,
    link_profiles: pd.DataFrame,
    *,
    period: str,
    start_min: int,
    end_min: int,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Append observed-speed-derived period and time-dependent volumes."""

    output = performance.copy()
    output["_observed_link_key"] = (
        output["link_id"].astype("string").str.strip()
    )
    expected_intervals = int(math.ceil((end_min - start_min) / interval_minutes))
    if link_profiles.empty:
        output["observed_derived_period_volume"] = np.nan
        output["observed_derived_profile_coverage_pct"] = 0.0
        output["observed_derived_tmc_codes"] = ""
        output["observed_derived_method"] = "no_eligible_observed_mapping"
        return output.drop(columns="_observed_link_key")

    summary_rows: List[Dict[str, object]] = []
    for link_id, group in link_profiles.groupby("link_id", sort=True):
        summary_rows.append(
            {
                "_observed_link_key": str(link_id),
                "observed_derived_period_volume": pd.to_numeric(
                    group["observed_derived_interval_volume"], errors="coerce"
                ).sum(min_count=1),
                "observed_derived_profile_coverage_pct": (
                    group["t_min"].nunique() / expected_intervals * 100.0
                ),
                "observed_derived_tmc_count": int(
                    len(
                        {
                            code
                            for text in group["tmc_codes"].dropna().astype(str)
                            for code in text.split(";")
                            if code
                        }
                    )
                ),
                "observed_derived_tmc_codes": ";".join(
                    sorted(
                        {
                            code
                            for text in group["tmc_codes"].dropna().astype(str)
                            for code in text.split(";")
                            if code
                        }
                    )
                ),
                "observed_derived_method": (
                    "CBI observed speed inverse-S3 per TMC; median per-lane "
                    "flow across mapped TMCs; multiplied by GMNS lanes"
                ),
                "observed_derived_profile_interval_minutes": interval_minutes,
            }
        )
    summary = pd.DataFrame(summary_rows)
    output = output.merge(
        summary,
        on="_observed_link_key",
        how="left",
        validate="many_to_one",
    )
    for source_column, prefix in (
        ("observed_speed_mph", "obs_speed_mph"),
        ("observed_derived_flow_vph", "obs_derived_flow_vph"),
        (
            "observed_derived_interval_volume",
            "obs_derived_volume",
        ),
    ):
        pivot = link_profiles.pivot(
            index="link_id",
            columns="time_of_day",
            values=source_column,
        )
        pivot.columns = [f"{prefix}_{value}" for value in pivot.columns]
        pivot.index = pivot.index.astype("string")
        pivot.index.name = "_observed_link_key"
        output = output.merge(
            pivot.reset_index(),
            on="_observed_link_key",
            how="left",
            validate="many_to_one",
        )
    output["observed_derived_period"] = period.upper()
    return output.drop(columns="_observed_link_key")


def period_summary(
    performance: pd.DataFrame,
    link_profiles: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    period: str,
    start_min: int,
    end_min: int,
    interval_minutes: int,
) -> Dict[str, object]:
    """Return concise coverage and reconciliation statistics."""

    observed_links = int(link_profiles["link_id"].nunique()) if not link_profiles.empty else 0
    source_links = int(
        performance["link_id"].astype("string").str.strip().nunique()
    )
    interval_total = (
        float(
            pd.to_numeric(
                link_profiles["observed_derived_interval_volume"],
                errors="coerce",
            ).sum()
        )
        if not link_profiles.empty
        else 0.0
    )
    unique_enriched = enriched.drop_duplicates("link_id", keep="last")
    period_total = float(
        pd.to_numeric(
            unique_enriched["observed_derived_period_volume"],
            errors="coerce",
        ).sum()
    )
    if not np.isclose(interval_total, period_total, rtol=1e-9, atol=1e-6):
        raise RuntimeError(
            f"Observed volume reconciliation failed for {period}: "
            f"{interval_total} != {period_total}"
        )
    return {
        "period": period.upper(),
        "start_min": start_min,
        "end_min": end_min,
        "profile_interval_minutes": interval_minutes,
        "expected_intervals_per_link": int(
            math.ceil((end_min - start_min) / interval_minutes)
        ),
        "source_link_performance_rows": int(len(performance)),
        "enriched_link_performance_rows": int(len(enriched)),
        "source_link_count": source_links,
        "observed_derived_link_count": observed_links,
        "observed_derived_link_coverage_pct": (
            observed_links / source_links * 100.0 if source_links else np.nan
        ),
        "observed_derived_profile_rows": int(len(link_profiles)),
        "observed_derived_period_volume": period_total,
        "minimum_interval_volume": (
            float(link_profiles["observed_derived_interval_volume"].min())
            if not link_profiles.empty
            else np.nan
        ),
        "maximum_interval_volume": (
            float(link_profiles["observed_derived_interval_volume"].max())
            if not link_profiles.empty
            else np.nan
        ),
        "reconciliation_status": "PASS",
    }


def write_period_outputs(
    stage_root: Path,
    *,
    period: str,
    enriched: pd.DataFrame,
    link_profiles: pd.DataFrame,
) -> Tuple[Path, Path]:
    """Write compressed enriched link performance and long-form profiles."""

    period_root = stage_root / period.lower()
    period_root.mkdir(parents=True, exist_ok=True)
    performance_path = period_root / "link_performance.csv.gz"
    profile_path = period_root / "observed_speed_derived_link_profiles.csv.gz"
    compression: Mapping[str, object] = {
        "method": "gzip",
        "compresslevel": 3,
        "mtime": 0,
    }
    enriched.to_csv(
        performance_path,
        index=False,
        compression=compression,
        float_format="%.6f",
    )
    link_profiles.to_csv(
        profile_path,
        index=False,
        compression=compression,
        float_format="%.6f",
    )
    return performance_path, profile_path
