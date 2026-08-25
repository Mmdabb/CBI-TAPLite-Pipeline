#!/usr/bin/env python3
"""Extract period links with low observed boundary speeds but weak modeled congestion."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corridor_measurement.cube_qvdf import qvdf_link_profile
from corridor_measurement.pipeline import load_settings


STAGE_NAME = "13-low-boundary-low-congestion-link-audit"
PERIODS = ("am", "md", "pm")
TAPLITE_KERNEL_COMMIT = "4ee2920346359f94cf9a7d85f6a25aa4b93ef75e"
TAPLITE_KERNEL_URL = (
    "https://github.com/asu-trans-ai-lab/TAPLite4MPO/blob/"
    f"{TAPLITE_KERNEL_COMMIT}/kernel/src/TAPLite.cpp#L5825-L6005"
)
REPRESENTATIVE_WEEKDAY_DATE = "2000-01-03"
RANDOM_SAMPLE_SEED = 20260810


def parser() -> argparse.ArgumentParser:
    codebase_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--cbi-corridors", type=Path, required=True)
    result.add_argument("--mapmatching-root", type=Path, required=True)
    result.add_argument("--assignment-root", type=Path, required=True)
    result.add_argument("--corridor-measurement-root", type=Path, required=True)
    result.add_argument("--replace", action="store_true")
    return result


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _prepare_performance(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy()
    frame["link_id"] = frame["link_id"].astype("string").str.strip()
    input_rows = len(frame)
    duplicate_rows = int(frame.duplicated("link_id", keep=False).sum())
    if "iteration_no" in frame:
        frame["iteration_no"] = pd.to_numeric(frame["iteration_no"], errors="coerce")
        frame = frame.sort_values(["link_id", "iteration_no"], na_position="first")
    frame = frame.drop_duplicates("link_id", keep="last")
    return frame, {
        "performance_input_rows": int(input_rows),
        "performance_duplicate_link_rows": duplicate_rows,
        "performance_unique_links": int(len(frame)),
    }


def select_links(link: pd.DataFrame, performance: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Left join performance to link.csv and retain only rows satisfying the requested rule."""
    link = link.copy()
    link["link_id"] = link["link_id"].astype("string").str.strip()
    if link["link_id"].duplicated().any():
        duplicates = link.loc[link["link_id"].duplicated(keep=False), "link_id"].head(10)
        raise ValueError(f"link.csv has duplicate link_id values, including {duplicates.tolist()}")

    performance, performance_stats = _prepare_performance(performance)
    performance = performance.rename(
        columns={column: f"performance_{column}" for column in performance if column != "link_id"}
    )
    joined = link.merge(performance, on="link_id", how="left", validate="one_to_one")

    required = [
        "qvdf_start_speed_mph",
        "qvdf_end_speed_mph",
        "performance_cutoff_speed_mph",
        "performance_congestion_ref_speed_mph",
        "performance_P",
        "performance_doc",
        "performance_speed_mph",
        "performance_volume",
        "performance_vt2_mph",
    ]
    missing = [column for column in required if column not in joined]
    if missing:
        raise KeyError(f"Required columns are missing after the left join: {missing}")
    _numeric(joined, required)

    start = joined["qvdf_start_speed_mph"]
    end = joined["qvdf_end_speed_mph"]
    vc = joined["performance_cutoff_speed_mph"]
    vref = joined["performance_congestion_ref_speed_mph"]
    period_speed = joined["performance_speed_mph"]
    vt2 = joined["performance_vt2_mph"].where(joined["performance_vt2_mph"].gt(0.0))

    joined["flag_start_below_cutoff_speed"] = start.notna() & vc.notna() & start.lt(vc)
    joined["flag_start_below_vref"] = start.notna() & vref.notna() & start.lt(vref)
    joined["flag_end_below_cutoff_speed"] = end.notna() & vc.notna() & end.lt(vc)
    joined["flag_end_below_vref"] = end.notna() & vref.notna() & end.lt(vref)
    joined["flag_start_below_cutoff_or_vref"] = (
        joined["flag_start_below_cutoff_speed"] | joined["flag_start_below_vref"]
    )
    joined["flag_end_below_cutoff_or_vref"] = (
        joined["flag_end_below_cutoff_speed"] | joined["flag_end_below_vref"]
    )
    joined["flag_both_observed_boundaries_available"] = start.notna() & end.notna()
    joined["flag_both_boundaries_below_cutoff_or_vref"] = (
        joined["flag_both_observed_boundaries_available"]
        & joined["flag_start_below_cutoff_or_vref"]
        & joined["flag_end_below_cutoff_or_vref"]
    )

    joined["flag_P_below_0_5_hours"] = (
        joined["performance_P"].notna() & joined["performance_P"].lt(0.5)
    )
    joined["flag_doc_below_0_5"] = (
        joined["performance_doc"].notna() & joined["performance_doc"].lt(0.5)
    )
    joined["flag_period_speed_above_both_boundaries"] = (
        period_speed.notna() & start.notna() & end.notna()
        & period_speed.gt(start) & period_speed.gt(end)
    )
    joined["flag_volume_equals_zero"] = (
        joined["performance_volume"].notna() & joined["performance_volume"].eq(0.0)
    )
    joined["flag_vt2_above_both_boundaries"] = (
        vt2.notna() & start.notna() & end.notna() & vt2.gt(start) & vt2.gt(end)
    )

    low_congestion_flags = [
        "flag_P_below_0_5_hours",
        "flag_doc_below_0_5",
        "flag_period_speed_above_both_boundaries",
        "flag_volume_equals_zero",
        "flag_vt2_above_both_boundaries",
    ]
    joined["flag_any_low_congestion_indicator"] = joined[low_congestion_flags].any(axis=1)
    joined["flag_selected"] = (
        joined["flag_both_boundaries_below_cutoff_or_vref"]
        & joined["flag_any_low_congestion_indicator"]
    )
    joined["selection_reasons"] = joined.apply(
        lambda row: ";".join(
            column.removeprefix("flag_") for column in low_congestion_flags if bool(row[column])
        ),
        axis=1,
    )

    selected = joined.loc[joined["flag_selected"]].copy()
    validation_failures = int(
        (~selected["flag_both_boundaries_below_cutoff_or_vref"]
         | ~selected["flag_any_low_congestion_indicator"]
         | ~selected["flag_selected"]).sum()
    )
    if validation_failures:
        raise AssertionError(
            f"{validation_failures} selected rows do not satisfy the complete selection rule"
        )
    selected_duplicate_links = int(selected["link_id"].duplicated().sum())
    if selected_duplicate_links:
        raise AssertionError(f"Selected output contains {selected_duplicate_links} duplicate links")
    audit_columns = [
        "flag_selected",
        "selection_reasons",
        "flag_both_observed_boundaries_available",
        "flag_both_boundaries_below_cutoff_or_vref",
        "flag_start_below_cutoff_or_vref",
        "flag_end_below_cutoff_or_vref",
        "flag_start_below_cutoff_speed",
        "flag_start_below_vref",
        "flag_end_below_cutoff_speed",
        "flag_end_below_vref",
        *low_congestion_flags,
        "flag_any_low_congestion_indicator",
    ]
    selected.insert(0, "analysis_period", "")
    ordered = ["analysis_period", "link_id", *audit_columns]
    ordered.extend(column for column in selected if column not in ordered)
    selected = selected[ordered]

    stats = {
        "link_rows": int(len(link)),
        "joined_rows": int(len(joined)),
        "links_with_performance_match": int(joined["performance_iteration_no"].notna().sum()),
        "links_with_both_observed_boundaries": int(
            joined["flag_both_observed_boundaries_available"].sum()
        ),
        "links_with_low_boundaries": int(
            joined["flag_both_boundaries_below_cutoff_or_vref"].sum()
        ),
        "selected_links": int(len(selected)),
        "selected_rule_validation_failures": validation_failures,
        "selected_duplicate_link_ids": selected_duplicate_links,
        **performance_stats,
    }
    for flag in low_congestion_flags:
        stats[f"selected_{flag.removeprefix('flag_')}"] = int(selected[flag].sum())
    return selected, stats


def _format_clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _attach_canonical_tmc(
    selected: pd.DataFrame,
    *,
    cbi_corridors_dir: Path,
    corridor_run_root: Path,
) -> pd.DataFrame:
    canonical_path = (
        cbi_corridors_dir.parent
        / "shared"
        / "network-mapping"
        / "canonical_node_pair_tmc.csv"
    )
    canonical = pd.read_csv(
        canonical_path,
        dtype={"link_id": "string", "tmc": "string"},
        low_memory=False,
    )
    canonical["link_id"] = canonical["link_id"].str.strip()
    canonical["tmc"] = canonical["tmc"].str.strip()
    canonical = canonical.sort_values(
        ["link_id", "node_pair_tmc_rank", "tmc"], kind="stable"
    ).drop_duplicates("link_id", keep="first")
    canonical = canonical.rename(columns={"tmc": "tmc_code"})
    keep = [
        "link_id",
        "tmc_code",
        "node_pair_tmc_rank",
        "node_pair_tmc_ranking_basis",
        "selected_for_node_pair_lookup",
        "link_tmc_rank",
        "link_tmc_ranking_basis",
        "tmc_link_rank",
        "tmc_link_ranking_basis",
        "distance_to_tmc_ft",
        "bearing_diff_deg",
    ]
    output = selected.merge(
        canonical[[column for column in keep if column in canonical]],
        on="link_id",
        how="left",
        validate="one_to_one",
    )

    mapping_path = (
        corridor_run_root
        / "04-network-mapping"
        / "corridor_tmc_gmns_link_mapping.csv"
    )
    mapping = pd.read_csv(
        mapping_path,
        dtype={"link_id": "string", "tmc_code": "string"},
        low_memory=False,
    )
    mapping["link_id"] = mapping["link_id"].str.strip()
    mapping["tmc_code"] = mapping["tmc_code"].str.strip()
    mapping["confidence"] = pd.to_numeric(mapping["confidence"], errors="coerce")
    mapping["eligible_for_comparison"] = mapping["eligible_for_comparison"].fillna(False)
    mapping = mapping.sort_values(
        ["link_id", "tmc_code", "eligible_for_comparison", "confidence", "corridor"],
        ascending=[True, True, False, False, True],
        kind="stable",
    ).drop_duplicates(["link_id", "tmc_code"], keep="first")
    mapping_keep = [
        "link_id",
        "tmc_code",
        "corridor",
        "direction",
        "road_order",
        "route_sequence",
        "qa_status",
        "confidence",
        "eligible_for_comparison",
    ]
    output = output.merge(
        mapping[[column for column in mapping_keep if column in mapping]],
        on=["link_id", "tmc_code"],
        how="left",
        validate="one_to_one",
    )
    output["corridor_mapping_method"] = np.where(
        output["corridor"].notna(), "exact_canonical_tmc_link_pair", ""
    )
    fallback = mapping.sort_values(
        ["tmc_code", "eligible_for_comparison", "confidence", "corridor"],
        ascending=[True, False, False, True],
        kind="stable",
    ).drop_duplicates("tmc_code", keep="first")
    fallback_columns = [column for column in mapping_keep if column not in {"link_id"}]
    fallback = fallback[fallback_columns].rename(
        columns={
            column: f"fallback_{column}"
            for column in fallback_columns
            if column != "tmc_code"
        }
    )
    output = output.merge(fallback, on="tmc_code", how="left", validate="many_to_one")
    for column in [item for item in mapping_keep if item not in {"link_id", "tmc_code"}]:
        fallback_column = f"fallback_{column}"
        missing = output[column].isna()
        output.loc[missing, column] = output.loc[missing, fallback_column]
        output = output.drop(columns=fallback_column)
    output.loc[
        output["corridor_mapping_method"].eq("") & output["corridor"].notna(),
        "corridor_mapping_method",
    ] = "canonical_tmc_corridor_fallback"
    output.loc[
        output["corridor_mapping_method"].eq("") & output["tmc_code"].notna(),
        "corridor_mapping_method",
    ] = "canonical_tmc_not_in_cbi_corridor_run"
    return output


def _simple_selected_links(selected: pd.DataFrame) -> pd.DataFrame:
    identifier_columns = [
        "analysis_period",
        "corridor",
        "tmc_code",
        "link_id",
        "from_node_id",
        "to_node_id",
        "lanes",
        "capacity",
        "allowed_use",
        "STREETNAME",
    ]
    toll_columns = sorted(
        column
        for column in selected
        if "toll" in column.lower() and not column.startswith("performance_")
    )
    condition_values = [
        "qvdf_start_speed_mph",
        "qvdf_end_speed_mph",
        "performance_cutoff_speed_mph",
        "performance_congestion_ref_speed_mph",
        "performance_P",
        "performance_doc",
        "performance_D",
        "performance_speed_mph",
        "performance_volume",
        "performance_vt2_mph",
        "performance_qvdf_profile_status",
    ]
    qvdf_inputs = [
        "vdf_free_speed_mph",
        "vdf_length_mi",
        "vdf_alpha",
        "vdf_beta",
        "vdf_plf",
        "vdf_cp",
        "vdf_cd",
        "vdf_n",
        "vdf_s",
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "qvdf_profile_mode",
    ]
    mapping_columns = [
        "direction",
        "road_order",
        "route_sequence",
        "qa_status",
        "confidence",
        "eligible_for_comparison",
        "corridor_mapping_method",
        "link_tmc_rank",
        "link_tmc_ranking_basis",
        "tmc_link_rank",
        "tmc_link_ranking_basis",
        "distance_to_tmc_ft",
        "bearing_diff_deg",
    ]
    audit_columns = [
        "selection_reasons",
        *[column for column in selected if column.startswith("flag_")],
    ]
    columns = [
        column
        for column in dict.fromkeys(
            identifier_columns
            + toll_columns
            + condition_values
            + qvdf_inputs
            + mapping_columns
            + audit_columns
        )
        if column in selected
    ]
    simple = selected[columns].copy()
    rename: dict[str, str] = {}
    if "toll" in simple and "TOLL" in simple:
        rename.update({"toll": "gmns_toll", "TOLL": "source_TOLL"})
    simple = simple.rename(columns=rename)
    casefolded = [column.casefold() for column in simple]
    if len(casefolded) != len(set(casefolded)):
        raise AssertionError("Simplified output contains case-insensitive duplicate headers")
    return simple


def _load_observed_weekday_profiles(
    cbi_corridors_dir: Path,
    selected_pairs: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for corridor, group in selected_pairs.dropna(subset=["corridor", "tmc_code"]).groupby(
        "corridor", sort=True
    ):
        source = (
            cbi_corridors_dir
            / str(corridor)
            / "03-profiles"
            / "average_weekday_profile.csv"
        )
        if not source.is_file():
            continue
        tmc_codes = set(group["tmc_code"].astype("string").str.strip())
        frame = pd.read_csv(
            source,
            usecols=["tmc_code", "t_min", "avg_weekday_speed_mph"],
            dtype={"tmc_code": "string"},
            low_memory=False,
        )
        frame["tmc_code"] = frame["tmc_code"].str.strip()
        frame = frame.loc[frame["tmc_code"].isin(tmc_codes)].copy()
        frame.insert(0, "corridor", str(corridor))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["corridor", "tmc_code", "t_min", "avg_weekday_speed_mph"]
        )
    observed = pd.concat(frames, ignore_index=True)
    _numeric(observed, ["t_min", "avg_weekday_speed_mph"])
    observed = observed.drop_duplicates(["corridor", "tmc_code", "t_min"], keep="first")
    return observed.sort_values(["corridor", "tmc_code", "t_min"], kind="stable")


def _observed_speed_at(
    profile: pd.DataFrame,
    minute: float,
) -> tuple[float, bool]:
    if profile.empty or not {"t_min", "avg_weekday_speed_mph"}.issubset(profile.columns):
        return float("nan"), False
    valid = profile.dropna(subset=["t_min", "avg_weekday_speed_mph"]).sort_values("t_min")
    if valid.empty:
        return float("nan"), False
    minutes = valid["t_min"].to_numpy(float)
    speeds = valid["avg_weekday_speed_mph"].to_numpy(float)
    exact = np.flatnonzero(np.isclose(minutes, float(minute)))
    if exact.size:
        return float(speeds[exact[0]]), True
    right = int(np.searchsorted(minutes, float(minute), side="right"))
    if right == 0 or right >= len(minutes):
        return float("nan"), False
    left = right - 1
    span = minutes[right] - minutes[left]
    if span <= 0.0 or span > 15.0 + 1e-9:
        return float("nan"), False
    factor = (float(minute) - minutes[left]) / span
    return float((1.0 - factor) * speeds[left] + factor * speeds[right]), False


def _observed_profile_lookup(
    profile: pd.DataFrame,
    target_minutes: list[int],
) -> dict[int, tuple[float, bool]]:
    if profile.empty or not {"t_min", "avg_weekday_speed_mph"}.issubset(profile.columns):
        return {minute: (float("nan"), False) for minute in target_minutes}
    valid = profile.dropna(subset=["t_min", "avg_weekday_speed_mph"]).sort_values("t_min")
    if valid.empty:
        return {minute: (float("nan"), False) for minute in target_minutes}
    source_minutes = valid["t_min"].to_numpy(float)
    source_speeds = valid["avg_weekday_speed_mph"].to_numpy(float)
    lookup: dict[int, tuple[float, bool]] = {}
    for minute in target_minutes:
        exact = np.flatnonzero(np.isclose(source_minutes, float(minute)))
        if exact.size:
            lookup[minute] = (float(source_speeds[exact[0]]), True)
            continue
        right = int(np.searchsorted(source_minutes, float(minute), side="right"))
        if right == 0 or right >= len(source_minutes):
            lookup[minute] = (float("nan"), False)
            continue
        left = right - 1
        span = source_minutes[right] - source_minutes[left]
        if span <= 0.0 or span > 15.0 + 1e-9:
            lookup[minute] = (float("nan"), False)
            continue
        factor = (float(minute) - source_minutes[left]) / span
        lookup[minute] = (
            float((1.0 - factor) * source_speeds[left] + factor * source_speeds[right]),
            False,
        )
    return lookup


def _minute_from_speed_column(column: str) -> int:
    hour, minute = column.removeprefix("spd_mph_").split(":")
    return int(hour) * 60 + int(minute)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _smoothstep01(value: float) -> float:
    position = min(1.0, max(0.0, value))
    return position * position * (3.0 - 2.0 * position)


def _kernel_reporting_profile(
    link_row: pd.Series,
    perf_row: pd.Series,
    *,
    period_start_min: int,
    period_end_min: int,
) -> dict[str, object]:
    """Mirror the current kernel dispatcher plus Link_QueueVDF reporting profile."""
    volume = _finite(perf_row.get("volume"))
    volume = volume if volume is not None and volume >= 0.0 else 0.0
    mode_value = _finite(link_row.get("qvdf_profile_mode"))
    profile_mode = int(mode_value) if mode_value is not None else -1
    link_type_value = _finite(link_row.get("link_type"))
    link_type = int(link_type_value) if link_type_value is not None else 0
    observed_t2 = _finite(link_row.get("t2_hour"))
    has_observed_t2 = observed_t2 is not None
    start_speed = _finite(link_row.get("qvdf_start_speed_mph"))
    end_speed = _finite(link_row.get("qvdf_end_speed_mph"))
    has_start = start_speed is not None
    has_end = end_speed is not None
    is_freeway = link_type == 1 or (link_type >= 100 and link_type % 100 == 1)

    if profile_mode == 0:
        eligible = False
        predicted_status = "flat_disabled"
    elif profile_mode == 1:
        eligible = True
        predicted_status = "generated_model"
    elif profile_mode == 2:
        eligible = has_observed_t2
        predicted_status = "generated_observed" if eligible else "flat_missing_observation"
    else:
        eligible = is_freeway or has_observed_t2
        predicted_status = (
            "generated_legacy_link_type"
            if eligible and is_freeway
            else "generated_legacy_observed_t2"
            if eligible
            else "flat_legacy_not_selected"
        )

    positive_volume = volume > 0.0
    do_qvdf = eligible and positive_volume
    if not positive_volume:
        predicted_status = "flat_zero_volume"

    parameters = link_row.to_dict()
    cutoff = _finite(perf_row.get("cutoff_speed_mph"))
    if cutoff is not None and cutoff > 0.0:
        parameters["cutoff_speed"] = cutoff
    analytical = qvdf_link_profile(
        parameters,
        volume,
        period_start_min=period_start_min,
        period_end_min=period_end_min,
        interval_minutes=5,
    )
    if do_qvdf:
        generated_boundary = max(
            float(analytical["congestion_ref_speed_mph"]),
            float(analytical["avg_queue_speed_mph"]),
        )
        vt2 = float(analytical["vt2_mph"])
        anchor_margin = max(2.0, 0.10 * max(0.0, generated_boundary - vt2))
        used_low_anchor_connector = has_observed_t2 and (
            (
                has_start
                and start_speed < generated_boundary
                and start_speed <= vt2 + anchor_margin
            )
            or (
                has_end
                and end_speed < generated_boundary
                and end_speed <= vt2 + anchor_margin
            )
        )
        if used_low_anchor_connector:
            predicted_status = "generated_low_anchor_connector"
        return {
            **analytical,
            "predicted_profile_status": predicted_status,
            "reconstruction_method": (
                "kernel_qvdf_with_boundary_anchoring"
                if has_start or has_end
                else "kernel_qvdf_unanchored_no_boundary_speed"
            ),
        }

    scalar_speed = _finite(perf_row.get("speed_mph"))
    if scalar_speed is None or scalar_speed <= 0.0:
        scalar_speed = _finite(link_row.get("vdf_free_speed_mph"))
    scalar_speed = scalar_speed if scalar_speed is not None and scalar_speed > 0.0 else 0.1
    minutes = list(range(period_start_min, period_end_min, 5))
    speed_by_minute = {minute: scalar_speed for minute in minutes}
    use_boundary_fallback = (
        positive_volume
        and profile_mode != 0
        and (has_start or has_end)
    )
    if use_boundary_fallback:
        predicted_status = (
            "smoothed_boundary_missing_observation"
            if profile_mode == 2 and not eligible
            else "smoothed_boundary_legacy_not_selected"
        )
        profile_start = period_start_min
        profile_last = max(period_start_min, period_end_min - 5)
        start = start_speed if has_start else scalar_speed
        end = end_speed if has_end else scalar_speed
        span = float(profile_last - profile_start)
        for minute in minutes:
            factor = (minute - profile_start) / span if span > 1e-9 else 0.0
            smooth = _smoothstep01(factor)
            speed_by_minute[minute] = (1.0 - smooth) * start + smooth * end
        speed_by_minute[profile_start] = start
        if profile_last > profile_start or has_end:
            speed_by_minute[profile_last] = end
        method = "kernel_smoothed_observed_boundary_fallback"
    else:
        method = "kernel_flat_assignment_speed_fallback"

    return {
        **analytical,
        "P": 0.0,
        "speed_by_minute": speed_by_minute,
        "predicted_profile_status": predicted_status,
        "reconstruction_method": method,
    }


def _forced_unanchored_qvdf_profile(
    link_row: pd.Series,
    perf_row: pd.Series,
    *,
    period_start_min: int,
    period_end_min: int,
) -> dict[str, object]:
    """Run Link_QueueVDF while forcing both observed speed anchors unavailable."""
    volume = _finite(perf_row.get("volume"))
    volume = volume if volume is not None and volume >= 0.0 else 0.0
    parameters = link_row.to_dict()
    parameters["qvdf_start_speed_mph"] = float("nan")
    parameters["qvdf_end_speed_mph"] = float("nan")
    cutoff = _finite(perf_row.get("cutoff_speed_mph"))
    if cutoff is not None and cutoff > 0.0:
        parameters["cutoff_speed"] = cutoff
    result = qvdf_link_profile(
        parameters,
        volume,
        period_start_min=period_start_min,
        period_end_min=period_end_min,
        interval_minutes=5,
    )
    return {
        **result,
        "reconstruction_method": "forced_unanchored_link_queue_vdf",
        "ignored_speed_boundaries": True,
    }


def _build_speed_comparison(
    selected: pd.DataFrame,
    performance: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    period: str,
    period_start_min: int,
    period_end_min: int,
) -> pd.DataFrame:
    performance, _ = _prepare_performance(performance)
    performance = performance.set_index("link_id", verify_integrity=True)
    speed_columns = {
        _minute_from_speed_column(column): column
        for column in performance
        if column.startswith("spd_mph_")
    }
    minutes = sorted(
        minute
        for minute in speed_columns
        if period_start_min <= minute < period_end_min
    )
    observed_groups = {
        (str(corridor), str(tmc)): group
        for (corridor, tmc), group in observed.groupby(
            ["corridor", "tmc_code"], sort=False
        )
    }
    observed_lookup = {
        (str(corridor), str(tmc)): _observed_profile_lookup(group, minutes)
        for (corridor, tmc), group in observed.groupby(
            ["corridor", "tmc_code"], sort=False
        )
    }
    rows: list[dict[str, object]] = []
    for _, link_row in selected.sort_values(["corridor", "tmc_code", "link_id"]).iterrows():
        link_id = str(link_row["link_id"])
        if link_id not in performance.index:
            continue
        perf_row = performance.loc[link_id]
        volume = pd.to_numeric(pd.Series([perf_row.get("volume")]), errors="coerce").iloc[0]
        if not np.isfinite(volume) or volume < 0.0:
            continue
        reconstructed = _forced_unanchored_qvdf_profile(
            link_row,
            perf_row,
            period_start_min=period_start_min,
            period_end_min=period_end_min,
        )
        start_available = pd.notna(link_row.get("qvdf_start_speed_mph"))
        end_available = pd.notna(link_row.get("qvdf_end_speed_mph"))
        boundary_mode = "forced_unanchored_ignoring_observed_start_end_speeds"
        corridor_value = link_row.get("corridor", "")
        tmc_value = link_row.get("tmc_code", "")
        corridor = "" if pd.isna(corridor_value) else str(corridor_value)
        tmc_code = "" if pd.isna(tmc_value) else str(tmc_value)
        observed_profile = observed_lookup.get((corridor, tmc_code), {})
        observed_raw_profile = observed_groups.get((corridor, tmc_code), pd.DataFrame())
        observed_t0 = _finite(link_row.get("t0_hour"))
        observed_t2 = _finite(link_row.get("t2_hour"))
        observed_t3 = _finite(link_row.get("t3_hour"))
        observed_duration = float("nan")
        if (
            observed_t0 is not None
            and observed_t2 is not None
            and observed_t3 is not None
            and observed_t0 < observed_t2 < observed_t3
        ):
            observed_duration = max(
                0.0,
                min(period_end_min / 60.0, observed_t3)
                - max(period_start_min / 60.0, observed_t0),
            )
        observed_vt2_speed = float("nan")
        if observed_t2 is not None:
            observed_vt2_speed, _ = _observed_speed_at(
                observed_raw_profile, observed_t2 * 60.0
            )
        reconstructed_p = float(reconstructed["P"])
        reconstructed_p_clipped = min(
            max(0.0, reconstructed_p),
            (period_end_min - period_start_min) / 60.0,
        )
        for minute in minutes:
            try:
                stored_speed = float(perf_row.get(speed_columns[minute]))
            except (TypeError, ValueError):
                stored_speed = float("nan")
            observed_speed, is_original = observed_profile.get(
                minute, (float("nan"), False)
            )
            reconstructed_speed = reconstructed["speed_by_minute"].get(minute, np.nan)
            clock = _format_clock(minute)
            timestamp = f"{REPRESENTATIVE_WEEKDAY_DATE} {clock}:00"
            rows.append(
                {
                    "analysis_period": period.upper(),
                    "corridor": corridor,
                    "tmc_code": tmc_code,
                    "link_id": link_id,
                    "representative_date": REPRESENTATIVE_WEEKDAY_DATE,
                    "time_of_day": f"{clock}:00",
                    "representative_average_weekday_datetime": timestamp,
                    "minute_of_day": minute,
                    "inrix_average_weekday_speed_mph": observed_speed,
                    "inrix_speed_is_original_15min_sample": is_original,
                    "link_performance_time_dependent_speed_mph": stored_speed,
                    "taplite_kernel_reconstructed_speed_mph": reconstructed_speed,
                    "stored_minus_reconstructed_speed_mph": (
                        stored_speed - reconstructed_speed
                        if np.isfinite(stored_speed) and np.isfinite(reconstructed_speed)
                        else np.nan
                    ),
                    "kernel_reconstruction_boundary_mode": boundary_mode,
                    "kernel_reconstruction_method": reconstructed[
                        "reconstruction_method"
                    ],
                    "performance_qvdf_profile_status": perf_row.get("qvdf_profile_status", ""),
                    "performance_volume": volume,
                    "performance_doc": perf_row.get("doc", np.nan),
                    "reconstructed_doc": reconstructed["doc"],
                    "performance_P_hours": perf_row.get("P", np.nan),
                    "reconstructed_P_hours": reconstructed_p,
                    "reconstructed_P_clipped_to_period_hours": reconstructed_p_clipped,
                    "reconstructed_vref_mph": reconstructed[
                        "congestion_ref_speed_mph"
                    ],
                    "reconstructed_avg_QVDF_period_speed_mph": reconstructed[
                        "avg_QVDF_period_speed_mph"
                    ],
                    "reconstructed_vt2_mph": reconstructed["vt2_mph"],
                    "reconstructed_t0_hour": reconstructed["t0"],
                    "reconstructed_t2_hour": reconstructed["t2"],
                    "reconstructed_t3_hour": reconstructed["t3"],
                    "observed_t0_hour": observed_t0,
                    "observed_t2_hour": observed_t2,
                    "observed_t3_hour": observed_t3,
                    "observed_congestion_duration_clipped_hours": observed_duration,
                    "observed_vt2_speed_mph": observed_vt2_speed,
                    "observed_start_boundary_speed_mph": link_row.get(
                        "qvdf_start_speed_mph", np.nan
                    ),
                    "observed_end_boundary_speed_mph": link_row.get(
                        "qvdf_end_speed_mph", np.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    group_keys = ["analysis_period", "tmc_code", "link_id"]
    result["inrix_period_average_speed_mph"] = result.groupby(group_keys)[
        "inrix_average_weekday_speed_mph"
    ].transform("mean")
    result["stored_profile_average_speed_mph"] = result.groupby(group_keys)[
        "link_performance_time_dependent_speed_mph"
    ].transform("mean")
    result["reconstructed_profile_average_speed_mph"] = result.groupby(group_keys)[
        "taplite_kernel_reconstructed_speed_mph"
    ].transform("mean")
    return result


def _plot_random_profiles(
    profiles: pd.DataFrame,
    *,
    period: str,
    output_path: Path,
    seed: int,
    sample_count: int = 10,
) -> pd.DataFrame:
    def display_number(value: object, digits: int = 1) -> str:
        number = _finite(value)
        return f"{number:,.{digits}f}" if number is not None else "n/a"

    pair_columns = ["corridor", "tmc_code", "link_id"]
    eligible = (
        profiles.groupby(pair_columns, as_index=False)
        .agg(
            inrix_values=("inrix_average_weekday_speed_mph", "count"),
            stored_values=("link_performance_time_dependent_speed_mph", "count"),
            reconstructed_values=("taplite_kernel_reconstructed_speed_mph", "count"),
        )
    )
    eligible = eligible.loc[
        eligible[["inrix_values", "stored_values", "reconstructed_values"]].min(axis=1).gt(0)
    ]
    count = min(sample_count, len(eligible))
    sample = eligible.sample(n=count, random_state=seed).sort_values(pair_columns)
    if count == 0:
        return sample

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    ncols = 2
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.8 * nrows), squeeze=False)
    legend_handles = None
    legend_labels = None
    for axis, (_, pair) in zip(axes.ravel(), sample.iterrows()):
        mask = np.logical_and.reduce(
            [profiles[column].astype(str).eq(str(pair[column])) for column in pair_columns]
        )
        group = profiles.loc[mask].sort_values("minute_of_day")
        time = pd.to_datetime(group["representative_average_weekday_datetime"])
        line1 = axis.plot(
            time,
            group["inrix_average_weekday_speed_mph"],
            color="#1F4E79",
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            markevery=3,
            label="INRIX weekday average",
        )
        line2 = axis.plot(
            time,
            group["link_performance_time_dependent_speed_mph"],
            color="#E67E22",
            linewidth=2.1,
            label="Stored link-performance profile",
        )
        line3 = axis.plot(
            time,
            group["taplite_kernel_reconstructed_speed_mph"],
            color="#2E8B57",
            linewidth=2.1,
            label="Unanchored kernel QVDF",
        )
        first = group.iloc[0]
        vref = _finite(first.get("reconstructed_vref_mph"))
        line4 = axis.axhline(
            vref if vref is not None else 0.0,
            color="#C00000",
            linewidth=1.8,
            linestyle="--",
            label="QVDF vref",
            visible=vref is not None,
        )
        if legend_handles is None:
            legend_handles = [line1[0], line2[0], line3[0], line4]
            legend_labels = [handle.get_label() for handle in legend_handles]
        axis.set_title(
            f"{pair['corridor']} | TMC {pair['tmc_code']} | Link {pair['link_id']}"
        )
        axis.set_ylabel("Speed (mph)")
        axis.grid(True, color="#D9E2F3", linewidth=0.8, alpha=0.8)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.tick_params(axis="x", rotation=35)
        annotation = "\n".join(
            [
                (
                    f"Volume={display_number(first.get('performance_volume'), 0)} veh  |  "
                    f"vref={display_number(vref)} mph"
                ),
                (
                    "Average mph: "
                    f"INRIX={display_number(first.get('inrix_period_average_speed_mph'))}, "
                    f"stored={display_number(first.get('stored_profile_average_speed_mph'))}, "
                    f"QVDF={display_number(first.get('reconstructed_profile_average_speed_mph'))}"
                ),
                (
                    "Congestion duration h: "
                    f"observed={display_number(first.get('observed_congestion_duration_clipped_hours'), 2)}, "
                    f"estimated clipped={display_number(first.get('reconstructed_P_clipped_to_period_hours'), 2)}"
                ),
                (
                    "v(t2) mph: "
                    f"observed={display_number(first.get('observed_vt2_speed_mph'))}, "
                    f"estimated={display_number(first.get('reconstructed_vt2_mph'))}"
                ),
                (
                    "Observed boundary mph: "
                    f"start={display_number(first.get('observed_start_boundary_speed_mph'))}, "
                    f"end={display_number(first.get('observed_end_boundary_speed_mph'))}"
                ),
            ]
        )
        axis.text(
            0.015,
            0.97,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#9EADBA",
                "alpha": 0.82,
            },
        )
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    fig.suptitle(
        f"{period.upper()} random selected link–TMC speed profiles",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=4,
            frameon=False,
        )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.945))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sample.insert(0, "analysis_period", period.upper())
    sample.insert(1, "random_seed", seed)
    return sample


def _replace_stage(stage_root: Path, run_root: Path, replace: bool) -> None:
    if stage_root.exists():
        if not replace:
            raise FileExistsError(f"Output stage already exists: {stage_root}")
        resolved = stage_root.resolve()
        if resolved.parent != run_root.resolve() or resolved.name != STAGE_NAME:
            raise ValueError(f"Unsafe replacement target: {stage_root}")
        shutil.rmtree(resolved)
    stage_root.mkdir(parents=True)


def main() -> None:
    args = parser().parse_args()
    run_root = args.corridor_measurement_root.resolve()
    settings, paths = load_settings(
        args.config.resolve(),
        cbi_corridors_dir=args.cbi_corridors,
        mapmatching_run_dir=args.mapmatching_root,
        taplite_assignment_dir=args.assignment_root,
        results_root=run_root,
    )
    stage_root = run_root / STAGE_NAME
    _replace_stage(stage_root, run_root, args.replace)
    simple_root = stage_root / "simple-selected-links"
    profile_root = stage_root / "speed-profile-comparisons"
    figure_root = stage_root / "figures"
    for directory in (simple_root, profile_root, figure_root):
        directory.mkdir(parents=True)

    period_results: dict[str, dict[str, object]] = {}
    sampled_pairs: list[pd.DataFrame] = []
    for period in PERIODS:
        period_root = paths.taplite_assignment_dir / period
        link_path = period_root / "link.csv"
        performance_path = period_root / "link_performance.csv"
        link = pd.read_csv(link_path, dtype={"link_id": "string"}, low_memory=False)
        performance = pd.read_csv(
            performance_path, dtype={"link_id": "string"}, low_memory=False
        )
        selected, stats = select_links(link, performance)
        selected["analysis_period"] = period.upper()
        selected = _attach_canonical_tmc(
            selected,
            cbi_corridors_dir=paths.cbi_corridors_dir,
            corridor_run_root=run_root,
        )
        stats["selected_links_with_canonical_tmc"] = int(selected["tmc_code"].notna().sum())
        stats["selected_links_missing_canonical_tmc"] = int(selected["tmc_code"].isna().sum())
        stats["selected_links_with_corridor"] = int(selected["corridor"].notna().sum())
        stats["selected_tmc_codes_not_in_cbi_corridor_run"] = sorted(
            selected.loc[selected["corridor"].isna(), "tmc_code"].dropna().unique().tolist()
        )

        output_name = f"{period}_selected_low_boundary_low_congestion_links.csv"
        output_path = stage_root / output_name
        selected.to_csv(output_path, index=False, float_format="%.6f")
        written = pd.read_csv(output_path, usecols=["link_id", "flag_selected"])
        if len(written) != len(selected) or not written["flag_selected"].all():
            raise AssertionError(f"Written output validation failed for {period.upper()}")
        stats["written_output_rows"] = int(len(written))

        simple = _simple_selected_links(selected)
        simple_name = f"{period}_selected_links_simple.csv"
        simple_path = simple_root / simple_name
        simple.to_csv(simple_path, index=False, float_format="%.6f")
        stats["simple_output_rows"] = int(len(simple))
        stats["simple_output_columns"] = int(simple.shape[1])

        observed = _load_observed_weekday_profiles(
            paths.cbi_corridors_dir,
            selected[["corridor", "tmc_code"]].drop_duplicates(),
        )
        definition = settings["periods"][period]
        profiles = _build_speed_comparison(
            selected,
            performance,
            observed,
            period=period,
            period_start_min=int(definition["start_min"]),
            period_end_min=int(definition["end_min"]),
        )
        profile_name = f"{period}_selected_link_tmc_three_speed_profiles.csv"
        profile_path = profile_root / profile_name
        profiles.to_csv(profile_path, index=False, float_format="%.6f")
        profile_keys = ["analysis_period", "tmc_code", "link_id", "minute_of_day"]
        if profiles.duplicated(profile_keys).any():
            raise AssertionError(f"Duplicate profile rows found for {period.upper()}")
        if not profiles["minute_of_day"].between(
            int(definition["start_min"]), int(definition["end_min"]) - 1
        ).all():
            raise AssertionError(f"Out-of-period profile timestamp found for {period.upper()}")
        stats["speed_profile_output_rows"] = int(len(profiles))
        stats["speed_profile_unique_link_tmc_pairs"] = int(
            profiles[["tmc_code", "link_id"]].drop_duplicates().shape[0]
        )
        stats["speed_profile_inrix_nonmissing_rows"] = int(
            profiles["inrix_average_weekday_speed_mph"].notna().sum()
        )
        stats["speed_profile_stored_nonmissing_rows"] = int(
            profiles["link_performance_time_dependent_speed_mph"].notna().sum()
        )
        stats["speed_profile_reconstructed_nonmissing_rows"] = int(
            profiles["taplite_kernel_reconstructed_speed_mph"].notna().sum()
        )
        absolute_difference = profiles["stored_minus_reconstructed_speed_mph"].abs()
        stats["stored_vs_reconstructed_mae_mph"] = float(absolute_difference.mean())
        stats["stored_vs_reconstructed_median_abs_error_mph"] = float(
            absolute_difference.median()
        )
        stats["stored_vs_reconstructed_max_abs_error_mph"] = float(
            absolute_difference.max()
        )
        stats["inrix_original_15min_rows"] = int(
            profiles["inrix_speed_is_original_15min_sample"].sum()
        )
        stats["inrix_interpolated_5min_rows"] = int(
            (
                profiles["inrix_average_weekday_speed_mph"].notna()
                & ~profiles["inrix_speed_is_original_15min_sample"]
            ).sum()
        )
        stats["kernel_reconstruction_boundary_modes"] = {
            str(key): int(value)
            for key, value in profiles[
                ["link_id", "kernel_reconstruction_boundary_mode"]
            ]
            .drop_duplicates("link_id")["kernel_reconstruction_boundary_mode"]
            .value_counts()
            .items()
        }
        per_link_profile = profiles.drop_duplicates("link_id")
        stats["kernel_reconstruction_methods"] = {
            str(key): int(value)
            for key, value in per_link_profile["kernel_reconstruction_method"]
            .value_counts(dropna=False)
            .items()
        }
        stats["performance_qvdf_profile_statuses"] = {
            str(key): int(value)
            for key, value in profiles[["link_id", "performance_qvdf_profile_status"]]
            .drop_duplicates("link_id")["performance_qvdf_profile_status"]
            .value_counts(dropna=False)
            .items()
        }

        figure_name = f"{period}_random_10_link_tmc_three_speed_profiles.png"
        sample = _plot_random_profiles(
            profiles,
            period=period,
            output_path=figure_root / figure_name,
            seed=RANDOM_SAMPLE_SEED + PERIODS.index(period),
            sample_count=10,
        )
        sampled_pairs.append(sample)
        stats["random_profile_pairs_plotted"] = int(len(sample))
        period_results[period.upper()] = {
            **stats,
            "link_source": str(link_path.resolve()),
            "link_performance_source": str(performance_path.resolve()),
            "output": output_name,
            "simple_output": str(Path("simple-selected-links") / simple_name),
            "speed_profile_output": str(Path("speed-profile-comparisons") / profile_name),
            "random_profile_figure": str(Path("figures") / figure_name),
        }

    sample_index = pd.concat(sampled_pairs, ignore_index=True)
    sample_index.to_csv(figure_root / "random_profile_sample_index.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_assignment": str(paths.taplite_assignment_dir.resolve()),
        "source_corridor_run": str(run_root.resolve()),
        "join": "link.csv LEFT JOIN final link_performance.csv row ON link_id",
        "boundary_rule": (
            "Both qvdf_start_speed_mph and qvdf_end_speed_mph must be present; each must "
            "be below cutoff_speed_mph OR congestion_ref_speed_mph."
        ),
        "low_congestion_rule": (
            "P < 0.5 OR doc < 0.5 OR speed_mph > both observed boundaries OR volume = 0 "
            "OR positive vt2_mph > both observed boundaries."
        ),
        "final_selection_rule": "boundary_rule AND low_congestion_rule",
        "canonical_tmc_mapping": str(
            (
                paths.cbi_corridors_dir.parent
                / "shared"
                / "network-mapping"
                / "canonical_node_pair_tmc.csv"
            ).resolve()
        ),
        "speed_profile_interval_minutes": 5,
        "inrix_profile_method": (
            "Final-QC average_weekday_speed_mph at original 15-minute observations; "
            "linear interpolation only between adjacent observations exactly 15 minutes apart."
        ),
        "representative_average_weekday_date": REPRESENTATIVE_WEEKDAY_DATE,
        "representative_date_note": (
            "A fixed Monday used only to form sortable plotting timestamps; it is not an "
            "INRIX observation date."
        ),
        "taplite_kernel_commit": TAPLITE_KERNEL_COMMIT,
        "taplite_kernel_source": TAPLITE_KERNEL_URL,
        "kernel_reconstruction": (
            "Python mirror of Link_QueueVDF forced to treat qvdf_start_speed_mph and "
            "qvdf_end_speed_mph as unavailable for every link. Observed congestion timing "
            "t0/t2/t3 is retained; no start/end speed anchor or dispatcher boundary fallback "
            "is applied to the reconstructed profile."
        ),
        "random_profile_sample_count_per_period": 10,
        "random_profile_seed_base": RANDOM_SAMPLE_SEED,
        "figure_congestion_duration": (
            "Observed t0-t3 and reconstructed analytical P are clipped to the assignment "
            "period for annotation; raw reconstructed P is also retained in the profile CSV."
        ),
        "periods": period_results,
    }
    (stage_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    readme = [
        "# Low-boundary / low-congestion link audit",
        "",
        "Each period CSV contains only selected network links. All original `link.csv`",
        "columns are retained, and the final `link_performance.csv` row is left-joined by",
        "`link_id` with a `performance_` prefix.",
        "",
        "A link is retained when both observed QVDF boundary speeds are available and each",
        "boundary is below `cutoff_speed_mph` or `congestion_ref_speed_mph`, AND at least",
        "one of these weak-congestion indicators is true:",
        "",
        "- `P < 0.5` hour",
        "- `doc < 0.5`",
        "- period `speed_mph` is above both observed boundary speeds",
        "- `volume = 0`",
        "- positive `vt2_mph` is above both observed boundary speeds",
        "",
        "The `flag_*` fields and `selection_reasons` show why every retained row qualified.",
        "",
        "## Simplified selected-link files",
        "",
        "`simple-selected-links/` contains one AM, MD, and PM CSV with the canonical winning",
        "TMC, link and node identifiers, lanes, capacity, allowed-use and toll fields, every",
        "condition value, QVDF input, selection flag, and matching diagnostic.",
        "",
        "## Three-speed comparison files",
        "",
        "`speed-profile-comparisons/` contains five-minute long-form files comparing:",
        "",
        "1. Final-QC INRIX average-weekday speed. Original 15-minute samples are identified",
        "   by a flag; only the two intervening five-minute values are linearly interpolated.",
        "2. The stored `spd_mph_HH:MM` value from `link_performance.csv`.",
        "3. An independent raw `Link_QueueVDF` reconstruction forced to ignore",
        "   `qvdf_start_speed_mph` and `qvdf_end_speed_mph` for every link. Observed",
        "   congestion timing `t0/t2/t3` remains available, but speed anchors and the",
        "   dispatcher boundary fallback are never applied to this reconstructed series.",
        "",
        f"The timestamp date `{REPRESENTATIVE_WEEKDAY_DATE}` is a fixed Monday used only for",
        "sorting and plotting an average weekday; it is not an INRIX observation date.",
        "Canonical TMCs that are absent from the CBI corridor run remain in the files with",
        "blank corridor and INRIX-speed fields; they are identified by `corridor_mapping_method`.",
        "",
        "## Random profile figures",
        "",
        "`figures/` contains a deterministic random sample of 10 link-TMC panels for each",
        "period plus `random_profile_sample_index.csv` with the selected pairs and seeds.",
        "Each panel includes a dashed reconstructed `vref` line and an annotation with volume,",
        "three period-average speeds, observed/estimated congestion duration and v(t2), and",
        "the observed start/end speed anchors that were intentionally ignored by reconstruction.",
        "The plotted estimated duration is clipped to the assignment period; the profile CSV",
        "retains both clipped and raw analytical `P` values.",
        "",
        f"Kernel source: {TAPLITE_KERNEL_URL}",
        "See `run_manifest.json` for source paths, methods, counts, and validation results.",
    ]
    (stage_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(stage_root), "periods": period_results}, indent=2))


if __name__ == "__main__":
    main()
