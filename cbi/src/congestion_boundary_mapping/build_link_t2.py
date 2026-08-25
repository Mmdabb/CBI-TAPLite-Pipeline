"""Assign accepted CBI T0/T2/T3 values by regional-link TMC map matching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd


PACKAGE_SRC_ROOT = Path(__file__).resolve().parents[1]

from congestion_boundary_mapping.propagate_t2_by_vdf import propagate_t2_by_vdf
from congestion_boundary_mapping.hybrid_t2 import apply_hybrid_t2, latest_spatial_output
from congestion_boundary_mapping.completion import (
    COMPLETION_MODES,
    complete_boundaries,
)
from cbi.workers import WorkerPlan, recommend_workers


PERIODS = ("AM", "MD", "PM")
PERIOD_ORDER = {"AM": 0, "MD": 1, "PM": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cbi-output-root",
        type=Path,
        required=True,
        help="Explicit CBI corridor-product root.",
    )
    parser.add_argument(
        "--supplemental-cbi-output-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional isolated CBI corridor root supplying approved direct "
            "observations. May be repeated. These roots affect direct mapping "
            "only; Ridge training remains an independent upstream operation."
        ),
    )
    parser.add_argument(
        "--canonical-node-pair-map",
        type=Path,
        default=None,
        help=(
            "Frozen canonical_node_pair_tmc.csv. Defaults to the selected "
            "CBI run shared/network-mapping resource."
        ),
    )
    parser.add_argument(
        "--am-map",
        type=Path,
        required=True,
        help="AM full_tmc_to_link.csv produced by tmc-matching.",
    )
    parser.add_argument(
        "--md-map",
        type=Path,
        required=True,
        help="MD full_tmc_to_link.csv produced by tmc-matching.",
    )
    parser.add_argument(
        "--pm-map",
        type=Path,
        required=True,
        help="PM full_tmc_to_link.csv produced by tmc-matching.",
    )
    parser.add_argument(
        "--network-root",
        type=Path,
        required=True,
        help="Network root containing am/md/pm/link.csv.",
    )
    parser.add_argument(
        "--spatial-output",
        type=Path,
        default=latest_spatial_output(PACKAGE_SRC_ROOT),
        help=(
            "Spatial expanded_link_t2.csv. The default is the mapper-owned "
            "versioned spatial resource."
        ),
    )
    parser.add_argument(
        "--completion-mode",
        choices=COMPLETION_MODES,
        default="ml",
        help=(
            "Final boundary hierarchy: direct-spatial-ML (default) or "
            "direct-spatial-VDF class."
        ),
    )
    parser.add_argument(
        "--ml-run-dir",
        type=Path,
        default=None,
        help=(
            "Ridge artifact directory for ml mode (default: mapper-owned "
            "versioned resources)."
        ),
    )
    parser.add_argument(
        "--comparison-run-dir",
        type=Path,
        default=None,
        help=(
            "Coverage-comparison artifact directory for ml validation "
            "(default: mapper-owned versioned resources)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Stable output directory (default: <cbi-output-root>/outputs/"
            "congestion-boundaries/link-t2)."
        ),
    )
    parser.add_argument(
        "--assignment-only",
        action="store_true",
        help=(
            "Reuse regional_link_t2_long.csv in --output-dir and write only "
            "the native AM/MD/PM period link files."
        ),
    )
    parser.add_argument(
        "--completion-only",
        action="store_true",
        help=(
            "Create only the selected final boundary product from an existing "
            "mapping run; do not rewrite candidate or period link inputs."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Explicit process-worker count. By default the system uses 70%% "
            "of currently free logical-core capacity."
        ),
    )
    parser.add_argument(
        "--worker-fraction",
        type=float,
        default=0.70,
        help="Share of currently free logical-core capacity to use (default: 0.70).",
    )
    return parser.parse_args(argv)


def normalize_tmc(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


def parse_bool(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    return str(path.resolve())


def screening_audit_path(accepted_path: Path) -> Path:
    names = {
        "detected_episodes_accepted.csv": "episode_filter_audit.csv",
        "daily_episodes_accepted.csv": "daily_episode_filter_audit.csv",
        "average_weekday_episodes_accepted.csv": (
            "average_weekday_episode_filter_audit.csv"
        ),
    }
    try:
        audit_name = names[accepted_path.name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported accepted episode input: {accepted_path}"
        ) from exc
    return accepted_path.with_name(audit_name)


def discover_episode_paths(
    output_root: Path,
    *,
    average_weekday: bool,
) -> list[Path]:
    """Discover the numbered contract, with read-only legacy fallback."""

    filename = (
        "average_weekday_episodes_accepted.csv"
        if average_weekday
        else "daily_episodes_accepted.csv"
    )
    current = sorted(
        output_root.glob(f"*/05-episode-filtering/{filename}")
    )
    if current:
        return current
    legacy = (
        "average_weekday_episodes_accepted.csv"
        if average_weekday
        else "detected_episodes_accepted.csv"
    )
    return sorted(output_root.glob(f"*/{legacy}"))


def read_episode_candidate_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"episode_id": "string", "tmc_code": "string"},
        low_memory=False,
    )
    audit_path = screening_audit_path(path)
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Accepted episode input requires its screening audit: {audit_path}"
        )
    audit = pd.read_csv(
        audit_path,
        dtype={"episode_id": "string"},
        low_memory=False,
    )
    required_audit = {"episode_id", "is_clean_valid_episode"}
    missing_audit = sorted(required_audit - set(audit.columns))
    if missing_audit:
        raise ValueError(
            f"{audit_path} is missing screening columns: {missing_audit}"
        )
    if frame["episode_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate episode IDs.")
    if audit["episode_id"].duplicated().any():
        raise ValueError(f"{audit_path} contains duplicate episode IDs.")
    file_ids = set(frame["episode_id"].dropna())
    audited_accepted_ids = set(
        audit.loc[
            parse_bool(audit["is_clean_valid_episode"]),
            "episode_id",
        ].dropna()
    )
    if file_ids != audited_accepted_ids:
        rejected = sorted(file_ids - audited_accepted_ids)[:5]
        omitted = sorted(audited_accepted_ids - file_ids)[:5]
        raise ValueError(
            f"{path} failed its accepted-only contract; "
            f"rejected IDs present={rejected}, accepted IDs omitted={omitted}"
        )
    frame["corridor_output"] = (
        path.parents[1].name
        if path.parent.name == "05-episode-filtering"
        else path.parent.name
    )
    frame["is_clean_valid_episode"] = True
    frame["screening_audit_source"] = str(audit_path.resolve())
    return frame


def load_episode_candidate_files(
    paths: list[Path],
    workers: int,
) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No episode candidate files were found.")
    if workers <= 1 or len(paths) == 1:
        frames = [read_episode_candidate_file(path) for path in paths]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(paths))
        ) as executor:
            frames = list(executor.map(read_episode_candidate_file, paths))
    nonempty_frames = [frame for frame in frames if not frame.empty]
    return pd.concat(
        nonempty_frames if nonempty_frames else frames,
        ignore_index=True,
    )


def normalize_episode_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    required = {
        "episode_id",
        "tmc_code",
        "period",
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "min_speed_mph",
        "P_hr",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(
            f"Accepted episode files are missing columns: {missing}"
        )
    if (
        "is_clean_valid_episode" not in candidates
        or not parse_bool(candidates["is_clean_valid_episode"]).all()
    ):
        raise ValueError(
            "T2 assignment received an episode that was not accepted by screening."
        )
    candidates["tmc"] = normalize_tmc(candidates["tmc_code"])
    candidates["period"] = candidates["period"].astype(str).str.upper()
    for column in ("t0_hour", "t2_hour", "t3_hour", "min_speed_mph", "P_hr"):
        if column in candidates:
            candidates[column] = pd.to_numeric(
                candidates[column], errors="coerce"
            )
    return candidates[
        candidates["tmc"].notna()
        & candidates["period"].isin(PERIODS)
        & candidates["t2_hour"].notna()
    ].copy()


def rank_episode_candidates(
    candidates: pd.DataFrame,
    group_columns: list[str],
    *,
    rank_column: str,
) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["_speed_sort"] = candidates["min_speed_mph"].fillna(np.inf)
    candidates["_duration_sort"] = -candidates["P_hr"].fillna(-np.inf)
    candidates["_t2_sort"] = candidates["t2_hour"].fillna(np.inf)
    candidates = candidates.sort_values(
        [
            *group_columns,
            "_speed_sort",
            "_duration_sort",
            "_t2_sort",
            "corridor_output",
            "episode_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates[rank_column] = (
        candidates.groupby(group_columns, sort=False).cumcount() + 1
    )
    return candidates


def select_episode_representatives(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = rank_episode_candidates(
        normalize_episode_candidates(candidates),
        ["tmc", "period"],
        rank_column="representative_rank",
    )
    candidates["is_representative"] = candidates["representative_rank"].eq(1)
    representatives = candidates[candidates["is_representative"]].copy()
    group_sizes = (
        candidates.groupby(["tmc", "period"], as_index=False)
        .size()
        .rename(columns={"size": "tmc_period_candidate_count"})
    )
    representatives = representatives.merge(
        group_sizes, on=["tmc", "period"], how="left"
    )
    candidates = candidates.drop(
        columns=["_speed_sort", "_duration_sort", "_t2_sort"]
    )
    representatives = representatives.drop(
        columns=["_speed_sort", "_duration_sort", "_t2_sort"], errors="ignore"
    )
    return candidates, representatives


def select_daily_probe_representatives(
    candidates: pd.DataFrame,
    eligible_tmc_periods: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = normalize_episode_candidates(candidates)
    if "date" not in daily:
        raise ValueError("Accepted daily episode inputs are missing date.")
    eligible = eligible_tmc_periods[["tmc", "period"]].drop_duplicates()
    daily = daily.merge(
        eligible,
        on=["tmc", "period"],
        how="inner",
        validate="many_to_one",
    )
    daily = rank_episode_candidates(
        daily,
        ["tmc", "period", "date"],
        rank_column="daily_representative_rank",
    )
    daily["is_daily_representative"] = daily[
        "daily_representative_rank"
    ].eq(1)
    daily_representatives = daily[daily["is_daily_representative"]].copy()

    episode_counts = (
        daily.groupby(["tmc", "period"], as_index=False)
        .size()
        .rename(columns={"size": "daily_probe_episode_count"})
    )
    summary = (
        daily_representatives.groupby(["tmc", "period"], as_index=False)
        .agg(
            t0_hour=("t0_hour", "mean"),
            t2_hour=("t2_hour", "mean"),
            t3_hour=("t3_hour", "mean"),
            P_hr=("P_hr", "mean"),
            min_speed_mph=("min_speed_mph", "mean"),
            corridor_output=("corridor_output", "first"),
            daily_probe_day_count=("date", "nunique"),
            daily_probe_t2_min_hour=("t2_hour", "min"),
            daily_probe_t2_max_hour=("t2_hour", "max"),
            daily_probe_t2_std_hour=(
                "t2_hour",
                lambda values: float(values.std(ddof=0)),
            ),
        )
        .merge(
            episode_counts,
            on=["tmc", "period"],
            how="left",
            validate="one_to_one",
        )
    )
    summary["episode_id"] = (
        "daily_probe_mean__"
        + summary["tmc"].astype(str)
        + "__"
        + summary["period"].astype(str)
    )
    summary["tmc_code"] = summary["tmc"]
    summary["date"] = "DailyMean"
    summary["representative_rank"] = 1
    summary["is_representative"] = True
    summary["tmc_period_candidate_count"] = summary[
        "daily_probe_day_count"
    ]
    summary["t2_source_method"] = "daily_probe_mean"
    daily = daily.drop(
        columns=["_speed_sort", "_duration_sort", "_t2_sort"]
    )
    daily_representatives = daily_representatives.drop(
        columns=["_speed_sort", "_duration_sort", "_t2_sort"]
    )
    return daily, daily_representatives, summary


def combine_average_and_daily_representatives(
    average_representatives: pd.DataFrame,
    daily_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    average_representatives = average_representatives.copy()
    daily_summary = daily_summary.copy()
    average_representatives["t2_source_method"] = "average_weekday"
    for column, default in (
        ("daily_probe_day_count", 0),
        ("daily_probe_episode_count", 0),
        ("daily_probe_t2_min_hour", np.nan),
        ("daily_probe_t2_max_hour", np.nan),
        ("daily_probe_t2_std_hour", np.nan),
    ):
        average_representatives[column] = default

    average_keys = average_representatives[
        ["tmc", "period"]
    ].drop_duplicates().assign(_average_available=True)
    daily_summary = daily_summary.merge(
        average_keys,
        on=["tmc", "period"],
        how="left",
        validate="one_to_one",
    )
    daily_summary["average_weekday_congestion_available"] = daily_summary[
        "_average_available"
    ].fillna(False).astype(bool)
    daily_summary["suppressed_no_average_weekday_congestion"] = ~daily_summary[
        "average_weekday_congestion_available"
    ]
    daily_summary["fallback_used"] = False
    daily_summary = daily_summary.drop(columns="_average_available")
    representatives = average_representatives.sort_values(
        ["tmc", "period"], kind="mergesort"
    ).copy()
    if representatives.duplicated(["tmc", "period"]).any():
        raise ValueError("Combined T2 representatives contain duplicate TMC-period keys.")
    return average_representatives, daily_summary, representatives


def load_episode_candidates(
    output_roots: Path | list[Path] | tuple[Path, ...],
    workers: int,
    eligible_tmc_periods: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    roots = (
        [Path(output_roots)]
        if isinstance(output_roots, (str, Path))
        else [Path(root) for root in output_roots]
    )
    average_paths = sorted(
        path
        for root in roots
        for path in discover_episode_paths(root, average_weekday=True)
    )
    daily_paths = sorted(
        path
        for root in roots
        for path in discover_episode_paths(root, average_weekday=False)
    )
    average_raw = load_episode_candidate_files(average_paths, workers)
    daily_raw = load_episode_candidate_files(daily_paths, workers)
    average_candidates, average_representatives = (
        select_episode_representatives(average_raw)
    )

    daily_candidates, daily_representatives, daily_summary = (
        select_daily_probe_representatives(
            daily_raw,
            eligible_tmc_periods,
        )
    )
    (
        average_representatives,
        daily_summary,
        representatives,
    ) = combine_average_and_daily_representatives(
        average_representatives,
        daily_summary,
    )
    return (
        average_candidates,
        average_representatives,
        daily_candidates,
        daily_representatives,
        daily_summary,
        representatives,
    )


def validate_accepted_t2_lineage(
    average_candidates: pd.DataFrame,
    average_representatives: pd.DataFrame,
    daily_candidates: pd.DataFrame,
    daily_representatives: pd.DataFrame,
    daily_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    long: pd.DataFrame,
) -> dict[str, object]:
    for label, frame in (
        ("average-weekday", average_candidates),
        ("daily", daily_candidates),
    ):
        if (
            "is_clean_valid_episode" not in frame
            or not parse_bool(frame["is_clean_valid_episode"]).all()
        ):
            raise ValueError(
                f"{label} T2 candidates include an episode rejected by screening."
            )

    average_ids = set(average_candidates["episode_id"].dropna())
    if not set(average_representatives["episode_id"].dropna()).issubset(
        average_ids
    ):
        raise ValueError(
            "An average-weekday representative is not in the accepted inputs."
        )
    daily_ids = set(daily_candidates["episode_id"].dropna())
    if not set(daily_representatives["episode_id"].dropna()).issubset(
        daily_ids
    ):
        raise ValueError(
            "A daily representative is not in the accepted inputs."
        )

    for boundary in ("t0_hour", "t2_hour", "t3_hour"):
        expected_daily_means = daily_representatives.groupby(
            ["tmc", "period"]
        )[boundary].mean()
        actual_daily_means = daily_summary.set_index(["tmc", "period"])[
            boundary
        ]
        if not expected_daily_means.index.equals(actual_daily_means.index):
            raise ValueError(
                f"Daily {boundary} summary keys do not match accepted daily probes."
            )
        if not np.allclose(
            expected_daily_means.to_numpy(dtype=float),
            actual_daily_means.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"A daily {boundary} summary is not the mean of accepted "
                "per-day episodes."
            )

    representative_keys = set(
        representatives[["tmc", "period", "episode_id"]]
        .itertuples(index=False, name=None)
    )
    selected = long[long["t2_hour"].notna()]
    selected_keys = set(
        selected[["selected_tmc", "period", "episode_id"]]
        .itertuples(index=False, name=None)
    )
    if not selected_keys.issubset(representative_keys):
        raise ValueError(
            "A regional-link T2 does not trace to an accepted representative."
        )

    return {
        "status": "PASS",
        "rejected_episode_rows_received": 0,
        "accepted_average_weekday_candidates": int(len(average_candidates)),
        "accepted_daily_candidates_for_mapped_periods": int(
            len(daily_candidates)
        ),
        "accepted_daily_representatives_by_day": int(
            len(daily_representatives)
        ),
        "regional_link_period_t2_traced_to_accepted_representative": int(
            len(selected)
        ),
    }


def load_period_map(period: str, path: Path, period_sequence: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"tmc": "string"}, low_memory=False)
    required = {"tmc", "link_id", "distance_to_tmc_ft"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing mapping columns: {missing}")
    frame["tmc"] = normalize_tmc(frame["tmc"])
    frame["link_id"] = pd.to_numeric(frame["link_id"], errors="coerce")
    frame["distance_to_tmc_ft"] = pd.to_numeric(
        frame["distance_to_tmc_ft"], errors="coerce"
    )
    frame = frame.dropna(subset=["tmc", "link_id"]).copy()
    frame["link_id"] = frame["link_id"].astype(np.int64)
    frame["period"] = period
    frame["map_source"] = str(path.resolve())
    frame["map_row_number"] = np.arange(1, len(frame) + 1)
    frame["_global_occurrence"] = (
        period_sequence * 10_000_000 + frame["map_row_number"]
    )
    open_column = f"{period.lower()}_is_open"
    frame["period_is_open"] = (
        parse_bool(frame[open_column])
        if open_column in frame
        else True
    )
    return frame


def load_period_map_task(
    task: tuple[str, Path, int],
) -> pd.DataFrame:
    return load_period_map(*task)


def load_period_maps(
    map_paths: dict[str, Path],
    workers: int,
) -> list[pd.DataFrame]:
    tasks = [
        (period, map_paths[period], PERIOD_ORDER[period])
        for period in PERIODS
    ]
    if workers <= 1:
        return [load_period_map_task(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks))
    ) as executor:
        return list(executor.map(load_period_map_task, tasks))


def rank_link_tmcs(
    mappings: pd.DataFrame, canonical_node_pair_map: Path
) -> pd.DataFrame:
    """Load the frozen combined-mapmatching winner; never rerank by period."""

    source = Path(canonical_node_pair_map)
    required = {
        "tmc",
        "link_id",
        "distance_to_tmc_ft",
        "node_pair_tmc_rank",
        "selected_for_node_pair_lookup",
        "node_pair_tmc_ranking_basis",
    }
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = sorted(required - header)
    if missing:
        raise ValueError(f"{source} is missing canonical columns: {missing}")
    canonical = pd.read_csv(
        source,
        usecols=sorted(required),
        dtype={"tmc": "string"},
        low_memory=False,
    )
    canonical["tmc"] = normalize_tmc(canonical["tmc"])
    canonical["link_id"] = pd.to_numeric(canonical["link_id"], errors="coerce")
    canonical["selected_for_node_pair_lookup"] = (
        canonical["selected_for_node_pair_lookup"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    canonical = canonical.loc[
        canonical["selected_for_node_pair_lookup"]
        & canonical["link_id"].notna()
    ].copy()
    canonical["link_id"] = canonical["link_id"].astype(np.int64)
    canonical["node_pair_tmc_rank"] = pd.to_numeric(
        canonical["node_pair_tmc_rank"], errors="coerce"
    ).astype("Int64")
    canonical["ranking_basis"] = canonical["node_pair_tmc_ranking_basis"]
    canonical["best_distance_to_tmc_ft"] = pd.to_numeric(
        canonical["distance_to_tmc_ft"], errors="coerce"
    )
    canonical = canonical[
        ["link_id", "tmc", "best_distance_to_tmc_ft", "node_pair_tmc_rank", "ranking_basis"]
    ].drop_duplicates()
    if canonical.duplicated("link_id").any():
        raise ValueError("Frozen canonical mapping contains duplicate link winners")
    mapped_links = set(mappings["link_id"].dropna().astype(np.int64))
    return canonical.loc[canonical["link_id"].isin(mapped_links)].copy()


def load_network_period(
    task: tuple[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period, path = task
    wanted = {
        "link_id",
        "from_node_id",
        "to_node_id",
        "LINKID",
        "STREETNAME",
        "length_in_mile",
        "lanes",
        "capacity",
        "free_speed",
        "link_type",
        "geometry",
    }
    frame = pd.read_csv(
        path, usecols=lambda column: column in wanted, low_memory=False
    )
    frame["link_id"] = pd.to_numeric(frame["link_id"], errors="coerce")
    frame = frame.dropna(subset=["link_id"]).copy()
    frame["link_id"] = frame["link_id"].astype(np.int64)
    frame["_period"] = period
    frame["_priority"] = {"MD": 0, "AM": 1, "PM": 2}[period]
    frame[f"available_{period.lower()}"] = True
    pairs = (
        frame[["link_id", "LINKID"]].dropna()
        if "LINKID" in frame
        else pd.DataFrame(columns=["link_id", "LINKID"])
    )
    return frame, pairs


def load_network_union(
    network_root: Path,
    workers: int = 1,
) -> tuple[pd.DataFrame, int]:
    tasks = [
        (period, network_root / period.lower() / "link.csv")
        for period in PERIODS
    ]
    if workers <= 1:
        results = [load_network_period(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks))
        ) as executor:
            results = list(executor.map(load_network_period, tasks))
    frames = [result[0] for result in results]
    linkid_pairs = [
        result[1] for result in results if not result[1].empty
    ]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    availability = (
        combined.groupby("link_id")["_period"]
        .agg(lambda periods: set(periods))
        .to_dict()
    )
    combined = combined.sort_values(
        ["link_id", "_priority"], kind="mergesort"
    ).drop_duplicates("link_id", keep="first")
    for period in PERIODS:
        combined[f"available_{period.lower()}"] = combined["link_id"].map(
            lambda link_id, p=period: p in availability.get(link_id, set())
        )

    linkid_mismatch_count = 0
    if linkid_pairs:
        pairs = pd.concat(linkid_pairs, ignore_index=True)
        linkid_mismatch_count = int(
            (pairs.groupby("link_id")["LINKID"].nunique(dropna=True) > 1).sum()
        )
    return combined.drop(columns=["_period", "_priority"]), linkid_mismatch_count


def build_assignments(
    network: pd.DataFrame,
    mappings: pd.DataFrame,
    pair_rank: pd.DataFrame,
    representatives: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period_maps = mappings[mappings["period_is_open"]].copy()
    period_maps = period_maps.merge(
        pair_rank,
        on=["link_id", "tmc"],
        how="left",
        validate="many_to_one",
    )
    period_maps = period_maps.sort_values(
        ["link_id", "period", "node_pair_tmc_rank", "map_row_number"],
        kind="mergesort",
    ).drop_duplicates(["link_id", "period", "tmc"], keep="first")

    rep_columns = [
        "tmc",
        "period",
        "episode_id",
        "corridor_output",
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "P_hr",
        "min_speed_mph",
        "tmc_period_candidate_count",
        "t2_source_method",
        "daily_probe_day_count",
        "daily_probe_episode_count",
        "daily_probe_t2_min_hour",
        "daily_probe_t2_max_hour",
        "daily_probe_t2_std_hour",
    ]
    available_rep_columns = [
        column for column in rep_columns if column in representatives
    ]
    audit = period_maps.merge(
        representatives[available_rep_columns],
        on=["tmc", "period"],
        how="left",
        validate="many_to_one",
    )
    for column in rep_columns:
        if column not in audit:
            audit[column] = np.nan
    audit["has_t2"] = audit["t2_hour"].notna()
    audit = audit.sort_values(
        ["link_id", "period", "node_pair_tmc_rank", "map_row_number"],
        kind="mergesort",
    )
    audit["eligible_candidate_order"] = (
        audit.groupby(["link_id", "period"], sort=False).cumcount() + 1
    )

    primary = pair_rank[pair_rank["node_pair_tmc_rank"].eq(1)][
        ["link_id", "tmc", "ranking_basis", "best_distance_to_tmc_ft"]
    ].rename(
        columns={
            "tmc": "primary_tmc",
            "ranking_basis": "primary_ranking_basis",
            "best_distance_to_tmc_ft": "primary_distance_to_tmc_ft",
        }
    )
    audit = audit.merge(primary, on="link_id", how="left", validate="many_to_one")

    primary_period = (
        audit[audit["tmc"].eq(audit["primary_tmc"])]
        [["link_id", "period", "has_t2"]]
        .drop_duplicates(["link_id", "period"])
        .rename(columns={"has_t2": "primary_has_t2"})
    )
    audit = audit.merge(
        primary_period,
        on=["link_id", "period"],
        how="left",
        validate="many_to_one",
    )
    audit["primary_available_for_period"] = audit["primary_has_t2"].notna()
    audit["primary_has_t2"] = audit["primary_has_t2"].fillna(False).astype(bool)
    audit["observed_no_congestion_protected"] = (
        audit["primary_available_for_period"] & ~audit["primary_has_t2"]
    )
    selectable = audit["has_t2"] & audit["tmc"].eq(audit["primary_tmc"])
    audit["_available_order"] = selectable.astype(int)
    audit["_available_order"] = audit.groupby(
        ["link_id", "period"], sort=False
    )["_available_order"].cumsum()
    audit["selected"] = selectable & audit["_available_order"].eq(1)
    audit["selection_reason"] = ""
    selected = audit["selected"]
    audit.loc[
        selected & audit["tmc"].eq(audit["primary_tmc"]),
        "selection_reason",
    ] = "selected_primary_ranked"
    audit.loc[
        audit["observed_no_congestion_protected"],
        "selection_reason",
    ] = "protected_primary_tmc_no_average_weekday_congestion"

    selected_rows = audit[audit["selected"]].copy()
    selected_rows = selected_rows.rename(
        columns={
            "tmc": "selected_tmc",
            "distance_to_tmc_ft": "selected_distance_to_tmc_ft",
            "ranking_basis": "selected_ranking_basis",
        }
    )

    grid = network[["link_id"]].drop_duplicates().assign(_join=1).merge(
        pd.DataFrame({"period": PERIODS, "_join": 1}), on="_join"
    ).drop(columns="_join")
    selected_columns = [
        "link_id",
        "period",
        "selected_tmc",
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "episode_id",
        "corridor_output",
        "selected_distance_to_tmc_ft",
        "node_pair_tmc_rank",
        "selected_ranking_basis",
        "selection_reason",
        "t2_source_method",
        "daily_probe_day_count",
        "daily_probe_episode_count",
        "daily_probe_t2_min_hour",
        "daily_probe_t2_max_hour",
        "daily_probe_t2_std_hour",
    ]
    long = grid.merge(
        selected_rows[selected_columns],
        on=["link_id", "period"],
        how="left",
        validate="one_to_one",
    )
    mapped_status = (
        audit.groupby(["link_id", "period"], as_index=False)
        .agg(
            mapped_tmc_count=("tmc", "nunique"),
            mapped_tmc_with_t2_count=("has_t2", "sum"),
        )
    )
    long = long.merge(
        mapped_status,
        on=["link_id", "period"],
        how="left",
        validate="one_to_one",
    )
    long["mapped_tmc_count"] = long["mapped_tmc_count"].fillna(0).astype(int)
    long["mapped_tmc_with_t2_count"] = (
        long["mapped_tmc_with_t2_count"].fillna(0).astype(int)
    )
    primary_status = (
        audit.sort_values(
            ["link_id", "period", "node_pair_tmc_rank"], kind="mergesort"
        )
        .groupby(["link_id", "period"], as_index=False)
        .agg(
            primary_tmc=("primary_tmc", "first"),
            primary_ranking_basis=("primary_ranking_basis", "first"),
            primary_distance_to_tmc_ft=("primary_distance_to_tmc_ft", "first"),
            primary_available_for_period=("primary_available_for_period", "max"),
            primary_has_average_weekday_congestion=("primary_has_t2", "max"),
            observed_no_congestion_protected=(
                "observed_no_congestion_protected",
                "max",
            ),
        )
    )
    long = long.merge(
        primary_status,
        on=["link_id", "period"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "primary_available_for_period",
        "primary_has_average_weekday_congestion",
        "observed_no_congestion_protected",
    ):
        long[column] = long[column].fillna(False).astype(bool)
    long["observation_status"] = np.select(
        [
            long["t2_hour"].notna(),
            long["observed_no_congestion_protected"],
            long["mapped_tmc_count"].gt(0),
        ],
        [
            "observed_average_weekday_congestion",
            "observed_no_average_weekday_congestion",
            "mapped_nonprimary_without_direct_boundary",
        ],
        default="no_open_mapped_tmc",
    )
    long.loc[
        long["selection_reason"].fillna("").eq("")
        & long["observed_no_congestion_protected"],
        "selection_reason",
    ] = "protected_primary_tmc_no_average_weekday_congestion"
    long.loc[
        long["selection_reason"].fillna("").eq("")
        & long["mapped_tmc_count"].eq(0),
        "selection_reason",
    ] = "no_open_mapped_tmc_for_period"
    long.loc[
        long["selection_reason"].fillna("").eq("")
        & long["mapped_tmc_count"].gt(0)
        & long["mapped_tmc_with_t2_count"].eq(0),
        "selection_reason",
    ] = "mapped_tmcs_without_detected_t2"

    wide_values: dict[str, pd.Series] = {}
    for period in PERIODS:
        suffix = period.lower()
        period_frame = long[long["period"].eq(period)].set_index("link_id")
        for source, target in (
            ("selected_tmc", f"tmc_{suffix}"),
            ("t0_hour", f"t0_{suffix}_hour"),
            ("t2_hour", f"t2_{suffix}_hour"),
            ("t3_hour", f"t3_{suffix}_hour"),
            ("episode_id", f"source_episode_id_{suffix}"),
            ("t2_source_method", f"t2_source_method_{suffix}"),
            ("daily_probe_day_count", f"daily_probe_day_count_{suffix}"),
            ("selection_reason", f"selection_rule_{suffix}"),
            ("selected_distance_to_tmc_ft", f"distance_to_tmc_ft_{suffix}"),
        ):
            wide_values[target] = period_frame[source]

    wide = network.copy().set_index("link_id")
    for target, values in wide_values.items():
        wide[target] = values
    wide = wide.reset_index()
    audit = audit.drop(columns=["_available_order"])
    return long, wide, audit


PERIOD_LINK_APPENDED_COLUMNS = [
    "t2_period",
    "t0_hour",
    "t2_hour",
    "t3_hour",
    "t2_source_tmc",
    "t2_source_episode_id",
    "t2_source_corridor",
    "t2_selection_rule",
    "t2_distance_to_tmc_ft",
    "t2_primary_tmc",
    "t2_mapped_tmc_count",
    "t2_source_method",
    "t2_daily_probe_day_count",
    "t2_daily_probe_episode_count",
    "t2_daily_probe_std_hour",
    "t2_observation_status",
    "t2_observed_no_congestion_protected",
]
PERIOD_LINK_SOURCE_COLUMNS = [
    "t0_hour",
    "t2_hour",
    "t3_hour",
    "selected_tmc",
    "episode_id",
    "corridor_output",
    "selection_reason",
    "selected_distance_to_tmc_ft",
    "primary_tmc",
    "mapped_tmc_count",
    "t2_source_method",
    "daily_probe_day_count",
    "daily_probe_episode_count",
    "daily_probe_t2_std_hour",
    "observation_status",
    "observed_no_congestion_protected",
]


def csv_suffix(values: list[object]) -> str:
    normalized: list[object] = []
    for value in values:
        if pd.isna(value):
            normalized.append("")
        elif isinstance(value, (float, np.floating)):
            normalized.append(format(float(value), ".12g"))
        elif isinstance(value, np.integer):
            normalized.append(int(value))
        else:
            normalized.append(value)
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="").writerow(normalized)
    return buffer.getvalue()


def write_period_link_file(
    task: tuple[str, Path, pd.DataFrame, Path],
) -> tuple[str, dict[str, object]]:
    period, source, assignment, target = task
    assignment = assignment.copy()
    assignment["link_id"] = pd.to_numeric(
        assignment["link_id"], errors="coerce"
    )
    if assignment["link_id"].isna().any():
        raise ValueError(f"{period} assignment has nonnumeric link_id values.")
    assignment["link_id"] = assignment["link_id"].astype(np.int64)
    if assignment["link_id"].duplicated().any():
        raise ValueError(f"{period} assignment has duplicate link_id values.")
    lookup = assignment.set_index("link_id")[
        PERIOD_LINK_SOURCE_COLUMNS
    ].to_dict(orient="index")

    target.parent.mkdir(parents=True, exist_ok=False)
    rows = 0
    assigned_t0 = 0
    assigned_t2 = 0
    assigned_t3 = 0
    seen_link_ids: set[int] = set()
    with source.open("r", encoding="utf-8", newline="") as input_stream, (
        target.open("w", encoding="utf-8", newline="")
    ) as output_stream:
        header_line = input_stream.readline()
        if not header_line:
            raise ValueError(f"{source} is empty.")
        header = next(csv.reader([header_line]))
        if "link_id" not in header:
            raise ValueError(f"{source} is missing link_id.")
        link_id_index = header.index("link_id")
        newline = "\r\n" if header_line.endswith("\r\n") else "\n"
        output_stream.write(
            header_line.rstrip("\r\n")
            + ","
            + csv_suffix(PERIOD_LINK_APPENDED_COLUMNS)
            + newline
        )
        for raw_line in input_stream:
            fields = next(csv.reader([raw_line]))
            if len(fields) != len(header):
                raise ValueError(
                    f"{source} contains a multiline or malformed CSV row."
                )
            link_id = int(fields[link_id_index])
            if link_id in seen_link_ids:
                raise ValueError(f"{source} has duplicate link_id {link_id}.")
            seen_link_ids.add(link_id)
            record = lookup.get(link_id)
            if record is None:
                raise ValueError(
                    f"{period} assignment is missing native link {link_id}."
                )
            assigned_t0 += int(pd.notna(record["t0_hour"]))
            assigned_t2 += int(pd.notna(record["t2_hour"]))
            assigned_t3 += int(pd.notna(record["t3_hour"]))
            output_stream.write(
                raw_line.rstrip("\r\n")
                + ","
                + csv_suffix(
                    [
                        period,
                        *[
                            record[column]
                            for column in PERIOD_LINK_SOURCE_COLUMNS
                        ],
                    ]
                )
                + newline
            )
            rows += 1

    return period, {
        "source": manifest_path(source),
        "output": manifest_path(target),
        "rows": rows,
        "assigned_t0": assigned_t0,
        "assigned_t2": assigned_t2,
        "assigned_t3": assigned_t3,
        "blank_t0": rows - assigned_t0,
        "blank_t2": rows - assigned_t2,
        "blank_t3": rows - assigned_t3,
        "sha256": sha256(target),
    }


def write_period_link_files(
    network_root: Path,
    long: pd.DataFrame,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, dict[str, object]]:
    required = {
        "link_id",
        "period",
        *PERIOD_LINK_SOURCE_COLUMNS,
    }
    missing = sorted(required - set(long.columns))
    if missing:
        raise ValueError(
            f"regional_link_t2_long.csv is missing columns: {missing}"
        )
    if long.duplicated(["link_id", "period"]).any():
        raise ValueError("Duplicate link-period rows in T2 assignment table.")

    period_root = output_dir / "period_link_files"
    if period_root.exists() and any(period_root.rglob("*")):
        raise FileExistsError(
            f"Period link output is not empty: {period_root}"
        )
    tasks = [
        (
            period,
            network_root / period.lower() / "link.csv",
            long[long["period"].eq(period)].copy(),
            period_root / period.lower() / "link.csv",
        )
        for period in PERIODS
    ]
    if workers <= 1:
        results = [write_period_link_file(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks))
        ) as executor:
            results = list(executor.map(write_period_link_file, tasks))
    return {period: result for period, result in results}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.assignment_only and args.completion_only:
        raise ValueError(
            "--assignment-only and --completion-only are mutually exclusive."
        )
    cbi_root = args.cbi_output_root.resolve()
    cbi_roots = [
        cbi_root,
        *[Path(root).resolve() for root in args.supplemental_cbi_output_root],
    ]
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.assignment_only or args.completion_only:
        raise ValueError(
            "--assignment-only and --completion-only require an explicit --output-dir."
        )
    else:
        output_dir = (
            cbi_root / "outputs" / "congestion-boundaries" / "link-t2"
        ).resolve()
    if args.completion_only:
        manifest_file = output_dir / "run_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(
                f"Completion-only mode requires {manifest_file}."
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest["boundary_completion"] = complete_boundaries(
            PACKAGE_SRC_ROOT,
            output_dir.parent,
            mode=args.completion_mode,
            ml_run_dir=args.ml_run_dir,
            comparison_run_dir=args.comparison_run_dir,
        )
        manifest_file.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest["boundary_completion"], indent=2))
        return 0

    average_candidate_paths = sorted(
        path
        for root in cbi_roots
        for path in discover_episode_paths(root, average_weekday=True)
    )
    daily_candidate_paths = sorted(
        path
        for root in cbi_roots
        for path in discover_episode_paths(root, average_weekday=False)
    )
    worker_plan = recommend_workers(
        max(
            len(average_candidate_paths),
            len(daily_candidate_paths),
            len(PERIODS),
        ),
        target_fraction=args.worker_fraction,
        explicit_workers=args.workers,
    )
    if args.assignment_only:
        long_path = output_dir / "regional_link_t2_long.csv"
        manifest_file = output_dir / "run_manifest.json"
        if not long_path.is_file() or not manifest_file.is_file():
            raise FileNotFoundError(
                "Assignment-only mode requires regional_link_t2_long.csv "
                f"and run_manifest.json under {output_dir}."
            )
        long = pd.read_csv(
            long_path,
            dtype={"selected_tmc": "string", "primary_tmc": "string"},
            low_memory=False,
        )
        period_link_files = write_period_link_files(
            args.network_root.resolve(),
            long,
            output_dir,
            workers=worker_plan.workers,
        )
        vdf_t2_postprocessing = propagate_t2_by_vdf(
            output_dir,
            workers=worker_plan.workers,
            worker_fraction=args.worker_fraction,
            update_parent_manifest=False,
        )
        for period, result in vdf_t2_postprocessing["periods"].items():
            period_link_files[period].update(
                {
                    "sha256": result["output_sha256"],
                    "vdf_grouping_column": result["vdf_column"],
                    "t2_est_carried_original": result[
                        "carried_original_t2_to_t2_est"
                    ],
                    "t2_est_propagated_by_vdf": result["propagated_t2_est"],
                    "t2_est_populated": result["populated_t2_est"],
                    "t2_est_blank": result["unmatched_blank_t2_est"],
                }
            )
        hybrid_t2_postprocessing = apply_hybrid_t2(
            output_dir,
            args.spatial_output.resolve(),
            workers=worker_plan.workers,
            worker_fraction=args.worker_fraction,
            update_parent_manifest=False,
        )
        for period, result in hybrid_t2_postprocessing[
            "period_products"
        ].items():
            period_link_files[period].update(
                {
                    "sha256": result["sha256"],
                    "hybrid_assignments_by_source": result[
                        "assignments_by_source"
                    ],
                    "final_t2_column": "t2_hybrid_hour",
                }
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest["period_link_files"] = period_link_files
        manifest["last_period_file_worker_plan"] = worker_plan.to_dict()
        manifest["vdf_t2_postprocessing"] = {
            "status": "PASS",
            "lookup_table": manifest_path(
                output_dir / "vdf_code_t2_lookup.csv"
            ),
            "manifest": manifest_path(
                output_dir / "vdf_t2_propagation_manifest.json"
            ),
            "worker_plan": vdf_t2_postprocessing["worker_plan"],
            "aggregation_rule": vdf_t2_postprocessing["aggregation_rule"],
            "fill_rule": vdf_t2_postprocessing["fill_rule"],
        }
        manifest["hybrid_t2_postprocessing"] = {
            "status": "PASS",
            "precedence": hybrid_t2_postprocessing["precedence"],
            "final_t2_column": "t2_hybrid_hour",
            "spatial_output": manifest_path(
                args.spatial_output.resolve()
            ),
            "spatial_output_sha256": hybrid_t2_postprocessing[
                "spatial_output_sha256"
            ],
            "manifest": manifest_path(
                output_dir / "hybrid_t2_manifest.json"
            ),
            "coverage_summary": manifest_path(
                output_dir / "hybrid_t2_coverage_summary.csv"
            ),
            "long_audit": manifest_path(
                output_dir / "hybrid_link_t2_long.csv"
            ),
            "worker_plan": hybrid_t2_postprocessing["worker_plan"],
        }
        manifest["boundary_completion"] = complete_boundaries(
            PACKAGE_SRC_ROOT,
            output_dir.parent,
            mode=args.completion_mode,
            ml_run_dir=args.ml_run_dir,
            comparison_run_dir=args.comparison_run_dir,
        )
        manifest_file.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "PASS", **period_link_files}, indent=2))
        return 0

    if output_dir.exists() and any(
        item.name not in {"logs", "qa"} for item in output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    map_paths = {
        "AM": args.am_map.resolve(),
        "MD": args.md_map.resolve(),
        "PM": args.pm_map.resolve(),
    }
    mapping_frames = load_period_maps(
        map_paths,
        worker_plan.workers,
    )
    mappings = pd.concat(mapping_frames, ignore_index=True, sort=False)
    canonical_node_pair_map = (
        args.canonical_node_pair_map.resolve()
        if args.canonical_node_pair_map is not None
        else (
            cbi_root
            / "shared"
            / "network-mapping"
            / "canonical_node_pair_tmc.csv"
        ).resolve()
    )
    if not canonical_node_pair_map.is_file():
        raise FileNotFoundError(
            "The selected CBI run does not contain its frozen canonical "
            f"node-pair mapping: {canonical_node_pair_map}"
        )
    pair_rank = rank_link_tmcs(mappings, canonical_node_pair_map)
    eligible_tmc_periods = mappings[mappings["period_is_open"]][
        ["tmc", "period"]
    ].drop_duplicates()
    (
        candidates,
        average_representatives,
        daily_candidates,
        daily_representatives,
        daily_summary,
        representatives,
    ) = load_episode_candidates(
        cbi_roots,
        worker_plan.workers,
        eligible_tmc_periods,
    )
    network, linkid_mismatch_count = load_network_union(
        args.network_root.resolve(),
        worker_plan.workers,
    )
    network_paths = {
        period: args.network_root.resolve()
        / period.lower()
        / "link.csv"
        for period in PERIODS
    }
    long, wide, audit = build_assignments(
        network, mappings, pair_rank, representatives
    )
    accepted_lineage_validation = validate_accepted_t2_lineage(
        candidates,
        average_representatives,
        daily_candidates,
        daily_representatives,
        daily_summary,
        representatives,
        long,
    )

    candidates.to_csv(output_dir / "tmc_period_t2_candidates.csv", index=False)
    average_representatives.to_csv(
        output_dir / "tmc_period_average_weekday_representative.csv",
        index=False,
    )
    daily_candidates.to_csv(
        output_dir / "tmc_period_daily_probe_candidates.csv",
        index=False,
    )
    daily_representatives.to_csv(
        output_dir / "tmc_period_daily_probe_representative_by_day.csv",
        index=False,
    )
    daily_summary.to_csv(
        output_dir / "tmc_period_daily_probe_summary.csv",
        index=False,
    )
    representatives.to_csv(
        output_dir / "tmc_period_t2_representative.csv", index=False
    )
    pair_rank.to_csv(output_dir / "regional_link_tmc_ranking.csv", index=False)
    audit.to_csv(output_dir / "link_t2_selection_audit.csv", index=False)
    long.to_csv(output_dir / "regional_link_t2_long.csv", index=False)
    wide.to_csv(output_dir / "regional_link_t2.csv", index=False)
    period_link_files = write_period_link_files(
        args.network_root.resolve(),
        long,
        output_dir,
        workers=worker_plan.workers,
    )
    vdf_t2_postprocessing = propagate_t2_by_vdf(
        output_dir,
        workers=worker_plan.workers,
        worker_fraction=args.worker_fraction,
        update_parent_manifest=False,
    )
    for period, result in vdf_t2_postprocessing["periods"].items():
        period_link_files[period].update(
            {
                "sha256": result["output_sha256"],
                "vdf_grouping_column": result["vdf_column"],
                "t2_est_carried_original": result[
                    "carried_original_t2_to_t2_est"
                ],
                "t2_est_propagated_by_vdf": result["propagated_t2_est"],
                "t2_est_populated": result["populated_t2_est"],
                "t2_est_blank": result["unmatched_blank_t2_est"],
            }
        )
    hybrid_t2_postprocessing = apply_hybrid_t2(
        output_dir,
        args.spatial_output.resolve(),
        workers=worker_plan.workers,
        worker_fraction=args.worker_fraction,
        update_parent_manifest=False,
    )
    for period, result in hybrid_t2_postprocessing[
        "period_products"
    ].items():
        period_link_files[period].update(
            {
                "sha256": result["sha256"],
                "hybrid_assignments_by_source": result[
                    "assignments_by_source"
                ],
                "final_t2_column": "t2_hybrid_hour",
            }
        )

    populated = long[long["t2_hour"].notna()].copy()
    daily_fallback_summary = daily_summary[
        daily_summary["fallback_used"]
    ]
    network_link_ids = set(network["link_id"])
    mapped_link_ids = set(pair_rank["link_id"]) & network_link_ids
    open_mapped_link_ids = (
        set(mappings.loc[mappings["period_is_open"], "link_id"])
        & network_link_ids
    )
    links_with_t2 = int(populated["link_id"].nunique())
    manifest = {
        "status": "PASS",
        "selection_rule": (
            "accepted episodes only; accepted average-weekday representative: "
            "lowest min speed, then longest duration, then earliest t2; missing "
            "TMC-period average-weekday congestion remains blank (daily probes "
            "are audit-only); regional link TMC: the frozen composite-ranked "
            "directed-node-pair winner from the selected CBI run, unchanged "
            "across AM, MD, and PM; a winning TMC with no average-weekday congestion is "
            "protected from alternate-TMC, spatial, VDF-class, and ML filling"
        ),
        "episode_input_contract": (
            "accepted-only files exactly cross-checked against "
            "is_clean_valid_episode=true in their paired screening audits"
        ),
        "accepted_episode_lineage_validation": accepted_lineage_validation,
        "worker_plan": worker_plan.to_dict(),
        "parallel_stages": [
            "period map loading",
            "average-weekday episode loading",
            "daily episode loading",
            "regional period-network loading",
            "period link-file writing",
            "AM/MD/PM direct-spatial candidate assignment",
            f"final {args.completion_mode} boundary completion",
        ],
        "periods": list(PERIODS),
        "cbi_direct_roots": [str(root) for root in cbi_roots],
        "regional_links": int(network["link_id"].nunique()),
        "links_with_at_least_one_mapped_tmc": len(mapped_link_ids),
        "links_with_at_least_one_open_mapped_tmc": len(
            open_mapped_link_ids
        ),
        "regional_link_period_rows": int(len(long)),
        "accepted_average_weekday_episode_candidates": int(len(candidates)),
        "average_weekday_representative_tmc_period_t2": int(
            len(average_representatives)
        ),
        "daily_probe_candidates_for_mapped_tmc_periods": int(
            len(daily_candidates)
        ),
        "daily_probe_representatives_by_day": int(
            len(daily_representatives)
        ),
        "daily_probe_tmc_period_summaries": int(len(daily_summary)),
        "daily_probe_fallback_tmc_period_t2": int(
            len(daily_fallback_summary)
        ),
        "daily_probe_suppressed_no_average_weekday_congestion": int(
            daily_summary[
                "suppressed_no_average_weekday_congestion"
            ].sum()
        ),
        "representative_tmc_period_t2": int(len(representatives)),
        "mapping_rows": {
            period: int(len(mapping_frames[PERIOD_ORDER[period]]))
            for period in PERIODS
        },
        "populated_link_period_t2": int(long["t2_hour"].notna().sum()),
        "links_with_at_least_one_t2": links_with_t2,
        "mapped_link_t2_coverage_pct": (
            100.0 * links_with_t2 / len(mapped_link_ids)
            if mapped_link_ids
            else 0.0
        ),
        "blank_link_period_t2": int(long["t2_hour"].isna().sum()),
        "populated_link_period_t2_by_source": {
            str(method): int(count)
            for method, count in populated[
                "t2_source_method"
            ].value_counts().items()
        },
        "selected_primary": int(
            long["selection_reason"].eq("selected_primary_ranked").sum()
        ),
        "selected_alternate_missing_t2": int(
            long["selection_reason"]
            .eq("selected_alternate_primary_missing_t2")
            .sum()
        ),
        "selected_alternate_period_unavailable": int(
            long["selection_reason"]
            .eq("selected_alternate_primary_not_open_or_mapped")
            .sum()
        ),
        "network_linkid_period_mismatches": linkid_mismatch_count,
        "period_link_files": period_link_files,
        "vdf_t2_postprocessing": {
            "status": "PASS",
            "lookup_table": manifest_path(
                output_dir / "vdf_code_t2_lookup.csv"
            ),
            "manifest": manifest_path(
                output_dir / "vdf_t2_propagation_manifest.json"
            ),
            "worker_plan": vdf_t2_postprocessing["worker_plan"],
            "aggregation_rule": vdf_t2_postprocessing["aggregation_rule"],
            "fill_rule": vdf_t2_postprocessing["fill_rule"],
        },
        "hybrid_t2_postprocessing": {
            "status": "PASS",
            "precedence": hybrid_t2_postprocessing["precedence"],
            "final_t2_column": "t2_hybrid_hour",
            "spatial_output": manifest_path(
                args.spatial_output.resolve()
            ),
            "spatial_output_sha256": hybrid_t2_postprocessing[
                "spatial_output_sha256"
            ],
            "manifest": manifest_path(
                output_dir / "hybrid_t2_manifest.json"
            ),
            "coverage_summary": manifest_path(
                output_dir / "hybrid_t2_coverage_summary.csv"
            ),
            "long_audit": manifest_path(
                output_dir / "hybrid_link_t2_long.csv"
            ),
            "worker_plan": hybrid_t2_postprocessing["worker_plan"],
        },
        "map_inputs": {
            manifest_path(path): sha256(path)
            for path in map_paths.values()
        },
        "network_inputs": {
            manifest_path(path): sha256(path)
            for path in network_paths.values()
        },
        "cbi_candidate_inputs": {
            "average_weekday": {
                manifest_path(path): {
                    "accepted_episode_sha256": sha256(path),
                    "screening_audit": manifest_path(
                        screening_audit_path(path)
                    ),
                    "screening_audit_sha256": sha256(
                        screening_audit_path(path)
                    ),
                }
                for path in average_candidate_paths
            },
            "daily": {
                manifest_path(path): {
                    "accepted_episode_sha256": sha256(path),
                    "screening_audit": manifest_path(
                        screening_audit_path(path)
                    ),
                    "screening_audit_sha256": sha256(
                        screening_audit_path(path)
                    ),
                }
                for path in daily_candidate_paths
            },
        },
    }
    manifest_file = output_dir / "run_manifest.json"
    manifest["boundary_completion"] = {
        "status": "PENDING",
        "mode": args.completion_mode,
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest["boundary_completion"] = complete_boundaries(
        PACKAGE_SRC_ROOT,
        output_dir.parent,
        mode=args.completion_mode,
        ml_run_dir=args.ml_run_dir,
        comparison_run_dir=args.comparison_run_dir,
    )
    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
