from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ExpansionConfig
from .workers import WorkerPlan, recommend_workers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_bool(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _episode_paths(
    corridor_dir: Path, contract: str
) -> Tuple[Path, Path, Path, Path]:
    if contract == "pre_filter":
        average_path = (
            corridor_dir
            / "04-episode-detection"
            / "average_weekday_episode_candidates.csv"
        )
        daily_path = (
            corridor_dir
            / "04-episode-detection"
            / "daily_episode_candidates.csv"
        )
    else:
        average_path = (
            corridor_dir
            / "05-episode-filtering"
            / "average_weekday_episodes_accepted.csv"
        )
        daily_path = (
            corridor_dir
            / "05-episode-filtering"
            / "daily_episodes_accepted.csv"
        )
    return (
        average_path,
        corridor_dir
        / "05-episode-filtering"
        / "average_weekday_episode_filter_audit.csv",
        daily_path,
        corridor_dir
        / "05-episode-filtering"
        / "daily_episode_filter_audit.csv",
    )


def _load_episode_pair(
    task: Tuple[str, str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    corridor_path, corridor_key, contract = task
    corridor_dir = Path(corridor_path)
    average_path, average_audit_path, daily_path, daily_audit_path = (
        _episode_paths(corridor_dir, contract)
    )
    used: List[str] = []

    def load_one(path: Path, audit_path: Path) -> pd.DataFrame:
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_csv(
            path,
            dtype={"episode_id": str, "tmc_code": str},
            low_memory=False,
        )
        used.append(str(path))
        if audit_path.is_file() and "episode_id" in frame:
            audit = pd.read_csv(
                audit_path,
                dtype={"episode_id": str},
                low_memory=False,
            )
            used.append(str(audit_path))
            columns = [
                column
                for column in ("episode_id", "is_clean_valid_episode")
                if column in audit
            ]
            if len(columns) == 2:
                audit = audit[columns].drop_duplicates("episode_id", keep="last")
                frame = frame.merge(
                    audit,
                    on="episode_id",
                    how="left",
                    validate="one_to_one",
                )
        if "is_clean_valid_episode" not in frame:
            frame["is_clean_valid_episode"] = contract == "accepted"
        frame["is_clean_valid_episode"] = _parse_bool(
            frame["is_clean_valid_episode"]
        )
        frame["corridor_output"] = corridor_key
        return frame

    return (
        load_one(average_path, average_audit_path),
        load_one(daily_path, daily_audit_path),
        used,
    )


def _normalize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {
        "episode_id",
        "tmc_code",
        "period",
        "t2_hour",
        "min_speed_mph",
        "P_hr",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Episode candidates are missing: " + ", ".join(missing))
    work = frame.copy()
    work["tmc"] = work["tmc_code"].astype(str).str.strip()
    work["period"] = work["period"].astype(str).str.upper()
    for column in (
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "min_speed_mph",
        "P_hr",
        "road_order",
    ):
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work[
        work["tmc"].ne("")
        & work["period"].isin({"AM", "MD", "PM"})
        & work["t2_hour"].notna()
    ].copy()
    return work


def _rank_representatives(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work["_speed"] = work["min_speed_mph"].fillna(np.inf)
    work["_duration"] = -work["P_hr"].fillna(-np.inf)
    work["_t2"] = work["t2_hour"].fillna(np.inf)
    work = work.sort_values(
        list(group_columns)
        + ["_speed", "_duration", "_t2", "corridor_output", "episode_id"],
        kind="mergesort",
    )
    return work.drop_duplicates(list(group_columns), keep="first").drop(
        columns=["_speed", "_duration", "_t2"]
    )


def _daily_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if "date" not in frame:
        raise ValueError("Daily episode candidates are missing date")
    selected = _rank_representatives(frame, ["tmc", "period", "date"])
    summary = (
        selected.groupby(["tmc", "period"], as_index=False)
        .agg(
            t0_hour=("t0_hour", "mean"),
            t2_hour=("t2_hour", "mean"),
            t3_hour=("t3_hour", "mean"),
            P_hr=("P_hr", "mean"),
            min_speed_mph=("min_speed_mph", "mean"),
            corridor_output=("corridor_output", "first"),
            corridor=("corridor", "first"),
            direction=("direction", "first"),
            road_order=("road_order", "first"),
            daily_probe_day_count=("date", "nunique"),
            daily_probe_t2_min_hour=("t2_hour", "min"),
            daily_probe_t2_max_hour=("t2_hour", "max"),
            daily_probe_t2_std_hour=(
                "t2_hour",
                lambda values: float(values.std(ddof=0)),
            ),
            screened_acceptance_share=(
                "is_clean_valid_episode",
                "mean",
            ),
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
    summary["is_clean_valid_episode"] = summary[
        "screened_acceptance_share"
    ].eq(1.0)
    summary["t2_source_method"] = "daily_probe_mean"
    return summary


def _combine_direct_representatives(
    average: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    average_rep = _rank_representatives(average, ["tmc", "period"])
    if not average_rep.empty:
        average_rep = average_rep.copy()
        average_rep["t2_source_method"] = "average_weekday"
        average_rep["daily_probe_day_count"] = 0
        average_rep["daily_probe_t2_min_hour"] = np.nan
        average_rep["daily_probe_t2_max_hour"] = np.nan
        average_rep["daily_probe_t2_std_hour"] = np.nan
        average_rep["screened_acceptance_share"] = average_rep[
            "is_clean_valid_episode"
        ].astype(float)
    daily_rep = _daily_summary(daily)
    if average_rep.empty:
        return daily_rep
    if daily_rep.empty:
        return average_rep
    average_keys = set(
        zip(average_rep["tmc"].astype(str), average_rep["period"].astype(str))
    )
    keep_daily = [
        (str(tmc), str(period)) not in average_keys
        for tmc, period in zip(daily_rep["tmc"], daily_rep["period"])
    ]
    return pd.concat(
        [average_rep, daily_rep.loc[keep_daily]],
        ignore_index=True,
        sort=False,
    )


def build_representative_snapshot(
    corridor_root: Path,
    config: ExpansionConfig,
    workers: int,
) -> Tuple[pd.DataFrame, List[Path]]:
    corridor_dirs = sorted(
        path
        for path in Path(corridor_root).iterdir()
        if path.is_dir()
    )
    tasks = [
        (str(path), path.name, config.episode_contract)
        for path in corridor_dirs
    ]
    if workers <= 1:
        results = [_load_episode_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            results = list(executor.map(_load_episode_pair, tasks))
    averages = [result[0] for result in results if not result[0].empty]
    dailies = [result[1] for result in results if not result[1].empty]
    used = [Path(path) for result in results for path in result[2]]
    average = _normalize_candidates(
        pd.concat(averages, ignore_index=True, sort=False)
        if averages
        else pd.DataFrame()
    )
    daily = _normalize_candidates(
        pd.concat(dailies, ignore_index=True, sort=False)
        if dailies
        else pd.DataFrame()
    )
    direct = _combine_direct_representatives(average, daily)

    if config.require_screened_acceptance_for_anchor:
        anchor_average = average[average["is_clean_valid_episode"]].copy()
        anchor_daily = daily[daily["is_clean_valid_episode"]].copy()
    else:
        anchor_average = average.copy()
        anchor_daily = daily.copy()
    anchor = _combine_direct_representatives(anchor_average, anchor_daily)
    anchor_columns = {
        "t2_hour": "anchor_t2_hour",
        "t2_source_method": "anchor_source_method",
        "episode_id": "anchor_episode_id",
        "daily_probe_day_count": "anchor_daily_probe_day_count",
        "daily_probe_t2_std_hour": "anchor_daily_t2_std_hour",
    }
    available = ["tmc", "period"] + [
        column for column in anchor_columns if column in anchor
    ]
    anchor = anchor[available].rename(columns=anchor_columns)
    representatives = direct.merge(
        anchor, on=["tmc", "period"], how="left", validate="one_to_one"
    )
    representatives["anchor_source_reliable"] = representatives[
        "anchor_t2_hour"
    ].notna()
    daily_anchor = representatives["anchor_source_method"].eq(
        "daily_probe_mean"
    )
    representatives.loc[daily_anchor, "anchor_source_reliable"] &= (
        pd.to_numeric(
            representatives.loc[daily_anchor, "anchor_daily_probe_day_count"],
            errors="coerce",
        ).fillna(0)
        >= config.minimum_daily_probe_days
    )
    representatives.loc[daily_anchor, "anchor_source_reliable"] &= (
        pd.to_numeric(
            representatives.loc[daily_anchor, "anchor_daily_t2_std_hour"],
            errors="coerce",
        ).fillna(np.inf)
        <= config.maximum_daily_t2_std_hours
    )
    representatives = representatives.sort_values(
        ["tmc", "period"], kind="mergesort"
    ).reset_index(drop=True)
    return representatives, sorted(set(used))


def _latest_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    metadata["tmc"] = metadata["tmc"].astype(str)
    metadata["_original_order"] = np.arange(len(metadata))
    first_order = metadata.groupby("tmc", sort=False)["_original_order"].min()
    if "active_start_date" in metadata:
        metadata["_active_start"] = pd.to_datetime(
            metadata["active_start_date"], errors="coerce", utc=True
        )
        metadata = metadata.sort_values(
            ["tmc", "_active_start", "_original_order"]
        )
    metadata = metadata.drop_duplicates("tmc", keep="last")
    metadata["_corridor_order"] = metadata["tmc"].map(first_order)
    metadata = metadata.sort_values("_corridor_order").reset_index(drop=True)
    metadata["profile_link_id"] = np.arange(1, len(metadata) + 1)
    return metadata


def _load_profile_task(
    task: Tuple[str, str, str, int, int]
) -> Tuple[pd.DataFrame, List[str]]:
    output_path, metadata_path, corridor_key, minimum_minute, maximum_minute = task
    output_dir = Path(output_path)
    metadata_file = Path(metadata_path)
    profile_path = (
        output_dir / "03-profiles" / "average_weekday_profile.csv"
    )
    fd_path = (
        output_dir / "02-fundamental-diagram" / "link_fd_context.csv"
    )
    if not (
        profile_path.is_file()
        and fd_path.is_file()
        and metadata_file.is_file()
    ):
        return pd.DataFrame(), []
    profile = pd.read_csv(profile_path, low_memory=False).rename(
        columns={"link_id": "profile_link_id"}
    )
    fd = pd.read_csv(fd_path, low_memory=False).rename(
        columns={"link_id": "profile_link_id"}
    )
    metadata = _latest_metadata(metadata_file)
    keep_meta = [
        column
        for column in (
            "tmc",
            "profile_link_id",
            "road",
            "direction",
            "road_order",
            "miles",
        )
        if column in metadata
    ]
    frame = profile.merge(
        metadata[keep_meta],
        on="profile_link_id",
        how="inner",
        validate="many_to_one",
    ).merge(
        fd[
            [
                column
                for column in (
                    "profile_link_id",
                    "capacity",
                    "free_flow_speed_posted",
                    "free_flow_speed_obs_99pct",
                )
                if column in fd
            ]
        ],
        on="profile_link_id",
        how="left",
        validate="many_to_one",
    )
    frame["t_min"] = pd.to_numeric(frame["t_min"], errors="coerce")
    frame = frame[
        frame["t_min"].between(
            float(minimum_minute), float(maximum_minute), inclusive="left"
        )
    ].copy()
    frame["speed_mph"] = pd.to_numeric(
        frame["avg_weekday_speed_mph"], errors="coerce"
    )
    frame["flow_vphpl"] = pd.to_numeric(
        frame.get("avg_weekday_flow_veh_per_hr_lane"), errors="coerce"
    )
    frame["freeflow_mph"] = pd.to_numeric(
        frame.get("free_flow_speed_posted"), errors="coerce"
    )
    observed_freeflow = (
        frame.groupby("profile_link_id")["speed_mph"]
        .transform(lambda values: values.quantile(0.95))
        .clip(lower=1.0)
    )
    frame["freeflow_mph"] = frame["freeflow_mph"].fillna(observed_freeflow)
    frame["normalized_speed"] = frame["speed_mph"] / frame[
        "freeflow_mph"
    ].where(frame["freeflow_mph"] > 1.0)
    frame["corridor_output"] = corridor_key
    frame["tmc"] = frame["tmc"].astype(str).str.strip()
    columns = [
        "tmc",
        "corridor_output",
        "road",
        "direction",
        "road_order",
        "miles",
        "profile_link_id",
        "t_min",
        "speed_mph",
        "flow_vphpl",
        "freeflow_mph",
        "capacity",
        "normalized_speed",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns], [str(profile_path), str(fd_path), str(metadata_file)]


def build_profile_snapshot(
    package_root: Path,
    config: ExpansionConfig,
    workers: int,
    *,
    cbi_output_root: Optional[Path] = None,
    corridor_input_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[Path]]:
    if cbi_output_root is None or corridor_input_root is None:
        raise ValueError(
            "cbi_output_root and corridor_input_root are required; automatic "
            "workspace/latest-run discovery is not supported"
        )
    output_root = Path(cbi_output_root).resolve()
    input_root = Path(corridor_input_root).resolve()
    minimum_minute = min(start for start, _ in config.periods.values())
    maximum_minute = max(end for _, end in config.periods.values())
    tasks = []
    for output_dir in sorted(output_root.iterdir()):
        if not output_dir.is_dir():
            continue
        tasks.append(
            (
                str(output_dir),
                str(input_root / output_dir.name / "TMC_Identification.csv"),
                output_dir.name,
                minimum_minute,
                maximum_minute,
            )
        )
    if workers <= 1:
        results = [_load_profile_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            results = list(executor.map(_load_profile_task, tasks))
    frames = [result[0] for result in results if not result[0].empty]
    if not frames:
        raise FileNotFoundError("No cleaned average-weekday profiles were found")
    profiles = pd.concat(frames, ignore_index=True, sort=False)
    profiles = profiles.sort_values(
        ["tmc", "t_min", "corridor_output"], kind="mergesort"
    ).drop_duplicates(["tmc", "t_min"], keep="first")
    used = sorted(
        set(Path(path) for result in results for path in result[1])
    )
    return profiles.reset_index(drop=True), used


def _select_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame]
    return frame[available].copy()


def build_map_and_network_snapshot(
    package_root: Path,
    config: ExpansionConfig,
    *,
    mapmatching_run: Optional[Path] = None,
    network_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Path]]:
    package_root = Path(package_root)
    map_frames: List[pd.DataFrame] = []
    route_frames: List[pd.DataFrame] = []
    network_frames: List[pd.DataFrame] = []
    used: List[Path] = []
    if mapmatching_run is None or network_root is None:
        raise ValueError(
            "mapmatching_run and network_root are required; automatic "
            "workspace/latest-run discovery is not supported"
        )
    mapmatching_run = Path(mapmatching_run).resolve()
    network_root = Path(network_root).resolve()
    for period in ("AM", "MD", "PM"):
        map_dir = mapmatching_run / period.lower()
        map_path = map_dir / "full_tmc_to_link.csv"
        summary_path = map_dir / "full_route_match_summary.csv"
        network_path = network_root / period.lower() / "link.csv"
        for path in (map_path, summary_path, network_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            used.append(path)

        mappings = pd.read_csv(map_path, dtype={"tmc": str}, low_memory=False)
        summary = pd.read_csv(
            summary_path, dtype={"tmc": str}, low_memory=False
        )
        summary_fields = _select_columns(
            summary,
            (
                "tmc",
                "road",
                "direction",
                "intersection",
                "road_order",
                "tmc_miles",
                "route_link_count",
                "route_link_ids",
                "o_node_id",
                "d_node_id",
                "start_link_id",
                "end_link_id",
                "route_length_mi",
                "confidence",
                "status",
                period.lower() + "_path_status",
                period.lower() + "_open_share",
            ),
        )
        summary_fields["period"] = period
        route_frames.append(summary_fields)

        summary_join = summary_fields.rename(
            columns={
                "status": "map_status",
                "confidence": "map_confidence",
                "tmc_miles": "summary_tmc_miles",
                "route_link_ids": "summary_route_link_ids",
                "o_node_id": "summary_o_node_id",
                "d_node_id": "summary_d_node_id",
            }
        )
        duplicate_columns = [
            column
            for column in ("road", "direction", "road_order", "period")
            if column in summary_join
        ]
        summary_join = summary_join.drop(columns=duplicate_columns)
        mappings["tmc"] = mappings["tmc"].astype(str).str.strip()
        mappings["period"] = period
        open_column = period.lower() + "_is_open"
        mappings["period_is_open"] = (
            _parse_bool(mappings[open_column])
            if open_column in mappings
            else True
        )
        mappings["map_row_number"] = np.arange(1, len(mappings) + 1)
        mappings = mappings.merge(
            summary_join,
            on="tmc",
            how="left",
            validate="many_to_one",
        )
        map_frames.append(
            _select_columns(
                mappings,
                (
                    "period",
                    "tmc",
                    "road",
                    "direction",
                    "road_order",
                    "sequence",
                    "link_id",
                    "from_node_id",
                    "to_node_id",
                    "length_mi",
                    "cumulative_mi",
                    "distance_to_tmc_ft",
                    "bearing_diff_deg",
                    "STREETNAME",
                    "allowed_use",
                    "lanes",
                    "capacity",
                    "free_speed",
                    "link_type",
                    "PROJECTID",
                    "LINKID",
                    "period_is_open",
                    "map_row_number",
                    "map_status",
                    "map_confidence",
                    "summary_tmc_miles",
                    "summary_route_link_ids",
                    "summary_o_node_id",
                    "summary_d_node_id",
                ),
            )
        )

        wanted_network = {
            "link_id",
            "from_node_id",
            "to_node_id",
            "length",
            "length_in_mile",
            "lanes",
            "capacity",
            "free_speed",
            "link_type",
            "allowed_use",
            "STREETNAME",
            "ref_volume",
            {"AM": "I4AMVOL", "MD": "I4MDVOL", "PM": "I4PMVOL"}[period],
        }
        network = pd.read_csv(
            network_path,
            usecols=lambda column: column in wanted_network,
            low_memory=False,
        )
        volume_column = {
            "AM": "I4AMVOL",
            "MD": "I4MDVOL",
            "PM": "I4PMVOL",
        }[period]
        if volume_column in network:
            network["period_volume"] = pd.to_numeric(
                network[volume_column], errors="coerce"
            )
        elif "ref_volume" in network:
            network["period_volume"] = pd.to_numeric(
                network["ref_volume"], errors="coerce"
            )
        else:
            network["period_volume"] = np.nan
        network["period"] = period
        network_frames.append(
            _select_columns(
                network,
                (
                    "period",
                    "link_id",
                    "from_node_id",
                    "to_node_id",
                    "length",
                    "length_in_mile",
                    "lanes",
                    "capacity",
                    "free_speed",
                    "link_type",
                    "allowed_use",
                    "STREETNAME",
                    "period_volume",
                ),
            )
        )
    return (
        pd.concat(map_frames, ignore_index=True, sort=False),
        pd.concat(route_frames, ignore_index=True, sort=False),
        pd.concat(network_frames, ignore_index=True, sort=False),
        used,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Dict[str, object]:
    frame.to_csv(path, index=False)
    return {
        "path": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256(path),
    }


def prepare_snapshot(
    package_root: Path,
    module_root: Path,
    config: ExpansionConfig,
    explicit_workers: Optional[int] = None,
    *,
    cbi_output_root: Optional[Path] = None,
    corridor_input_root: Optional[Path] = None,
    mapmatching_run: Optional[Path] = None,
    network_root: Optional[Path] = None,
) -> Dict[str, object]:
    package_root = Path(package_root).resolve()
    module_root = Path(module_root).resolve()
    snapshot_dir = module_root / "input-snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if any(
        value is None
        for value in (
            cbi_output_root,
            corridor_input_root,
            mapmatching_run,
            network_root,
        )
    ):
        raise ValueError(
            "prepare requires explicit CBI outputs, corridor inputs, "
            "map-matching outputs, and network root"
        )
    resolved_cbi_root = Path(cbi_output_root).resolve()
    corridor_dirs = [
        path
        for path in resolved_cbi_root.iterdir()
        if path.is_dir()
    ]
    worker_plan = recommend_workers(
        max(1, len(corridor_dirs)),
        target_fraction=config.worker_fraction,
        explicit_workers=explicit_workers,
    )
    representatives, episode_sources = build_representative_snapshot(
        resolved_cbi_root,
        config,
        worker_plan.workers,
    )
    profiles, profile_sources = build_profile_snapshot(
        package_root,
        config,
        worker_plan.workers,
        cbi_output_root=resolved_cbi_root,
        corridor_input_root=corridor_input_root,
    )
    mappings, routes, network, map_sources = build_map_and_network_snapshot(
        package_root,
        config,
        mapmatching_run=mapmatching_run,
        network_root=network_root,
    )
    products = {
        "tmc_period_representatives": _write_csv(
            representatives,
            snapshot_dir / "tmc_period_representatives.csv",
        ),
        "tmc_profiles": _write_csv(
            profiles, snapshot_dir / "tmc_profiles.csv"
        ),
        "map_matches": _write_csv(
            mappings, snapshot_dir / "map_matches.csv"
        ),
        "route_summary": _write_csv(
            routes, snapshot_dir / "route_summary.csv"
        ),
        "regional_network": _write_csv(
            network, snapshot_dir / "regional_network.csv"
        ),
    }
    source_paths = sorted(
        set(episode_sources + profile_sources + map_sources), key=str
    )
    sources = {
        str(path.relative_to(package_root)): {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in source_paths
    }
    manifest = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "package_root_at_snapshot": str(package_root),
        "module_root": str(module_root),
        "config": config.to_dict(),
        "worker_plan": worker_plan.to_dict(),
        "products": products,
        "source_files": sources,
    }
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
