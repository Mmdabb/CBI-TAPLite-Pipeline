"""Apply direct-spatial-class T2 precedence to mapped period link files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import uuid

import numpy as np
import pandas as pd

from cbi.workers import recommend_workers
PERIOD_DIRECTORIES = {"AM": "am", "MD": "md", "PM": "pm"}
PERIOD_LIMITS = {"AM": (6.0, 9.0), "MD": (9.0, 15.0), "PM": (15.0, 19.0)}
HYBRID_COLUMNS = [
    "t0_hybrid_hour",
    "t2_hybrid_hour",
    "t3_hybrid_hour",
    "t2_hybrid_source",
    "t2_hybrid_detail",
    "t2_hybrid_precedence_rank",
    "t2_direct_hour",
    "t2_direct_origin",
    "t2_spatial_hour",
    "t2_spatial_tier",
    "t2_spatial_method",
    "t2_spatial_confidence",
    "t2_class_hour",
    "t2_observation_status",
    "t2_observed_no_congestion_protected",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_spatial_output(package_root: Path) -> Path:
    """Return the mapper-owned spatial expansion artifact."""

    del package_root
    path = (
        Path(__file__).resolve().parent
        / "resources"
        / "ridge_completion"
        / "spatial_run"
        / "outputs"
        / "expanded_link_t2.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"The bundled spatial expanded_link_t2.csv is missing: {path}"
        )
    return path


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_spatial_assignments(path: Path) -> pd.DataFrame:
    spatial = pd.read_csv(path, low_memory=False)
    required = {
        "period",
        "link_id",
        "t2_hour",
        "assignment_tier",
        "assignment_method",
        "assignment_confidence",
    }
    missing = sorted(required - set(spatial.columns))
    if missing:
        raise ValueError(f"Spatial expansion output is missing columns: {missing}")
    spatial["period"] = spatial["period"].astype(str).str.upper()
    _numeric(spatial, ("link_id", "t0_hour", "t2_hour", "t3_hour"))
    if spatial["link_id"].isna().any():
        raise ValueError("Spatial expansion output contains a blank link_id.")
    spatial["link_id"] = spatial["link_id"].astype(np.int64)
    if spatial.duplicated(["period", "link_id"]).any():
        raise ValueError(
            "Spatial expansion output contains duplicate period/link_id rows."
        )
    unknown_periods = sorted(
        set(spatial["period"].dropna()) - set(PERIOD_DIRECTORIES)
    )
    if unknown_periods:
        raise ValueError(
            f"Spatial expansion output contains unknown periods: {unknown_periods}"
        )
    snapshot_path = (
        Path(path).resolve().parent.parent
        / "input-snapshot"
        / "tmc_period_representatives.csv"
    )
    if (
        snapshot_path.is_file()
        and "episode_id" in spatial
        and (
            "t0_hour" not in spatial
            or "t3_hour" not in spatial
            or spatial.loc[
                spatial["assignment_tier"].eq("A_direct"),
                [column for column in ("t0_hour", "t3_hour") if column in spatial],
            ].isna().any().any()
        )
    ):
        representatives = pd.read_csv(
            snapshot_path,
            usecols=["period", "episode_id", "t0_hour", "t3_hour"],
            dtype={"episode_id": "string"},
            low_memory=False,
        )
        representatives["period"] = (
            representatives["period"].astype(str).str.upper()
        )
        if representatives.duplicated(["period", "episode_id"]).any():
            raise ValueError(
                "Spatial representative snapshot contains duplicate "
                "period/episode_id keys."
            )
        spatial["episode_id"] = spatial["episode_id"].astype("string")
        spatial = spatial.merge(
            representatives.rename(
                columns={
                    "t0_hour": "snapshot_t0_hour",
                    "t3_hour": "snapshot_t3_hour",
                }
            ),
            on=["period", "episode_id"],
            how="left",
            validate="many_to_one",
        )
        for boundary in ("t0_hour", "t3_hour"):
            snapshot_boundary = f"snapshot_{boundary}"
            if boundary in spatial:
                spatial[boundary] = spatial[boundary].combine_first(
                    spatial[snapshot_boundary]
                )
            else:
                spatial[boundary] = spatial[snapshot_boundary]
            spatial = spatial.drop(columns=snapshot_boundary)
    return spatial


def load_mapping_period(path: Path, period: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"link_id", "t0_hour", "t2_hour", "t3_hour", "t2_est"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(
            f"{period} mapped link file is missing columns: {missing}"
        )
    optional = [
        column
        for column in (
            "t2_source_method",
            "t2_observation_status",
            "t2_observed_no_congestion_protected",
        )
        if column in header
    ]
    frame = pd.read_csv(
        path,
        usecols=[*required, *optional],
        low_memory=False,
    )
    _numeric(frame, ("link_id", "t0_hour", "t2_hour", "t3_hour", "t2_est"))
    if frame["link_id"].isna().any():
        raise ValueError(f"{period} mapped link file contains blank link_id.")
    frame["link_id"] = frame["link_id"].astype(np.int64)
    if frame["link_id"].duplicated().any():
        raise ValueError(
            f"{period} mapped link file contains duplicate link_id values."
        )
    if "t2_source_method" not in frame:
        frame["t2_source_method"] = ""
    if "t2_observation_status" not in frame:
        frame["t2_observation_status"] = ""
    if "t2_observed_no_congestion_protected" not in frame:
        frame["t2_observed_no_congestion_protected"] = False
    frame["t2_observed_no_congestion_protected"] = (
        frame["t2_observed_no_congestion_protected"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    return frame


def build_hybrid_assignments(
    mapping: pd.DataFrame,
    spatial: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    period = str(period).upper()
    if period not in PERIOD_DIRECTORIES:
        raise ValueError(f"Unknown period: {period}")
    main = mapping.copy()
    if "t2_observation_status" not in main:
        main["t2_observation_status"] = ""
    if "t2_observed_no_congestion_protected" not in main:
        main["t2_observed_no_congestion_protected"] = False
    spatial_period = spatial.loc[spatial["period"].eq(period)].copy()
    spatial_columns = [
        "link_id",
        "t2_hour",
        "assignment_tier",
        "assignment_method",
        "assignment_confidence",
    ]
    for column in ("t0_hour", "t3_hour"):
        if column in spatial_period:
            spatial_columns.append(column)
    spatial_period = spatial_period[spatial_columns].rename(
        columns={
            "t0_hour": "prototype_t0_hour",
            "t2_hour": "prototype_t2_hour",
            "t3_hour": "prototype_t3_hour",
            "assignment_tier": "prototype_assignment_tier",
            "assignment_method": "prototype_assignment_method",
            "assignment_confidence": "prototype_assignment_confidence",
        }
    )
    joined = main.merge(
        spatial_period,
        on="link_id",
        how="left",
        validate="one_to_one",
    )
    for column in ("prototype_t0_hour", "prototype_t3_hour"):
        if column not in joined:
            joined[column] = np.nan

    protected_no_congestion = joined[
        "t2_observed_no_congestion_protected"
    ].fillna(False).astype(bool)
    if joined.loc[
        protected_no_congestion,
        ["t0_hour", "t2_hour", "t3_hour"],
    ].notna().any().any():
        raise ValueError(
            f"{period} protected no-congestion rows contain direct boundaries."
        )
    mapping_direct = joined["t2_hour"].notna() & ~protected_no_congestion
    prototype_value = pd.to_numeric(
        joined["prototype_t2_hour"], errors="coerce"
    )
    prototype_direct = (
        ~protected_no_congestion
        & ~mapping_direct
        & prototype_value.notna()
        & joined["prototype_assignment_tier"].eq("A_direct")
    )
    any_direct = mapping_direct | prototype_direct
    spatial_candidate = (
        ~protected_no_congestion
        & ~any_direct
        & prototype_value.notna()
        & joined["prototype_assignment_tier"].isin(
            {"B_bracketed", "C_one_sided"}
        )
    )
    class_candidate = (
        ~protected_no_congestion
        & ~any_direct
        & ~spatial_candidate
        & joined["t2_est"].notna()
    )

    result = pd.DataFrame({"link_id": joined["link_id"]})
    result["period"] = period
    result["t2_direct_hour"] = joined["t2_hour"]
    result.loc[prototype_direct, "t2_direct_hour"] = prototype_value.loc[
        prototype_direct
    ]
    result["t2_direct_origin"] = ""
    mapping_method = joined["t2_source_method"].fillna("").astype(str)
    result.loc[mapping_direct, "t2_direct_origin"] = (
        "boundary_mapping_" + mapping_method.loc[mapping_direct]
    ).str.rstrip("_")
    result.loc[prototype_direct, "t2_direct_origin"] = (
        "prototype_prefilter_direct"
    )

    result["t2_spatial_hour"] = np.nan
    result.loc[spatial_candidate, "t2_spatial_hour"] = prototype_value.loc[
        spatial_candidate
    ]
    result["t2_spatial_tier"] = ""
    result.loc[spatial_candidate, "t2_spatial_tier"] = joined.loc[
        spatial_candidate, "prototype_assignment_tier"
    ]
    result["t2_spatial_method"] = ""
    result.loc[spatial_candidate, "t2_spatial_method"] = joined.loc[
        spatial_candidate, "prototype_assignment_method"
    ]
    result["t2_spatial_confidence"] = ""
    result.loc[spatial_candidate, "t2_spatial_confidence"] = joined.loc[
        spatial_candidate, "prototype_assignment_confidence"
    ]

    result["t2_class_hour"] = joined["t2_est"].where(
        ~mapping_direct & ~protected_no_congestion
    )
    result["t2_hybrid_hour"] = result["t2_direct_hour"].combine_first(
        result["t2_spatial_hour"]
    ).combine_first(result["t2_class_hour"])
    result["t2_hybrid_source"] = np.select(
        [
            protected_no_congestion,
            any_direct,
            spatial_candidate,
            class_candidate,
        ],
        ["observed_no_congestion", "direct", "spatial", "class"],
        default="unassigned",
    )
    result["t2_hybrid_detail"] = ""
    result.loc[
        protected_no_congestion, "t2_hybrid_detail"
    ] = "protected_best_match_tmc_no_average_weekday_congestion"
    result.loc[any_direct, "t2_hybrid_detail"] = result.loc[
        any_direct, "t2_direct_origin"
    ]
    result.loc[spatial_candidate, "t2_hybrid_detail"] = (
        result.loc[spatial_candidate, "t2_spatial_tier"].astype(str)
        + "__"
        + result.loc[spatial_candidate, "t2_spatial_method"].astype(str)
    )
    result.loc[class_candidate, "t2_hybrid_detail"] = "vdf_link_type_mean"
    result.loc[
        result["t2_hybrid_source"].eq("unassigned"), "t2_hybrid_detail"
    ] = "no_available_t2"
    result["t2_hybrid_precedence_rank"] = pd.Series(
        np.select(
            [
                protected_no_congestion,
                any_direct,
                spatial_candidate,
                class_candidate,
            ],
            [0, 1, 2, 3],
            default=np.nan,
        ),
        index=result.index,
    ).astype("Int64")

    result["t0_hybrid_hour"] = joined["t0_hour"]
    result["t3_hybrid_hour"] = joined["t3_hour"]
    result.loc[prototype_direct, "t0_hybrid_hour"] = joined.loc[
        prototype_direct, "prototype_t0_hour"
    ]
    result.loc[prototype_direct, "t3_hybrid_hour"] = joined.loc[
        prototype_direct, "prototype_t3_hour"
    ]
    result.loc[
        protected_no_congestion,
        ["t0_hybrid_hour", "t2_hybrid_hour", "t3_hybrid_hour"],
    ] = np.nan
    result["t2_observation_status"] = joined[
        "t2_observation_status"
    ].fillna("")
    result["t2_observed_no_congestion_protected"] = protected_no_congestion

    if not np.allclose(
        result.loc[mapping_direct, "t2_hybrid_hour"],
        joined.loc[mapping_direct, "t2_hour"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{period} hybrid output changed a mapped direct T2.")
    if joined.loc[mapping_direct, ["t0_hour", "t3_hour"]].isna().any().any():
        raise ValueError(
            f"{period} mapped direct T2 rows do not all have T0 and T3."
        )
    ordered = (
        joined.loc[mapping_direct, "t0_hour"]
        <= joined.loc[mapping_direct, "t2_hour"]
    ) & (
        joined.loc[mapping_direct, "t2_hour"]
        <= joined.loc[mapping_direct, "t3_hour"]
    )
    if not ordered.all():
        raise ValueError(f"{period} mapped direct T0/T2/T3 are not ordered.")
    if joined.loc[
        prototype_direct,
        ["prototype_t0_hour", "prototype_t3_hour"],
    ].isna().any().any():
        raise ValueError(
            f"{period} prototype direct T2 rows do not all have T0 and T3."
        )
    prototype_ordered = (
        joined.loc[prototype_direct, "prototype_t0_hour"]
        <= joined.loc[prototype_direct, "prototype_t2_hour"]
    ) & (
        joined.loc[prototype_direct, "prototype_t2_hour"]
        <= joined.loc[prototype_direct, "prototype_t3_hour"]
    )
    if not prototype_ordered.all():
        raise ValueError(f"{period} prototype direct T0/T2/T3 are not ordered.")
    if not np.allclose(
        result.loc[spatial_candidate, "t2_hybrid_hour"],
        result.loc[spatial_candidate, "t2_spatial_hour"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{period} spatial T2 precedence validation failed.")
    if not np.allclose(
        result.loc[class_candidate, "t2_hybrid_hour"],
        result.loc[class_candidate, "t2_class_hour"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{period} class T2 fallback validation failed.")
    if result.loc[
        protected_no_congestion,
        ["t0_hybrid_hour", "t2_hybrid_hour", "t3_hybrid_hour"],
    ].notna().any().any():
        raise ValueError(
            f"{period} protected no-congestion boundaries were overwritten."
        )
    lower, upper = PERIOD_LIMITS[period]
    assigned = result["t2_hybrid_hour"].notna()
    if not result.loc[assigned, "t2_hybrid_hour"].between(
        lower, upper, inclusive="left"
    ).all():
        raise ValueError(f"{period} hybrid T2 contains an out-of-period value.")
    return result[
        ["link_id", "period", *HYBRID_COLUMNS]
    ].sort_values("link_id", kind="mergesort").reset_index(drop=True)


def _csv_suffix(values: Iterable[object]) -> str:
    normalized = []
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


def write_hybrid_link_file(
    source: Path,
    target: Path,
    assignments: pd.DataFrame,
) -> Dict[str, object]:
    if assignments["link_id"].duplicated().any():
        raise ValueError("Hybrid assignment table contains duplicate link_id values.")
    lookup = assignments.set_index("link_id")[HYBRID_COLUMNS].to_dict(
        orient="index"
    )
    target.parent.mkdir(parents=True, exist_ok=False)
    rows = 0
    seen: set[int] = set()
    with source.open("r", encoding="utf-8", newline="") as input_stream, (
        target.open("w", encoding="utf-8", newline="")
    ) as output_stream:
        header_line = input_stream.readline()
        if not header_line:
            raise ValueError(f"{source} is empty.")
        header = next(csv.reader([header_line]))
        duplicate_hybrid_headers = sorted(
            column
            for column in HYBRID_COLUMNS
            if header.count(column) > 1
        )
        if duplicate_hybrid_headers:
            raise ValueError(
                f"{source} contains duplicate hybrid columns: "
                f"{duplicate_hybrid_headers}"
            )
        if "link_id" not in header:
            raise ValueError(f"{source} is missing link_id.")
        link_index = header.index("link_id")
        retained_indexes = [
            index
            for index, column in enumerate(header)
            if column not in HYBRID_COLUMNS
        ]
        retained_header = [header[index] for index in retained_indexes]
        refreshing_existing_hybrid = len(retained_indexes) != len(header)
        newline = "\r\n" if header_line.endswith("\r\n") else "\n"
        output_stream.write(
            (
                _csv_suffix(retained_header)
                if refreshing_existing_hybrid
                else header_line.rstrip("\r\n")
            )
            + ","
            + _csv_suffix(HYBRID_COLUMNS)
            + newline
        )
        for raw_line in input_stream:
            fields = next(csv.reader([raw_line]))
            if len(fields) != len(header):
                raise ValueError(
                    f"{source} contains a multiline or malformed CSV row."
                )
            link_id = int(fields[link_index])
            if link_id in seen:
                raise ValueError(f"{source} has duplicate link_id {link_id}.")
            seen.add(link_id)
            record = lookup.get(link_id)
            if record is None:
                raise ValueError(f"Hybrid assignment is missing link {link_id}.")
            output_stream.write(
                (
                    _csv_suffix(fields[index] for index in retained_indexes)
                    if refreshing_existing_hybrid
                    else raw_line.rstrip("\r\n")
                )
                + ","
                + _csv_suffix(record[column] for column in HYBRID_COLUMNS)
                + newline
            )
            rows += 1
    if rows != len(assignments):
        raise ValueError(
            f"Hybrid output wrote {rows} rows but expected {len(assignments)}."
        )
    return {
        "path": str(target),
        "rows": rows,
        "sha256": sha256(target),
        "assignments_by_source": {
            str(source_name): int(count)
            for source_name, count in assignments[
                "t2_hybrid_source"
            ].value_counts().items()
        },
    }


def _coverage_rows(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in [*PERIOD_DIRECTORIES, "ALL"]:
        group = (
            assignments
            if period == "ALL"
            else assignments.loc[assignments["period"].eq(period)]
        )
        total = len(group)
        counts = group["t2_hybrid_source"].value_counts()
        assigned = int(group["t2_hybrid_hour"].notna().sum())
        rows.append(
            {
                "period": period,
                "network_link_period_rows": int(total),
                "direct": int(counts.get("direct", 0)),
                "spatial": int(counts.get("spatial", 0)),
                "class": int(counts.get("class", 0)),
                "unassigned": int(counts.get("unassigned", 0)),
                "observed_no_congestion": int(
                    counts.get("observed_no_congestion", 0)
                ),
                "assigned": assigned,
                "coverage_pct": 100.0 * assigned / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def process_hybrid_period(
    task: Tuple[str, Path, Path, Path],
) -> Dict[str, object]:
    period, source, spatial_output, target = task
    mapping = load_mapping_period(source, period)
    spatial = load_spatial_assignments(spatial_output)
    assignments = build_hybrid_assignments(mapping, spatial, period)
    result = write_hybrid_link_file(source, target, assignments)
    audit_path = target.parent / "hybrid_assignments.csv"
    assignments.to_csv(audit_path, index=False)
    result.update(
        {
            "period": period,
            "assignment_audit": str(audit_path),
            "assignment_audit_sha256": sha256(audit_path),
        }
    )
    return result


def update_run_manifest(
    mapping_link_root: Path,
    hybrid_manifest: Dict[str, object],
) -> None:
    path = mapping_link_root / "run_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    period_link_files = manifest.setdefault("period_link_files", {})
    for period, result in hybrid_manifest["period_products"].items():
        item = period_link_files.setdefault(period, {})
        item.update(
            {
                "sha256": result["sha256"],
                "hybrid_assignments_by_source": result[
                    "assignments_by_source"
                ],
                "final_t2_column": "t2_hybrid_hour",
            }
        )
    manifest["hybrid_t2_postprocessing"] = {
        "status": hybrid_manifest["status"],
        "precedence": hybrid_manifest["precedence"],
        "spatial_output": hybrid_manifest["spatial_output"],
        "spatial_output_sha256": hybrid_manifest["spatial_output_sha256"],
        "manifest": "hybrid_t2_manifest.json",
        "coverage_summary": "hybrid_t2_coverage_summary.csv",
        "long_audit": "hybrid_link_t2_long.csv",
        "worker_plan": hybrid_manifest["worker_plan"],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def apply_hybrid_t2(
    mapping_link_root: Path,
    spatial_output: Path,
    *,
    workers: Optional[int] = None,
    worker_fraction: float = 0.70,
    update_parent_manifest: bool = True,
) -> Dict[str, object]:
    mapping_link_root = Path(mapping_link_root).resolve()
    spatial_output = Path(spatial_output).resolve()
    if not spatial_output.is_file():
        raise FileNotFoundError(spatial_output)
    sources = {
        period: (
            mapping_link_root
            / "period_link_files"
            / directory
            / "link.csv"
        )
        for period, directory in PERIOD_DIRECTORIES.items()
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing period link files: {missing}")
    worker_plan = recommend_workers(
        len(PERIOD_DIRECTORIES),
        target_fraction=worker_fraction,
        explicit_workers=workers,
    )
    input_hashes = {
        str(source): sha256(source)
        for source in sources.values()
    }
    stage_root = (
        mapping_link_root
        / f".hybrid_t2_mapping_{uuid.uuid4().hex}"
    )
    stage_root.mkdir(parents=False, exist_ok=False)
    committed = False
    try:
        tasks = [
            (
                period,
                sources[period],
                spatial_output,
                stage_root
                / "period_link_files"
                / directory
                / "link.csv",
            )
            for period, directory in PERIOD_DIRECTORIES.items()
        ]
        if worker_plan.workers <= 1:
            results = [process_hybrid_period(task) for task in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=worker_plan.workers
            ) as executor:
                results = list(executor.map(process_hybrid_period, tasks))
        results.sort(
            key=lambda item: list(PERIOD_DIRECTORIES).index(item["period"])
        )
        assignments = [
            pd.read_csv(result["assignment_audit"], low_memory=False)
            for result in results
        ]
        combined = pd.concat(assignments, ignore_index=True, sort=False)
        if combined.duplicated(["period", "link_id"]).any():
            raise ValueError("Combined hybrid output contains duplicate keys.")

        audit_path = stage_root / "hybrid_link_t2_long.csv"
        combined.to_csv(audit_path, index=False)
        coverage = _coverage_rows(combined)
        coverage_path = stage_root / "hybrid_t2_coverage_summary.csv"
        coverage.to_csv(coverage_path, index=False)
        period_results = {
            str(result["period"]): {
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"period", "assignment_audit"}
                },
                "path": str(sources[str(result["period"])]),
            }
            for result in results
        }
        manifest: Dict[str, object] = {
            "status": "PASS",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "mapped_hybrid_full_network",
            "precedence": ["direct", "spatial", "class"],
            "final_t2_column": "t2_hybrid_hour",
            "direct_rule": (
                "Use mapped direct T2 first; where absent, retain an "
                "A_direct pre-filter observation from spatial expansion."
            ),
            "spatial_rule": (
                "Use only B_bracketed or C_one_sided spatial assignments "
                "after all direct candidates are empty."
            ),
            "class_rule": (
                "Use t2_est, the period/link-type class mean, only after "
                "direct and spatial candidates are empty."
            ),
            "mapping_link_root": str(mapping_link_root),
            "spatial_output": str(spatial_output),
            "spatial_output_sha256": sha256(spatial_output),
            "mapping_inputs_before_hybrid": input_hashes,
            "worker_plan": worker_plan.to_dict(),
            "parallel_stages": [
                "AM hybrid assignment",
                "MD hybrid assignment",
                "PM hybrid assignment",
            ],
            "period_products": period_results,
            "hybrid_long": {
                "path": str(
                    mapping_link_root / "hybrid_link_t2_long.csv"
                ),
                "rows": int(len(combined)),
                "sha256": sha256(audit_path),
            },
            "coverage_summary": {
                "path": str(
                    mapping_link_root
                    / "hybrid_t2_coverage_summary.csv"
                ),
                "rows": int(len(coverage)),
                "sha256": sha256(coverage_path),
            },
        }
        staged_manifest = stage_root / "hybrid_t2_manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        for period, directory in PERIOD_DIRECTORIES.items():
            staged = (
                stage_root
                / "period_link_files"
                / directory
                / "link.csv"
            )
            os.replace(staged, sources[period])
        os.replace(
            audit_path,
            mapping_link_root / "hybrid_link_t2_long.csv",
        )
        os.replace(
            coverage_path,
            mapping_link_root / "hybrid_t2_coverage_summary.csv",
        )
        os.replace(
            staged_manifest,
            mapping_link_root / "hybrid_t2_manifest.json",
        )
        committed = True
        if update_parent_manifest:
            update_run_manifest(mapping_link_root, manifest)
        return manifest
    finally:
        if stage_root.is_dir():
            if committed:
                shutil.rmtree(stage_root)
            else:
                print(
                    "Hybrid mapping failed before commit; staged "
                    f"diagnostics remain at {stage_root}",
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply direct-spatial-class T2 precedence in place."
    )
    parser.add_argument("mapping_link_root", type=Path)
    parser.add_argument("spatial_output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--worker-fraction", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = apply_hybrid_t2(
        args.mapping_link_root,
        args.spatial_output,
        workers=args.workers,
        worker_fraction=args.worker_fraction,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
