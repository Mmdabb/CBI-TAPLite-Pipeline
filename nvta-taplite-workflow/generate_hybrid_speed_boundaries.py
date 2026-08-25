"""Build the complete observed/assignment hybrid speed-anchor resource.

Hierarchy:
1. Keep existing post-QC CBI anchors for covered canonical winners.
2. Use direct regional weekday-average boundary speeds for other winners.
3. Use stable assignment ``speed_mph`` anchors for non-canonical links.

The script does not convert a network or run TAPlite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from generate_assignment_speed_boundaries import BOUNDARY_DTYPE, _available_mean, _load_period


PERIODS = {"AM": (360, 540), "MD": (540, 900), "PM": (900, 1140)}
BOUNDARY_FIELDS = [name for name in BOUNDARY_DTYPE.names or () if "speed_mph" in name]
PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def packed_key(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["from_node_id"].to_numpy(dtype=np.uint64) << np.uint64(32)
    ) | frame["to_node_id"].to_numpy(dtype=np.uint64)


def load_canonical(path: Path) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    required = {"tmc", "from_node_id", "to_node_id"}
    missing = sorted(required - header)
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    optional = [column for column in ("link_id", "road", "direction") if column in header]
    frame = pd.read_csv(path, usecols=[*required, *optional], dtype={"tmc": "string"})
    frame["tmc"] = frame["tmc"].str.strip().str.upper()
    for column in ("from_node_id", "to_node_id"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["tmc", "from_node_id", "to_node_id"]).copy()
    frame[["from_node_id", "to_node_id"]] = frame[["from_node_id", "to_node_id"]].astype("uint32")
    frame["packed_key"] = packed_key(frame)
    if frame.duplicated("packed_key").any():
        raise ValueError("Canonical mapping has duplicate directed node pairs")
    return frame


def load_existing(path: Path) -> pd.DataFrame:
    lookup = np.load(path, allow_pickle=False)
    if lookup.dtype != BOUNDARY_DTYPE:
        raise ValueError(f"Unexpected existing lookup dtype: {lookup.dtype.descr}")
    frame = pd.DataFrame({name: lookup[name] for name in lookup.dtype.names or ()})
    if frame.duplicated("packed_key").any():
        raise ValueError("Existing observed lookup has duplicate keys")
    return frame


def load_assignment(root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    combined: pd.DataFrame | None = None
    sources: dict[str, Path] = {}
    for period in ("am", "md", "pm"):
        frame, source = _load_period(root, period)
        sources[period.upper()] = source
        combined = frame if combined is None else combined.merge(
            frame, on=["from_node_id", "to_node_id"], how="outer", validate="one_to_one"
        )
    assert combined is not None
    combined["qvdf_start_speed_mph_am"] = combined["speed_mph_am"]
    combined["qvdf_end_speed_mph_am"] = _available_mean(combined["speed_mph_am"], combined["speed_mph_md"])
    combined["qvdf_start_speed_mph_md"] = combined["qvdf_end_speed_mph_am"]
    combined["qvdf_end_speed_mph_md"] = _available_mean(combined["speed_mph_md"], combined["speed_mph_pm"])
    combined["qvdf_start_speed_mph_pm"] = combined["qvdf_end_speed_mph_md"]
    combined["qvdf_end_speed_mph_pm"] = combined["speed_mph_pm"]
    combined["packed_key"] = packed_key(combined)
    if combined.duplicated("packed_key").any():
        raise ValueError("Stable assignment has duplicate directed node pairs")
    return combined, sources


def regional_boundaries(path: Path, wanted: set[str]) -> pd.DataFrame:
    minutes = sorted({minute for pair in PERIODS.values() for minute in pair})
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["tmc_code", "measurement_tstamp", "speed"],
        dtype={"tmc_code": "string"},
        chunksize=1_000_000,
    ):
        chunk["tmc"] = chunk["tmc_code"].str.strip().str.upper()
        chunk = chunk[chunk["tmc"].isin(wanted)].copy()
        if chunk.empty:
            continue
        chunk["datetime"] = pd.to_datetime(chunk["measurement_tstamp"], errors="coerce")
        chunk["speed"] = pd.to_numeric(chunk["speed"], errors="coerce")
        chunk = chunk[
            chunk["datetime"].notna()
            & chunk["datetime"].dt.weekday.lt(5)
            & chunk["speed"].between(1.0, 150.0)
        ].copy()
        chunk["t_min"] = chunk["datetime"].dt.hour * 60 + chunk["datetime"].dt.minute
        chunk = chunk[chunk["t_min"].isin(minutes)]
        if not chunk.empty:
            pieces.append(chunk[["tmc", "t_min", "speed"]])
    if not pieces:
        raise ValueError("No regional weekday boundary observations were found")
    wide = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["tmc", "t_min"], sort=False)["speed"]
        .mean()
        .unstack("t_min")
        .reindex(columns=minutes)
        .reset_index()
    )
    for period, (start, end) in PERIODS.items():
        suffix = period.lower()
        wide[f"qvdf_start_speed_mph_{suffix}"] = wide[start]
        wide[f"qvdf_end_speed_mph_{suffix}"] = wide[end]
    return wide[["tmc", *BOUNDARY_FIELDS]]


def virtual_boundaries(root: Path, wanted: set[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(Path(root).glob("*/Readings.csv")):
        frame = pd.read_csv(
            path,
            usecols=["tmc_code", "measurement_tstamp", "speed"],
            dtype={"tmc_code": "string"},
        )
        frame["tmc"] = frame["tmc_code"].str.strip().str.upper()
        frame = frame[frame["tmc"].isin(wanted)].copy()
        if frame.empty:
            continue
        frame["datetime"] = pd.to_datetime(
            frame["measurement_tstamp"], errors="coerce"
        )
        frame["speed"] = pd.to_numeric(frame["speed"], errors="coerce")
        frame = frame[
            frame["datetime"].notna() & frame["speed"].between(1.0, 150.0)
        ].copy()
        frame["t_min"] = (
            frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
        )
        minutes = sorted({minute for pair in PERIODS.values() for minute in pair})
        frame = frame[frame["t_min"].isin(minutes)]
        if not frame.empty:
            pieces.append(frame[["tmc", "t_min", "speed"]])
    if not pieces:
        raise ValueError(
            "No virtual weekday boundary observations were found for the "
            "requested canonical TMCs"
        )
    minutes = sorted({minute for pair in PERIODS.values() for minute in pair})
    wide = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["tmc", "t_min"], sort=False)["speed"]
        .mean()
        .unstack("t_min")
        .reindex(columns=minutes)
        .reset_index()
    )
    for period, (start, end) in PERIODS.items():
        suffix = period.lower()
        wide[f"qvdf_start_speed_mph_{suffix}"] = wide[start]
        wide[f"qvdf_end_speed_mph_{suffix}"] = wide[end]
    return wide[["tmc", *BOUNDARY_FIELDS]]


def save_lookup(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.sort_values("packed_key", kind="stable")
    lookup = np.empty(len(frame), dtype=BOUNDARY_DTYPE)
    for field in BOUNDARY_DTYPE.names or ():
        lookup[field] = frame[field].to_numpy(dtype=BOUNDARY_DTYPE[field])
    np.save(path, lookup, allow_pickle=False)
    restored = np.load(path, allow_pickle=False)
    if restored.dtype != BOUNDARY_DTYPE or len(restored) != len(frame):
        raise ValueError("Lookup round-trip failed")
    if len(restored) and np.any(np.diff(restored["packed_key"]) <= 0):
        raise ValueError("Lookup keys are not uniquely sorted")


def backup_resource(resource: Path, output: Path) -> dict[str, object]:
    backup = output / "previous_resource_backup"
    backup.mkdir(parents=True, exist_ok=False)
    files: dict[str, object] = {}
    for name in ("observed_link_speed_boundaries.npy", "boundary_completeness_report.csv", "metadata.json"):
        source = resource / name
        if source.is_file():
            target = backup / name
            shutil.copy2(source, target)
            files[name] = {"path": str(target), "sha256": sha256(target)}
    return {"directory": str(backup), "files": files}


def install_resource(output: Path, resource: Path) -> None:
    resource.mkdir(parents=True, exist_ok=True)
    for name in ("observed_link_speed_boundaries.npy", "boundary_completeness_report.csv", "metadata.json"):
        temporary = resource / f".{name}.installing"
        shutil.copy2(output / name, temporary)
        os.replace(temporary, resource / name)


def build_hybrid_lookup(
    canonical_path: Path,
    readings_path: Path,
    assignment_root: Path,
    existing_path: Path,
    output: Path,
    virtual_corridor_inputs: Path | None = None,
    install_to: Path | None = None,
) -> dict[str, object]:
    canonical_path, readings_path, assignment_root, existing_path = (
        canonical_path.resolve(), readings_path.resolve(), assignment_root.resolve(), existing_path.resolve()
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backup = backup_resource(install_to.resolve(), output) if install_to is not None else None

    canonical = load_canonical(canonical_path)
    existing = load_existing(existing_path)
    assignment, assignment_sources = load_assignment(assignment_root)
    canonical_keys = set(canonical["packed_key"].astype("uint64"))
    assignment_keys = set(assignment["packed_key"].astype("uint64"))
    if canonical_keys - assignment_keys:
        raise ValueError("Canonical node pairs are missing from the stable assignment")
    existing = existing[existing["packed_key"].isin(canonical_keys)].copy()
    existing_keys = set(existing["packed_key"].astype("uint64"))
    observed = canonical[["packed_key", "tmc"]].merge(
        existing[["packed_key", *BOUNDARY_FIELDS]],
        on="packed_key",
        how="left",
        validate="one_to_one",
    )
    for field in BOUNDARY_FIELDS:
        observed[f"{field}_source"] = np.where(
            observed[field].notna(),
            "existing_cbi_observed_weekday_average",
            "",
        )
    needs_fallback = observed[BOUNDARY_FIELDS].isna().any(axis=1)
    needed = observed.loc[needs_fallback, ["packed_key", "tmc"]].copy()
    actual_needed = needed[~needed["tmc"].str.startswith("VIRTUAL-")]
    virtual_needed = needed[needed["tmc"].str.startswith("VIRTUAL-")]

    def fill_fallback(boundaries: pd.DataFrame, targets: pd.DataFrame, source: str) -> None:
        direct = targets.merge(
            boundaries,
            on="tmc",
            how="left",
            validate="many_to_one",
        ).set_index("packed_key")
        indexed = observed.set_index("packed_key")
        for field in BOUNDARY_FIELDS:
            fillable = indexed.loc[direct.index, field].isna() & direct[field].notna()
            keys = direct.index[fillable]
            indexed.loc[keys, field] = direct.loc[keys, field].astype(float)
            indexed.loc[keys, f"{field}_source"] = source
        observed.update(indexed.reset_index())

    if not actual_needed.empty:
        fill_fallback(
            regional_boundaries(readings_path, set(actual_needed["tmc"])),
            actual_needed,
            "regional_direct_weekday_average",
        )
    if not virtual_needed.empty:
        if virtual_corridor_inputs is None:
            raise ValueError(
                "Partial virtual canonical anchors require --virtual-corridor-inputs"
            )
        fill_fallback(
            virtual_boundaries(
                Path(virtual_corridor_inputs).resolve(), set(virtual_needed["tmc"])
            ),
            virtual_needed,
            "virtual_pre_qc_weekday_average",
        )
    if observed[BOUNDARY_FIELDS].isna().any(axis=None):
        incomplete = observed.loc[
            observed[BOUNDARY_FIELDS].isna().any(axis=1), "tmc"
        ].tolist()
        raise ValueError(
            "Canonical speed-anchor coverage remains incomplete for: "
            + ", ".join(incomplete[:20])
        )

    result = assignment.copy()
    result["is_canonical_winner"] = result["packed_key"].isin(canonical_keys)
    result["canonical_tmc"] = result["packed_key"].map(canonical.set_index("packed_key")["tmc"])
    for field in BOUNDARY_FIELDS:
        result[f"{field}_source"] = np.where(
            result[field].notna(), "stable_assignment_speed_mph", "stable_assignment_period_link_absent"
        )
    result = result.set_index("packed_key", drop=False)
    observed = observed.set_index("packed_key", drop=False)
    for field in BOUNDARY_FIELDS:
        result.loc[observed.index, field] = observed[field].astype(float)
        result.loc[observed.index, f"{field}_source"] = observed[
            f"{field}_source"
        ].astype(str)
    result = result.reset_index(drop=True)
    canonical_result = result[result["is_canonical_winner"]]
    if len(canonical_result) != len(canonical) or canonical_result[BOUNDARY_FIELDS].isna().any(axis=None):
        raise ValueError("Canonical speed-anchor coverage is incomplete")
    values = result[BOUNDARY_FIELDS].to_numpy(dtype=float)
    if (np.isfinite(values) & ((values <= 0) | (values > 150))).any():
        raise ValueError("Boundary speed is outside (0, 150] mph")

    lookup_path = output / "observed_link_speed_boundaries.npy"
    save_lookup(result, lookup_path)
    audit_columns = ["packed_key", "from_node_id", "to_node_id", "canonical_tmc", "is_canonical_winner"]
    audit_columns += [column for column in ("speed_mph_am", "speed_mph_md", "speed_mph_pm") if column in result]
    for field in BOUNDARY_FIELDS:
        audit_columns += [field, f"{field}_source"]
    audit_path = output / "hybrid_speed_boundary_audit.csv"
    result[audit_columns].sort_values("packed_key").to_csv(audit_path, index=False)

    reports: list[pd.DataFrame] = []
    for period, (start, end) in PERIODS.items():
        suffix = period.lower()
        sf, ef = f"qvdf_start_speed_mph_{suffix}", f"qvdf_end_speed_mph_{suffix}"
        report = result[["packed_key", "from_node_id", "to_node_id", "canonical_tmc", "is_canonical_winner", sf, ef, f"{sf}_source", f"{ef}_source"]].copy()
        report = report.rename(columns={sf: "qvdf_start_speed_mph", ef: "qvdf_end_speed_mph", f"{sf}_source": "start_speed_source", f"{ef}_source": "end_speed_source"})
        report["period"], report["start_minute"], report["end_minute"] = period, start, end
        start_ok, end_ok = report["qvdf_start_speed_mph"].notna(), report["qvdf_end_speed_mph"].notna()
        report["boundary_status"] = np.select(
            [start_ok & end_ok, start_ok & ~end_ok, ~start_ok & end_ok],
            ["both", "start_only", "end_only"],
            default="neither",
        )
        reports.append(report)
    completeness = pd.concat(reports, ignore_index=True)
    completeness_path = output / "boundary_completeness_report.csv"
    completeness.to_csv(completeness_path, index=False)

    metadata: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "NumPy .npy structured array sorted by packed_key",
        "key_definition": "(uint64(from_node_id) << 32) | uint64(to_node_id)",
        "record_dtype": BOUNDARY_DTYPE.descr,
        "speed_unit": "mph",
        "boundary_minutes": {period: list(bounds) for period, bounds in PERIODS.items()},
        "hierarchy": [
            "existing post-QC CBI weekday-average anchors for covered canonical winners",
            "direct regional weekday-average boundary speeds for remaining canonical winners",
            "stable assignment speed_mph anchors for non-canonical network pairs",
        ],
        "assignment_boundary_logic": {
            "AM_start": "AM speed_mph",
            "AM_end_and_MD_start": "mean of available AM and MD speed_mph",
            "MD_end_and_PM_start": "mean of available MD and PM speed_mph",
            "PM_end": "PM speed_mph",
        },
        "counts": {
            "network_node_pairs": int(len(result)),
            "canonical_winner_pairs": int(len(canonical)),
            "canonical_existing_cbi_observed_pairs": int(len(existing)),
            "canonical_regional_direct_pairs": int(
                observed[[f"{field}_source" for field in BOUNDARY_FIELDS]]
                .eq("regional_direct_weekday_average")
                .any(axis=1)
                .sum()
            ),
            "canonical_virtual_direct_pairs": int(
                observed[[f"{field}_source" for field in BOUNDARY_FIELDS]]
                .eq("virtual_pre_qc_weekday_average")
                .any(axis=1)
                .sum()
            ),
            "assignment_fallback_noncanonical_pairs": int((~result["is_canonical_winner"]).sum()),
            "canonical_pairs_with_all_six_boundaries": int(canonical_result[BOUNDARY_FIELDS].notna().all(axis=1).sum()),
            "network_pairs_with_all_six_boundaries": int(result[BOUNDARY_FIELDS].notna().all(axis=1).sum()),
        },
        "sources": {
            "canonical_mapping": {"path": str(canonical_path), "sha256": sha256(canonical_path)},
            "regional_readings": {"path": str(readings_path), "sha256": sha256(readings_path)},
            "virtual_corridor_inputs": (
                str(Path(virtual_corridor_inputs).resolve())
                if virtual_corridor_inputs is not None
                else None
            ),
            "existing_observed_lookup": {"path": str(existing_path), "sha256": sha256(existing_path)},
            "stable_assignment": {period: {"path": str(path), "sha256": sha256(path)} for period, path in assignment_sources.items()},
        },
        "products": {
            "lookup": {"path": str(lookup_path), "sha256": sha256(lookup_path)},
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
            "completeness": {"path": str(completeness_path), "sha256": sha256(completeness_path)},
        },
        "installation": {"resource_directory": str(install_to.resolve()), "previous_resource": backup} if install_to is not None else None,
        "periods": {},
    }
    for period in PERIODS:
        subset = completeness[completeness["period"].eq(period)]
        metadata["periods"][period] = {
            "boundary_status_counts": subset["boundary_status"].value_counts().to_dict(),
            "start_source_counts": subset["start_speed_source"].value_counts().to_dict(),
            "end_source_counts": subset["end_speed_source"].value_counts().to_dict(),
        }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if install_to is not None:
        install_resource(output, install_to.resolve())
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--regional-readings", type=Path, required=True)
    parser.add_argument("--stable-assignment", type=Path, required=True)
    parser.add_argument("--existing-observed-lookup", type=Path, required=True)
    parser.add_argument("--virtual-corridor-inputs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install-to", type=Path, default=None)
    args = parser.parse_args()
    metadata = build_hybrid_lookup(
        args.canonical,
        args.regional_readings,
        args.stable_assignment,
        args.existing_observed_lookup,
        args.output_dir,
        args.virtual_corridor_inputs,
        args.install_to,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
