"""Build QVDF period-edge speed anchors from an existing TAPlite assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BOUNDARY_DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("qvdf_start_speed_mph_am", "<f4"),
        ("qvdf_end_speed_mph_am", "<f4"),
        ("qvdf_start_speed_mph_md", "<f4"),
        ("qvdf_end_speed_mph_md", "<f4"),
        ("qvdf_start_speed_mph_pm", "<f4"),
        ("qvdf_end_speed_mph_pm", "<f4"),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_period(assignment_root: Path, period: str) -> tuple[pd.DataFrame, Path]:
    source = assignment_root / period / "link_performance.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing assignment link performance: {source}")
    header = pd.read_csv(source, nrows=0)
    required = {"link_id", "from_node_id", "to_node_id", "speed_mph"}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(sorted(missing))}")
    columns = ["link_id", "from_node_id", "to_node_id", "speed_mph"]
    if "iteration_no" in header.columns:
        columns.insert(0, "iteration_no")
    frame = pd.read_csv(source, usecols=columns, low_memory=False)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "iteration_no" in frame.columns:
        frame = frame.loc[frame["iteration_no"].eq(frame["iteration_no"].max())]
    frame = frame.dropna(subset=["from_node_id", "to_node_id"]).copy()
    frame[["from_node_id", "to_node_id"]] = frame[
        ["from_node_id", "to_node_id"]
    ].astype("uint32")
    frame.loc[~np.isfinite(frame["speed_mph"]) | frame["speed_mph"].le(0), "speed_mph"] = np.nan
    duplicates = frame.duplicated(["from_node_id", "to_node_id"], keep=False)
    if duplicates.any():
        differing = (
            frame.loc[duplicates]
            .groupby(["from_node_id", "to_node_id"])["speed_mph"]
            .nunique(dropna=False)
            .gt(1)
        )
        if differing.any():
            raise ValueError(
                f"{source} has directed node pairs with conflicting period speeds"
            )
        frame = frame.drop_duplicates(["from_node_id", "to_node_id"], keep="last")
    return (
        frame[["link_id", "from_node_id", "to_node_id", "speed_mph"]].rename(
            columns={"link_id": f"link_id_{period}", "speed_mph": f"speed_mph_{period}"}
        ),
        source,
    )


def _available_mean(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([left, right], axis=1).mean(axis=1, skipna=True)


def build_lookup(assignment_root: Path, output_directory: Path) -> dict[str, object]:
    assignment_root = assignment_root.resolve()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    sources: dict[str, Path] = {}
    combined: pd.DataFrame | None = None
    for period in ("am", "md", "pm"):
        frame, source = _load_period(assignment_root, period)
        sources[period] = source
        combined = frame if combined is None else combined.merge(
            frame, on=["from_node_id", "to_node_id"], how="outer", validate="one_to_one"
        )
    assert combined is not None
    # A small set of period-specific/managed links can be absent from a
    # terminal assignment period. Use the closest available modeled period
    # speed only when that terminal value does not exist.
    combined["qvdf_start_speed_mph_am"] = (
        combined[["speed_mph_am", "speed_mph_md", "speed_mph_pm"]]
        .bfill(axis=1)
        .iloc[:, 0]
    )
    combined["qvdf_end_speed_mph_am"] = _available_mean(
        combined["speed_mph_am"], combined["speed_mph_md"]
    )
    combined["qvdf_start_speed_mph_md"] = combined["qvdf_end_speed_mph_am"]
    combined["qvdf_end_speed_mph_md"] = _available_mean(
        combined["speed_mph_md"], combined["speed_mph_pm"]
    )
    combined["qvdf_start_speed_mph_pm"] = combined["qvdf_end_speed_mph_md"]
    combined["qvdf_end_speed_mph_pm"] = (
        combined[["speed_mph_pm", "speed_mph_md", "speed_mph_am"]]
        .bfill(axis=1)
        .iloc[:, 0]
    )
    from_nodes = combined["from_node_id"].to_numpy(dtype=np.uint64)
    to_nodes = combined["to_node_id"].to_numpy(dtype=np.uint64)
    combined["packed_key"] = (from_nodes << np.uint64(32)) | to_nodes
    combined = combined.sort_values("packed_key", kind="stable").reset_index(drop=True)
    records = np.empty(len(combined), dtype=BOUNDARY_DTYPE)
    for field in BOUNDARY_DTYPE.names or ():
        records[field] = combined[field].to_numpy(dtype=BOUNDARY_DTYPE[field])
    lookup_path = output_directory / "observed_link_speed_boundaries.npy"
    np.save(lookup_path, records, allow_pickle=False)
    audit_path = output_directory / "assignment_speed_boundary_audit.csv"
    combined.to_csv(audit_path, index=False)
    boundary_fields = [name for name in BOUNDARY_DTYPE.names or () if "speed_mph" in name]
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_root": str(assignment_root),
        "logic": {
            "AM_start": "AM speed_mph",
            "AM_end_and_MD_start": "mean of available AM and MD speed_mph",
            "MD_end_and_PM_start": "mean of available MD and PM speed_mph",
            "PM_end": "PM speed_mph",
            "missing_adjacent_period_rule": "mean of the available adjacent value; blank only when both are missing",
            "missing_terminal_period_rule": "closest available modeled period speed",
        },
        "node_pairs": int(len(combined)),
        "complete_six_boundaries": int(combined[boundary_fields].notna().all(axis=1).sum()),
        "sources": {
            period.upper(): {"path": str(path), "sha256": _sha256(path)}
            for period, path in sources.items()
        },
        "lookup": {"path": str(lookup_path), "sha256": _sha256(lookup_path)},
        "audit_csv": str(audit_path),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_lookup(args.assignment_root, args.output_directory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
