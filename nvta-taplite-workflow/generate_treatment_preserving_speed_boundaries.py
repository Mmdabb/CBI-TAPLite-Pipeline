"""Create stage-two anchors while retaining actual and virtual observations.

The stage-one assignment supplies average-speed anchors for links whose
baseline anchor source is an assignment fallback. Baseline anchors classified
as actual or virtual are copied unchanged. The generated resource is isolated;
this script never installs it into the packaged TAPlite resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from generate_assignment_speed_boundaries import BOUNDARY_DTYPE, build_lookup


BOUNDARY_FIELDS = tuple(
    field for field in BOUNDARY_DTYPE.names or () if "speed_mph" in field
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_lookup(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype != BOUNDARY_DTYPE:
        raise ValueError(f"Unexpected speed-boundary dtype in {path}: {values.dtype}")
    if len(values) and np.any(np.diff(values["packed_key"]) <= 0):
        raise ValueError(f"Speed-boundary lookup is not uniquely sorted: {path}")
    return values


def build_treatment_preserving_lookup(
    assignment_root: Path,
    baseline_anchors: Path,
    baseline_audit: Path,
    output_dir: Path,
) -> dict[str, object]:
    assignment_root = assignment_root.resolve()
    baseline_anchors = baseline_anchors.resolve()
    baseline_audit = baseline_audit.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    assignment_dir = output_dir / "assignment-derived"
    assignment_manifest = build_lookup(assignment_root, assignment_dir)
    assignment = _load_lookup(
        assignment_dir / "observed_link_speed_boundaries.npy"
    )
    baseline_path = baseline_anchors / "observed_link_speed_boundaries.npy"
    baseline = _load_lookup(baseline_path)
    if not np.array_equal(assignment["packed_key"], baseline["packed_key"]):
        only_assignment = np.setdiff1d(
            assignment["packed_key"], baseline["packed_key"]
        )
        only_baseline = np.setdiff1d(
            baseline["packed_key"], assignment["packed_key"]
        )
        raise ValueError(
            "Stage-one assignment and baseline anchors have different node-pair "
            f"coverage: assignment-only={len(only_assignment):,}, "
            f"baseline-only={len(only_baseline):,}"
        )

    audit = pd.read_csv(baseline_audit, low_memory=False)
    required = {"packed_key", "anchor_source"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Baseline audit is missing: {', '.join(sorted(missing))}")
    audit["packed_key"] = pd.to_numeric(audit["packed_key"], errors="raise").astype(
        "uint64"
    )
    if audit.duplicated("packed_key").any():
        raise ValueError("Baseline anchor audit contains duplicate packed keys")
    audit = audit.set_index("packed_key").reindex(baseline["packed_key"])
    if audit["anchor_source"].isna().any():
        raise ValueError("Baseline anchor audit does not cover every baseline node pair")

    source = audit["anchor_source"].astype("string")
    retain_observed = (
        source.str.startswith(("actual", "virtual"))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    result = assignment.copy()
    for field in BOUNDARY_FIELDS:
        result[field][retain_observed] = baseline[field][retain_observed]
    if not all(np.isfinite(result[field]).all() for field in BOUNDARY_FIELDS):
        raise ValueError("Generated stage-two anchor lookup has incomplete boundaries")

    lookup_path = output_dir / "observed_link_speed_boundaries.npy"
    np.save(lookup_path, result, allow_pickle=False)
    restored = _load_lookup(lookup_path)
    if not np.array_equal(restored, result):
        raise ValueError("Generated anchor lookup failed round-trip verification")

    output_audit = pd.DataFrame(
        {field: result[field] for field in BOUNDARY_DTYPE.names or ()}
    )
    output_audit["baseline_anchor_source"] = source.to_numpy()
    output_audit["stage2_anchor_source"] = np.where(
        retain_observed,
        source.to_numpy(),
        "stage1_assignment_speed_mph",
    )
    for field in BOUNDARY_FIELDS:
        output_audit[f"baseline_{field}"] = baseline[field]
        output_audit[f"stage1_assignment_{field}"] = assignment[field]
    audit_path = output_dir / "hybrid_speed_boundary_audit.csv"
    output_audit.to_csv(audit_path, index=False)

    reports: list[pd.DataFrame] = []
    for period in ("am", "md", "pm"):
        report = output_audit[
            [
                "packed_key",
                "from_node_id",
                "to_node_id",
                "stage2_anchor_source",
            ]
        ].copy()
        report["period"] = period.upper()
        report["qvdf_start_speed_mph"] = result[
            f"qvdf_start_speed_mph_{period}"
        ]
        report["qvdf_end_speed_mph"] = result[f"qvdf_end_speed_mph_{period}"]
        report["boundary_status"] = "both"
        reports.append(report)
    completeness_path = output_dir / "boundary_completeness_report.csv"
    pd.concat(reports, ignore_index=True).to_csv(completeness_path, index=False)

    source_counts = (
        output_audit["stage2_anchor_source"].value_counts().sort_index().to_dict()
    )
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "retain baseline actual/virtual; replace only assignment fallbacks",
        "speed_unit": "mph",
        "assignment_boundary_logic": assignment_manifest["logic"],
        "node_pairs": int(len(result)),
        "complete_six_boundaries": int(len(result)),
        "stage2_anchor_source_counts": {
            str(label): int(count) for label, count in source_counts.items()
        },
        "sources": {
            "stage1_assignment": str(assignment_root),
            "baseline_lookup": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "baseline_audit": {
                "path": str(baseline_audit),
                "sha256": _sha256(baseline_audit),
            },
        },
        "products": {
            "lookup": {"path": str(lookup_path), "sha256": _sha256(lookup_path)},
            "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "completeness": {
                "path": str(completeness_path),
                "sha256": _sha256(completeness_path),
            },
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--baseline-anchors", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_treatment_preserving_lookup(
        args.assignment_root,
        args.baseline_anchors,
        args.baseline_audit,
        args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
