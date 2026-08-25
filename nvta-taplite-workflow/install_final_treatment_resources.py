from __future__ import annotations

"""Build, audit, back up, and install final treatment-aware TAPlite resources."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from generate_assignment_speed_boundaries import BOUNDARY_DTYPE, build_lookup


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.installing")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _validated_lookup(path: Path, expected_dtype: np.dtype | None = None) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if expected_dtype is not None and values.dtype != expected_dtype:
        raise ValueError(f"Unexpected dtype in {path}: {values.dtype}")
    if len(values) and np.any(np.diff(values["packed_key"]) <= 0):
        raise ValueError(f"Lookup is not uniquely sorted: {path}")
    return values


def _merge_observed_overrides(
    baseline: np.ndarray,
    actual: np.ndarray,
    virtual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not (baseline.dtype == actual.dtype == virtual.dtype):
        raise ValueError("Speed-boundary lookup dtypes differ")
    if len(np.intersect1d(actual["packed_key"], virtual["packed_key"])):
        raise ValueError("Actual and virtual speed-anchor node pairs overlap")
    result = baseline.copy()
    scope = np.full(len(result), "assignment", dtype="U48")
    keys = result["packed_key"]
    for label, overrides in (("actual", actual), ("virtual", virtual)):
        positions = np.searchsorted(keys, overrides["packed_key"])
        found = (positions < len(keys)) & (keys[np.minimum(positions, len(keys) - 1)] == overrides["packed_key"])
        if not found.all():
            raise ValueError(f"{label} speed anchors contain node pairs absent from assignment")
        boundary_fields = [
            field for field in result.dtype.names or () if "speed_mph" in field
        ]
        complete = np.ones(len(overrides), dtype=bool)
        for field in boundary_fields:
            finite = np.isfinite(overrides[field])
            complete &= finite
            result[field][positions[finite]] = overrides[field][finite]
        scope[positions] = np.where(
            complete,
            label,
            f"{label}_partial_assignment_fallback",
        )
    return result, scope


def _merge_disjoint(actual: np.ndarray, virtual: np.ndarray) -> np.ndarray:
    if actual.dtype != virtual.dtype:
        raise ValueError("Actual and virtual observed-T2 dtypes differ")
    if len(np.intersect1d(actual["packed_key"], virtual["packed_key"])):
        raise ValueError("Actual and virtual observed-T2 node pairs overlap")
    result = np.concatenate([actual, virtual])
    result.sort(order="packed_key")
    if len(result) and np.any(np.diff(result["packed_key"]) <= 0):
        raise ValueError("Combined observed-T2 lookup is not uniquely sorted")
    return result


def install(
    assignment_root: Path,
    actual_resources: Path,
    virtual_resources: Path,
    boundary_lookup: Path,
    workflow_resources: Path,
    output_dir: Path,
) -> dict[str, object]:
    assignment_root = Path(assignment_root).resolve()
    actual_resources = Path(actual_resources).resolve()
    virtual_resources = Path(virtual_resources).resolve()
    boundary_lookup = Path(boundary_lookup).resolve()
    workflow_resources = Path(workflow_resources).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    backup = output_dir / "pre-final-workflow-resources-backup"
    shutil.copytree(
        workflow_resources,
        backup,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    assignment_dir = output_dir / "assignment-speed-boundaries"
    build_lookup(assignment_root, assignment_dir)
    baseline_path = assignment_dir / "observed_link_speed_boundaries.npy"
    baseline = _validated_lookup(baseline_path, BOUNDARY_DTYPE)
    actual_speed_path = actual_resources / "observed-link-speed-boundaries/observed_link_speed_boundaries.npy"
    virtual_speed_path = virtual_resources / "observed-link-speed-boundaries/observed_link_speed_boundaries.npy"
    actual_speed = _validated_lookup(actual_speed_path, BOUNDARY_DTYPE)
    virtual_speed = _validated_lookup(virtual_speed_path, BOUNDARY_DTYPE)
    speed, scope = _merge_observed_overrides(baseline, actual_speed, virtual_speed)
    final_speed_dir = output_dir / "observed_link_speed_boundary_lookup"
    final_speed_dir.mkdir()
    final_speed_path = final_speed_dir / "observed_link_speed_boundaries.npy"
    np.save(final_speed_path, speed, allow_pickle=False)
    speed_audit = pd.DataFrame({name: speed[name] for name in speed.dtype.names or ()})
    speed_audit["anchor_source"] = scope
    speed_audit_path = final_speed_dir / "hybrid_speed_boundary_audit.csv"
    speed_audit.to_csv(speed_audit_path, index=False)
    reports = []
    for period in ("am", "md", "pm"):
        report = speed_audit[["packed_key", "from_node_id", "to_node_id", "anchor_source"]].copy()
        report["period"] = period.upper()
        report["qvdf_start_speed_mph"] = speed_audit[f"qvdf_start_speed_mph_{period}"]
        report["qvdf_end_speed_mph"] = speed_audit[f"qvdf_end_speed_mph_{period}"]
        report["boundary_status"] = np.where(
            report[["qvdf_start_speed_mph", "qvdf_end_speed_mph"]].notna().all(axis=1),
            "both",
            "incomplete",
        )
        reports.append(report)
    completeness_path = final_speed_dir / "boundary_completeness_report.csv"
    pd.concat(reports, ignore_index=True).to_csv(completeness_path, index=False)
    speed_metadata = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "speed_unit": "mph",
        "hierarchy": ["actual observed", "virtual observed", "first-pass assignment speed_mph"],
        "assignment_boundary_logic": {
            "AM_start": "AM speed_mph",
            "AM_end_and_MD_start": "mean(AM, MD)",
            "MD_end_and_PM_start": "mean(MD, PM)",
            "PM_end": "PM speed_mph",
        },
        "counts": {
            "network_node_pairs": int(len(speed)),
            "anchor_source": {
                str(label): int(count)
                for label, count in zip(*np.unique(scope, return_counts=True))
            },
            "complete_six_boundaries": int(
                pd.DataFrame({name: speed[name] for name in speed.dtype.names or ()})[
                    [name for name in speed.dtype.names or () if "speed_mph" in name]
                ].notna().all(axis=1).sum()
            ),
        },
        "lookup_sha256": sha256(final_speed_path),
        "audit_sha256": sha256(speed_audit_path),
    }
    speed_metadata_path = final_speed_dir / "metadata.json"
    speed_metadata_path.write_text(json.dumps(speed_metadata, indent=2) + "\n", encoding="utf-8")

    actual_t2_path = actual_resources / "observed-link-t2/observed_link_t2.npy"
    virtual_t2_path = virtual_resources / "observed-link-t2/observed_link_t2.npy"
    actual_t2 = _validated_lookup(actual_t2_path)
    virtual_t2 = _validated_lookup(virtual_t2_path, actual_t2.dtype)
    observed_t2 = _merge_disjoint(actual_t2, virtual_t2)
    final_t2_dir = output_dir / "observed_link_t2_lookup"
    final_t2_dir.mkdir()
    final_t2_path = final_t2_dir / "observed_link_t2.npy"
    np.save(final_t2_path, observed_t2, allow_pickle=False)
    t2_metadata = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hierarchy": ["actual direct", "virtual direct"],
        "no_episode_rule": "NaN triplet protects the direct matched link-period from completion",
        "actual_rows": int(len(actual_t2)),
        "virtual_rows": int(len(virtual_t2)),
        "combined_rows": int(len(observed_t2)),
        "lookup_sha256": sha256(final_t2_path),
    }
    t2_metadata_path = final_t2_dir / "metadata.json"
    t2_metadata_path.write_text(json.dumps(t2_metadata, indent=2) + "\n", encoding="utf-8")

    final_boundary_dir = output_dir / "congestion_t_node_pair_lookup"
    shutil.copytree(boundary_lookup, final_boundary_dir)

    qvdf_manifest = {
        "status": "PASS",
        "scope": "actual CBI daily accepted episodes only",
        "source_manifest": str(actual_resources / "manifest.json"),
        "source_manifest_sha256": sha256(actual_resources / "manifest.json"),
        "installed_link_qvdf_sha256": sha256(workflow_resources / "link_qvdf.csv"),
    }
    qvdf_manifest_path = output_dir / "link_qvdf_source_manifest.json"
    qvdf_manifest_path.write_text(json.dumps(qvdf_manifest, indent=2) + "\n", encoding="utf-8")

    targets = [
        (final_speed_path, workflow_resources / "observed_link_speed_boundary_lookup/observed_link_speed_boundaries.npy"),
        (completeness_path, workflow_resources / "observed_link_speed_boundary_lookup/boundary_completeness_report.csv"),
        (speed_metadata_path, workflow_resources / "observed_link_speed_boundary_lookup/metadata.json"),
        (final_t2_path, workflow_resources / "observed_link_t2_lookup/observed_link_t2.npy"),
        (t2_metadata_path, workflow_resources / "observed_link_t2_lookup/metadata.json"),
        (qvdf_manifest_path, workflow_resources / "link_qvdf_source_manifest.json"),
    ]
    for name in ("am_node_pair_boundaries.npy", "md_node_pair_boundaries.npy", "pm_node_pair_boundaries.npy", "metadata.json"):
        targets.append((final_boundary_dir / name, workflow_resources / "congestion_t_node_pair_lookup" / name))
    installed = []
    for source, target in targets:
        atomic_copy(source, target)
        installed.append({"target": str(target), "sha256": sha256(target)})

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backup": str(backup),
        "speed_anchors": speed_metadata,
        "observed_t2": t2_metadata,
        "congestion_lookup_source": str(boundary_lookup),
        "installed": installed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--actual-resources", type=Path, required=True)
    parser.add_argument("--virtual-resources", type=Path, required=True)
    parser.add_argument("--boundary-lookup", type=Path, required=True)
    parser.add_argument("--workflow-resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
