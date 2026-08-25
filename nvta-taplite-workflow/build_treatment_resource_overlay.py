from __future__ import annotations

"""Build and optionally install the first-pass actual/virtual resource overlay.

The network-wide QVDF remains actual-only. Direct PLF values are combined from
the actual and virtual CBI products after proving that their directed node-pair
keys are disjoint. Before installation, the complete workflow resource folder
is copied into the output directory as a recoverable backup.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PLF_FILE = "observed_link_plf_overrides.npy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def merge_plf_lookups(actual_path: Path, virtual_path: Path) -> np.ndarray:
    actual = np.load(actual_path, allow_pickle=False)
    virtual = np.load(virtual_path, allow_pickle=False)
    if actual.dtype != virtual.dtype:
        raise ValueError(
            f"Actual and virtual PLF dtypes differ: {actual.dtype} != {virtual.dtype}"
        )
    for label, values in (("actual", actual), ("virtual", virtual)):
        keys = values["packed_key"]
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"{label} PLF lookup contains duplicate node-pair keys")
    overlap = np.intersect1d(actual["packed_key"], virtual["packed_key"])
    if len(overlap):
        raise ValueError(
            f"Actual and virtual PLF scopes overlap on {len(overlap)} node pairs"
        )
    combined = np.concatenate([actual, virtual])
    combined.sort(order="packed_key")
    if len(combined) and np.any(np.diff(combined["packed_key"]) <= 0):
        raise ValueError("Combined PLF lookup is not uniquely sorted")
    return combined


def file_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.installing")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def build_overlay(
    actual_resources: Path,
    virtual_resources: Path,
    workflow_resources: Path,
    output_dir: Path,
    *,
    install: bool = False,
) -> dict[str, object]:
    actual_resources = Path(actual_resources).resolve()
    virtual_resources = Path(virtual_resources).resolve()
    workflow_resources = Path(workflow_resources).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    backup = output_dir / "workflow-resources-backup"
    shutil.copytree(
        workflow_resources,
        backup,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    before = file_inventory(backup)

    staged = output_dir / "first-pass-resources"
    staged.mkdir()
    actual_qvdf = actual_resources / "daily" / "link_qvdf.csv"
    staged_qvdf = staged / "link_qvdf.csv"
    shutil.copy2(actual_qvdf, staged_qvdf)

    actual_plf = actual_resources / "observed-link-plf" / PLF_FILE
    virtual_plf = virtual_resources / "observed-link-plf" / PLF_FILE
    combined = merge_plf_lookups(actual_plf, virtual_plf)
    staged_plf_dir = staged / "observed_link_plf_lookup"
    staged_plf_dir.mkdir()
    staged_plf = staged_plf_dir / PLF_FILE
    np.save(staged_plf, combined, allow_pickle=False)

    source_labels = np.concatenate(
        [
            np.full(len(np.load(actual_plf, allow_pickle=False)), "actual"),
            np.full(len(np.load(virtual_plf, allow_pickle=False)), "virtual"),
        ]
    )
    unsorted = np.concatenate(
        [np.load(actual_plf, allow_pickle=False), np.load(virtual_plf, allow_pickle=False)]
    )
    order = np.argsort(unsorted["packed_key"], kind="stable")
    audit = pd.DataFrame({name: unsorted[name][order] for name in unsorted.dtype.names or ()})
    audit["resource_scope"] = source_labels[order]
    audit_path = staged_plf_dir / "combined_plf_audit.csv"
    audit.to_csv(audit_path, index=False)
    plf_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "NumPy .npy structured array sorted by packed_key",
        "key_definition": "(uint64(from_node_id) << 32) | uint64(to_node_id)",
        "unit": "dimensionless",
        "record_dtype": combined.dtype.descr,
        "precedence": ["actual direct", "virtual direct"],
        "actual_rows": int(len(np.load(actual_plf, allow_pickle=False))),
        "virtual_rows": int(len(np.load(virtual_plf, allow_pickle=False))),
        "combined_rows": int(len(combined)),
        "overlapping_node_pairs": 0,
        "lookup_sha256": sha256(staged_plf),
        "audit_sha256": sha256(audit_path),
        "sources": {
            "actual": {"path": str(actual_plf), "sha256": sha256(actual_plf)},
            "virtual": {"path": str(virtual_plf), "sha256": sha256(virtual_plf)},
        },
    }
    staged_metadata = staged_plf_dir / "metadata.json"
    staged_metadata.write_text(json.dumps(plf_metadata, indent=2) + "\n", encoding="utf-8")

    installed: list[dict[str, object]] = []
    if install:
        targets = [
            (staged_qvdf, workflow_resources / "link_qvdf.csv"),
            (staged_plf, workflow_resources / "observed_link_plf_lookup" / PLF_FILE),
            (staged_metadata, workflow_resources / "observed_link_plf_lookup" / "metadata.json"),
        ]
        for source, target in targets:
            atomic_copy(source, target)
            installed.append(
                {"target": str(target), "sha256": sha256(target), "source": str(source)}
            )

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope_rules": {
            "network_qvdf": "actual CBI daily accepted episodes only",
            "direct_plf": "actual direct plus approved virtual direct, disjoint node pairs",
            "speed_anchors": "unchanged in first pass",
            "congestion_boundaries": "unchanged in first pass",
        },
        "backup": {"directory": str(backup), "files": before},
        "staged": {
            "link_qvdf": {"path": str(staged_qvdf), "sha256": sha256(staged_qvdf)},
            "combined_plf": {"path": str(staged_plf), "sha256": sha256(staged_plf)},
            "combined_plf_rows": int(len(combined)),
        },
        "installed": installed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-resources", type=Path, required=True)
    parser.add_argument("--virtual-resources", type=Path, required=True)
    parser.add_argument("--workflow-resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    manifest = build_overlay(
        args.actual_resources,
        args.virtual_resources,
        args.workflow_resources,
        args.output_dir,
        install=args.install,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
