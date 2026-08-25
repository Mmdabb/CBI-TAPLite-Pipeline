"""Overlay period QVDF parameters onto newly converted TAPlite link files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = {
    "plf": "vdf_plf",
    "qdf": "vdf_qdf",
    "n": "vdf_n",
    "s": "vdf_s",
    "cp": "vdf_cp",
    "cd": "vdf_cd",
    "alpha": "vdf_alpha",
    "beta": "vdf_beta",
}
PERIOD_SEQUENCE = {"am": 1, "md": 2, "pm": 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_dictionary(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError("QVDF override resource must contain one Python dictionary")
    rows = []
    for key, value in payload.items():
        if not isinstance(key, tuple) or len(key) != 2 or not isinstance(value, dict):
            raise ValueError("Invalid QVDF node-pair dictionary entry")
        row = dict(value)
        row["from_node_id"] = int(key[0])
        row["to_node_id"] = int(key[1])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError("QVDF override dictionary contains duplicate node-pair keys")
    return frame


def apply_overrides(
    assignment_root: Path,
    dictionary_path: Path,
    backup_root: Path,
    *,
    expected_profile_mode: int = 1,
) -> dict[str, object]:
    assignment_root = assignment_root.resolve()
    dictionary_path = dictionary_path.resolve()
    backup_root = backup_root.resolve()
    if not dictionary_path.is_file():
        raise FileNotFoundError(dictionary_path)
    if backup_root.exists():
        raise FileExistsError(f"QVDF pre-overlay backup already exists: {backup_root}")
    lookup = _load_dictionary(dictionary_path)
    results: list[dict[str, object]] = []
    for period, sequence in PERIOD_SEQUENCE.items():
        link_path = assignment_root / period / "link.csv"
        if not link_path.is_file():
            raise FileNotFoundError(link_path)
        frame = pd.read_csv(link_path, low_memory=False)
        required = {
            "from_node_id",
            "to_node_id",
            "vdf_type",
            "qvdf_profile_mode",
            *PARAMETER_COLUMNS.values(),
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{link_path} is missing: {', '.join(sorted(missing))}")
        source_columns = {
            f"QVDF_{parameter}{sequence}": target
            for parameter, target in PARAMETER_COLUMNS.items()
        }
        absent = set(source_columns).difference(lookup.columns)
        if absent:
            raise ValueError(
                f"QVDF dictionary is missing {period.upper()} fields: "
                + ", ".join(sorted(absent))
            )
        selected = lookup[["from_node_id", "to_node_id", *source_columns]].rename(
            columns=source_columns
        )
        merged = frame.merge(
            selected,
            on=["from_node_id", "to_node_id"],
            how="left",
            validate="many_to_one",
            suffixes=("", "__override"),
            indicator=True,
        )
        unmatched = int(merged["_merge"].ne("both").sum())
        if unmatched:
            raise ValueError(
                f"{period.upper()} converted link.csv has {unmatched:,} node pairs "
                "missing from the requested QVDF dictionary"
            )
        vdf_type = pd.to_numeric(merged["vdf_type"], errors="coerce")
        if not vdf_type.eq(2).all():
            raise ValueError(
                f"{period.upper()} link.csv must have vdf_type=2 before overlay"
            )
        mode = pd.to_numeric(merged["qvdf_profile_mode"], errors="coerce")
        if not mode.eq(expected_profile_mode).all():
            raise ValueError(
                f"{period.upper()} link.csv must have "
                f"qvdf_profile_mode={expected_profile_mode} before overlay"
            )
        changed_cells = 0
        for target in PARAMETER_COLUMNS.values():
            override = pd.to_numeric(merged[f"{target}__override"], errors="coerce")
            if override.isna().any() or not np.isfinite(override).all():
                raise ValueError(f"{period.upper()} override {target} has missing/nonfinite values")
            original = pd.to_numeric(merged[target], errors="coerce")
            changed_cells += int(
                (~np.isclose(original, override, equal_nan=True)).sum()
            )
            merged[target] = override
        merged = merged[frame.columns]
        period_backup = backup_root / period
        period_backup.mkdir(parents=True, exist_ok=False)
        backup_path = period_backup / "link.csv"
        shutil.copy2(link_path, backup_path)
        before_hash = _sha256(link_path)
        temp_path = link_path.with_suffix(".csv.qvdf-overlay.tmp")
        merged.to_csv(temp_path, index=False)
        temp_path.replace(link_path)
        after_hash = _sha256(link_path)
        results.append(
            {
                "period": period.upper(),
                "links": int(len(merged)),
                "matched_links": int(len(merged)),
                "changed_parameter_cells": changed_cells,
                "link_csv": str(link_path),
                "pre_overlay_backup": str(backup_path),
                "sha256_before": before_hash,
                "sha256_after": after_hash,
            }
        )
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_root": str(assignment_root),
        "dictionary_path": str(dictionary_path),
        "dictionary_sha256": _sha256(dictionary_path),
        "dictionary_node_pairs": int(len(lookup)),
        "parameter_mapping": PARAMETER_COLUMNS,
        "period_sequence": {key.upper(): value for key, value in PERIOD_SEQUENCE.items()},
        "expected_qvdf_profile_mode": expected_profile_mode,
        "periods": results,
    }
    manifest_path = backup_root.parent / "qvdf_override_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def validate_overrides(
    assignment_root: Path,
    dictionary_path: Path,
    *,
    expected_profile_mode: int = 1,
) -> dict[str, object]:
    assignment_root = assignment_root.resolve()
    dictionary_path = dictionary_path.resolve()
    lookup = _load_dictionary(dictionary_path)
    results: list[dict[str, object]] = []
    for period, sequence in PERIOD_SEQUENCE.items():
        link_path = assignment_root / period / "link.csv"
        frame = pd.read_csv(link_path, low_memory=False)
        source_columns = {
            f"QVDF_{parameter}{sequence}": target
            for parameter, target in PARAMETER_COLUMNS.items()
        }
        selected = lookup[["from_node_id", "to_node_id", *source_columns]].rename(
            columns={source: f"expected_{target}" for source, target in source_columns.items()}
        )
        merged = frame.merge(
            selected,
            on=["from_node_id", "to_node_id"],
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{period.upper()} has links absent from QVDF dictionary")
        maximum_difference: dict[str, float] = {}
        for target in PARAMETER_COLUMNS.values():
            actual = pd.to_numeric(merged[target], errors="coerce")
            expected = pd.to_numeric(merged[f"expected_{target}"], errors="coerce")
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError(f"{period.upper()} {target} has missing/nonfinite values")
            difference = (actual - expected).abs()
            if not np.isclose(actual, expected, atol=1e-9, rtol=1e-9).all():
                raise ValueError(f"{period.upper()} {target} differs from QVDF dictionary")
            maximum_difference[target] = float(difference.max())
        if not pd.to_numeric(merged["vdf_type"], errors="coerce").eq(2).all():
            raise ValueError(f"{period.upper()} vdf_type changed after overlay")
        if not pd.to_numeric(
            merged["qvdf_profile_mode"], errors="coerce"
        ).eq(expected_profile_mode).all():
            raise ValueError(f"{period.upper()} qvdf_profile_mode changed after overlay")
        results.append(
            {
                "period": period.upper(),
                "links": int(len(merged)),
                "maximum_absolute_difference": maximum_difference,
            }
        )
    return {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_root": str(assignment_root),
        "dictionary_path": str(dictionary_path),
        "dictionary_sha256": _sha256(dictionary_path),
        "expected_qvdf_profile_mode": expected_profile_mode,
        "periods": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment_root", type=Path)
    parser.add_argument("dictionary_path", type=Path)
    parser.add_argument("backup_root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--qvdf-profile-mode", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    result = (
        validate_overrides(
            args.assignment_root,
            args.dictionary_path,
            expected_profile_mode=args.qvdf_profile_mode,
        )
        if args.verify_only
        else apply_overrides(
            args.assignment_root,
            args.dictionary_path,
            args.backup_root,
            expected_profile_mode=args.qvdf_profile_mode,
        )
    )
    if args.report is not None:
        args.report.resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
