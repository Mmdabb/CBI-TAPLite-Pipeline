from __future__ import annotations

"""Identify observed corridor slices that have no frozen network mapping."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corridor-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(args.mapping, dtype={"tmc": "string"}, usecols=["tmc"])
    mapped = set(mapping["tmc"].astype(str).str.strip().str.upper())
    rows: list[dict[str, object]] = []
    for folder in sorted(path for path in args.corridor_root.iterdir() if path.is_dir()):
        metadata_path = folder / "TMC_Identification.csv"
        readings_path = folder / "Readings.csv"
        if not metadata_path.is_file() or not readings_path.is_file():
            continue
        metadata = pd.read_csv(metadata_path, dtype={"tmc": "string"}, low_memory=False)
        tmcs = metadata["tmc"].astype(str).str.strip().str.upper()
        mapped_count = int(tmcs.isin(mapped).sum())
        if mapped_count:
            continue
        rows.append(
            {
                "corridor": folder.name,
                "road": str(metadata.iloc[0]["road"]),
                "direction": str(metadata.iloc[0]["direction"]),
                "tmc_count": int(len(metadata)),
                "tmc_codes": ";".join(tmcs),
                "mapped_tmc_count": 0,
                "decision": "excluded_from_link_based_cbi",
                "reason": "observed TMC corridor has no frozen canonical or approved supplemental directed network mapping",
                "required_resolution": "review mapmatching and establish a defensible directed link before CBI",
            }
        )
    exclusions = pd.DataFrame(rows)
    csv_path = output / "actual_corridors_without_network_mapping.csv"
    exclusions.to_csv(csv_path, index=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "corridor_root": str(args.corridor_root.resolve()),
        "mapping": str(args.mapping.resolve()),
        "excluded_corridor_count": int(len(exclusions)),
        "excluded_corridors": exclusions.get("corridor", pd.Series(dtype=str)).tolist(),
        "output": str(csv_path),
    }
    (output / "actual_cbi_mapping_coverage_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
