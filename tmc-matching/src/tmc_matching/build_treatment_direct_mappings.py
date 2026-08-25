from __future__ import annotations

"""Publish direct mapping inputs for isolated actual and virtual CBI runs."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def managed_mapping_rows(
    supplemental: pd.DataFrame,
    canonical_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in supplemental.itertuples(index=False):
        row = {column: np.nan for column in canonical_columns}
        row.update(
            {
                "tmc": item.source_tmc_primary,
                "road": item.source_road,
                "direction": item.direction,
                "road_order": item.source_road_order,
                "link_id": item.link_id,
                "from_node_id": item.from_node_id,
                "to_node_id": item.to_node_id,
                "STREETNAME": item.STREETNAME,
                "length_mi": item.length_mi,
                "lanes": item.lanes,
                "capacity": item.capacity,
                "free_speed": item.free_speed,
                "link_type": item.link_type,
                "sequence": 1,
                "cumulative_mi": item.length_mi,
                "distance_to_tmc_ft": 0.0,
                "bearing_diff_deg": getattr(
                    item, "managed_direction_difference_deg", 0.0
                ),
                "route_link_count": 1,
                "tmc_miles": item.length_mi,
                "route_length_mi": item.length_mi,
                "length_ratio": 1.0,
                "match_confidence": 100.0,
                "match_status": "supplemental_managed_actual",
                "geometry_overlap_pct": 100.0,
                "geometry_overlap_source": "managed_treatment_direct",
                "first_map_occurrence": 10_000_000 + int(item.link_id),
                "link_tmc_rank": 1,
                "link_tmc_ranking_basis": "only_supplemental_actual_candidate",
                "tmc_link_rank": 1,
                "tmc_link_ranking_basis": "supplemental_managed_actual",
                "node_pair_tmc_rank": 1,
                "node_pair_tmc_ranking_basis": "supplemental_managed_actual",
                "selected_for_node_pair_lookup": True,
                "node_pair_has_observed_candidate": True,
                "node_pair_winner_has_observation": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns)


def build(
    coverage_root: Path,
    canonical_path: Path,
) -> dict[str, object]:
    coverage = Path(coverage_root).resolve()
    destination = coverage / "actual/combined-direct-mapping"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    canonical = pd.read_csv(canonical_path, dtype={"tmc": "string"}, low_memory=False)
    selected = canonical["selected_for_node_pair_lookup"].fillna(False).astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    canonical = canonical[selected].copy()
    supplemental_path = (
        coverage
        / "actual/managed-canonical/supplemental_managed_actual.csv"
    )
    supplemental = pd.read_csv(supplemental_path, low_memory=False)
    additions = managed_mapping_rows(supplemental, list(canonical.columns))
    combined = pd.concat([canonical, additions], ignore_index=True, sort=False)
    if combined.duplicated(["from_node_id", "to_node_id"]).any():
        duplicates = combined[
            combined.duplicated(["from_node_id", "to_node_id"], keep=False)
        ][["tmc", "link_id", "from_node_id", "to_node_id"]]
        raise ValueError(
            "Actual treatment mapping conflicts by node pair: "
            + duplicates.head(10).to_json(orient="records")
        )
    output_path = destination / "actual_tmc_to_link.csv"
    combined.to_csv(output_path, index=False)
    additions.to_csv(destination / "supplemental_managed_mapping_rows.csv", index=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "canonical_source": str(Path(canonical_path).resolve()),
        "canonical_source_sha256": sha256(canonical_path),
        "canonical_rows": int(len(canonical)),
        "supplemental_managed_actual_rows": int(len(additions)),
        "combined_actual_rows": int(len(combined)),
        "unique_directed_node_pairs": int(
            combined[["from_node_id", "to_node_id"]].drop_duplicates().shape[0]
        ),
        "canonical_rows_modified": 0,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.coverage_root, args.canonical)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
