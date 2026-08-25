"""Validate converted QVDF mode and assignment-derived speed anchors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def validate(
    assignment_root: Path,
    boundary_lookup: Path,
    output_path: Path,
    *,
    expected_profile_mode: int = 1,
    expected_vdf_type: int = 2,
) -> dict[str, object]:
    records = np.load(boundary_lookup.resolve(), allow_pickle=False)
    lookup = pd.DataFrame.from_records(records)
    results: list[dict[str, object]] = []
    for period in ("am", "md", "pm"):
        link_path = assignment_root.resolve() / period / "link.csv"
        frame = pd.read_csv(
            link_path,
            usecols=[
                "from_node_id",
                "to_node_id",
                "allowed_use",
                "vdf_type",
                "qvdf_profile_mode",
                "qvdf_start_speed_mph",
                "qvdf_end_speed_mph",
            ],
            low_memory=False,
        )
        expected = lookup[
            [
                "from_node_id",
                "to_node_id",
                f"qvdf_start_speed_mph_{period}",
                f"qvdf_end_speed_mph_{period}",
            ]
        ].rename(
            columns={
                f"qvdf_start_speed_mph_{period}": "expected_start",
                f"qvdf_end_speed_mph_{period}": "expected_end",
            }
        )
        merged = frame.merge(
            expected,
            on=["from_node_id", "to_node_id"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{period.upper()} has links absent from the boundary lookup")
        actual_start = pd.to_numeric(merged["qvdf_start_speed_mph"], errors="coerce")
        actual_end = pd.to_numeric(merged["qvdf_end_speed_mph"], errors="coerce")
        expected_start = pd.to_numeric(merged["expected_start"], errors="coerce")
        expected_end = pd.to_numeric(merged["expected_end"], errors="coerce")
        if not np.isfinite(expected_start).all() or not np.isfinite(expected_end).all():
            raise ValueError(f"{period.upper()} active links have incomplete expected anchors")
        start_matches = np.isclose(actual_start, expected_start, atol=1e-4, rtol=0)
        end_matches = np.isclose(actual_end, expected_end, atol=1e-4, rtol=0)
        if not start_matches.all() or not end_matches.all():
            bad = merged.loc[
                ~(start_matches & end_matches),
                [
                    "from_node_id",
                    "to_node_id",
                    "allowed_use",
                    "qvdf_start_speed_mph",
                    "expected_start",
                    "qvdf_end_speed_mph",
                    "expected_end",
                ],
            ]
            raise ValueError(
                f"{period.upper()} anchors differ from the lookup on {len(bad):,} "
                "links; sample=" + bad.head(5).to_dict(orient="records").__repr__()
            )
        vdf_type = pd.to_numeric(merged["vdf_type"], errors="coerce")
        profile_mode = pd.to_numeric(merged["qvdf_profile_mode"], errors="coerce")
        if not vdf_type.eq(expected_vdf_type).all():
            raise ValueError(f"{period.upper()} does not have vdf_type={expected_vdf_type}")
        if not profile_mode.eq(expected_profile_mode).all():
            raise ValueError(
                f"{period.upper()} does not have qvdf_profile_mode={expected_profile_mode}"
            )
        results.append(
            {
                "period": period.upper(),
                "links": int(len(merged)),
                "vdf_type": expected_vdf_type,
                "qvdf_profile_mode": expected_profile_mode,
                "complete_start_anchors": int(actual_start.notna().sum()),
                "complete_end_anchors": int(actual_end.notna().sum()),
                "max_start_difference_mph": float((actual_start - expected_start).abs().max()),
                "max_end_difference_mph": float((actual_end - expected_end).abs().max()),
            }
        )
    report = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_root": str(assignment_root.resolve()),
        "boundary_lookup": str(boundary_lookup.resolve()),
        "periods": results,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment_root", type=Path)
    parser.add_argument("boundary_lookup", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--expected-profile-mode", type=int, default=1)
    parser.add_argument("--expected-vdf-type", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.assignment_root,
                args.boundary_lookup,
                args.output_path,
                expected_profile_mode=args.expected_profile_mode,
                expected_vdf_type=args.expected_vdf_type,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
