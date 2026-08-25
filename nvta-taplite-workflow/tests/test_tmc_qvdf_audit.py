from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_tmc_qvdf_inputs import audit_assignment
from generate_assignment_speed_boundaries import BOUNDARY_DTYPE


T2_DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        *[
            (f"observed_{boundary}_hour_{period}", "<f4")
            for period in ("am", "md", "pm")
            for boundary in ("t0", "t2", "t3")
        ],
    ]
)


def _packed(from_node: int, to_node: int) -> int:
    return (from_node << 32) | to_node


def test_completion_boundaries_are_allowed_outside_direct_tmc_mapping(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    speed_dir = resources / "observed_link_speed_boundary_lookup"
    t2_dir = resources / "observed_link_t2_lookup"
    speed_dir.mkdir(parents=True)
    t2_dir.mkdir(parents=True)

    speed = np.empty(2, dtype=BOUNDARY_DTYPE)
    for index, (from_node, to_node) in enumerate(((10, 20), (30, 40))):
        speed[index]["packed_key"] = _packed(from_node, to_node)
        speed[index]["from_node_id"] = from_node
        speed[index]["to_node_id"] = to_node
        for field in BOUNDARY_DTYPE.names or ():
            if "speed_mph" in field:
                speed[index][field] = 45.0
    np.save(speed_dir / "observed_link_speed_boundaries.npy", speed, allow_pickle=False)

    direct = np.empty(1, dtype=T2_DTYPE)
    direct[0]["packed_key"] = _packed(10, 20)
    direct[0]["from_node_id"] = 10
    direct[0]["to_node_id"] = 20
    for field in T2_DTYPE.names or ():
        if field.startswith("observed_"):
            direct[0][field] = np.nan
    np.save(t2_dir / "observed_link_t2.npy", direct, allow_pickle=False)

    assignment = tmp_path / "assignment"
    for period in ("am", "md", "pm"):
        period_dir = assignment / period
        period_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "link_id": 1,
                    "from_node_id": 10,
                    "to_node_id": 20,
                    "allowed_use": "all",
                    "t0_hour": np.nan,
                    "t2_hour": np.nan,
                    "t3_hour": np.nan,
                    "qvdf_start_speed_mph": 45.0,
                    "qvdf_end_speed_mph": 45.0,
                },
                {
                    "link_id": 2,
                    "from_node_id": 30,
                    "to_node_id": 40,
                    "allowed_use": "all",
                    "t0_hour": {"am": 6.2, "md": 9.2, "pm": 15.2}[period],
                    "t2_hour": {"am": 7.0, "md": 12.0, "pm": 17.0}[period],
                    "t3_hour": {"am": 8.0, "md": 14.0, "pm": 18.5}[period],
                    "qvdf_start_speed_mph": 45.0,
                    "qvdf_end_speed_mph": 45.0,
                },
            ]
        ).to_csv(period_dir / "link.csv", index=False)

    audit, summary = audit_assignment(assignment, resources)

    assert summary["status"] == "PASS"
    assert summary["total_failures"] == 0
    assert not audit["unexpected_boundaries_without_congestion"].any()
    assert summary["periods"]["AM"]["direct_observation_node_pairs"] == 1
    assert summary["periods"]["AM"]["speed_anchor_node_pairs"] == 2
