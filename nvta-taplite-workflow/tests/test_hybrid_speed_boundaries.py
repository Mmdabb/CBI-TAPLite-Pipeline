from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT))

from generate_assignment_speed_boundaries import BOUNDARY_DTYPE
from generate_hybrid_speed_boundaries import BOUNDARY_FIELDS, build_hybrid_lookup


class HybridSpeedBoundaryTests(unittest.TestCase):
    def test_partial_virtual_anchor_uses_virtual_pre_qc_profile_only_for_blank_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.csv"
            pd.DataFrame(
                [{
                    "tmc": "VIRTUAL-A",
                    "link_id": 1,
                    "from_node_id": 1,
                    "to_node_id": 2,
                }]
            ).to_csv(canonical, index=False)
            existing = np.empty(1, dtype=BOUNDARY_DTYPE)
            existing["packed_key"] = (np.uint64(1) << np.uint64(32)) | np.uint64(2)
            existing["from_node_id"], existing["to_node_id"] = 1, 2
            for field in BOUNDARY_FIELDS:
                existing[field] = 35.0
            existing["qvdf_end_speed_mph_pm"] = np.nan
            existing_path = root / "existing.npy"
            np.save(existing_path, existing, allow_pickle=False)
            readings = root / "readings.csv"
            pd.DataFrame(
                columns=["tmc_code", "measurement_tstamp", "speed"]
            ).to_csv(readings, index=False)
            virtual_root = root / "virtual"
            virtual_corridor = virtual_root / "VIRTUAL_TEST"
            virtual_corridor.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "tmc_code": "VIRTUAL-A",
                        "measurement_tstamp": f"2025-10-01 {minute // 60:02d}:{minute % 60:02d}:00",
                        "speed": speed,
                    }
                    for minute, speed in (
                        (360, 30.0),
                        (540, 40.0),
                        (900, 50.0),
                        (1140, 55.0),
                    )
                ]
            ).to_csv(virtual_corridor / "Readings.csv", index=False)
            assignment = root / "assignment"
            for period in ("am", "md", "pm"):
                folder = assignment / period
                folder.mkdir(parents=True)
                pd.DataFrame(
                    [{
                        "iteration_no": 10,
                        "link_id": 1,
                        "from_node_id": 1,
                        "to_node_id": 2,
                        "speed_mph": 70.0,
                    }]
                ).to_csv(folder / "link_performance.csv", index=False)

            output = root / "output"
            metadata = build_hybrid_lookup(
                canonical,
                readings,
                assignment,
                existing_path,
                output,
                virtual_corridor_inputs=virtual_root,
            )

            lookup = np.load(output / "observed_link_speed_boundaries.npy")
            self.assertEqual(float(lookup[0]["qvdf_start_speed_mph_am"]), 35.0)
            self.assertEqual(float(lookup[0]["qvdf_end_speed_mph_pm"]), 55.0)
            self.assertEqual(metadata["counts"]["canonical_virtual_direct_pairs"], 1)

    def test_complete_existing_coverage_does_not_require_regional_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.csv"
            pd.DataFrame(
                [{"tmc": "A", "link_id": 1, "from_node_id": 1, "to_node_id": 2}]
            ).to_csv(canonical, index=False)
            existing = np.empty(1, dtype=BOUNDARY_DTYPE)
            existing["packed_key"] = (np.uint64(1) << np.uint64(32)) | np.uint64(2)
            existing["from_node_id"], existing["to_node_id"] = 1, 2
            for field in BOUNDARY_FIELDS:
                existing[field] = 35.0
            existing_path = root / "existing.npy"
            np.save(existing_path, existing, allow_pickle=False)
            readings = root / "readings.csv"
            pd.DataFrame(
                columns=["tmc_code", "measurement_tstamp", "speed"]
            ).to_csv(readings, index=False)
            assignment = root / "assignment"
            for period in ("am", "md", "pm"):
                folder = assignment / period
                folder.mkdir(parents=True)
                pd.DataFrame(
                    [{
                        "iteration_no": 10,
                        "link_id": 1,
                        "from_node_id": 1,
                        "to_node_id": 2,
                        "speed_mph": 40.0,
                    }]
                ).to_csv(folder / "link_performance.csv", index=False)

            output = root / "output"
            metadata = build_hybrid_lookup(
                canonical, readings, assignment, existing_path, output
            )

            self.assertEqual(
                metadata["counts"]["canonical_existing_cbi_observed_pairs"], 1
            )
            self.assertEqual(
                metadata["counts"]["canonical_regional_direct_pairs"], 0
            )

    def test_observed_regional_assignment_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.csv"
            pd.DataFrame(
                [
                    {"tmc": "A", "link_id": 1, "from_node_id": 1, "to_node_id": 2},
                    {"tmc": "B", "link_id": 2, "from_node_id": 2, "to_node_id": 3},
                ]
            ).to_csv(canonical, index=False)

            existing = np.empty(1, dtype=BOUNDARY_DTYPE)
            existing["packed_key"] = (np.uint64(1) << np.uint64(32)) | np.uint64(2)
            existing["from_node_id"], existing["to_node_id"] = 1, 2
            for index, field in enumerate(BOUNDARY_FIELDS):
                existing[field] = 20.0 + index
            existing_path = root / "existing.npy"
            np.save(existing_path, existing, allow_pickle=False)

            readings = root / "readings.csv"
            observations = []
            for date in ("2025-10-01", "2025-10-02"):
                for minute, speed in ((360, 30), (540, 40), (900, 50), (1140, 60)):
                    observations.append(
                        {
                            "tmc_code": "B",
                            "measurement_tstamp": f"{date} {minute // 60:02d}:{minute % 60:02d}:00",
                            "speed": speed,
                        }
                    )
            pd.DataFrame(observations).to_csv(readings, index=False)

            assignment = root / "assignment"
            for period, speeds in {"am": (70, 71, 72), "md": (60, 61, 62), "pm": (50, 51, 52)}.items():
                folder = assignment / period
                folder.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {"iteration_no": 10, "link_id": 1, "from_node_id": 1, "to_node_id": 2, "speed_mph": speeds[0]},
                        {"iteration_no": 10, "link_id": 2, "from_node_id": 2, "to_node_id": 3, "speed_mph": speeds[1]},
                        {"iteration_no": 10, "link_id": 3, "from_node_id": 3, "to_node_id": 4, "speed_mph": speeds[2]},
                    ]
                ).to_csv(folder / "link_performance.csv", index=False)

            output = root / "output"
            metadata = build_hybrid_lookup(canonical, readings, assignment, existing_path, output)
            lookup = np.load(output / "observed_link_speed_boundaries.npy", allow_pickle=False)
            pairs = {(int(row["from_node_id"]), int(row["to_node_id"])): row for row in lookup}
            self.assertEqual(float(pairs[(1, 2)]["qvdf_start_speed_mph_am"]), 20.0)
            self.assertEqual(float(pairs[(2, 3)]["qvdf_start_speed_mph_am"]), 30.0)
            self.assertEqual(float(pairs[(2, 3)]["qvdf_end_speed_mph_am"]), 40.0)
            self.assertEqual(float(pairs[(3, 4)]["qvdf_start_speed_mph_am"]), 72.0)
            self.assertEqual(float(pairs[(3, 4)]["qvdf_end_speed_mph_am"]), 67.0)
            self.assertEqual(float(pairs[(3, 4)]["qvdf_end_speed_mph_pm"]), 52.0)
            self.assertEqual(metadata["counts"]["canonical_existing_cbi_observed_pairs"], 1)
            self.assertEqual(metadata["counts"]["canonical_regional_direct_pairs"], 1)
            self.assertEqual(metadata["counts"]["assignment_fallback_noncanonical_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
