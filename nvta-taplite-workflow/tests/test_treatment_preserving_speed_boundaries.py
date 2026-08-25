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
from generate_treatment_preserving_speed_boundaries import (
    BOUNDARY_FIELDS,
    build_treatment_preserving_lookup,
)


class TreatmentPreservingSpeedBoundaryTests(unittest.TestCase):
    def test_retains_actual_and_replaces_assignment_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignment = root / "assignment"
            period_speeds = {
                "am": (51.0, 61.0),
                "md": (53.0, 63.0),
                "pm": (55.0, 65.0),
            }
            for period, speeds in period_speeds.items():
                folder = assignment / period
                folder.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "iteration_no": 10,
                            "link_id": 1,
                            "from_node_id": 1,
                            "to_node_id": 2,
                            "speed_mph": speeds[0],
                        },
                        {
                            "iteration_no": 10,
                            "link_id": 2,
                            "from_node_id": 2,
                            "to_node_id": 3,
                            "speed_mph": speeds[1],
                        },
                    ]
                ).to_csv(folder / "link_performance.csv", index=False)

            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            baseline = np.empty(2, dtype=BOUNDARY_DTYPE)
            baseline["packed_key"] = [
                (np.uint64(1) << np.uint64(32)) | np.uint64(2),
                (np.uint64(2) << np.uint64(32)) | np.uint64(3),
            ]
            baseline["from_node_id"] = [1, 2]
            baseline["to_node_id"] = [2, 3]
            for index, field in enumerate(BOUNDARY_FIELDS):
                baseline[field] = [20.0 + index, 30.0 + index]
            np.save(
                baseline_dir / "observed_link_speed_boundaries.npy",
                baseline,
                allow_pickle=False,
            )
            audit = root / "baseline-audit.csv"
            pd.DataFrame(
                {
                    "packed_key": baseline["packed_key"],
                    "anchor_source": ["actual", "assignment"],
                }
            ).to_csv(audit, index=False)

            output = root / "output"
            result = build_treatment_preserving_lookup(
                assignment,
                baseline_dir,
                audit,
                output,
            )
            lookup = np.load(
                output / "observed_link_speed_boundaries.npy", allow_pickle=False
            )
            self.assertEqual(float(lookup[0]["qvdf_start_speed_mph_am"]), 20.0)
            self.assertEqual(float(lookup[0]["qvdf_end_speed_mph_pm"]), 25.0)
            self.assertEqual(float(lookup[1]["qvdf_start_speed_mph_am"]), 61.0)
            self.assertEqual(float(lookup[1]["qvdf_end_speed_mph_am"]), 62.0)
            self.assertEqual(float(lookup[1]["qvdf_end_speed_mph_pm"]), 65.0)
            self.assertEqual(
                result["stage2_anchor_source_counts"],
                {"actual": 1, "stage1_assignment_speed_mph": 1},
            )


if __name__ == "__main__":
    unittest.main()
