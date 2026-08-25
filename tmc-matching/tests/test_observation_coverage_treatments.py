import math
import sys
import unittest
from pathlib import Path

import pandas as pd


MATCHER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MATCHER_ROOT / "src"))

from tmc_matching.build_observation_coverage_treatments import combine_profiles


class ObservationCoverageTreatmentTests(unittest.TestCase):
    def test_internal_interpolation_uses_barycentric_weights(self):
        profiles = {
            "A": pd.DataFrame(
                {"speed": [30.0], "historical_average_speed": [32.0], "reference_speed": [60.0]},
                index=pd.Index([360], name="time_minute"),
            ),
            "B": pd.DataFrame(
                {"speed": [50.0], "historical_average_speed": [52.0], "reference_speed": [60.0]},
                index=pd.Index([360], name="time_minute"),
            ),
        }
        result = combine_profiles(
            {
                "treatment": "virtual_internal_interpolation",
                "source_tmc_upstream": "A",
                "source_tmc_downstream": "B",
                "weight_upstream": 0.75,
                "weight_downstream": 0.25,
            },
            profiles,
            (1.0, 100.0),
        )
        self.assertAlmostEqual(float(result.iloc[0]["speed"]), 35.0)

    def test_terminal_extrapolation_damps_toward_reference(self):
        profiles = {
            "A": pd.DataFrame(
                {"speed": [30.0], "historical_average_speed": [30.0], "reference_speed": [60.0]},
                index=pd.Index([360], name="time_minute"),
            )
        }
        result = combine_profiles(
            {
                "treatment": "virtual_terminal_extrapolation_0_1mi",
                "source_tmc_upstream": "A",
                "distance_decay_weight": math.exp(-1.0),
            },
            profiles,
            (1.0, 100.0),
        )
        self.assertGreater(float(result.iloc[0]["speed"]), 30.0)
        self.assertLess(float(result.iloc[0]["speed"]), 60.0)


if __name__ == "__main__":
    unittest.main()
