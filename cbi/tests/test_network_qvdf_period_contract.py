from __future__ import annotations

import numpy as np
import pandas as pd

from cbi.config import PipelineSettings
from cbi.episodes import period_clipped_qdf


def test_cross_period_qdf_uses_only_overlap_with_t2_nvta_period() -> None:
    timestamps = pd.date_range("2025-10-01 06:00", "2025-10-01 09:45", freq="15min")
    observations = pd.DataFrame(
        {"datetime": timestamps, "flow_vph": [100.0] * len(timestamps)}
    )
    result = period_clipped_qdf(
        observations,
        period="AM",
        t0_timestamp="2025-10-01 07:00",
        t2_timestamp="2025-10-01 08:00",
        t3_timestamp="2025-10-01 10:00",
        settings=PipelineSettings(),
    )
    assert np.isclose(result["qdf_episode_demand"], 200.0)
    assert np.isclose(result["qdf_period_demand"], 300.0)
    assert np.isclose(result["qdf"], 2.0 / 3.0)
    assert np.isclose(result["plf"], 0.5)
    assert result["qdf_period_duration_hours"] == 3.0
    assert result["qdf_integration_rule"] == "episode_period_overlap_over_period_total"
