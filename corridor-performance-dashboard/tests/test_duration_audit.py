from __future__ import annotations

import pandas as pd
import pytest

from corridor_measurement.duration_audit import (
    build_tmc_period_audit,
    threshold_duration_hours,
)


def test_threshold_duration_counts_only_valid_intervals() -> None:
    frame = pd.DataFrame({"speed": [40.0, 55.0, None], "threshold": [50.0, 50.0, 50.0]})
    assert threshold_duration_hours(
        frame, speed_column="speed", threshold_column="threshold", interval_minutes=15
    ) == 0.25


def test_build_tmc_period_audit_joins_same_tmc_and_time() -> None:
    measurement = pd.DataFrame(
        {
            "corridor": ["I66_EB"] * 2,
            "tmc_code": ["110+00001"] * 2,
            "period": ["AM"] * 2,
            "direction": ["EASTBOUND"] * 2,
            "road_order": [1.0, 1.0],
            "t_min": [360, 375],
            "observed_tmc_speed_mph": [40.0, 60.0],
            "model_tmc_speed_mph": [45.0, 55.0],
            "cbi_tmc_congestion_threshold_mph": [50.0, 50.0],
            "taplite_period_p_hours": [0.4, 0.4],
            "taplite_period_doc": [0.8, 0.8],
            "taplite_period_volume": [1000.0, 1000.0],
            "gmns_link_count": [1, 1],
        }
    )
    cbi = pd.DataFrame(
        {
            "corridor": ["I66_EB"] * 2,
            "tmc_code": ["110+00001"] * 2,
            "t_min": [360, 375],
            "speed_qvdf_model": [48.0, 49.0],
            "congestion_threshold_mph": [50.0, 50.0],
        }
    )
    episodes = pd.DataFrame(
        {
            "corridor": ["I66_EB"],
            "tmc_code": ["110+00001"],
            "period": ["AM"],
            "reconstruction_selected": [True],
            "P_hr": [0.5],
            "demand_capacity_ratio": [0.7],
            "episode_demand": [2100.0],
            "capacity_reference_hours": [3.0],
            "capacity_volume_veh_per_lane": [6000.0],
        }
    )
    result = build_tmc_period_audit(
        measurement,
        cbi,
        episodes,
        periods={"am": {"start_min": 360, "end_min": 540}},
    )
    assert len(result) == 1
    assert result.loc[0, "cbi_qvdf_same_threshold_p_hours"] == 0.5
    assert result.loc[0, "taplite_same_threshold_p_hours"] == 0.25
    assert result.loc[0, "cbi_selected_episode_p_hours"] == 0.5
    assert result.loc[0, "cbi_doc_minus_taplite_doc"] == pytest.approx(-0.1)
