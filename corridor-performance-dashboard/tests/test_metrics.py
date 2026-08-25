import math

import pandas as pd

from corridor_measurement.metrics import (
    congestion_episodes,
    congestion_fit_metrics,
    speed_profile_metrics,
    weighted_harmonic_mean,
)


def test_weighted_harmonic_mean_is_distance_over_travel_time():
    value = weighted_harmonic_mean([60.0, 30.0], [1.0, 1.0])
    assert math.isclose(value, 40.0)


def test_speed_profile_metrics():
    result = speed_profile_metrics(
        pd.Series([50.0, 40.0]),
        pd.Series([45.0, 50.0]),
        mape_minimum_observed_speed_mph=5.0,
    )
    assert math.isclose(result["mae_mph"], 7.5)
    assert math.isclose(result["mape_pct"], 17.5)
    assert result["matched_interval_count"] == 2


def test_congestion_episodes_split_on_time_gap():
    frame = pd.DataFrame(
        {
            "t_min": [360, 375, 405],
            "flag": [True, True, True],
        }
    )
    episodes = congestion_episodes(frame, "flag", interval_minutes=15)
    assert episodes == [
        {"start_min": 360, "end_min": 390, "duration_min": 30},
        {"start_min": 405, "end_min": 420, "duration_min": 15},
    ]


def test_congestion_fit_duration_and_overlap():
    frame = pd.DataFrame(
        {
            "t_min": [360, 375, 390, 405],
            "observed_congested": [False, True, True, False],
            "model_congested": [False, False, True, True],
        }
    )
    result = congestion_fit_metrics(frame, interval_minutes=15)
    assert result["observed_congestion_duration_min"] == 30
    assert result["model_congestion_duration_min"] == 30
    assert result["congestion_overlap_min"] == 15
    assert math.isclose(result["congestion_iou_pct"], 100.0 / 3.0)
