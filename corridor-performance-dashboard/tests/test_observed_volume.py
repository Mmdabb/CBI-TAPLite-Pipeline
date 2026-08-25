from __future__ import annotations

import numpy as np
import pandas as pd

from corridor_measurement.observed_volume import (
    derive_period_link_profiles,
    enrich_link_performance,
    inverse_s3_flow_from_speed,
    period_summary,
)


def test_inverse_s3_flow_is_bounded_and_nonnegative() -> None:
    speed = np.array([60.0, 42.0, 20.0])
    free_flow = np.full(3, 60.0)
    capacity = np.full(3, 2000.0)
    critical_density = np.full(3, 2000.0 / 42.0)
    shape = np.full(3, 2.0 * np.log(2.0) / np.log(60.0 / 42.0))

    flow = inverse_s3_flow_from_speed(
        speed,
        free_flow_speed_mph=free_flow,
        critical_density_vpmpl=critical_density,
        shape_m=shape,
        capacity_vphpl=capacity,
    )

    assert np.isfinite(flow).all()
    assert (flow >= 0).all()
    assert (flow <= 2000.0).all()
    assert np.isclose(flow[1], 2000.0)


def test_complete_period_profiles_are_appended_and_reconciled() -> None:
    shape = 2.0 * np.log(2.0) / np.log(60.0 / 42.0)
    profiles = pd.DataFrame(
        {
            "corridor": ["C", "C"],
            "tmc_code": ["T1", "T1"],
            "t_min": [360, 375],
            "avg_weekday_speed_mph": [55.0, 42.0],
            "free_flow_speed_model_mph": [60.0, 60.0],
            "critical_density_veh_per_mile_lane": [2000.0 / 42.0] * 2,
            "s3_shape_m": [shape, shape],
            "capacity_vphpl": [2000.0, 2000.0],
        }
    )
    profiles["observed_derived_flow_vphpl"] = inverse_s3_flow_from_speed(
        profiles["avg_weekday_speed_mph"],
        free_flow_speed_mph=profiles["free_flow_speed_model_mph"],
        critical_density_vpmpl=profiles[
            "critical_density_veh_per_mile_lane"
        ],
        shape_m=profiles["s3_shape_m"],
        capacity_vphpl=profiles["capacity_vphpl"],
    )
    mapping = pd.DataFrame(
        {
            "corridor": ["C"],
            "period": ["AM"],
            "tmc_code": ["T1"],
            "link_id": ["10"],
            "eligible_for_comparison": [True],
        }
    )
    performance = pd.DataFrame(
        {
            "iteration_no": [10, 10],
            "link_id": ["10", "11"],
            "volume": [500.0, 300.0],
            "link_capacity": [4000.0, 2000.0],
            "lane_capacity": [2000.0, 2000.0],
        }
    )

    link_profiles = derive_period_link_profiles(
        profiles,
        mapping,
        performance,
        period="AM",
        start_min=360,
        end_min=390,
        interval_minutes=15,
    )
    enriched = enrich_link_performance(
        performance,
        link_profiles,
        period="AM",
        start_min=360,
        end_min=390,
        interval_minutes=15,
    )
    summary = period_summary(
        performance,
        link_profiles,
        enriched,
        period="AM",
        start_min=360,
        end_min=390,
        interval_minutes=15,
    )

    assert len(link_profiles) == 2
    assert set(link_profiles["gmns_lanes"]) == {2.0}
    assert "obs_derived_volume_06:00" in enriched
    assert "obs_derived_volume_06:15" in enriched
    link = enriched.set_index("link_id").loc["10"]
    assert np.isclose(
        link["observed_derived_period_volume"],
        link["obs_derived_volume_06:00"]
        + link["obs_derived_volume_06:15"],
    )
    assert link["observed_derived_profile_coverage_pct"] == 100.0
    assert pd.isna(
        enriched.set_index("link_id").loc["11", "observed_derived_period_volume"]
    )
    assert summary["reconciliation_status"] == "PASS"
