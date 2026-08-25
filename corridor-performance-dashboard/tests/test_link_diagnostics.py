from __future__ import annotations

import numpy as np
import pandas as pd

from corridor_measurement.link_diagnostics import (
    eligible_episodes,
    integrate_profile_window,
    kernel_reconciliation,
)
from corridor_measurement.link_volume_scatter import (
    build_unique_matched_link_periods,
    extract_problematic_tmc_link_matches,
    origin_constrained_fit,
)


def test_kernel_reconciliation_matches_link_queue_vdf_equations() -> None:
    frame = pd.DataFrame(
        {
            "volume": [3600.0],
            "D": [400.0],
            "doc": [0.2],
            "vdf_plf": [1.0],
            "lane_capacity": [2000.0],
            "link_capacity": [6000.0],
        }
    )
    result = kernel_reconciliation(frame, period_duration_hours=3.0)
    assert np.isclose(result.loc[0, "kernel_recomputed_D_vphpl"], 400.0)
    assert np.isclose(result.loc[0, "kernel_recomputed_doc"], 0.2)
    assert np.isclose(result.loc[0, "kernel_recomputed_plf"], 1.0)
    assert result.loc[0, "kernel_formula_status"] == "reconciled"


def test_episode_eligibility_uses_t2_and_clips_t0_t3() -> None:
    episodes = pd.DataFrame(
        {
            "period": ["AM", "AM"],
            "t0_hour": [5.0, 8.0],
            "t2_hour": [8.0, 9.0],
            "t3_hour": [10.0, 10.0],
        }
    )
    result = eligible_episodes(
        episodes, {"am": {"start_min": 360, "end_min": 540}}
    )
    assert len(result) == 1
    assert result.iloc[0]["clipped_t0_min"] == 360
    assert result.iloc[0]["clipped_t3_min"] == 540
    assert result.iloc[0]["clipped_duration_hours"] == 3.0


def test_profile_integration_honors_partial_bins() -> None:
    profile = pd.DataFrame(
        {
            "t_min": [360, 375],
            "flow": [1000.0, 2000.0],
        }
    )
    volume = integrate_profile_window(
        profile,
        start_min=367.5,
        end_min=382.5,
        value_column="flow",
        interval_minutes=15,
    )
    assert np.isclose(volume, 375.0)


def test_problematic_export_uses_exact_zero_or_low_doc_definition() -> None:
    frame = pd.DataFrame(
        {
            "corridor": ["A", "A", "A"],
            "period": ["AM", "AM", "AM"],
            "road_order": [1, 2, 3],
            "tmc_code": ["t1", "t2", "t3"],
            "link_id": ["l1", "l2", "l3"],
            "assignment_volume": [0.0, 100.0, 100.0],
            "assignment_doc": [0.5, 0.10, 0.11],
        }
    )
    result = extract_problematic_tmc_link_matches(frame)
    assert result["link_id"].tolist() == ["l1", "l2"]
    assert result["problem_zero_volume_flag"].tolist() == [True, False]
    assert result["problem_doc_le_0_10_flag"].tolist() == [False, True]


def test_scatter_data_deduplicates_physical_link_period_and_keeps_provenance() -> None:
    frame = pd.DataFrame(
        {
            "corridor": ["A", "B", "A"],
            "period": ["AM", "AM", "PM"],
            "tmc_code": ["t1", "t2", "t3"],
            "link_id": ["l1", "l1", "l1"],
            "assignment_volume": [100.0, 100.0, 200.0],
            "assignment_doc": [0.05, 0.05, 0.20],
            "cube_period_volume": [120.0, 120.0, 220.0],
            "synthetic_period_volume": [150.0, 150.0, 250.0],
        }
    )
    result = build_unique_matched_link_periods(frame)
    assert len(result) == 2
    am = result[result["period"].eq("AM")].iloc[0]
    assert am["tmc_match_count"] == 2
    assert am["corridor_match_count"] == 2
    assert am["matched_tmc_codes"] == "t1|t2"
    assert bool(am["problematic_flag"])


def test_origin_constrained_fit_returns_y_equals_two_x() -> None:
    fit = origin_constrained_fit([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])
    assert fit["n"] == 3
    assert np.isclose(fit["slope"], 2.0)
    assert np.isclose(fit["slope_minus_parity_percent"], 100.0)
    assert np.isclose(fit["origin_r_squared"], 1.0)
