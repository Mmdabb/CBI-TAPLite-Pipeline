from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from cbi.calibration import fit_qvdf
from cbi.config import (
    PipelineSettings,
    WORKFLOW_PERIODS,
    period_duration_hours,
)
from cbi.episodes import (
    detector_config,
    episode_candidate_audit,
    episode_filter_audit,
)
from cbi.output_contract import STEP_FOLDERS
from cbi.network_mapping import load_canonical_mapping
from cbi.outputs import DAILY_PAQ_COLUMNS, HANDOFF_COLUMNS, STAGE0_COLUMNS
from cbi.figures import _pick_validation_links
from cbi.reconstruction import (
    predict_minimum_speed,
    predicted_bounds_about_t2,
    qvdf_queue_shape,
    reconstruct_episode_speed,
    select_nonoverlapping_episodes,
)
from cbi.state_transition import (
    StateTransitionConfig,
    _episode_from_bounds,
    episode_time_period_tag,
)
from cbi.workers import recommend_workers


def test_cbi_uses_taplite_workflow_period_boundaries():
    settings = PipelineSettings()
    assert WORKFLOW_PERIODS == {
        "AM": (6 * 60, 9 * 60),
        "MD": (9 * 60, 15 * 60),
        "PM": (15 * 60, 19 * 60),
        "NT": (19 * 60, 6 * 60),
    }
    assert settings.period_for_minute(5 * 60 + 59) is None
    assert settings.period_for_minute(6 * 60) == "AM"
    assert settings.period_for_minute(8 * 60 + 59) == "AM"
    assert settings.period_for_minute(9 * 60) == "MD"
    assert settings.period_for_minute(14 * 60 + 59) == "MD"
    assert settings.period_for_minute(15 * 60) == "PM"
    assert settings.period_for_minute(18 * 60 + 59) == "PM"
    assert settings.period_for_minute(19 * 60) is None


def test_demand_capacity_uses_full_period_duration():
    assert period_duration_hours("AM") == 3.0
    assert period_duration_hours("MD") == 6.0
    assert period_duration_hours("PM") == 4.0
    assert period_duration_hours("NT") == 11.0

    timestamps = pd.date_range("2026-01-05 06:00", periods=4, freq="15min")
    entity = pd.DataFrame(
        {
            "entity_id": ["TMC-1"] * 4,
            "timestamp": timestamps,
            "date": ["2026-01-05"] * 4,
            "time_of_day": timestamps.strftime("%H:%M:%S"),
            "speed_mph": [40.0, 35.0, 30.0, 42.0],
            "flow": [100.0] * 4,
            "per_lane_hourly_capacity": [1000.0] * 4,
            "aggregation_mode": ["daily"] * 4,
        }
    )
    episode = _episode_from_bounds(
        entity,
        0,
        3,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )
    assert episode is not None
    assert episode["capacity_reference_hours"] == 3.0
    assert episode["capacity_volume_veh_per_lane"] == 3000.0
    assert np.isclose(episode["episode_demand"], 100.0)
    assert np.isclose(episode["demand_capacity_ratio"], 1.0 / 30.0)


def _boundary_repair_entity(clean_speed, pre_qc_speed):
    timestamps = pd.date_range(
        "2026-01-05 06:00", periods=len(clean_speed), freq="15min"
    )
    return pd.DataFrame(
        {
            "entity_id": ["TMC-boundary"] * len(clean_speed),
            "timestamp": timestamps,
            "date": ["2026-01-05"] * len(clean_speed),
            "time_of_day": timestamps.strftime("%H:%M:%S"),
            "speed_mph": clean_speed,
            "speed_mph_pre_qc": pre_qc_speed,
            "flow": [1000.0] * len(clean_speed),
            "per_lane_hourly_capacity": [2000.0] * len(clean_speed),
            "aggregation_mode": ["daily"] * len(clean_speed),
        }
    )


def test_missing_t3_shifts_to_nearby_post_qc_recovery():
    entity = _boundary_repair_entity(
        [55.0, 30.0, 35.0, 40.0, np.nan, np.nan, 50.0],
        [55.0, 30.0, 35.0, 40.0, 42.0, 44.0, 50.0],
    )
    episode = _episode_from_bounds(
        entity,
        0,
        4,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )

    assert episode is not None
    assert episode["original_t3"] == "07:00:00"
    assert episode["t3"] == "07:30:00"
    assert episode["t3_boundary_speed_source"] == "shifted_post_qc"
    assert episode["t3_boundary_shift_intervals"] == 2
    assert episode["t3_boundary_speed_mph"] == 50.0
    assert episode["duration_min"] == 105.0
    assert episode["discharge_window_valid"]
    assert episode["is_valid_for_mu"]


def test_missing_t3_uses_physical_pre_qc_speed_after_search_fails():
    entity = _boundary_repair_entity(
        [55.0, 30.0, 35.0, 40.0, np.nan],
        [55.0, 30.0, 35.0, 40.0, 42.0],
    )
    episode = _episode_from_bounds(
        entity,
        0,
        4,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )

    assert episode is not None
    assert episode["t3"] == episode["original_t3"]
    assert episode["t3_boundary_speed_source"] == "pre_qc_fallback"
    assert episode["t3_boundary_shift_intervals"] == 0
    assert episode["t3_boundary_speed_mph"] == 42.0
    assert episode["original_t3_speed_pre_qc_mph"] == 42.0
    assert episode["discharge_speed_gain_mph"] == 12.0
    assert episode["discharge_window_valid"]
    assert episode["is_valid_for_mu"]


def test_missing_t3_rejects_nonphysical_pre_qc_speed():
    entity = _boundary_repair_entity(
        [55.0, 30.0, 35.0, 40.0, np.nan],
        [55.0, 30.0, 35.0, 40.0, 95.0],
    )
    episode = _episode_from_bounds(
        entity,
        0,
        4,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )

    assert episode is not None
    assert episode["t3_boundary_speed_source"] == "missing"
    assert np.isnan(episode["t3_boundary_speed_mph"])
    assert not episode["discharge_window_valid"]
    assert not episode["is_valid_for_mu"]


def test_missing_t0_shifts_to_nearby_post_qc_onset():
    entity = _boundary_repair_entity(
        [55.0, 50.0, np.nan, 30.0, 25.0, 35.0, 50.0],
        [55.0, 50.0, 43.0, 30.0, 25.0, 35.0, 50.0],
    )
    episode = _episode_from_bounds(
        entity,
        2,
        6,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )

    assert episode is not None
    assert episode["original_t0"] == "06:30:00"
    assert episode["t0"] == "06:15:00"
    assert episode["t2"] == "07:00:00"
    assert episode["t0_boundary_speed_source"] == "shifted_post_qc"
    assert episode["t0_boundary_shift_intervals"] == -1
    assert episode["t0_boundary_speed_mph"] == 50.0
    assert episode["duration_min"] == 90.0


def test_missing_t0_uses_physical_pre_qc_speed_after_search_fails():
    entity = _boundary_repair_entity(
        [np.nan, 30.0, 25.0, 35.0, 50.0],
        [42.0, 30.0, 25.0, 35.0, 50.0],
    )
    episode = _episode_from_bounds(
        entity,
        0,
        4,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )

    assert episode is not None
    assert episode["t0"] == episode["original_t0"]
    assert episode["t0_boundary_speed_source"] == "pre_qc_fallback"
    assert episode["t0_boundary_shift_intervals"] == 0
    assert episode["t0_boundary_speed_mph"] == 42.0
    assert episode["original_t0_speed_pre_qc_mph"] == 42.0


def test_missing_t0_rejects_nonphysical_pre_qc_speed():
    entity = _boundary_repair_entity(
        [np.nan, 30.0, 25.0, 35.0, 50.0],
        [95.0, 30.0, 25.0, 35.0, 50.0],
    )
    episode = _episode_from_bounds(
        entity,
        0,
        4,
        threshold_used=45.0,
        freeflow_speed_mph=65.0,
        interval_minutes=15.0,
        config=StateTransitionConfig(),
        notes="test",
    )

    assert episode is not None
    assert episode["t0_boundary_speed_source"] == "missing"
    assert np.isnan(episode["t0_boundary_speed_mph"])


def test_detector_audit_tags_use_taplite_workflow_boundaries():
    assert episode_time_period_tag(
        "2000-01-03 08:45:00", "2000-01-03 09:15:00", 15
    ) == "AM;MD"
    assert episode_time_period_tag(
        "2000-01-03 14:45:00", "2000-01-03 15:15:00", 15
    ) == "MD;PM"
    assert episode_time_period_tag(
        "2000-01-03 05:45:00", "2000-01-03 06:15:00", 15
    ) == "NT;AM"


def test_worker_plan_respects_explicit_count_and_task_limit():
    plan = recommend_workers(
        3,
        explicit_workers=8,
        sample_seconds=0.01,
    )
    assert plan.workers == 3
    assert plan.task_count == 3


def test_validation_link_selection_includes_congested_and_recovered_links():
    rows = []
    for link_id, speeds, free_flow, cutoff in (
        (1, [15.0, 20.0], 65.0, 45.0),
        (2, [30.0, 64.0], 65.0, 45.0),
        (3, [50.0, 60.0], 65.0, 45.0),
        (4, [35.0, 62.0], 65.0, 45.0),
    ):
        for index, speed in enumerate(speeds):
            rows.append(
                {
                    "link_id": link_id,
                    "t_min": 360 + index * 15,
                    "speed_raw": speed,
                    "obs_ff": free_flow,
                    "cutoff": cutoff,
                }
            )
    links, tags = _pick_validation_links(pd.DataFrame(rows), n_links=4)
    assert 1 in links
    assert "congested" in tags
    assert "free-flow recovery" in tags


def test_robust_qvdf_recovers_power_law():
    ratio = np.linspace(0.4, 2.0, 20)
    duration = 0.8 * ratio**1.4
    minimum = 50.0 / (1 + 0.3 * duration**1.2)
    fit = fit_qvdf(ratio, duration, minimum, 50.0)
    assert fit.status == "ok"
    assert abs(fit.n - 1.4) < 0.05
    assert abs(fit.s - 1.2) < 0.05


def test_state_transition_settings_are_scaled_to_fifteen_minutes():
    config = detector_config(PipelineSettings(interval_minutes=15))
    assert config.default_interval_minutes == 15
    assert config.detection_smoothing_window_intervals == 3
    assert config.rolling_mean_extension_intervals == 2
    assert config.max_noncongested_gap_intervals == 1


def test_predicted_bounds_keep_detected_t2():
    t0, t2, t3 = predicted_bounds_about_t2(7.0, 8.0, 10.0, 1.5)
    assert t2 == 8.0
    assert np.isclose(t2 - t0, 0.5)
    assert np.isclose(t3 - t2, 1.0)


def test_episode_reconstruction_minimum_occurs_at_detected_t2():
    time = np.arange(6 * 60, 11 * 60 + 1, 15)
    model = reconstruct_episode_speed(
        time,
        t0_hour=7.0,
        t2_hour=8.0,
        t3_hour=10.0,
        minimum_speed_mph=22.0,
        cutoff_mph=45.0,
        free_flow_mph=65.0,
        length_mi=1.0,
        discharge_vphpl=1800.0,
        window_start_minute=6 * 60,
        window_end_minute=11 * 60,
    )
    minimum_time = time[int(np.argmin(model))] / 60.0
    assert minimum_time == 8.0
    assert np.isclose(model.min(), 22.0)


def test_qvdf_queue_shape_is_zero_at_bounds_and_peaks_at_t2_fraction():
    x = np.array([0.0, 0.25, 0.4, 0.75, 1.0])
    shape = qvdf_queue_shape(x, 0.4)
    assert shape[0] == 0.0
    assert shape[-1] == 0.0
    assert np.isclose(shape[2], 1.0)
    assert int(np.argmax(shape)) == 2


def test_episode_reconstruction_keeps_smoothstep_outer_shoulders():
    time = np.arange(6 * 60, 11 * 60 + 1, 15)
    model = reconstruct_episode_speed(
        time,
        t0_hour=7.0,
        t2_hour=8.0,
        t3_hour=10.0,
        minimum_speed_mph=22.0,
        cutoff_mph=45.0,
        free_flow_mph=65.0,
        length_mi=1.0,
        discharge_vphpl=1800.0,
        window_start_minute=6 * 60,
        window_end_minute=11 * 60,
        left_anchor_mph=60.0,
        right_anchor_mph=62.0,
    )
    assert np.isclose(model[0], 60.0)
    assert np.isclose(model[np.where(time == 7 * 60)[0][0]], 45.0)
    assert np.isclose(model[np.where(time == 10 * 60)[0][0]], 45.0)
    assert np.isclose(model[-1], 62.0)
    assert np.all((model[(time < 7 * 60)] >= 45.0) & (model[(time < 7 * 60)] <= 60.0))
    assert np.all((model[(time > 10 * 60)] >= 45.0) & (model[(time > 10 * 60)] <= 62.0))


def test_nonoverlapping_episode_selection_keeps_compatible_episodes():
    episodes = pd.DataFrame(
        [
            {"episode_id": "deep", "t0_hour": 7.0, "t2_hour": 8.0, "t3_hour": 9.0, "P_hr": 2.0, "min_speed_mph": 15.0},
            {"episode_id": "overlap", "t0_hour": 8.5, "t2_hour": 9.0, "t3_hour": 9.5, "P_hr": 1.0, "min_speed_mph": 25.0},
            {"episode_id": "later", "t0_hour": 10.0, "t2_hour": 10.5, "t3_hour": 11.0, "P_hr": 1.0, "min_speed_mph": 30.0},
        ]
    )
    accepted, dropped = select_nonoverlapping_episodes(episodes)
    assert accepted["episode_id"].tolist() == ["deep", "later"]
    assert dropped["episode_id"].tolist() == ["overlap"]
    assert dropped.loc[0, "overlap_resolution_reason"].startswith(
        "overlaps_higher_priority_episode:"
    )


def test_canonical_mapping_uses_closest_then_first_occurrence(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "tmc": ["A", "B", "C", "C"],
            "link_id": [10, 10, 11, 12],
            "distance_to_tmc_ft": [30.0, 10.0, np.nan, np.nan],
        }
    ).to_csv(path, index=False)
    mapping = load_canonical_mapping(path)
    link_primary = mapping[mapping["link_tmc_rank"].eq(1)].set_index("link_id")
    assert link_primary.loc[10, "tmc"] == "B"
    tmc_c = mapping[mapping["tmc"].eq("C")].sort_values("tmc_link_rank")
    assert tmc_c["link_id"].tolist() == [11, 12]


def test_canonical_mapping_composite_score_can_outweigh_distance(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "tmc": ["GEOMETRY_ONLY", "CONSISTENT"],
            "road": ["I-66", "I-66"],
            "direction": ["EASTBOUND", "EASTBOUND"],
            "link_id": [10, 10],
            "from_node_id": [1, 1],
            "to_node_id": [2, 2],
            "length_mi": [1.0, 1.0],
            "cumulative_mi": [1.0, 1.0],
            "route_length_mi": [1.0, 1.0],
            "tmc_miles": [1.0, 1.0],
            "distance_to_tmc_ft": [0.0, 25.0],
            "geometry_overlap_pct": [100.0, 95.0],
            "bearing_diff_deg": [85.0, 2.0],
            "STREETNAME": ["Unrelated Road", "I-66"],
            "link_type": [305, 301],
        }
    ).to_csv(path, index=False)
    mapping = load_canonical_mapping(path)
    primary = mapping[mapping["link_tmc_rank"].eq(1)].iloc[0]
    node_pair_primary = mapping[
        mapping["selected_for_node_pair_lookup"]
    ].iloc[0]
    assert primary["tmc"] == "CONSISTENT"
    assert node_pair_primary["tmc"] == "CONSISTENT"
    assert node_pair_primary["node_pair_tmc_rank"] == 1
    assert primary["link_tmc_ranking_basis"] == "highest_composite_match_score"
    assert primary["composite_match_score"] > mapping.loc[
        mapping["tmc"].eq("GEOMETRY_ONLY"), "composite_match_score"
    ].iloc[0]


def test_node_pair_winner_is_ranked_across_different_link_ids(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "tmc": ["LOW", "HIGH"],
            "road": ["I-66", "I-66"],
            "direction": ["EASTBOUND", "EASTBOUND"],
            "link_id": [10, 11],
            "from_node_id": [1, 1],
            "to_node_id": [2, 2],
            "distance_to_tmc_ft": [1.0, 20.0],
            "geometry_overlap_pct": [100.0, 95.0],
            "bearing_diff_deg": [89.0, 1.0],
            "STREETNAME": ["Unrelated", "I-66"],
            "link_type": [305, 301],
            "length_mi": [1.0, 1.0],
            "cumulative_mi": [1.0, 1.0],
            "route_length_mi": [1.0, 1.0],
            "tmc_miles": [1.0, 1.0],
        }
    ).to_csv(path, index=False)

    mapping = load_canonical_mapping(path)
    winner = mapping[mapping["selected_for_node_pair_lookup"]].iloc[0]

    assert winner["tmc"] == "HIGH"
    assert winner["node_pair_tmc_ranking_basis"] == "highest_composite_match_score"
    assert mapping["selected_for_node_pair_lookup"].sum() == 1


def test_canonical_mapping_observation_quality_is_a_scored_component(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "tmc": ["LOW_Q", "HIGH_Q"],
            "road": ["US-50", "US-50"],
            "link_id": [10, 10],
            "length_mi": [1.0, 1.0],
            "cumulative_mi": [1.0, 1.0],
            "route_length_mi": [1.0, 1.0],
            "tmc_miles": [1.0, 1.0],
            "distance_to_tmc_ft": [5.0, 5.0],
            "bearing_diff_deg": [2.0, 2.0],
            "STREETNAME": ["US-50", "US-50"],
            "link_type": [302, 302],
        }
    ).to_csv(path, index=False)
    quality = pd.DataFrame(
        {"tmc": ["LOW_Q", "HIGH_Q"], "observation_quality_score": [0.1, 0.9]}
    )
    mapping = load_canonical_mapping(path, observation_quality=quality)
    primary = mapping[mapping["link_tmc_rank"].eq(1)].iloc[0]
    assert primary["tmc"] == "HIGH_Q"


def test_current_output_contract_column_order():
    assert STAGE0_COLUMNS[0] == "link_id"
    assert DAILY_PAQ_COLUMNS[:6] == [
        "link_id",
        "sensor_uid",
        "tmc_code",
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
    ]
    assert "demand_capacity_basis" in DAILY_PAQ_COLUMNS
    assert "demand_is_proxy" in DAILY_PAQ_COLUMNS
    assert len(HANDOFF_COLUMNS) == 28
    assert HANDOFF_COLUMNS[1:4] == [
        "sensor_uid",
        "tmc_code",
        "network_link_id",
    ]
    assert "emis_co2_g_model" in HANDOFF_COLUMNS
    assert HANDOFF_COLUMNS[-1] == "emissions_method"


def test_output_steps_are_numbered_and_named():
    assert list(STEP_FOLDERS.values()) == [
        "01-input-and-qc",
        "02-fundamental-diagram",
        "03-profiles",
        "04-episode-detection",
        "05-episode-filtering",
        "06-qvdf-calibration",
        "07-reconstruction-and-handoff",
        "08-quality-assurance",
        "09-summary-tables",
        "10-figures",
        "11-run-metadata",
    ]


def test_minimum_speed_law():
    parameters = {"f_p": 0.3, "s": 1.2}
    expected = 50.0 / (1.0 + 0.3 * 2.0**1.2)
    assert np.isclose(predict_minimum_speed(50.0, 2.0, parameters), expected)


def test_prefilter_episode_audit_retains_tmc_and_t2():
    candidates = pd.DataFrame(
        [
            {
                "episode_id": "candidate-1",
                "sensor_uid": "corridor::tmc-1",
                "tmc_code": "tmc-1",
                "corridor": "corridor",
                "link_id": 1,
                "date": "Weekday",
                "period": "AM",
                "t0_hour": 7.0,
                "t2_hour": 8.0,
                "t3_hour": 9.0,
                "original_t0_hour": 7.25,
                "original_t3_hour": 8.5,
                "t0_boundary_speed_mph": 52.0,
                "t0_boundary_speed_source": "shifted_post_qc",
                "t0_boundary_shift_intervals": -1,
                "t3_boundary_speed_mph": 48.0,
                "t3_boundary_speed_source": "shifted_post_qc",
                "t3_boundary_shift_intervals": 2,
            }
        ]
    )
    audit = episode_candidate_audit(candidates)
    assert audit.loc[0, "tmc_code"] == "tmc-1"
    assert audit.loc[0, "t2_hour"] == 8.0
    assert audit.loc[0, "original_t0_hour"] == 7.25
    assert audit.loc[0, "t0_boundary_speed_source"] == "shifted_post_qc"
    assert audit.loc[0, "t0_boundary_shift_intervals"] == -1
    assert audit.loc[0, "original_t3_hour"] == 8.5
    assert audit.loc[0, "t3_boundary_speed_source"] == "shifted_post_qc"
    assert audit.loc[0, "t3_boundary_shift_intervals"] == 2
    assert "is_clean_valid_episode" not in audit.columns


def test_filter_audit_is_self_contained_and_keeps_candidate_id():
    screened = pd.DataFrame(
        [
            {
                "episode_id": "candidate-1",
                "sensor_uid": "corridor::tmc-1",
                "tmc_code": "tmc-1",
                "corridor": "corridor",
                "link_id": 1,
                "date": "Weekday",
                "period": "AM",
                "t0_hour": 7.0,
                "t2_hour": 8.0,
                "t3_hour": 9.0,
                "duration_min": 120.0,
                "threshold_used": 45.0,
                "mean_speed_mph": 30.0,
                "episode_demand": 1200.0,
                "magnitude": 0.5,
                "road_order": 1,
                "direction": "EB",
                "flag_vt2_too_low": False,
                "n_hard_flags": 0,
                "n_soft_flags": 0,
                "measured_outlier_flag": False,
                "measured_outlier_reasons": "",
                "is_clean_valid_episode": True,
            }
        ]
    )
    candidate = episode_candidate_audit(screened)
    audit = episode_filter_audit(screened)
    assert audit["episode_id"].tolist() == candidate["episode_id"].tolist()
    for column in (
        "duration_min",
        "threshold_used",
        "mean_speed_mph",
        "episode_demand",
        "magnitude",
        "road_order",
        "direction",
    ):
        assert column in audit.columns
    assert "flag_vt2_too_low" in audit.columns
