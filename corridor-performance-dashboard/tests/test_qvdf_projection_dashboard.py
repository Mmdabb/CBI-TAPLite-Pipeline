from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corridor_dashboard.qvdf_projection_dashboard.analysis import (
    _expanded_network_candidates,
    _select_tmc_period_candidates,
    collect_cbi_products,
)
from corridor_dashboard.qvdf_projection_dashboard.assignment import (
    build_assignment_extract,
)
from corridor_dashboard.qvdf_projection_dashboard.metrics import reconstruction_curves
from corridor_dashboard.qvdf_projection_dashboard.settings import DashboardSettings


def test_dashboard_uses_current_workflow_periods() -> None:
    settings = DashboardSettings()
    assert settings.periods == {
        "AM": (6 * 60, 9 * 60),
        "MD": (9 * 60, 15 * 60),
        "PM": (15 * 60, 19 * 60),
    }


def test_assignment_extract_uses_volume_over_capacity(tmp_path: Path) -> None:
    assignment_root = tmp_path / "assignment"
    expected_dc = {"am": 2.0 / 3.0, "md": 0.5, "pm": 1.0}
    for period, volume in (("am", 200.0), ("md", 300.0), ("pm", 400.0)):
        folder = assignment_root / period
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "iteration_no": [8, 9],
                "link_id": [10, 10],
                "volume": [1.0, volume],
                "link_capacity": [100.0, 100.0],
                "doc": [0.01, expected_dc[period]],
                "P": [0.0, 0.0],
                "t0": [7.0, 7.0],
                "t2": [7.0, 7.0],
                "t3": [7.0, 7.0],
                "free_speed_mph": [65.0, 65.0],
                "spd_mph_06:00": [65.0, 65.0],
                "spd_mph_06:15": [65.0, 65.0],
            }
        ).to_csv(folder / "link_performance.csv", index=False)
    settings = DashboardSettings(
        assignment_root=assignment_root,
        output_root=tmp_path / "output",
    )
    settings.dashboard_data_root.mkdir(parents=True)
    _, result = build_assignment_extract(settings)
    by_period = result.set_index("period")
    assert np.isclose(by_period.loc["AM", "dc_dta_vol"], 2.0 / 3.0)
    assert np.isclose(by_period.loc["MD", "dc_dta_vol"], 0.5)
    assert np.isclose(by_period.loc["PM", "dc_dta_vol"], 1.0)
    assert by_period["assignment_period_duration_hours"].to_dict() == {
        "AM": 3.0,
        "MD": 6.0,
        "PM": 4.0,
    }
    assert np.allclose(result["dc_dta_vol"], result["dc_dta_doc"])
    assert set(result["assignment_boundary_source"]) == {
        "taplite_link_performance"
    }
    assert set(result["assignment_curve_source"]) == {"taplite_spd_profile"}
    assert result["assignment_speed_profile_json"].str.len().gt(0).all()


def test_unordered_taplite_boundaries_use_speed_profile() -> None:
    row = pd.Series(
        {
            "projection_status": "ready",
            "has_accepted_episode": False,
            "has_assignment_boundaries": False,
            "has_assignment_speed_profile": True,
            "assignment_speed_profile_json": (
                '{"time_minutes":[360,365,370],'
                '"speed_mph":[65.0,50.0,60.0]}'
            ),
            "assignment_free_speed_mph": 65.0,
            "assignment_cutoff_speed_mph": 48.75,
            "network_length_mi": 1.0,
        }
    )
    time = np.array([360.0, 365.0, 370.0])
    cbi, projected = reconstruction_curves(row, time)
    assert np.allclose(cbi, [65.0, 65.0, 65.0])
    assert np.allclose(projected, [65.0, 50.0, 60.0])


def test_frozen_node_pair_winner_is_retained_without_period_alternate(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "canonical_node_pair_tmc.csv"
    pd.DataFrame(
        {
            "tmc": ["B", "A"],
            "link_id": [100, 101],
            "from_node_id": [1, 2],
            "to_node_id": [2, 3],
            "length_mi": [1.0, 1.2],
            "sequence": [1, 2],
            "distance_to_tmc_ft": [10.0, 5.0],
            "bearing_diff_deg": [1.0, 1.0],
            "road_order": [20, 10],
            "direction": ["EASTBOUND"] * 2,
            "lanes": [3] * 2,
            "capacity": [2100] * 2,
            "node_pair_tmc_rank": [1, 1],
            "selected_for_node_pair_lookup": [True, True],
            "tmc_link_rank": [1, 1],
            "first_map_occurrence": [1, 2],
        }
    ).to_csv(mapping, index=False)
    episodes = pd.DataFrame(
        {
            "corridor": ["C", "C", "C"],
            "tmc_code": ["A", "B", "A"],
            "period": ["AM", "AM", "PM"],
            "road_order": [1, 2, 1],
            "direction": ["EB", "EB", "EB"],
            "lanes": [2, 2, 2],
        }
    )
    expanded = _expanded_network_candidates(episodes, mapping)
    assert expanded["road_order"].tolist() == [10, 20, 10]
    assert expanded["observed_road_order"].tolist() == [1, 2, 1]
    assert "network_road_order" in expanded.columns
    assignment = pd.DataFrame(
        {
            "net_link_id": [100, 101, 100, 101],
            "period": ["AM", "AM", "PM", "PM"],
            "assignment_t0_hour": [6.2, pd.NA, 15.2, 15.3],
            "assignment_t2_hour": [6.8, pd.NA, 16.0, 16.1],
            "assignment_t3_hour": [7.4, pd.NA, 16.8, 16.9],
            "assignment_vt2_mph": [35.0, pd.NA, 34.0, 33.0],
        }
    )
    expanded = expanded.merge(
        assignment,
        on=["net_link_id", "period"],
        how="left",
        validate="many_to_one",
    )
    selected = _select_tmc_period_candidates(expanded)

    assert not selected.duplicated(["corridor", "tmc_code", "period"]).any()
    assert set(selected.loc[selected["period"].eq("AM"), "tmc_code"]) == {"A", "B"}
    selected_by_tmc_period = selected.set_index(["tmc_code", "period"])
    assert selected_by_tmc_period.loc[("A", "AM"), "net_link_id"] == 101
    assert (
        selected_by_tmc_period.loc[("A", "AM"), "link_selection_basis"]
        == "frozen_node_pair_winner_without_period_coverage"
    )
    assert selected_by_tmc_period.loc[("B", "AM"), "net_link_id"] == 100
    assert selected_by_tmc_period.loc[("A", "PM"), "net_link_id"] == 101


def test_product_collection_skips_run_summary_folder(
    tmp_path: Path,
) -> None:
    corridor = tmp_path / "I66_EB"
    (tmp_path / "_run-summary").mkdir()
    inputs = {
        "03-profiles/average_weekday_profile.csv": pd.DataFrame(
            {
                "corridor": ["I66_EB"],
                "tmc_code": ["TMC-1"],
                "network_link_id": [100],
            }
        ),
        (
            "05-episode-filtering/average_weekday_episodes_accepted.csv"
        ): pd.DataFrame({"corridor": ["I66_EB"], "tmc_code": ["TMC-1"]}),
        "05-episode-filtering/daily_episodes_accepted.csv": pd.DataFrame(
            {"corridor": ["I66_EB"], "tmc_code": ["TMC-1"]}
        ),
        "06-qvdf-calibration/qvdf_selected_parameters.csv": pd.DataFrame(
            {"corridor": ["I66_EB"], "sensor_uid": ["I66_EB::TMC-1"]}
        ),
    }
    for relative, frame in inputs.items():
        path = corridor / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    products = collect_cbi_products(tmp_path)

    assert products.coverage["corridor"].tolist() == ["I66_EB"]


def test_product_collection_excludes_managed_only_corridor(
    tmp_path: Path,
) -> None:
    for corridor_name, tmc_code in (
        ("I66_EB", "GP-1"),
        ("VA7BR_NB", "MANAGED-1"),
    ):
        corridor = tmp_path / corridor_name
        inputs = {
            "03-profiles/average_weekday_profile.csv": pd.DataFrame(
                {
                    "corridor": [corridor_name],
                    "tmc_code": [tmc_code],
                    "network_link_id": [100],
                }
            ),
            (
                "05-episode-filtering/average_weekday_episodes_accepted.csv"
            ): pd.DataFrame(
                {"corridor": [corridor_name], "tmc_code": [tmc_code]}
            ),
            "05-episode-filtering/daily_episodes_accepted.csv": pd.DataFrame(
                {"corridor": [corridor_name], "tmc_code": [tmc_code]}
            ),
            "06-qvdf-calibration/qvdf_selected_parameters.csv": pd.DataFrame(
                {
                    "corridor": [corridor_name],
                    "tmc_code": [tmc_code],
                    "sensor_uid": [f"{corridor_name}::{tmc_code}"],
                }
            ),
        }
        for relative, frame in inputs.items():
            path = corridor / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)

    products = collect_cbi_products(
        tmp_path,
        eligible_tmc_codes={"GP-1"},
    )

    assert products.coverage["corridor"].tolist() == ["I66_EB"]
    assert products.profiles["tmc_code"].tolist() == ["GP-1"]
