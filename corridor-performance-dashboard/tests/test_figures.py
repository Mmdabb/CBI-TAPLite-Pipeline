import pandas as pd

from corridor_measurement.figures import (
    create_corridor_figures,
    select_representative_tmcs,
)


def test_select_representative_tmcs_uses_first_middle_last_order():
    frame = pd.DataFrame(
        {
            "corridor": ["A"] * 5,
            "tmc_code": ["t1", "t2", "t3", "t4", "t5"],
            "direction": ["EASTBOUND"] * 5,
            "road_order": [1, 2, 3, 4, 5],
            "model_tmc_speed_mph": [50, 50, 50, 50, 50],
        }
    )
    selected = select_representative_tmcs(frame, count=3)
    assert selected["tmc_code"].tolist() == ["t1", "t3", "t5"]
    assert selected["selection_position"].tolist() == ["first", "middle", "last"]


def test_select_representative_tmcs_uses_five_without_reducing_membership():
    rows = []
    for index in range(8):
        for minute, observed in ((360, 60.0 - index), (375, 35.0 + index)):
            rows.append(
                {
                    "corridor": "A",
                    "tmc_code": f"t{index + 1}",
                    "direction": "EASTBOUND",
                    "road_order": index + 1,
                    "t_min": minute,
                    "observed_tmc_speed_mph": observed,
                    "model_tmc_speed_mph": observed + (index - 3),
                }
            )
    profiles = pd.DataFrame(rows)

    selected = select_representative_tmcs(profiles, count=5)

    assert selected["tmc_code"].nunique() == 5
    assert selected.iloc[0]["tmc_code"] == "t1"
    assert selected.iloc[-1]["tmc_code"] == "t8"
    assert profiles["tmc_code"].nunique() == 8


def test_select_representative_tmcs_uses_all_when_fewer_than_five():
    frame = pd.DataFrame(
        {
            "corridor": ["A"] * 4,
            "tmc_code": ["t1", "t2", "t3", "t4"],
            "direction": ["EASTBOUND"] * 4,
            "road_order": [1, 2, 3, 4],
            "observed_tmc_speed_mph": [50, 45, 40, 35],
            "model_tmc_speed_mph": [48, 44, 42, 37],
        }
    )

    selected = select_representative_tmcs(frame, count=5)

    assert selected["tmc_code"].tolist() == ["t1", "t2", "t3", "t4"]


def test_figure_output_uses_numbered_folders_and_includes_error_heatmap(tmp_path):
    rows = []
    for corridor in ("A_EB", "B_EB"):
        for order, tmc in enumerate(("t1", "t2", "t3"), start=1):
            for minute in (360, 375):
                rows.append(
                    {
                        "corridor": corridor,
                        "tmc_code": tmc,
                        "direction": "EASTBOUND",
                        "road_order": order,
                        "t_min": minute,
                        "period": "AM",
                        "observed_tmc_speed_mph": 55.0 - order - (minute - 360) / 15,
                        "model_tmc_speed_mph": 52.0 - order,
                        "cube_qvdf_tmc_speed_mph": 50.0 - order + (minute - 360) / 30,
                        "cbi_tmc_congestion_threshold_mph": 35.0,
                        "gmns_link_count": 1,
                        "gmns_link_ids": str(100 + order),
                        "taplite_period_volume": 1000.0,
                        "taplite_period_doc": 0.7,
                        "taplite_period_p_hours": 1.0,
                        "cube_period_volume": 1200.0,
                        "cube_period_doc": 0.8,
                        "cube_period_p_hours": 1.1,
                    }
                )
    settings = {
        "periods": {"am": {"start_min": 360, "end_min": 390}},
        "comparison_interval_minutes": 15,
        "selected_tmc_count_per_corridor": 5,
        "figure_dpi": 72,
        "heatmap_speed_min_mph": 0.0,
        "heatmap_speed_max_mph": 75.0,
        "heatmap_error_max_mph": 40.0,
    }

    manifest, selected_profiles, _ = create_corridor_figures(
        pd.DataFrame(rows), tmp_path, settings=settings, workers=2
    )

    assert selected_profiles.groupby("corridor")["tmc_code"].nunique().eq(3).all()
    assert len(manifest) == 2
    assert manifest["selected_tmc_count"].eq(3).all()
    assert (tmp_path / "06-figures/selected-tmc-profiles/A_EB.png").is_file()
    assert (tmp_path / "06-figures/speed-heatmaps/A_EB.png").is_file()
    assert (
        tmp_path
        / "06-figures/absolute-error-heatmaps/taplite-vs-observed/A_EB.png"
    ).is_file()
    assert (
        tmp_path
        / "06-figures/absolute-error-heatmaps/cube-vs-observed/A_EB.png"
    ).is_file()
    assert (
        tmp_path
        / "06-figures/absolute-error-heatmaps/taplite-vs-cube/A_EB.png"
    ).is_file()
    assert {
        "taplite_vs_observed_error_heatmap_figure",
        "cube_vs_observed_error_heatmap_figure",
        "taplite_vs_cube_error_heatmap_figure",
    }.issubset(manifest.columns)
    assert (tmp_path / "06-figures/FIGURES.md").is_file()
