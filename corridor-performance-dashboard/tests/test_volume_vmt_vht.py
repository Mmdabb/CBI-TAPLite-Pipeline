from pathlib import Path

import numpy as np
import pandas as pd

from corridor_measurement.volume_vmt_vht import (
    build_corridor_period_comparison,
    build_scatter_metrics,
    build_network_scatter_metrics,
    create_scatter_figures,
    load_period_link_comparison,
)


def _link_comparison(tmp_path: Path) -> pd.DataFrame:
    performance_source = tmp_path / "link_performance.csv"
    link_source = tmp_path / "link.csv"
    pd.DataFrame(
        [
            {
                "iteration_no": 10,
                "link_id": "1",
                "volume": 80.0,
                "VMT": 160.0,
                "VHT": 4.5,
                "travel_time": 3.0,
                "speed_mph": 40.0,
                "doc": 0.70,
            },
            {
                "iteration_no": 10,
                "link_id": "2",
                "volume": 0.0,
                "VMT": 0.0,
                "VHT": 0.0,
                "travel_time": 2.4,
                "speed_mph": 25.0,
                "doc": 0.10,
            },
        ]
    ).to_csv(performance_source, index=False)
    pd.DataFrame(
        [
            {
                "link_id": "1",
                "vdf_length_mi": 2.0,
                "I4AMVOL": 100.0,
                "I4AMSPD": 40.0,
                "I4AMVC": 0.50,
            },
            {
                "link_id": "2",
                "vdf_length_mi": 1.0,
                "I4AMVOL": 50.0,
                "I4AMSPD": 25.0,
                "I4AMVC": 0.20,
            },
        ]
    ).to_csv(link_source, index=False)
    return load_period_link_comparison(
        performance_source,
        link_source,
        period="am",
        cube_volume_column="I4AMVOL",
        cube_speed_column="I4AMSPD",
        cube_voc_column="I4AMVC",
    )


def test_link_comparison_uses_recorded_taplite_and_derived_cube_measures(tmp_path):
    comparison = _link_comparison(tmp_path).set_index("link_id")

    assert comparison.loc["1", "taplite_volume"] == 80.0
    assert comparison.loc["1", "taplite_recorded_vmt"] == 160.0
    assert comparison.loc["1", "taplite_recorded_vht"] == 4.5
    assert comparison.loc["1", "taplite_vmt"] == 160.0
    assert comparison.loc["1", "taplite_travel_time_hours"] == 0.05
    assert comparison.loc["1", "taplite_vht"] == 4.0
    assert comparison.loc["1", "cube_volume"] == 100.0
    assert comparison.loc["1", "cube_vmt"] == 200.0
    assert comparison.loc["1", "cube_travel_time_hours"] == 0.05
    assert comparison.loc["1", "cube_vht"] == 5.0
    assert comparison.loc["1", "cube_speed_column"] == "I4AMSPD"
    assert comparison.loc["1", "cube_voc"] == 0.50
    assert comparison.loc["1", "taplite_doc"] == 0.70


def test_corridor_comparison_deduplicates_and_sums_mapped_links(tmp_path):
    links = _link_comparison(tmp_path)
    membership = pd.DataFrame(
        {
            "period": ["AM", "AM", "AM"],
            "corridor": ["A_EB", "A_EB", "A_EB"],
            "link_id": ["1", "1", "2"],
        }
    )

    corridor = build_corridor_period_comparison(links, membership).iloc[0]

    assert corridor["gmns_link_count"] == 2
    assert corridor["cube_volume"] == 150.0
    assert corridor["taplite_volume"] == 80.0
    assert corridor["cube_vmt"] == 250.0
    assert corridor["cube_vht"] == 7.0
    assert corridor["taplite_vht"] == 4.0


def test_scatter_metrics_and_figures_cover_both_scopes(tmp_path):
    links = _link_comparison(tmp_path)
    membership = pd.DataFrame(
        {
            "period": ["AM", "AM"],
            "corridor": ["A_EB", "B_EB"],
            "link_id": ["1", "2"],
        }
    )
    corridors = build_corridor_period_comparison(links, membership)
    metrics = build_scatter_metrics(links, corridors)
    network_metrics = build_network_scatter_metrics(links)

    manifest = create_scatter_figures(
        links,
        corridors,
        metrics,
        tmp_path,
        network_scatter_metrics=network_metrics,
        figure_dpi=72,
    )

    assert len(metrics) == 6
    assert set(metrics["scope"]) == {"all_links", "corridors"}
    assert np.isfinite(metrics["mae"]).all()
    assert len(network_metrics) == 12
    assert set(network_metrics["period"]) == {"ALL", "AM"}
    assert len(manifest) == 6
    assert (
        tmp_path
        / "08-volume-vmt-vht-comparison/figures/link-level/AM.png"
    ).is_file()
    assert (
        tmp_path
        / "08-volume-vmt-vht-comparison/figures/corridor-level/AM.png"
    ).is_file()
    assert (
        tmp_path
        / "08-volume-vmt-vht-comparison/figures/all-network/"
        "cube-vs-taplite-volume-travel-time-speed.png"
    ).is_file()
    assert (
        tmp_path
        / "08-volume-vmt-vht-comparison/figures/all-network/"
        "cube-vs-taplite-vmt-vht-doc.png"
    ).is_file()
    assert (
        tmp_path
        / "08-volume-vmt-vht-comparison/figures/all-network/by-period/AM/"
        "cube-vs-taplite-volume-travel-time-speed.png"
    ).is_file()
    assert (
        tmp_path
        / "08-volume-vmt-vht-comparison/figures/all-network/by-period/AM/"
        "cube-vs-taplite-vmt-vht-doc.png"
    ).is_file()
