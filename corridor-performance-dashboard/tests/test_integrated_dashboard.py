from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from corridor_dashboard.builder import (
    DashboardBuildSettings,
    HTML_TEMPLATE,
    _load_mapmatching,
    _load_tmc_map_metrics,
    _parse_linestring_wgs84,
    build_dashboard,
)


def test_dashboard_quality_table_omits_r2_columns() -> None:
    assert "Duration R²" not in HTML_TEMPLATE
    assert "Speed R²" not in HTML_TEMPLATE
    assert "Duration MAPE" in HTML_TEMPLATE


def test_route_map_excludes_non_gp_tmc_geometry(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "full_tmc_to_link.csv",
        [
            {
                "tmc": "GP-1",
                "facility_class": "gp",
                "geometry_wgs84": "LINESTRING (-77.5 38.8, -77.4 38.9)",
            },
            {
                "tmc": "MANAGED-1",
                "facility_class": "managed",
                "geometry_wgs84": "LINESTRING (-77.6 38.8, -77.5 38.9)",
            },
        ],
    )
    _write_csv(
        tmp_path / "full_route_match_summary.csv",
        [
            {"tmc": "GP-1", "status": "matched"},
            {"tmc": "MANAGED-1", "status": "matched"},
        ],
    )

    geometry, summary, eligible = _load_mapmatching(tmp_path)

    assert eligible == {"GP-1"}
    assert set(geometry) == {"GP-1"}
    assert set(summary) == {"GP-1"}


def test_route_map_tmc_metrics_use_daily_observed_model_pairs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "02-tmc-results" / "tmc_daily_profiles.csv"
    _write_csv(
        source,
        [
            {
                "corridor": "I66_EB",
                "tmc_code": "TMC-1",
                "observed_tmc_speed_mph": 40,
                "model_tmc_speed_mph": 44,
            },
            {
                "corridor": "I66_EB",
                "tmc_code": "TMC-1",
                "observed_tmc_speed_mph": 50,
                "model_tmc_speed_mph": 45,
            },
        ],
    )

    metrics = _load_tmc_map_metrics(tmp_path)[("I66_EB", "TMC-1")]

    assert metrics["observed_average_speed_mph"] == 45
    assert metrics["model_average_speed_mph"] == 44.5
    assert metrics["speed_mae_mph"] == 4.5
    assert metrics["speed_mape_pct"] == 10
    assert metrics["matched_interval_count"] == 2
from corridor_dashboard.combined_profile import (
    _apply_profile_selection_overrides,
    _assignment_summary,
    _load_assignment_parameters,
    _load_primary_link_mapping,
    _filter_general_purpose_profiles,
    _load_ranked_primary_link_mapping,
    _native_link_profile,
    _select_canonical_representative_tmcs,
    _tmc_link_label,
)


def test_profile_override_is_idempotent_when_replacement_already_selected(
    tmp_path: Path,
) -> None:
    override = tmp_path / "overrides.csv"
    _write_csv(
        override,
        [{
            "corridor": "I66_EB",
            "replace_tmc_code": "OLD",
            "replacement_tmc_code": "BETTER",
        }],
    )
    selection = pd.DataFrame([{
        "corridor": "I66_EB",
        "tmc_code": "BETTER",
        "road_order": 24.0,
        "selection_position": "behavior_diverse_2_of_5",
    }])
    profiles = pd.DataFrame({
        "corridor": ["I66_EB"],
        "tmc_code": ["BETTER"],
        "observed_tmc_speed_mph": [60.0],
        "model_tmc_speed_mph": [59.0],
    })
    ranked = pd.DataFrame({
        "corridor": ["I66_EB"],
        "tmc_code": ["BETTER"],
        "road_order": [24.0],
        "selected_for_node_pair_lookup": [True],
    })

    result = _apply_profile_selection_overrides(
        selection, profiles, ranked, override
    )

    assert result["tmc_code"].tolist() == ["BETTER"]
    assert not result["selection_override_applied"].any()


def test_dashboard_profile_filter_uses_facility_class(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "full_tmc_to_link.csv",
        [
            {"tmc": "GP-1", "facility_class": "gp"},
            {"tmc": "MANAGED-1", "facility_class": "managed"},
            {"tmc": "BLANK", "facility_class": ""},
        ],
    )
    profiles = pd.DataFrame(
        {
            "tmc_code": ["GP-1", "MANAGED-1", "BLANK"],
            "corridor": ["I66_EB", "I66_EB", "I66_EB"],
        }
    )

    filtered = _filter_general_purpose_profiles(profiles, tmp_path)

    assert filtered["tmc_code"].tolist() == ["GP-1"]


def test_dashboard_ranks_canonical_links_only_within_gp_membership(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "I66_EB" / "01-input-and-qc" / "link_reference.csv",
        [
            {
                "tmc_code": "MANAGED-1",
                "network_link_id": 100,
                "network_from_node_id": 1,
                "network_to_node_id": 2,
                "network_match_score": 0.99,
                "road_order": 1,
                "network_node_pair_tmc_rank": 1,
                "network_selected_for_node_pair_lookup": True,
            },
            {
                "tmc_code": "GP-1",
                "network_link_id": 100,
                "network_from_node_id": 1,
                "network_to_node_id": 2,
                "network_match_score": 0.80,
                "road_order": 2,
                "network_node_pair_tmc_rank": 2,
                "network_selected_for_node_pair_lookup": False,
            },
        ],
    )

    ranked = _load_ranked_primary_link_mapping(
        tmp_path, eligible_tmc_codes={"GP-1"}
    )

    assert ranked["tmc_code"].tolist() == ["GP-1"]
    assert ranked["selected_for_node_pair_lookup"].tolist() == [False]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dashboard_primary_link_mapping_allows_shared_link(tmp_path: Path) -> None:
    selected = pd.DataFrame(
        {
            "corridor": ["I66_EB", "I66_EB"],
            "tmc_code": pd.Series(["TMC-A", "TMC-B"], dtype="string"),
        }
    )
    _write_csv(
        tmp_path / "I66_EB" / "01-input-and-qc" / "link_reference.csv",
        [
            {
                "tmc_code": "TMC-A",
                "network_link_id": 38189,
                "network_from_node_id": 1,
                "network_to_node_id": 2,
                "network_match_distance_ft": 5.0,
                "network_bearing_diff_deg": 1.0,
                "network_mapping_status": "mapped_primary_link",
                "network_node_pair_tmc_rank": 1,
                "network_selected_for_node_pair_lookup": True,
            },
            {
                "tmc_code": "TMC-B",
                "network_link_id": 38189,
                "network_from_node_id": 1,
                "network_to_node_id": 2,
                "network_match_distance_ft": 7.0,
                "network_bearing_diff_deg": 2.0,
                "network_mapping_status": "mapped_primary_link",
                "network_node_pair_tmc_rank": 2,
                "network_selected_for_node_pair_lookup": False,
            },
        ],
    )

    mapping = _load_primary_link_mapping(selected, tmp_path)

    assert mapping["primary_link_id"].tolist() == ["38189", "38189"]
    assert mapping[["corridor", "tmc_code"]].drop_duplicates().shape[0] == 2
    assert mapping["selected_for_node_pair_lookup"].tolist() == [True, False]


def test_dashboard_representatives_exclude_noncanonical_shared_tmc() -> None:
    profile_rows = []
    for order, tmc_code in enumerate(
        ["TMC-A", "TMC-B", "TMC-C", "TMC-D", "TMC-E", "TMC-NONCANONICAL"],
        start=1,
    ):
        for minute in (360, 375):
            profile_rows.append(
                {
                    "corridor": "I66_EB",
                    "tmc_code": tmc_code,
                    "road_order": order,
                    "t_min": minute,
                    "observed_tmc_speed_mph": 70.0 - order,
                    "model_tmc_speed_mph": 68.0 - order,
                    "cube_qvdf_tmc_speed_mph": 69.0 - order,
                }
            )
    ranked_mapping = pd.DataFrame(
        {
            "corridor": ["I66_EB"] * 6,
            "tmc_code": [
                "TMC-A", "TMC-B", "TMC-C", "TMC-D", "TMC-E",
                "TMC-NONCANONICAL",
            ],
            "selected_for_node_pair_lookup": [True, True, True, True, True, False],
        }
    )

    selected = _select_canonical_representative_tmcs(
        pd.DataFrame(profile_rows), ranked_mapping, count=5
    )

    assert selected["tmc_code"].tolist() == [
        "TMC-A", "TMC-B", "TMC-C", "TMC-D", "TMC-E"
    ]
    assert "TMC-NONCANONICAL" not in set(selected["tmc_code"])


def test_dashboard_selects_most_congested_tmc_in_each_spatial_segment() -> None:
    profile_rows = []
    tmc_codes = [f"TMC-{index:02d}" for index in range(1, 11)]
    for order, tmc_code in enumerate(tmc_codes, start=1):
        observed_speed = 45.0 if order % 2 else 25.0
        for minute in (360, 375):
            profile_rows.append(
                {
                    "corridor": "I66_EB",
                    "tmc_code": tmc_code,
                    "road_order": order,
                    "t_min": minute,
                    "observed_tmc_speed_mph": observed_speed,
                    "model_tmc_speed_mph": observed_speed + 1.0,
                    "cube_qvdf_tmc_speed_mph": observed_speed + 2.0,
                    "cbi_tmc_congestion_threshold_mph": 40.0,
                }
            )
    ranked_mapping = pd.DataFrame(
        {
            "corridor": ["I66_EB"] * 10,
            "tmc_code": tmc_codes,
            "selected_for_node_pair_lookup": [True] * 10,
        }
    )

    selected = _select_canonical_representative_tmcs(
        pd.DataFrame(profile_rows), ranked_mapping, count=5
    )

    assert selected["tmc_code"].tolist() == [
        "TMC-02", "TMC-04", "TMC-06", "TMC-08", "TMC-10"
    ]
    assert selected["selection_position"].tolist() == [
        "segment_1_most_congested",
        "segment_2_most_congested",
        "segment_3_most_congested",
        "segment_4_most_congested",
        "segment_5_most_congested",
    ]


def test_native_dashboard_profile_uses_only_requested_best_link() -> None:
    assignment = {
        "AM": pd.DataFrame(
            {
                "spd_mph_06:00": [70.0, 40.0],
                "spd_mph_06:05": [69.0, 39.0],
                "volume": [1000.0, 2000.0],
                "doc": [0.5, 0.9],
                "P": [0.0, 1.0],
                "link_capacity": [6000.0, 4000.0],
            },
            index=pd.Index(["38189", "39167"], name="link_id"),
        )
    }

    minutes, speeds = _native_link_profile("38189", assignment)
    summary = _assignment_summary("38189", "AM", assignment)

    assert minutes == [360, 365]
    assert speeds == [70.0, 69.0]
    assert summary["volume"] == 1000.0
    assert summary["doc"] == 0.5


def test_dashboard_cube_values_come_from_period_i4_fields(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "am" / "link_performance.csv",
        [
            {
                "iteration_no": 10,
                "link_id": 38189,
                "volume": 900.0,
                "doc": 0.3,
                "spd_mph_06:00": 65.0,
            }
        ],
    )
    _write_csv(
        tmp_path / "am" / "link.csv",
        [
            {
                "link_id": 38189,
                "STREETNAME": "I-66",
                "allowed_use": "closed",
                "I4AMVOL": 1234.0,
                "I4AMVC": 0.42,
            }
        ],
    )

    assignment = _load_assignment_parameters(tmp_path, {"38189"})
    summary = _assignment_summary("38189", "AM", assignment)

    assert summary["cube_volume"] == 1234.0
    assert summary["cube_vc"] == 0.42
    assert summary["allowed_use"] == "closed"
    assert summary["is_closed"] is True


def test_assignment_summary_does_not_infer_closure_from_zero_volume() -> None:
    assignment = {
        "AM": pd.DataFrame(
            {"volume": [0.0], "allowed_use": ["sov;hov2;hov3;trk;apv;com"]},
            index=pd.Index(["38189"], name="link_id"),
        )
    }

    summary = _assignment_summary("38189", "AM", assignment)

    assert summary["volume"] == 0.0
    assert summary["is_closed"] is False


def test_assignment_summary_clips_p_to_period_duration() -> None:
    assignment = {
        period: pd.DataFrame(
            {"P": [9.0]},
            index=pd.Index(["38189"], name="link_id"),
        )
        for period in ("AM", "MD", "PM")
    }

    assert _assignment_summary("38189", "AM", assignment)["p_hours"] == 3.0
    assert _assignment_summary("38189", "MD", assignment)["p_hours"] == 6.0
    assert _assignment_summary("38189", "PM", assignment)["p_hours"] == 4.0


def _write_measurement_fixture(root: Path) -> None:
    _write_csv(
        root / "01-corridor-results" / "corridor_metrics.csv",
        [
            {
                "corridor": "I66_EB",
                "matched_interval_count": 52,
                "mae_mph": 8.25,
                "mape_pct": 15.22,
                "cube_vs_observed_mae_mph": 8.39,
                "cube_vs_observed_mape_pct": 15.47,
                "taplite_vs_cube_mae_mph": 0.29,
                "observed_congestion_duration_min": 60,
                "model_congestion_duration_min": 45,
                "cube_vs_observed_model_congestion_duration_min": 50,
                "congestion_duration_absolute_error_min": 15,
                "congestion_iou_pct": 50,
                "result_status": "complete",
            }
        ],
    )
    _write_csv(
        root / "01-corridor-results" / "overall_metrics.csv",
        [
            {
                "corridor_count": 1,
                "corridor_mean_speed_mae_mph": 8.25,
                "corridor_mean_speed_mape_pct": 15.22,
                "corridor_mean_cube_vs_observed_speed_mae_mph": 8.39,
                "corridor_mean_taplite_vs_cube_speed_mae_mph": 0.29,
                "congestion_duration_mae_min": 15,
            }
        ],
    )
    _write_csv(
        root / "01-corridor-results" / "corridor_period_metrics.csv",
        [{"corridor": "I66_EB", "period": "AM", "mae_mph": 8.0}],
    )
    _write_csv(
        root / "01-corridor-results" / "daily_corridor_profiles.csv",
        [
            {
                "corridor": "I66_EB",
                "time_minutes": 360,
                "observed_speed_mph": 55,
                "model_speed_mph": 57,
            }
        ],
    )
    _write_csv(
        root / "02-tmc-results" / "selected_tmc_period_metrics.csv",
        [{"corridor": "I66_EB", "tmc_code": "TMC-1", "period": "AM"}],
    )
    _write_csv(
        root / "02-tmc-results" / "selected_tmc_profiles.csv",
        [
            {
                "corridor": "I66_EB",
                "period": "AM",
                "tmc_code": "TMC-1",
                "road_order": 1,
                "selection_position": "corridor_start",
                "t_min": 360,
                "observed_tmc_speed_mph": 55,
                "model_tmc_speed_mph": 52,
                "cbi_tmc_congestion_threshold_mph": 45,
                "taplite_period_volume": 1000,
                "taplite_period_doc": 0.5,
                "taplite_period_p_hours": 0.25,
                "cube_period_volume": 1200,
                "cube_period_doc": 0.4,
                "cube_period_p_hours": 0.5,
                "gmns_link_count": 1,
            }
        ],
    )
    _write_csv(
        root / "03-congestion-results" / "congestion_episodes.csv",
        [{"corridor": "I66_EB", "source": "observed"}],
    )
    _write_csv(
        root
        / "08-volume-vmt-vht-comparison"
        / "data"
        / "scatter_metrics.csv",
        [{"scope": "all_links", "period": "AM", "metric": "volume"}],
    )
    _write_csv(
        root
        / "08-volume-vmt-vht-comparison"
        / "data"
        / "corridor_period_comparison.csv",
        [{"corridor": "I66_EB", "period": "AM"}],
    )
    figure_values = {
        "selected_profile_figure": (
            "06-figures/selected-tmc-profiles/I66_EB.png"
        ),
        "speed_heatmap_figure": "06-figures/speed-heatmaps/I66_EB.png",
        "taplite_vs_observed_error_heatmap_figure": (
            "06-figures/absolute-error-heatmaps/"
            "taplite-vs-observed/I66_EB.png"
        ),
        "cube_vs_observed_error_heatmap_figure": (
            "06-figures/absolute-error-heatmaps/"
            "cube-vs-observed/I66_EB.png"
        ),
        "taplite_vs_cube_error_heatmap_figure": (
            "06-figures/absolute-error-heatmaps/"
            "taplite-vs-cube/I66_EB.png"
        ),
    }
    _write_csv(
        root / "06-figures" / "figure_manifest.csv",
        [
            {
                "corridor": "I66_EB",
                "tmc_count": 24,
                "selected_tmc_count": 5,
                "selected_tmc_codes": "TMC-1;TMC-2;TMC-3;TMC-4;TMC-5",
                **figure_values,
            }
        ],
    )
    for relative in figure_values.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"figure")
    scatter_figure = (
        "08-volume-vmt-vht-comparison/figures/link-level/AM.png"
    )
    _write_csv(
        root
        / "08-volume-vmt-vht-comparison"
        / "scatter_manifest.csv",
        [{"scope": "all_links", "period": "AM", "figure": scatter_figure}],
    )
    scatter_path = root / scatter_figure
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    scatter_path.write_bytes(b"scatter")
    manifest = root / "07-run-metadata" / "run_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"status":"PASS"}', encoding="utf-8")
    _write_csv(
        root
        / "09-observed-speed-derived-volume"
        / "observed_volume_summary.csv",
        [
            {
                "period": "AM",
                "observed_derived_link_count": 1,
                "observed_derived_period_volume": 1000,
                "reconciliation_status": "PASS",
            }
        ],
    )
    _write_csv(
        root
        / "10-link-volume-diagnostics"
        / "corridor_period_summary.csv",
        [
            {
                "corridor": "I66_EB",
                "period": "AM",
                "mapped_physical_link_count": 24,
                "zero_assignment_link_count": 4,
                "doc_le_0_10_link_count": 6,
                "doc_le_0_25_link_count": 8,
                "mapmatching_review_link_count": 1,
                "formula_review_link_count": 0,
                "median_assignment_volume": 1000,
                "median_synthetic_period_volume": 2500,
                "zero_assignment_positive_synthetic_count": 4,
            }
        ],
    )
    _write_csv(
        root / "10-link-volume-diagnostics" / "manual_review_sample.csv",
        [{"corridor": "I66_EB", "period": "AM", "link_id": 100}],
    )
    _write_csv(
        root
        / "10-link-volume-diagnostics"
        / "kernel_formula_reconciliation.csv",
        [{"period": "AM", "review_count": 0}],
    )


def test_parse_wgs84_linestring() -> None:
    assert _parse_linestring_wgs84(
        "LINESTRING (-77.5 38.8, -77.4 38.9)"
    ) == [[38.8, -77.5], [38.9, -77.4]]
    assert _parse_linestring_wgs84("") == []


def test_tmc_link_label_lists_ordered_path_links() -> None:
    assert _tmc_link_label(2, "38189;39167") == (
        "TMC covers 2 links: 38189 → 39167"
    )
    assert _tmc_link_label(1, "40899") == "TMC covers 1 link: 40899"
    assert _tmc_link_label(2) == "TMC covers 2 links"


def test_builds_static_map_and_stages_report(tmp_path: Path) -> None:
    package = tmp_path / "nvta-cbi-package"
    results = package / "outputs" / "current-run"
    corridor = results / "I66_EB"
    (corridor / "11-run-metadata").mkdir(parents=True)
    (corridor / "11-run-metadata" / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "raw_qc_pass_rate": 0.99,
                "episodes_detected": 10,
                "episodes_clean": 8,
                "calibration_rows": 3,
                "average_weekday_calibration_rows": 2,
                "figures": 4,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        corridor / "01-input-and-qc" / "link_reference.csv",
        [
            {
                "tmc_code": "TMC-1",
                "direction": "EASTBOUND",
                "length_mi": 1.25,
            }
        ],
    )
    _write_csv(
        corridor
        / "07-reconstruction-and-handoff"
        / "average_weekday_time_dependent.csv",
        [
            {
                "tmc_code": "TMC-1",
                "t_min": 360,
                "speed_qvdf_model": 50,
                "congestion_threshold_mph": 45,
            }
        ],
    )
    _write_csv(
        results / "_run-summary" / "_QUALITY_SUMMARY.csv",
        [
            {
                "key": "I66_EB",
                "period": "AM",
                "n_links": 1,
                "step1_DC_P_R2": 0.9,
                "step2_P_mag_R2": 0.8,
                "P_MAPE_pct": 4.0,
                "vt2_MAPE_pct": 5.0,
                "t0_MAE_min": 6.0,
                "gates_pass": "7/7",
            }
        ],
    )
    map_product = package / "data" / "tmc_mapmatching" / "current"
    _write_csv(
        map_product / "full_tmc_to_link.csv",
        [
            {
                "tmc": "TMC-1",
                "facility_class": "gp",
                "geometry_wgs84": (
                    "LINESTRING (-77.5 38.8, -77.4 38.9)"
                ),
            }
        ],
    )
    _write_csv(
        map_product / "full_route_match_summary.csv",
        [{"tmc": "TMC-1", "status": "matched"}],
    )
    reports = package / "outputs" / "qvdf-current"
    _write_csv(
        reports / "data" / "corridor_coverage.csv",
        [
            {
                "corridor": "I66_EB",
                "coverage_status": "ready",
                "average_accepted_episodes": 2,
                "daily_accepted_episodes": 8,
                "calibrated_link_periods": 3,
                "selected_link_periods": 3,
                "ready_link_periods": 3,
                "assignment_links": 1,
            }
        ],
    )
    report_dir = reports / "corridors" / "I66_EB"
    report_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text(
        '<style></style><a href="../../index.html">All corridors</a>'
        '<h2>Assignment projection analysis</h2>'
        '<h2>Daily analysis</h2>'
        '<h3>Sensor versus model, full day</h3>'
        '<h3>Speed heatmap</h3><h3>Speed and volume</h3>',
        encoding="utf-8",
    )
    (report_dir / "projection.png").write_bytes(b"staged projection")
    figure_dir = corridor / "10-figures"
    figure_dir.mkdir(parents=True)
    (figure_dir / "qvdf_projection.png").write_bytes(b"projection")
    ritis_15min = (
        package
        / "input-data"
        / "shared"
        / "ritis"
        / "NOVA-Oct1-31-2025--Avg-at-15min-.csv"
    )
    ritis_15min.parent.mkdir(parents=True)
    ritis_15min.write_text(
        "tmc_code,measurement_tstamp,speed\n",
        encoding="utf-8",
    )
    output = package / "outputs" / "integrated"
    measurement = package / "outputs" / "measurement"
    _write_measurement_fixture(measurement)
    # The current measurement workflow does not create the legacy
    # observed-speed-derived volume export.  Dashboard generation must not
    # fail when that optional download is absent.
    (
        measurement
        / "09-observed-speed-derived-volume"
        / "observed_volume_summary.csv"
    ).unlink()
    manifest = build_dashboard(
        DashboardBuildSettings(
            package_root=package,
            corridor_results_root=results,
            mapmatching_product_root=map_product,
            qvdf_report_root=reports,
            corridor_measurement_root=measurement,
            assignment_root=package / "assignment",
            ritis_15min_path=ritis_15min,
            output_root=output,
        )
    )

    assert manifest["corridors"] == 1
    assert manifest["staged_reports"] == 1
    assert manifest["missing_tmc_geometry"] == 0
    assert manifest["sources"]["ritis_15min"].endswith(
        "NOVA-Oct1-31-2025--Avg-at-15min-.csv"
    )
    assert (output / ".nojekyll").is_file()
    staged_report = (
        output / "reports" / "corridors" / "I66_EB" / "index.html"
    ).read_text(encoding="utf-8")
    assert "Interactive corridor map" in staged_report
    assert "TMC-aligned profile diagnostics" in staged_report
    assert "Assignment projection analysis" not in staged_report
    assert "projection.png" not in staged_report
    assert "tmc_observed_qvdf_taplite.png" in staged_report
    assert ".tmc-profile-figure{" in staged_report
    assert "corridor-profile-measurement/assets/06-figures" not in staged_report
    assert "CBI speed and flow consistency" in staged_report
    assert "CBI reconstruction speed heatmap" not in staged_report
    assert "speed_heatmap.png" not in staged_report
    assert "speed_volume_link" not in staged_report
    assert "sensor_vs_model_fullday.png" not in staged_report
    assert (
        output
        / "reports"
        / "corridors"
        / "I66_EB"
        / "daily_analysis"
        / "tmc_observed_qvdf_taplite.png"
    ).is_file()
    dashboard = (output / "index.html").read_text(encoding="utf-8")
    assert "reports/corridors/I66_EB/index.html" in dashboard
    assert "corridor-profile-measurement/index.html" in dashboard
    assert "Mean TAPlite vs CBI speed MAE" not in dashboard
    profile_page = (
        output / "corridor-profile-measurement" / "index.html"
    ).read_text(encoding="utf-8")
    assert "Corridor Profile Measurement" in profile_page
    assert "Mean TAPlite vs CBI speed MAE" in profile_page
    assert "Download data" in profile_page
    assert "Download all corridor data" in profile_page
    assert "Observed-speed-derived volume summary" not in profile_page
    assert "Behavior-diverse selected TMC profiles" not in profile_page
    assert 'id="profileFigure"' not in profile_page
    assert "Volume, VMT, and VHT comparison" not in profile_page
    assert "I66_EB.png" in profile_page
    assert "Assignment loading and mapped-link diagnostic" in profile_page
    assert "Observed and TAPlite heatmaps" in profile_page
    assert "<th>Cube-CBI MAE</th>" not in profile_page
    assert "<th>Cube-CBI MAPE</th>" not in profile_page
    assert "<th>TAPlite-Cube MAE</th>" not in profile_page
    assert "<th>Cube congestion</th>" not in profile_page
    assert (
        "<th>TAPlite-CBI MAPE</th><th>Observed congestion</th>"
        in profile_page
    )
    assert "TAPlite vs observed error" not in profile_page
    assert 'id="tapliteError"' not in profile_page
    assert 'id="cubeError"' not in profile_page
    assert 'id="tapliteCubeError"' not in profile_page
    assert "Learn more" in profile_page
    methods_page = (output / "learn-more" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "mathjax@3" in methods_page
    assert 'class="worked-example"' in methods_page
    assert 'id="network-scatter-diagnostics"' in methods_page
    assert 'id="error-heatmaps"' not in methods_page
    assert 'id="assignment-projection"' not in methods_page
    assert "legacy CBI assignment-projection heatmap is no longer displayed" in methods_page
    assert r"\widehat y=a x" in methods_page
    assert (
        output
        / "corridor-profile-measurement"
        / "downloads"
        / "corridor-profile-measurement-data.zip"
    ).is_file()
    assert manifest["corridor_profile_measurement"]["corridors"] == 1
    assert not (
        output / "reports" / "corridors" / "I66_EB" / "projection.png"
    ).exists()


def test_settings_keep_explicit_corridor_measurement_run(
    tmp_path: Path,
) -> None:
    package = tmp_path / "nvta-cbi-package"
    corridor_root = package / "outputs" / "current" / "corridors"
    corridor_root.mkdir(parents=True)
    older = (
        package
        / "outputs"
        / "corridor-profile-measurement"
        / "corridor-profile-measurement-2026-07-30-10-00"
    )
    newest = (
        package
        / "outputs"
        / "corridor-profile-measurement"
        / "corridor-profile-measurement-2026-07-31-10-00"
    )
    _write_measurement_fixture(older)
    _write_measurement_fixture(newest)

    settings = DashboardBuildSettings(
        package_root=package,
        corridor_results_root=corridor_root,
        mapmatching_product_root=package / "mapping",
        qvdf_report_root=package / "projection",
        corridor_measurement_root=newest,
        assignment_root=package / "assignment",
        ritis_15min_path=package / "observed.csv",
        output_root=package / "outputs" / "dashboard",
    )

    assert settings.corridor_measurement_root == newest.resolve()
