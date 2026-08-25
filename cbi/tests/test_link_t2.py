from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from congestion_boundary_mapping.build_link_t2 import (
    build_assignments,
    combine_average_and_daily_representatives,
    rank_link_tmcs,
    select_daily_probe_representatives,
    select_episode_representatives,
)
from congestion_boundary_mapping.propagate_t2_by_vdf import resolve_t2_est
from congestion_boundary_mapping.hybrid_t2 import (
    HYBRID_COLUMNS,
    apply_hybrid_t2,
    build_hybrid_assignments,
)


def _write_canonical_winners(
    tmp_path: Path, winners: list[tuple[int, str, float]]
) -> Path:
    path = tmp_path / "canonical_node_pair_tmc.csv"
    pd.DataFrame(
        [
            {
                "link_id": link_id,
                "tmc": tmc,
                "distance_to_tmc_ft": distance,
                "node_pair_tmc_rank": 1,
                "selected_for_node_pair_lookup": True,
                "node_pair_tmc_ranking_basis": "highest_composite_match_score",
            }
            for link_id, tmc, distance in winners
        ]
    ).to_csv(path, index=False)
    return path


def test_representative_episode_uses_speed_duration_then_t2():
    candidates = pd.DataFrame(
        [
            {
                "episode_id": "slowest",
                "tmc_code": "lowest",
                "period": "AM",
                "t0_hour": 7.0,
                "t2_hour": 8.5,
                "t3_hour": 9.0,
                "min_speed_mph": 19.0,
                "P_hr": 1.0,
            },
            {
                "episode_id": "longer-later",
                "tmc_code": "tie",
                "period": "PM",
                "t0_hour": 16.0,
                "t2_hour": 17.0,
                "t3_hour": 18.0,
                "min_speed_mph": 20.0,
                "P_hr": 3.0,
            },
            {
                "episode_id": "shorter",
                "tmc_code": "tie",
                "period": "PM",
                "t0_hour": 15.5,
                "t2_hour": 16.0,
                "t3_hour": 17.5,
                "min_speed_mph": 20.0,
                "P_hr": 2.0,
            },
            {
                "episode_id": "longer-earlier",
                "tmc_code": "tie",
                "period": "PM",
                "t0_hour": 15.75,
                "t2_hour": 16.5,
                "t3_hour": 18.75,
                "min_speed_mph": 20.0,
                "P_hr": 3.0,
            },
        ]
    )
    candidates["corridor_output"] = "corridor"
    candidates["is_clean_valid_episode"] = True

    _, representatives = select_episode_representatives(candidates)
    selected = representatives.set_index(["tmc", "period"])["episode_id"]

    assert selected[("lowest", "AM")] == "slowest"
    assert selected[("tie", "PM")] == "longer-earlier"


def test_closest_tmc_without_congestion_is_not_replaced_by_alternate(tmp_path):
    mappings = pd.DataFrame(
        [
            {
                "link_id": 1,
                "tmc": "near",
                "period": "AM",
                "distance_to_tmc_ft": 10.0,
                "map_row_number": 1,
                "_global_occurrence": 1,
                "period_is_open": True,
                "map_source": "am",
            },
            {
                "link_id": 1,
                "tmc": "far",
                "period": "AM",
                "distance_to_tmc_ft": 20.0,
                "map_row_number": 2,
                "_global_occurrence": 2,
                "period_is_open": True,
                "map_source": "am",
            },
            {
                "link_id": 1,
                "tmc": "near",
                "period": "MD",
                "distance_to_tmc_ft": 10.0,
                "map_row_number": 1,
                "_global_occurrence": 10_000_001,
                "period_is_open": True,
                "map_source": "md",
            },
            {
                "link_id": 1,
                "tmc": "far",
                "period": "MD",
                "distance_to_tmc_ft": 20.0,
                "map_row_number": 2,
                "_global_occurrence": 10_000_002,
                "period_is_open": True,
                "map_source": "md",
            },
        ]
    )
    representatives = pd.DataFrame(
        [
            {
                "tmc": "near",
                "period": "AM",
                "episode_id": "near-am",
                "t2_hour": 8.0,
            },
            {
                "tmc": "far",
                "period": "AM",
                "episode_id": "far-am",
                "t2_hour": 8.5,
            },
            {
                "tmc": "far",
                "period": "MD",
                "episode_id": "far-md",
                "t2_hour": 12.0,
            },
        ]
    )
    network = pd.DataFrame([{"link_id": 1, "LINKID": "regional-1"}])
    rank = rank_link_tmcs(
        mappings, _write_canonical_winners(tmp_path, [(1, "near", 10.0)])
    )
    long, _, _ = build_assignments(network, mappings, rank, representatives)

    am = long[long["period"].eq("AM")].iloc[0]
    md = long[long["period"].eq("MD")].iloc[0]
    pm = long[long["period"].eq("PM")].iloc[0]
    assert am["selected_tmc"] == "near"
    assert am["t2_hour"] == 8.0
    assert pd.isna(md["selected_tmc"])
    assert pd.isna(md["t2_hour"])
    assert md["observed_no_congestion_protected"]
    assert (
        md["selection_reason"]
        == "protected_primary_tmc_no_average_weekday_congestion"
    )
    assert pd.isna(pm["t2_hour"])


def test_one_tmc_to_many_links_copies_t2_and_keeps_missing_period_blank(tmp_path):
    mappings = pd.DataFrame(
        [
            {
                "link_id": link_id,
                "tmc": "shared",
                "period": "AM",
                "distance_to_tmc_ft": distance,
                "map_row_number": row_number,
                "_global_occurrence": row_number,
                "period_is_open": True,
                "map_source": "am",
            }
            for link_id, distance, row_number in ((1, 5.0, 1), (2, 8.0, 2))
        ]
    )
    representatives = pd.DataFrame(
        [
            {
                "tmc": "shared",
                "period": "AM",
                "episode_id": "shared-am",
                "t2_hour": 7.75,
            }
        ]
    )
    network = pd.DataFrame(
        [
            {"link_id": 1, "LINKID": "regional-1"},
            {"link_id": 2, "LINKID": "regional-2"},
        ]
    )

    rank = rank_link_tmcs(
        mappings,
        _write_canonical_winners(
            tmp_path, [(1, "shared", 5.0), (2, "shared", 8.0)]
        ),
    )
    long, _, _ = build_assignments(network, mappings, rank, representatives)

    am = long[long["period"].eq("AM")].sort_values("link_id")
    assert am["selected_tmc"].tolist() == ["shared", "shared"]
    assert am["t2_hour"].tolist() == [7.75, 7.75]
    assert long[long["period"].eq("MD")]["t2_hour"].isna().all()
    assert long[long["period"].eq("PM")]["t2_hour"].isna().all()


def test_missing_distance_uses_frozen_winner(tmp_path):
    mappings = pd.DataFrame(
        [
            {
                "link_id": 1,
                "tmc": tmc,
                "period": "AM",
                "distance_to_tmc_ft": float("nan"),
                "map_row_number": row_number,
                "_global_occurrence": row_number,
                "period_is_open": True,
                "map_source": "am",
            }
            for tmc, row_number in (("first", 1), ("second", 2))
        ]
    )
    representatives = pd.DataFrame(
        [
            {
                "tmc": tmc,
                "period": "AM",
                "episode_id": f"{tmc}-am",
                "t2_hour": t2,
            }
            for tmc, t2 in (("first", 8.0), ("second", 8.5))
        ]
    )
    network = pd.DataFrame([{"link_id": 1, "LINKID": "regional-1"}])

    rank = rank_link_tmcs(
        mappings, _write_canonical_winners(tmp_path, [(1, "first", np.nan)])
    )
    long, _, _ = build_assignments(network, mappings, rank, representatives)

    first_rank = rank.sort_values("node_pair_tmc_rank").iloc[0]
    am = long[long["period"].eq("AM")].iloc[0]
    assert first_rank["tmc"] == "first"
    assert first_rank["ranking_basis"] == "highest_composite_match_score"
    assert am["selected_tmc"] == "first"


def test_period_closed_primary_does_not_use_open_alternate(tmp_path):
    mappings = pd.DataFrame(
        [
            {
                "link_id": 1,
                "tmc": "near",
                "period": "MD",
                "distance_to_tmc_ft": 5.0,
                "map_row_number": 1,
                "_global_occurrence": 10_000_001,
                "period_is_open": False,
                "map_source": "md",
            },
            {
                "link_id": 1,
                "tmc": "far",
                "period": "MD",
                "distance_to_tmc_ft": 20.0,
                "map_row_number": 2,
                "_global_occurrence": 10_000_002,
                "period_is_open": True,
                "map_source": "md",
            },
        ]
    )
    representatives = pd.DataFrame(
        [
            {
                "tmc": "near",
                "period": "MD",
                "episode_id": "near-md",
                "t2_hour": 12.0,
            },
            {
                "tmc": "far",
                "period": "MD",
                "episode_id": "far-md",
                "t2_hour": 12.5,
            },
        ]
    )
    network = pd.DataFrame([{"link_id": 1, "LINKID": "regional-1"}])

    rank = rank_link_tmcs(
        mappings, _write_canonical_winners(tmp_path, [(1, "near", 5.0)])
    )
    long, _, audit = build_assignments(
        network, mappings, rank, representatives
    )

    md = long[long["period"].eq("MD")].iloc[0]
    assert pd.isna(md["selected_tmc"])
    assert pd.isna(md["t2_hour"])
    assert not audit["selected"].any()


def test_daily_probe_selects_one_episode_per_day_then_averages_t2():
    daily = pd.DataFrame(
        [
            {
                "episode_id": "day1-early",
                "tmc_code": "tmc-a",
                "period": "AM",
                "date": "2026-01-05",
                "t0_hour": 6.5,
                "t2_hour": 7.0,
                "t3_hour": 8.0,
                "min_speed_mph": 30.0,
                "P_hr": 1.5,
                "corridor_output": "corridor",
            },
            {
                "episode_id": "day1-slowest",
                "tmc_code": "tmc-a",
                "period": "AM",
                "date": "2026-01-05",
                "t0_hour": 7.0,
                "t2_hour": 8.0,
                "t3_hour": 9.0,
                "min_speed_mph": 20.0,
                "P_hr": 2.0,
                "corridor_output": "corridor",
            },
            {
                "episode_id": "day2",
                "tmc_code": "tmc-a",
                "period": "AM",
                "date": "2026-01-06",
                "t0_hour": 6.5,
                "t2_hour": 7.0,
                "t3_hour": 8.0,
                "min_speed_mph": 25.0,
                "P_hr": 1.5,
                "corridor_output": "corridor",
            },
        ]
    )
    daily["is_clean_valid_episode"] = True
    eligible = pd.DataFrame([{"tmc": "tmc-a", "period": "AM"}])

    _, by_day, summary = select_daily_probe_representatives(
        daily,
        eligible,
    )

    assert by_day["episode_id"].tolist() == ["day1-slowest", "day2"]
    assert summary.loc[0, "t2_hour"] == 7.5
    assert summary.loc[0, "t0_hour"] == 6.75
    assert summary.loc[0, "t3_hour"] == 8.5
    assert summary.loc[0, "daily_probe_day_count"] == 2
    assert summary.loc[0, "daily_probe_episode_count"] == 3
    assert summary.loc[0, "t2_source_method"] == "daily_probe_mean"


def test_rejected_episode_is_refused_by_t2_selection():
    candidates = pd.DataFrame(
        [
            {
                "episode_id": "rejected",
                "tmc_code": "tmc-a",
                "period": "AM",
                "t0_hour": 7.0,
                "t2_hour": 8.0,
                "t3_hour": 9.0,
                "min_speed_mph": 20.0,
                "P_hr": 2.0,
                "corridor_output": "corridor",
                "is_clean_valid_episode": False,
            }
        ]
    )

    try:
        select_episode_representatives(candidates)
    except ValueError as exc:
        assert "not accepted" in str(exc)
    else:
        raise AssertionError("Rejected episode was allowed into T2 selection.")


def test_vdf_postprocessing_carries_original_and_fills_only_missing_t2():
    means = {"code-a": 7.5}

    assert resolve_t2_est(7.0, "code-a", means) == (
        7.0,
        "carried_original",
    )
    assert resolve_t2_est(None, "code-a", means) == (
        7.5,
        "propagated_vdf_mean",
    )
    assert resolve_t2_est(None, "code-b", means) == (
        None,
        "unmatched_blank",
    )


def test_average_weekday_is_authoritative_and_daily_probe_stays_audit_only():
    average = pd.DataFrame(
        [
            {
                "tmc": "tmc-a",
                "period": "AM",
                "t2_hour": 8.25,
                "episode_id": "average-a",
            }
        ]
    )
    daily_summary = pd.DataFrame(
        [
            {
                "tmc": "tmc-a",
                "period": "AM",
                "t2_hour": 7.5,
                "episode_id": "daily-a",
                "t2_source_method": "daily_probe_mean",
            },
            {
                "tmc": "tmc-b",
                "period": "MD",
                "t2_hour": 12.0,
                "episode_id": "daily-b",
                "t2_source_method": "daily_probe_mean",
            },
        ]
    )

    _, audited_daily, combined = combine_average_and_daily_representatives(
        average,
        daily_summary,
    )
    selected = combined.set_index(["tmc", "period"])

    assert selected.loc[("tmc-a", "AM"), "t2_hour"] == 8.25
    assert selected.loc[("tmc-a", "AM"), "t2_source_method"] == "average_weekday"
    assert ("tmc-b", "MD") not in selected.index
    assert audited_daily.set_index(["tmc", "period"]).loc[
        ("tmc-a", "AM"), "fallback_used"
    ] == False
    daily_b = audited_daily.set_index(["tmc", "period"]).loc[
        ("tmc-b", "MD")
    ]
    assert daily_b["fallback_used"] == False
    assert daily_b["suppressed_no_average_weekday_congestion"]


def test_selected_episode_boundaries_follow_t2_to_link_outputs(tmp_path):
    mappings = pd.DataFrame(
        [
            {
                "link_id": 1,
                "tmc": "tmc-a",
                "period": "AM",
                "distance_to_tmc_ft": 5.0,
                "map_row_number": 1,
                "_global_occurrence": 1,
                "period_is_open": True,
                "map_source": "am",
            }
        ]
    )
    representatives = pd.DataFrame(
        [
            {
                "tmc": "tmc-a",
                "period": "AM",
                "episode_id": "episode-a",
                "t0_hour": 6.5,
                "t2_hour": 7.5,
                "t3_hour": 8.75,
            }
        ]
    )
    network = pd.DataFrame([{"link_id": 1, "LINKID": "regional-1"}])

    rank = rank_link_tmcs(
        mappings, _write_canonical_winners(tmp_path, [(1, "tmc-a", 5.0)])
    )
    long, wide, _ = build_assignments(
        network, mappings, rank, representatives
    )

    am = long[long["period"].eq("AM")].iloc[0]
    assert (am["t0_hour"], am["t2_hour"], am["t3_hour"]) == (
        6.5,
        7.5,
        8.75,
    )
    assert (
        wide.loc[0, "t0_am_hour"],
        wide.loc[0, "t2_am_hour"],
        wide.loc[0, "t3_am_hour"],
    ) == (6.5, 7.5, 8.75)


def test_hybrid_precedence_is_direct_then_spatial_then_class():
    integration = pd.DataFrame(
        {
            "link_id": [1, 2, 3, 4, 5],
            "t0_hour": [6.5, np.nan, np.nan, np.nan, np.nan],
            "t2_hour": [7.0, np.nan, np.nan, np.nan, np.nan],
            "t3_hour": [8.0, np.nan, np.nan, np.nan, np.nan],
            "t2_est": [7.0, 7.8, 7.9, 8.0, np.nan],
            "t2_source_method": [
                "average_weekday",
                "",
                "",
                "",
                "",
            ],
        }
    )
    spatial = pd.DataFrame(
        {
            "period": ["AM", "AM", "AM"],
            "link_id": [1, 2, 3],
            "t0_hour": [6.25, 6.75, np.nan],
            "t2_hour": [7.4, 7.2, 7.3],
            "t3_hour": [8.25, 8.1, np.nan],
            "assignment_tier": [
                "B_bracketed",
                "A_direct",
                "B_bracketed",
            ],
            "assignment_method": [
                "linear_t2_interpolation",
                "direct_mapped_tmc",
                "linear_t2_interpolation",
            ],
            "assignment_confidence": ["medium", "high", "medium"],
        }
    )

    hybrid = build_hybrid_assignments(integration, spatial, "AM").set_index(
        "link_id"
    )

    assert hybrid.loc[1, "t2_hybrid_hour"] == 7.0
    assert hybrid.loc[1, "t2_hybrid_source"] == "direct"
    assert hybrid.loc[2, "t2_hybrid_hour"] == 7.2
    assert hybrid.loc[2, "t2_hybrid_source"] == "direct"
    assert hybrid.loc[2, "t0_hybrid_hour"] == 6.75
    assert hybrid.loc[2, "t3_hybrid_hour"] == 8.1
    assert hybrid.loc[3, "t2_hybrid_hour"] == 7.3
    assert hybrid.loc[3, "t2_hybrid_source"] == "spatial"
    assert hybrid.loc[4, "t2_hybrid_hour"] == 8.0
    assert hybrid.loc[4, "t2_hybrid_source"] == "class"
    assert pd.isna(hybrid.loc[5, "t2_hybrid_hour"])
    assert hybrid.loc[5, "t2_hybrid_source"] == "unassigned"


def test_hybrid_protects_observed_no_congestion_from_all_fallbacks():
    integration = pd.DataFrame(
        {
            "link_id": [6],
            "t0_hour": [np.nan],
            "t2_hour": [np.nan],
            "t3_hour": [np.nan],
            "t2_est": [8.0],
            "t2_source_method": [""],
            "t2_observation_status": [
                "observed_no_average_weekday_congestion"
            ],
            "t2_observed_no_congestion_protected": [True],
        }
    )
    spatial = pd.DataFrame(
        {
            "period": ["AM"],
            "link_id": [6],
            "t0_hour": [6.5],
            "t2_hour": [7.5],
            "t3_hour": [8.5],
            "assignment_tier": ["A_direct"],
            "assignment_method": ["direct_mapped_tmc"],
            "assignment_confidence": ["high"],
        }
    )

    row = build_hybrid_assignments(integration, spatial, "AM").iloc[0]

    assert row["t2_hybrid_source"] == "observed_no_congestion"
    assert row["t2_hybrid_precedence_rank"] == 0
    assert pd.isna(row["t0_hybrid_hour"])
    assert pd.isna(row["t2_hybrid_hour"])
    assert pd.isna(row["t3_hybrid_hour"])


def test_integrated_hybrid_stage_is_idempotent(tmp_path: Path):
    integration_root = tmp_path / "link-t2"
    spatial_rows = []
    periods = {
        "AM": ("am", 7.0, 7.4, 8.0),
        "MD": ("md", 11.0, 12.0, 13.0),
        "PM": ("pm", 16.0, 17.0, 18.0),
    }
    for period, (directory, direct_t2, spatial_t2, class_t2) in periods.items():
        period_dir = integration_root / "period_link_files" / directory
        period_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "link_id": [1, 2, 3],
                "native_value": ["kept-a", "kept-b", "kept-c"],
                "t0_hour": [direct_t2 - 0.5, np.nan, np.nan],
                "t2_hour": [direct_t2, np.nan, np.nan],
                "t3_hour": [direct_t2 + 0.5, np.nan, np.nan],
                "t2_source_method": ["average_weekday", "", ""],
                "t2_est": [direct_t2, direct_t2 + 0.2, class_t2],
            }
        ).to_csv(period_dir / "link.csv", index=False)
        spatial_rows.append(
            {
                "period": period,
                "link_id": 2,
                "t0_hour": np.nan,
                "t2_hour": spatial_t2,
                "t3_hour": np.nan,
                "assignment_tier": "B_bracketed",
                "assignment_method": "linear_t2_interpolation",
                "assignment_confidence": "medium",
            }
        )
    spatial_path = tmp_path / "expanded_link_t2.csv"
    pd.DataFrame(spatial_rows).to_csv(spatial_path, index=False)

    first = apply_hybrid_t2(
        integration_root,
        spatial_path,
        workers=1,
        update_parent_manifest=False,
    )
    second = apply_hybrid_t2(
        integration_root,
        spatial_path,
        workers=1,
        update_parent_manifest=False,
    )

    assert first["precedence"] == ["direct", "spatial", "class"]
    assert second["status"] == "PASS"
    for period, (directory, _, spatial_t2, class_t2) in periods.items():
        path = integration_root / "period_link_files" / directory / "link.csv"
        header = pd.read_csv(path, nrows=0).columns.tolist()
        assert all(header.count(column) == 1 for column in HYBRID_COLUMNS)
        frame = pd.read_csv(path).set_index("link_id")
        assert frame["native_value"].tolist() == [
            "kept-a",
            "kept-b",
            "kept-c",
        ]
        assert frame["t2_hybrid_source"].tolist() == [
            "direct",
            "spatial",
            "class",
        ]
        assert frame.loc[2, "t2_hybrid_hour"] == spatial_t2
        assert frame.loc[3, "t2_hybrid_hour"] == class_t2
    coverage = pd.read_csv(
        integration_root / "hybrid_t2_coverage_summary.csv"
    )
    assert coverage.loc[coverage["period"].eq("ALL"), "coverage_pct"].iloc[0] == 100.0
    manifest = json.loads(
        (integration_root / "hybrid_t2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["final_t2_column"] == "t2_hybrid_hour"
