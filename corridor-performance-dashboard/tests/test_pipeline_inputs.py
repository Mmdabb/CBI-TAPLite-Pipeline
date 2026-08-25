import pandas as pd

from corridor_measurement.pipeline import (
    _complete_period_corridor_grid,
    build_membership_audit,
    load_general_purpose_tmc_codes,
    load_link_performance,
    load_period_mapping,
)


def test_period_mapping_keeps_only_frozen_node_pair_winner(tmp_path):
    mapping_dir = tmp_path / "am-product"
    mapping_dir.mkdir()
    pd.DataFrame(
        {
            "tmc": ["WINNER", "LOSER"],
            "road": ["I-66", "I-66"],
            "direction": ["EASTBOUND", "EASTBOUND"],
            "road_order": [1, 2],
            "sequence": [1, 1],
            "link_id": [100, 100],
            "from_node_id": [10, 10],
            "to_node_id": [20, 20],
            "length_mi": [1.0, 1.0],
            "facility_class": ["gp", "gp"],
        }
    ).to_csv(mapping_dir / "full_tmc_to_link.csv", index=False)
    pd.DataFrame(
        {
            "tmc": ["WINNER", "LOSER"],
            "route_link_count": [1, 1],
            "confidence": [90.0, 80.0],
            "status": ["matched", "matched"],
        }
    ).to_csv(mapping_dir / "full_route_match_summary.csv", index=False)
    canonical = tmp_path / "canonical_node_pair_tmc.csv"
    pd.DataFrame(
        {
            "tmc": ["WINNER"],
            "from_node_id": [10],
            "to_node_id": [20],
            "selected_for_node_pair_lookup": [True],
        }
    ).to_csv(canonical, index=False)

    mapping, _ = load_period_mapping(
        mapping_dir,
        canonical_node_pair_mapping=canonical,
        strict_qa_only=False,
        strict_qa_statuses=("matched",),
    )

    assert mapping["tmc"].tolist() == ["WINNER"]
    assert mapping["canonical_node_pair_winner"].all()


def test_visualization_membership_is_strictly_general_purpose(tmp_path):
    products = {"am": "am-product", "md": "md-product", "pm": "pm-product"}
    rows_by_period = {
        "am": [
            {"tmc": "GP-1", "facility_class": "gp"},
            {"tmc": "MANAGED-1", "facility_class": "managed"},
            {"tmc": "CONFLICT", "facility_class": "gp"},
            {"tmc": "BLANK", "facility_class": ""},
        ],
        "md": [
            {"tmc": "GP-1", "facility_class": "GP"},
            {"tmc": "MANAGED-1", "facility_class": "managed"},
            {"tmc": "CONFLICT", "facility_class": "managed"},
        ],
        "pm": [
            {"tmc": "GP-1", "facility_class": "gp"},
            {"tmc": "MANAGED-1", "facility_class": "managed"},
        ],
    }
    for period, product in products.items():
        destination = tmp_path / product / "full_tmc_to_link.csv"
        destination.parent.mkdir(parents=True)
        pd.DataFrame(rows_by_period[period]).to_csv(destination, index=False)

    eligible, audit = load_general_purpose_tmc_codes(tmp_path, products)

    assert eligible == {"GP-1"}
    assert audit["managed_tmc_count"] == 2
    assert audit["unclassified_tmc_count"] == 1
    assert audit["conflicting_tmc_count"] == 1


def test_link_performance_retains_speed_and_period_metrics(tmp_path):
    source = tmp_path / "link_performance.csv"
    pd.DataFrame(
        [
            {
                "link_id": "101",
                "volume": 1200.0,
                "doc": 0.75,
                "P": 1.25,
                "spd_mph_06:00": 42.0,
            }
        ]
    ).to_csv(source, index=False)
    frame, minutes = load_link_performance(source)
    assert minutes == {"spd_mph_06:00": 360}
    assert frame.loc["101", "volume"] == 1200.0
    assert frame.loc["101", "doc"] == 0.75
    assert frame.loc["101", "P"] == 1.25


def test_membership_audit_reconciles_dashboard_and_observed_counts():
    profiles = pd.DataFrame(
        {
            "corridor": ["I66_EB", "I66_EB"],
            "tmc_code": ["a", "b"],
        }
    )
    reference = pd.DataFrame(
        {
            "corridor": ["I66_EB", "I66_EB"],
            "tmc_code": ["a", "b"],
            "network_link_id": ["1", "2"],
            "road_order": [1.0, 2.0],
        }
    )
    audit = build_membership_audit(profiles, reference).iloc[0]
    assert audit["dashboard_link_reference_tmc_count"] == 2
    assert audit["observed_profile_tmc_count"] == 2
    assert bool(audit["membership_counts_match"])


def test_complete_period_grid_retains_corridor_with_no_eligible_links():
    aligned = pd.DataFrame(
        {
            "corridor": ["OPEN"],
            "t_min": [900],
            "model_speed_mph": [42.0],
        }
    )
    reference = pd.DataFrame(
        {
            "corridor": ["OPEN", "CLOSED"],
            "tmc_code": ["a", "b"],
        }
    )

    complete = _complete_period_corridor_grid(
        aligned,
        reference,
        start_min=900,
        end_min=930,
        interval_minutes=15,
    )

    assert len(complete) == 4
    closed = complete[complete["corridor"].eq("CLOSED")]
    assert closed["t_min"].tolist() == [900, 915]
    assert closed["model_speed_mph"].isna().all()
