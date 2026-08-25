from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbi.network_qvdf import (
    OBSERVED_LINK_PLF_DTYPE,
    OBSERVED_LINK_SPEED_BOUNDARY_DTYPE,
    OBSERVED_LINK_T2_DTYPE,
    RESOURCE_COLUMNS,
    build_observed_link_plf_overrides,
    build_observed_link_speed_boundaries,
    build_observed_link_t2_lookup,
    build_qvdf_resources,
    build_resource_from_episodes,
    load_observed_primary_links,
    load_nvta_network_link_types,
)


def _episodes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    period_t2 = {"AM": 7.0, "MD": 12.0, "PM": 17.0}
    for period in ("AM", "MD", "PM"):
        for index, ratio in enumerate((0.6, 0.8, 1.0, 1.2), start=1):
            duration = 0.7 * ratio**1.3
            cutoff = 50.0
            minimum = cutoff / (1.0 + 0.25 * duration**1.1)
            t2_hour = period_t2[period] + index / 100.0
            rows.append(
                {
                    "period": period,
                    "network_link_id": 10,
                    "network_link_type": 1,
                    "tmc_code": "TMC-10",
                    "corridor": "X",
                    "episode_id": f"{period}-10-{index}",
                    "t0_hour": t2_hour - duration / 2.0,
                    "t2_hour": t2_hour,
                    "t3_hour": t2_hour + duration / 2.0,
                    "demand_capacity_ratio": ratio,
                    "P_hr": duration,
                    "min_speed_mph": minimum,
                    "threshold_used": cutoff,
                    "qdf": 0.5 + index / 100.0,
                }
            )
        rows.append(
            {
                "period": period,
                "network_link_id": 20,
                "network_link_type": 2,
                "tmc_code": "TMC-20",
                "corridor": "X",
                "episode_id": f"{period}-20",
                "t0_hour": period_t2[period] - 0.3,
                "t2_hour": period_t2[period],
                "t3_hour": period_t2[period] + 0.3,
                "demand_capacity_ratio": 0.9,
                "P_hr": 0.6,
                "min_speed_mph": 42.0,
                "threshold_used": 50.0,
                "qdf": 0.55,
            }
        )
    return pd.DataFrame(rows)


def _write_network(root: Path, *, conflict: bool = False) -> Path:
    for period in ("am", "md", "pm"):
        folder = root / period
        folder.mkdir(parents=True)
        types = [1, 2, 3]
        if conflict and period == "pm":
            types[0] = 9
        frame = pd.DataFrame({"link_id": [10, 20, 30], "link_type": types})
        if period == "md":
            frame = pd.concat(
                [frame, pd.DataFrame({"link_id": [40], "link_type": [3]})],
                ignore_index=True,
            )
        frame.to_csv(folder / "link.csv", index=False)
    return root


def _write_weekday_profile(root: Path) -> Path:
    path = root / "X" / "03-profiles" / "average_weekday_profile.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    speeds = {
        "TMC-10": (61.0, 49.0, 57.0, 52.0),
        "TMC-20": (55.0, 51.0, 54.0, 53.0),
    }
    for tmc_code, values in speeds.items():
        for minute, speed in zip((360, 540, 900, 1140), values):
            rows.append(
                {
                    "corridor": "X",
                    "tmc_code": tmc_code,
                    "t_min": minute,
                    "avg_weekday_speed_mph": speed,
                    "avg_weekday_speed_mph_pre_qc": speed,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_frozen_mapping(root: Path) -> Path:
    rows: list[dict[str, object]] = []
    for source in root.glob("**/01-input-and-qc/link_reference.csv"):
        reference = pd.read_csv(source)
        for row in reference.itertuples(index=False):
            rows.append(
                {
                    "tmc": row.tmc_code,
                    "link_id": row.network_link_id,
                    "from_node_id": row.network_from_node_id,
                    "to_node_id": row.network_to_node_id,
                    "distance_to_tmc_ft": row.network_match_distance_ft,
                    "composite_match_score": 100.0,
                    "node_pair_tmc_rank": 1,
                    "selected_for_node_pair_lookup": True,
                }
            )
    path = root / "shared" / "network-mapping" / "canonical_node_pair_tmc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_network_qvdf_resource_has_taplite_schema_and_complete_network_codes() -> None:
    resource, audit = build_resource_from_episodes(
        _episodes(), minimum_episodes=3, vdf_codes=[1, 2, 3]
    )
    assert resource.columns.tolist() == RESOURCE_COLUMNS
    assert resource["vdf_code"].astype(str).tolist() == ["1", "2", "3", "all"]
    assert resource.iloc[-1]["vdf_code"] == "all"
    assert np.isfinite(
        resource.filter(regex=r"^QVDF_").to_numpy(dtype=float)
    ).all()
    for code in ("2", "3"):
        rows = audit[audit["vdf_code"].astype(str).eq(code)]
        assert set(rows["calibration_source"]) == {"network_all_fallback"}


def test_qvdf_plf_uses_nvta_period_durations() -> None:
    resource, _ = build_resource_from_episodes(_episodes(), minimum_episodes=3)
    row = resource[resource["vdf_code"].astype(str).eq("all")].iloc[0]
    assert np.isclose(row["QVDF_plf1"], 1.0 / (row["QVDF_qdf1"] * 3.0))
    assert np.isclose(row["QVDF_plf2"], 1.0 / (row["QVDF_qdf2"] * 6.0))
    assert np.isclose(row["QVDF_plf3"], 1.0 / (row["QVDF_qdf3"] * 4.0))


def test_nvta_network_consensus_accepts_period_presence_differences(tmp_path: Path) -> None:
    lookup, audit, metadata = load_nvta_network_link_types(
        _write_network(tmp_path / "network")
    )
    assert lookup.loc[10] == "1"
    assert metadata["cross_period_type_conflicts"] == 0
    assert metadata["partial_period_link_ids"] == 1
    extra = audit[audit["link_id"].eq(40)].iloc[0]
    assert extra["consensus_status"] == "consistent_partial_period_presence"


def test_observed_links_expand_one_frozen_tmc_to_all_winning_node_pairs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    reference = run_root / "corridors" / "X" / "01-input-and-qc" / "link_reference.csv"
    reference.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "A",
                "road_order": 1,
                "network_link_id": 10,
                "network_from_node_id": 100,
                "network_to_node_id": 101,
                "network_match_distance_ft": 1.0,
            }
        ]
    ).to_csv(reference, index=False)

    mapping = run_root / "shared" / "network-mapping" / "canonical_node_pair_tmc.csv"
    mapping.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "tmc": "A",
                "link_id": 10,
                "from_node_id": 100,
                "to_node_id": 101,
                "distance_to_tmc_ft": 1.0,
                "composite_match_score": 0.95,
                "node_pair_tmc_rank": 1,
                "selected_for_node_pair_lookup": True,
            },
            {
                "tmc": "A",
                "link_id": 11,
                "from_node_id": 101,
                "to_node_id": 102,
                "distance_to_tmc_ft": 2.0,
                "composite_match_score": 0.90,
                "node_pair_tmc_rank": 1,
                "selected_for_node_pair_lookup": True,
            },
        ]
    ).to_csv(mapping, index=False)

    links = load_observed_primary_links(run_root)

    assert links["tmc_code"].tolist() == ["A", "A"]
    assert links["network_link_id"].tolist() == [10, 11]
    assert list(
        links[["network_from_node_id", "network_to_node_id"]].itertuples(
            index=False, name=None
        )
    ) == [(100, 101), (101, 102)]
    assert links["network_selected_for_node_pair_lookup"].all()
    assert links["network_node_pair_tmc_rank"].eq(1).all()


def test_end_to_end_builder_writes_reviewable_products(tmp_path: Path) -> None:
    run = tmp_path / "run" / "corridors" / "X" / "05-episode-filtering"
    run.mkdir(parents=True)
    episodes = _episodes().drop(columns="network_link_type")
    episodes.to_csv(run / "daily_episodes_accepted.csv", index=False)
    episodes.to_csv(run / "average_weekday_episodes_accepted.csv", index=False)
    reference = run.parents[0] / "01-input-and-qc" / "link_reference.csv"
    reference.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "TMC-10",
                "network_link_id": 10,
                "network_from_node_id": 100,
                "network_to_node_id": 101,
                "network_match_distance_ft": 1.0,
            },
            {
                "corridor": "X",
                "tmc_code": "TMC-20",
                "network_link_id": 20,
                "network_from_node_id": 200,
                "network_to_node_id": 201,
                "network_match_distance_ft": 2.0,
            },
        ]
    ).to_csv(reference, index=False)
    _write_frozen_mapping(tmp_path / "run")
    _write_weekday_profile(tmp_path / "run" / "corridors")
    output = tmp_path / "network-qvdf"
    manifest = build_qvdf_resources(
        tmp_path / "run",
        output,
        network_root=_write_network(tmp_path / "network"),
        minimum_episodes=3,
    )
    assert manifest["authoritative_basis"] == "daily"
    assert manifest["period_durations_hours"] == {"AM": 3.0, "MD": 6.0, "PM": 4.0}
    assert (output / "daily" / "link_qvdf.csv").is_file()
    assert (output / "average-weekday" / "link_qvdf.csv").is_file()
    assert (output / "network_link_type_consensus.csv").is_file()
    assert (
        output / "observed-link-plf" / "observed_link_plf_overrides.npy"
    ).is_file()
    assert (
        output
        / "observed-link-speed-boundaries"
        / "observed_link_speed_boundaries.npy"
    ).is_file()
    assert (output / "observed-link-t2" / "observed_link_t2.npy").is_file()
    assert (output / "review" / "REVIEW_SUMMARY.md").is_file()
    assert (output / "review" / "direct_fit_quality_flags.csv").is_file()
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["network"]["partial_period_link_ids"] == 1


def test_observed_link_plf_uses_weekday_qdf_and_neutral_no_congestion(
    tmp_path: Path,
) -> None:
    cbi_root = tmp_path / "cbi"
    reference = cbi_root / "X" / "01-input-and-qc" / "link_reference.csv"
    reference.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "A",
                "network_link_id": 1,
                "network_from_node_id": 10,
                "network_to_node_id": 20,
                "network_match_distance_ft": 3.0,
            },
            {
                "corridor": "X",
                "tmc_code": "B",
                "network_link_id": 2,
                "network_from_node_id": 30,
                "network_to_node_id": 40,
                "network_match_distance_ft": 4.0,
            },
        ]
    ).to_csv(reference, index=False)
    _write_frozen_mapping(cbi_root)
    episodes = pd.DataFrame(
        [
            {"corridor": "X", "tmc_code": "A", "period": "AM", "qdf": 0.5},
            {"corridor": "X", "tmc_code": "A", "period": "AM", "qdf": 0.7},
        ]
    )
    output = tmp_path / "observed-plf"
    metadata = build_observed_link_plf_overrides(cbi_root, episodes, output)
    lookup = np.load(
        output / "observed_link_plf_overrides.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert lookup.dtype == OBSERVED_LINK_PLF_DTYPE
    assert metadata["unique_node_pair_rows"] == 2
    assert metadata["unit"] == "dimensionless"
    row_a = lookup[lookup["from_node_id"] == 10][0]
    row_b = lookup[lookup["from_node_id"] == 30][0]
    assert np.isclose(row_a["plf_am"], 1.0 / (0.6 * 3.0))
    assert np.isclose(row_a["plf_md"], 1.0)
    assert np.isclose(row_a["plf_pm"], 1.0)
    assert np.isclose(row_b["plf_am"], 1.0)
    assert np.isclose(row_b["plf_md"], 1.0)
    assert np.isclose(row_b["plf_pm"], 1.0)


def test_observed_link_speed_boundaries_use_exact_period_edges(
    tmp_path: Path,
) -> None:
    cbi_root = tmp_path / "cbi"
    reference = cbi_root / "X" / "01-input-and-qc" / "link_reference.csv"
    reference.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "TMC-10",
                "network_link_id": 1,
                "network_from_node_id": 10,
                "network_to_node_id": 20,
                "network_match_distance_ft": 3.0,
            },
            {
                "corridor": "X",
                "tmc_code": "TMC-20",
                "network_link_id": 2,
                "network_from_node_id": 30,
                "network_to_node_id": 40,
                "network_match_distance_ft": 4.0,
            },
        ]
    ).to_csv(reference, index=False)
    _write_frozen_mapping(cbi_root)
    _write_weekday_profile(cbi_root)
    profile_path = (
        cbi_root / "X" / "03-profiles" / "average_weekday_profile.csv"
    )
    profile = pd.read_csv(profile_path)
    profile.loc[
        profile["tmc_code"].eq("TMC-20") & profile["t_min"].eq(900),
        "avg_weekday_speed_mph",
    ] = np.nan
    profile.to_csv(profile_path, index=False)

    output = tmp_path / "speed-boundaries"
    metadata = build_observed_link_speed_boundaries(cbi_root, output)
    lookup = np.load(
        output / "observed_link_speed_boundaries.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    assert lookup.dtype == OBSERVED_LINK_SPEED_BOUNDARY_DTYPE
    assert metadata["unique_node_pair_rows"] == 2
    row_a = lookup[lookup["from_node_id"] == 10][0]
    row_b = lookup[lookup["from_node_id"] == 30][0]
    assert np.isclose(row_a["qvdf_start_speed_mph_am"], 61.0)
    assert np.isclose(row_a["qvdf_end_speed_mph_am"], 49.0)
    assert np.isclose(row_a["qvdf_start_speed_mph_md"], 49.0)
    assert np.isclose(row_a["qvdf_end_speed_mph_md"], 57.0)
    assert np.isclose(row_a["qvdf_start_speed_mph_pm"], 57.0)
    assert np.isclose(row_a["qvdf_end_speed_mph_pm"], 52.0)
    assert np.isclose(row_b["qvdf_end_speed_mph_md"], 54.0)
    assert np.isclose(row_b["qvdf_start_speed_mph_pm"], 54.0)
    completeness = pd.read_csv(
        output / "boundary_completeness_report.csv"
    )
    tmc_20_md = completeness[
        completeness["tmc_code"].eq("TMC-20")
        & completeness["period"].eq("MD")
    ].iloc[0]
    tmc_20_pm = completeness[
        completeness["tmc_code"].eq("TMC-20")
        & completeness["period"].eq("PM")
    ].iloc[0]
    assert tmc_20_md["boundary_status"] == "both"
    assert tmc_20_pm["boundary_status"] == "both"
    assert tmc_20_md["end_missing_cause"] == "available"
    assert (
        tmc_20_md["end_speed_source"]
        == "pre_qc_weekday_average_fallback"
    )
    assert np.isnan(tmc_20_md["end_post_qc_speed_mph"])
    assert np.isclose(tmc_20_md["end_pre_qc_speed_mph"], 54.0)


def test_observed_link_t2_uses_representative_accepted_weekday_episode(
    tmp_path: Path,
) -> None:
    cbi_root = tmp_path / "cbi"
    reference = cbi_root / "X" / "01-input-and-qc" / "link_reference.csv"
    reference.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "A",
                "network_link_id": 1,
                "network_from_node_id": 10,
                "network_to_node_id": 20,
                "network_match_distance_ft": 3.0,
            },
            {
                "corridor": "X",
                "tmc_code": "B",
                "network_link_id": 2,
                "network_from_node_id": 30,
                "network_to_node_id": 40,
                "network_match_distance_ft": 4.0,
            },
        ]
    ).to_csv(reference, index=False)
    _write_frozen_mapping(cbi_root)
    episodes = pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "A",
                "period": "AM",
                "episode_id": "shallower",
                "t0_hour": 6.25,
                "t2_hour": 7.0,
                "t3_hour": 8.25,
                "min_speed_mph": 30.0,
                "P_hr": 2.0,
            },
            {
                "corridor": "X",
                "tmc_code": "A",
                "period": "AM",
                "episode_id": "deeper",
                "t0_hour": 5.5,
                "t2_hour": 8.0,
                "t3_hour": 9.5,
                "min_speed_mph": 20.0,
                "P_hr": 1.0,
            },
            {
                "corridor": "X",
                "tmc_code": "A",
                "period": "PM",
                "episode_id": "pm",
                "t0_hour": 14.5,
                "t2_hour": 17.25,
                "t3_hour": 20.0,
                "min_speed_mph": 25.0,
                "P_hr": 1.5,
            },
        ]
    )
    output = tmp_path / "observed-t2"
    metadata = build_observed_link_t2_lookup(cbi_root, episodes, output)
    lookup = np.load(
        output / "observed_link_t2.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    assert lookup.dtype == OBSERVED_LINK_T2_DTYPE
    assert metadata["unique_node_pair_rows"] == 2
    row_a = lookup[lookup["from_node_id"] == 10][0]
    row_b = lookup[lookup["from_node_id"] == 30][0]
    assert np.isclose(row_a["observed_t0_hour_am"], 5.5)
    assert np.isclose(row_a["observed_t2_hour_am"], 8.0)
    assert np.isclose(row_a["observed_t3_hour_am"], 9.5)
    assert np.isnan(row_a["observed_t2_hour_md"])
    assert np.isclose(row_a["observed_t0_hour_pm"], 14.5)
    assert np.isclose(row_a["observed_t2_hour_pm"], 17.25)
    assert np.isclose(row_a["observed_t3_hour_pm"], 20.0)
    assert np.isnan(row_b["observed_t2_hour_am"])
    audit = pd.read_csv(output / "observed_link_t2.csv")
    selected_a = audit[
        audit["selected_for_node_pair_lookup"]
        & audit["tmc_code"].eq("A")
    ].iloc[0]
    assert selected_a["observed_t2_episode_id_am"] == "deeper"
    assert selected_a["accepted_episode_count_am"] == 2


def test_observed_link_t2_can_audit_and_omit_invalid_virtual_triplet(
    tmp_path: Path,
) -> None:
    cbi_root = tmp_path / "cbi"
    reference = cbi_root / "X" / "01-input-and-qc" / "link_reference.csv"
    reference.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "VIRTUAL-A",
                "network_link_id": 1,
                "network_from_node_id": 10,
                "network_to_node_id": 20,
                "network_match_distance_ft": 0.0,
            }
        ]
    ).to_csv(reference, index=False)
    _write_frozen_mapping(cbi_root)
    episodes = pd.DataFrame(
        [
            {
                "corridor": "X",
                "tmc_code": "VIRTUAL-A",
                "period": "PM",
                "episode_id": "invalid-flat-entry",
                "t0_hour": 18.0,
                "t2_hour": 18.0,
                "t3_hour": 19.0,
                "min_speed_mph": 25.0,
                "P_hr": 1.0,
            }
        ]
    )
    output = tmp_path / "observed-t2"
    metadata = build_observed_link_t2_lookup(
        cbi_root,
        episodes,
        output,
        invalid_episode_policy="omit",
    )
    lookup = np.load(output / "observed_link_t2.npy", allow_pickle=False)
    assert np.isnan(lookup[0]["observed_t2_hour_pm"])
    assert metadata["invalid_episode_rows_omitted"] == 1
    invalid = pd.read_csv(output / "invalid_observed_episode_triplets.csv")
    assert invalid["invalid_observed_triplet_reason"].tolist() == [
        "missing_or_unordered_t0_t2_t3"
    ]
