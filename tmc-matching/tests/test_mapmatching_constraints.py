from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path

import pandas as pd

MATCHER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MATCHER_ROOT / "src"))

from tmc_matching.run_tmc_mapmatching import (
    build_corridor_link_outputs,
    derive_corridor_name,
    infer_corridor_lane_class,
    load_all_tmc,
    match_corridor,
    order_corridor_summary_columns,
    select_corridor_links_full,
    select_transition_links_full,
    write_period_products,
)
from tmc_matching.tmc_line_matcher import (
    MatchConfig,
    load_base2025_physical_links as _load_links,
    load_base2025_physical_nodes as _load_nodes,
)


def _nvta_input_root() -> Path:
    configured = os.environ.get("TMC_MATCHING_NVTA_INPUT_DIR")
    if not configured:
        raise unittest.SkipTest(
            "Set TMC_MATCHING_NVTA_INPUT_DIR to run NVTA regression tests."
        )
    return Path(configured).resolve()


def load_all_tmc():
    return __import__(
        "tmc_matching.run_tmc_mapmatching", fromlist=["load_all_tmc"]
    ).load_all_tmc(_nvta_input_root() / "TMC_Identification.csv")


def load_base2025_physical_links():
    root = _nvta_input_root() / "network"
    return _load_links(
        root / "am" / "link.csv",
        root / "pm" / "link.csv",
        root / "md" / "link.csv",
    )


def load_base2025_physical_nodes():
    root = _nvta_input_root() / "network"
    return _load_nodes(
        root / "am" / "node.csv",
        root / "pm" / "node.csv",
        root / "md" / "node.csv",
    )


class PeriodProductContractTest(unittest.TestCase):
    def test_period_products_use_authoritative_open_status(self) -> None:
        summary = pd.DataFrame([{"tmc": "A", "route_link_count": 2}])
        long = pd.DataFrame(
            [
                {"tmc": "A", "link_id": 1, "am_is_open": True, "md_is_open": True, "pm_is_open": False},
                {"tmc": "A", "link_id": 2, "am_is_open": False, "md_is_open": True, "pm_is_open": True},
            ]
        )
        corridors = pd.DataFrame([{"road": "I-1", "direction": "NORTHBOUND"}])
        corridor_links = long.copy()
        with tempfile.TemporaryDirectory() as directory:
            union = Path(directory) / "combined"
            union.mkdir()
            products = write_period_products(
                union, summary, long, corridors, corridor_links
            )
            root = Path(directory)
            am = pd.read_csv(
                root
                / "am"
                / "full_tmc_to_link.csv"
            )
            md = pd.read_csv(
                root
                / "md"
                / "full_tmc_to_link.csv"
            )
            pm = pd.read_csv(
                root
                / "pm"
                / "full_tmc_to_link.csv"
            )
        self.assertEqual(am["link_id"].tolist(), [1])
        self.assertEqual(md["link_id"].tolist(), [1, 2])
        self.assertEqual(pm["link_id"].tolist(), [2])
        self.assertEqual(products["MD"]["route_link_rows"], 2)


class CorridorContinuityRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmc = load_all_tmc()
        group = tmc[(tmc["road"] == "I-66") & (tmc["direction"] == "WESTBOUND")].copy()
        summary, _, _, _ = match_corridor(
            MatchConfig(road="I-66", direction="WESTBOUND", lane_class="all_open"),
            group,
            load_base2025_physical_links(),
            load_base2025_physical_nodes(),
        )
        cls.summary = summary.set_index("tmc")

    def test_i66_west_stays_on_the_connected_pm_branch(self) -> None:
        previous = self.summary.loc["110+04174"]
        current = self.summary.loc["110P04174"]
        following = self.summary.loc["110+04175"]

        self.assertEqual(previous["route_link_ids"], "32136;32130")
        self.assertEqual(current["route_link_ids"], "30296;32551")
        self.assertEqual(following["route_link_ids"], "32552")
        self.assertEqual(current["corridor_transition_status"], "node_connected")

    def test_i66_west_has_no_disconnected_source_chain_transition(self) -> None:
        self.assertFalse(self.summary["corridor_transition_status"].eq("disconnected").any())


class CorridorLinkOutputRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmc = load_all_tmc()
        links = load_base2025_physical_links()
        group = tmc[(tmc["road"] == "I-395") & (tmc["direction"] == "NORTHBOUND")].copy()
        summary, _, _, _ = match_corridor(
            MatchConfig(road="I-395", direction="NORTHBOUND", lane_class="all_open"),
            group,
            links,
            load_base2025_physical_nodes(),
        )
        cls.corridor_summary, cls.corridor_links = build_corridor_link_outputs(summary, links)

    def test_i395_north_inserts_transition_links_in_order(self) -> None:
        link_ids = self.corridor_links["link_id"].tolist()
        start = link_ids.index(26271)
        self.assertEqual(link_ids[start : start + 4], [26271, 26803, 26422, 26274])
        roles = self.corridor_links.set_index("link_id")["link_role"]
        self.assertIn(roles.loc[26803], {"route", "transition", "route_and_transition"})
        self.assertIn(roles.loc[26422], {"route", "transition", "route_and_transition"})

    def test_corridor_summary_has_am_md_pm_status_fields(self) -> None:
        row = self.corridor_summary.iloc[0]
        self.assertEqual(row["am_open_tmc_count"] + row["am_closed_tmc_count"] + row["am_partial_tmc_count"] + row["am_no_path_tmc_count"], 23)
        self.assertEqual(row["md_open_tmc_count"] + row["md_closed_tmc_count"] + row["md_partial_tmc_count"] + row["md_no_path_tmc_count"], 23)
        self.assertEqual(row["pm_open_tmc_count"] + row["pm_closed_tmc_count"] + row["pm_partial_tmc_count"] + row["pm_no_path_tmc_count"], 23)
        self.assertIn("26803;26422;26274", row["corridor_links"])

    def test_requested_corridor_column_order_removes_lane_class(self) -> None:
        combined = pd.DataFrame(
            [
                {
                    "road": "I-395",
                    "direction": "NORTHBOUND",
                    "lane_class": "all_open",
                    **self.corridor_summary.iloc[0].to_dict(),
                    "tmc_count": 23,
                }
            ]
        )
        ordered = order_corridor_summary_columns(combined)
        self.assertNotIn("lane_class", ordered.columns)
        self.assertEqual(
            ordered.columns[:7].tolist(),
            [
                "road",
                "direction",
                "corridor_links",
                "corridor_link_count",
                "am_corridor_link_status",
                "md_corridor_link_status",
                "pm_corridor_link_status",
            ],
        )


class LaneClassSeparationRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmc = load_all_tmc()
        cls.links = load_base2025_physical_links()

    def test_auto_lane_class_separates_explicit_hov_companions(self) -> None:
        roads = set(self.tmc["road"].dropna().astype(str))
        self.assertEqual(infer_corridor_lane_class("I-95", "auto", roads), "gp")
        self.assertEqual(infer_corridor_lane_class("I-95 (HOV)", "auto", roads), "managed")
        self.assertEqual(infer_corridor_lane_class("I-395", "auto", roads), "gp")
        self.assertEqual(infer_corridor_lane_class("I-395 (HOV)", "auto", roads), "managed")
        self.assertEqual(infer_corridor_lane_class("I-66", "auto", roads), "all_open")

    def test_i95_gp_candidate_pool_excludes_express_and_restricted_links(self) -> None:
        group = self.tmc[(self.tmc["road"] == "I-95") & (self.tmc["direction"] == "SOUTHBOUND")].copy()
        selected = select_corridor_links_full(
            MatchConfig(road="I-95", direction="SOUTHBOUND", lane_class="gp"),
            group,
            self.links,
        )
        self.assertFalse(selected.empty)
        self.assertFalse(selected["street_upper"].str.contains("HOV|EXPRESS|HOT", regex=True).any())
        self.assertTrue(pd.to_numeric(selected["AMLIMIT"], errors="coerce").fillna(0).astype(int).eq(0).all())
        self.assertTrue(pd.to_numeric(selected["MDLIMIT"], errors="coerce").fillna(0).astype(int).eq(0).all())
        self.assertTrue(pd.to_numeric(selected["PMLIMIT"], errors="coerce").fillna(0).astype(int).eq(0).all())
        self.assertTrue(selected["physical_gp"].fillna(False).astype(bool).all())

    def test_i95_hov_candidate_pool_contains_only_managed_links(self) -> None:
        group = self.tmc[
            (self.tmc["road"] == "I-95 (HOV)") & (self.tmc["direction"] == "SOUTHBOUND")
        ].copy()
        selected = select_corridor_links_full(
            MatchConfig(road="I-95 (HOV)", direction="SOUTHBOUND", lane_class="managed"),
            group,
            self.links,
        )
        self.assertFalse(selected.empty)
        self.assertTrue(selected["physical_managed"].fillna(False).astype(bool).all())

    def test_limit_profiles_use_only_documented_codes(self) -> None:
        am_codes = set(pd.to_numeric(self.links["AMLIMIT"], errors="coerce").dropna().astype(int))
        md_codes = set(pd.to_numeric(self.links["MDLIMIT"], errors="coerce").dropna().astype(int))
        pm_codes = set(pd.to_numeric(self.links["PMLIMIT"], errors="coerce").dropna().astype(int))
        self.assertEqual(am_codes | md_codes | pm_codes, {0, 2, 4, 5, 9})
        inactive = self.links["AMLIMIT"].eq(9) & self.links["MDLIMIT"].eq(9) & self.links["PMLIMIT"].eq(9)
        self.assertFalse(self.links.loc[inactive, "physical_active"].any())

    def test_md_union_keeps_reversible_links_missing_from_one_peak_network(self) -> None:
        by_id = self.links.set_index(self.links["link_id"].astype(int), drop=False)
        self.assertEqual(len(self.links), 49336)
        self.assertEqual(int(by_id.loc[8671, "AMLIMIT"]), 9)
        self.assertFalse(bool(by_id.loc[8671, "am_is_open"]))
        self.assertTrue(bool(by_id.loc[8671, "md_is_open"]))
        self.assertTrue(bool(by_id.loc[8671, "pm_is_open"]))
        self.assertEqual(int(by_id.loc[8673, "PMLIMIT"]), 9)
        self.assertTrue(bool(by_id.loc[8673, "am_is_open"]))
        self.assertTrue(bool(by_id.loc[8673, "md_is_open"]))
        self.assertFalse(bool(by_id.loc[8673, "pm_is_open"]))

    def test_limit_classifies_gp_and_reversible_links(self) -> None:
        by_id = self.links.set_index(self.links["link_id"].astype(int), drop=False)
        self.assertEqual(by_id.loc[38529, "facility_class"], "gp")
        self.assertEqual(by_id.loc[39101, "facility_class"], "managed")
        self.assertEqual(by_id.loc[39112, "facility_class"], "managed")
        self.assertEqual(by_id.loc[47382, "facility_class"], "managed")

    def test_managed_corridor_uses_matched_facility_name(self) -> None:
        matched_links = pd.DataFrame(
            {
                "STREETNAME": ["I-95 Express Lanes", "I-95 Express Lanes", "I-95 HOV Ramp"],
                "length_mi": [1.0, 1.5, 0.1],
            }
        )
        name = derive_corridor_name(
            MatchConfig(road="I-95 (HOV)", direction="SOUTHBOUND", lane_class="managed"),
            matched_links,
        )
        self.assertEqual(name, "I-95 Express Lanes")


class ManagedTransitionConnectivityRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmc = load_all_tmc()
        cls.links = load_base2025_physical_links()
        group = tmc[(tmc["road"] == "I-95 (HOV)") & (tmc["direction"] == "SOUTHBOUND")].copy()
        config = MatchConfig(road="I-95 (HOV)", direction="SOUTHBOUND", lane_class="managed")
        cls.transition_candidates = select_transition_links_full(config, group, cls.links)
        summary, _, _, _ = match_corridor(
            config,
            group,
            cls.links,
            load_base2025_physical_nodes(),
        )
        cls.summary = summary.set_index("tmc")
        _, cls.corridor_links = build_corridor_link_outputs(summary, cls.links)

    def test_unnamed_managed_connector_is_in_transition_topology(self) -> None:
        self.assertIn(39101, self.transition_candidates["link_id"].astype(int).tolist())

    def test_i95_express_transition_inserts_link_39101(self) -> None:
        connector_rows = self.summary[
            self.summary["corridor_transition_link_ids"].fillna("").astype(str).eq("39101")
        ]
        self.assertEqual(len(connector_rows), 1)
        self.assertEqual(connector_rows.iloc[0]["corridor_transition_status"], "short_connector")

    def test_i95_express_corridor_links_are_continuous_at_39101(self) -> None:
        ordered = self.corridor_links.sort_values("sequence").reset_index(drop=True)
        link_ids = ordered["link_id"].astype(int).tolist()
        start = link_ids.index(39098)
        self.assertEqual(link_ids[start : start + 3], [39098, 39101, 39103])
        segment = ordered.iloc[start : start + 3]
        self.assertEqual(segment.iloc[0]["to_node_id"], segment.iloc[1]["from_node_id"])
        self.assertEqual(segment.iloc[1]["to_node_id"], segment.iloc[2]["from_node_id"])


class SoftRoadNameRouteRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmc = load_all_tmc()
        group = tmc[(tmc["road"] == "US-1") & (tmc["direction"] == "NORTHBOUND")].copy()
        summary, long, _, _ = match_corridor(
            MatchConfig(road="US-1", direction="NORTHBOUND", lane_class="all_open"),
            group,
            load_base2025_physical_links(),
            load_base2025_physical_nodes(),
        )
        cls.summary = summary.set_index("tmc")
        cls.long = long

    def test_us1_name_change_remains_on_continuous_route(self) -> None:
        row = self.summary.loc["110+09528"]
        self.assertEqual(row["route_link_ids"], "38524;40425;38529;40656;39702;39057")
        self.assertEqual(row["status"], "matched")
        self.assertEqual(row["facility_class_switch_count"], 0)
        self.assertEqual(row["attribute_discontinuity_count"], 0)
        self.assertGreater(row["candidate_score_margin"], 2.0)

    def test_us1_route_keeps_fraley_links_as_primary_route_links(self) -> None:
        route = self.long[self.long["tmc"].astype(str).eq("110+09528")]
        self.assertEqual(route["link_id"].astype(int).tolist(), [38524, 40425, 38529, 40656, 39702, 39057])
        self.assertEqual(
            route.loc[route["link_id"].astype(int).isin([38529, 40656, 39702]), "STREETNAME"].tolist(),
            ["Fraley Boulevard", "Fraley Boulevard", "Fraley Boulevard"],
        )


if __name__ == "__main__":
    unittest.main()
