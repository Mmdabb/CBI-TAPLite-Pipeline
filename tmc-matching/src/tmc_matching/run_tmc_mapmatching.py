from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from pathlib import Path
from typing import Sequence

import networkx as nx
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .tmc_line_matcher import (
    BASE2025_CRS_EPSG,
    MODEL_CRS_EPSG,
    PERIODS,
    MatchConfig,
    bearing,
    build_adjacency,
    build_corridor_summary,
    classify_match,
    directed_angle_diff,
    endpoint_candidates,
    load_base2025_physical_links,
    load_base2025_physical_nodes,
    route_metrics,
    shortest_link_path,
)


GEOMETRY_OVERLAP_BUFFER_FT = 250.0

SOURCE_CHAIN_TOLERANCE_FT = 100.0
TRANSITION_CONNECTOR_CUTOFF_MI = 1.0
SHORT_CONNECTOR_BASE_PENALTY = 20.0
SHORT_CONNECTOR_PER_MILE_PENALTY = 80.0
DISCONNECTED_BRANCH_PENALTY = 250.0
MANAGED_ROAD_MARKERS = ("HOV", "EXPRESS", "HOT")


def geometry_overlap_pct(
    link_geometry: LineString,
    tmc_line: LineString,
    buffer_ft: float = GEOMETRY_OVERLAP_BUFFER_FT,
) -> float:
    """Return the share of a matched link inside the buffered TMC alignment."""

    if link_geometry is None or link_geometry.is_empty or link_geometry.length <= 0:
        return float("nan")
    corridor = tmc_line.buffer(float(buffer_ft), cap_style=2)
    overlap_length = float(link_geometry.intersection(corridor).length)
    return max(0.0, min(100.0, 100.0 * overlap_length / link_geometry.length))


def load_all_tmc(tmc_file: Path, model_crs_epsg: int = MODEL_CRS_EPSG) -> pd.DataFrame:
    tmc = pd.read_csv(tmc_file, dtype={"tmc": "string"}, low_memory=False)
    tmc = tmc.sort_values(["road", "direction", "road_order", "tmc"]).reset_index(drop=True)
    transformer = Transformer.from_crs(4326, model_crs_epsg, always_xy=True)
    sx, sy = transformer.transform(tmc["start_longitude"].to_numpy(), tmc["start_latitude"].to_numpy())
    ex, ey = transformer.transform(tmc["end_longitude"].to_numpy(), tmc["end_latitude"].to_numpy())
    tmc["start_x"] = sx
    tmc["start_y"] = sy
    tmc["end_x"] = ex
    tmc["end_y"] = ey
    tmc["geometry_obj"] = [
        LineString([(x1, y1), (x2, y2)])
        for x1, y1, x2, y2 in zip(tmc["start_x"], tmc["start_y"], tmc["end_x"], tmc["end_y"])
    ]
    tmc["tmc_bearing"] = tmc["geometry_obj"].map(bearing)
    return tmc


def road_name_terms(road: str) -> list[str]:
    road_upper = road.upper().strip()
    base = re.sub(r"\s*\([^)]*\)", "", road_upper).strip()
    terms = [road_upper, base, base.replace("-", " ")]
    if "-BR" in base:
        terms.append(base.replace("-BR", " BR").replace("-", " "))
        terms.append(base.replace("-BR", " BUSINESS").replace("-", " "))
    return [term for term in dict.fromkeys(terms) if term]


def base_road_label(road: object) -> str:
    """Return the base road label used to pair GP and managed TMC inventories."""
    return re.sub(r"\s*\([^)]*\)", "", str(road).upper()).strip()


def is_managed_road_label(road: object) -> bool:
    road_upper = str(road).upper()
    return any(marker in road_upper for marker in MANAGED_ROAD_MARKERS)


def infer_corridor_lane_class(
    road: object,
    requested_lane_class: str,
    available_roads: set[str],
) -> str:
    """Resolve automatic GP/managed separation from explicit TMC road labels.

    When the source inventory contains both a base road and a managed variant
    such as ``I-95`` and ``I-95 (HOV)``, the two candidate pools must remain
    separate. Roads without an explicit managed companion retain the combined
    physical-network pool because some inventories (notably I-66) store GP and
    reversible TMC chains under the same road label.
    """
    if requested_lane_class != "auto":
        return requested_lane_class
    if is_managed_road_label(road):
        return "managed"
    managed_bases = {
        base_road_label(candidate)
        for candidate in available_roads
        if is_managed_road_label(candidate)
    }
    if base_road_label(road) in managed_bases:
        return "gp"
    return "all_open"


def derive_corridor_name(config: MatchConfig, match_long: pd.DataFrame) -> str:
    """Use the dominant matched facility name for managed corridors."""
    if config.lane_class != "managed" or match_long.empty or "STREETNAME" not in match_long.columns:
        return config.road
    names = match_long[["STREETNAME", "length_mi"]].copy()
    names["STREETNAME"] = names["STREETNAME"].fillna("").astype(str).str.strip()
    names = names[names["STREETNAME"].str.contains("HOV|EXPRESS|HOT", case=False, regex=True)].copy()
    if names.empty:
        return config.road
    names["name_key"] = names["STREETNAME"].str.upper()
    names["length_mi"] = pd.to_numeric(names["length_mi"], errors="coerce").fillna(0.0)
    ranked = (
        names.groupby("name_key", as_index=False)
        .agg(total_length_mi=("length_mi", "sum"), appearances=("name_key", "size"))
        .sort_values(["total_length_mi", "appearances", "name_key"], ascending=[False, False, True])
    )
    best_key = str(ranked.iloc[0]["name_key"])
    return str(names.loc[names["name_key"] == best_key, "STREETNAME"].iloc[0])


def source_chain_predecessors(
    tmc_group: pd.DataFrame,
    tolerance_ft: float = SOURCE_CHAIN_TOLERANCE_FT,
) -> dict[str, tuple[str, float]]:
    """Find the earlier TMC whose source endpoint feeds each TMC start.

    `road_order` can interleave multiple TMC chains on the same named corridor,
    so adjacency is recovered from projected source endpoints instead of from
    consecutive dataframe rows alone.
    """
    ordered = tmc_group.sort_values(["road_order", "tmc"]).reset_index(drop=True)
    predecessors: dict[str, tuple[str, float]] = {}
    prior_rows: list[pd.Series] = []
    for _, current in ordered.iterrows():
        best: tuple[float, float, int, str] | None = None
        current_order = pd.to_numeric(pd.Series([current.get("road_order")]), errors="coerce").iloc[0]
        for prior_position, prior in enumerate(prior_rows):
            gap_ft = math.hypot(
                float(prior["end_x"]) - float(current["start_x"]),
                float(prior["end_y"]) - float(current["start_y"]),
            )
            if gap_ft > tolerance_ft:
                continue
            prior_order = pd.to_numeric(pd.Series([prior.get("road_order")]), errors="coerce").iloc[0]
            order_gap = (
                abs(float(current_order) - float(prior_order))
                if pd.notna(current_order) and pd.notna(prior_order)
                else float("inf")
            )
            rank = (gap_ft, order_gap, -prior_position, str(prior["tmc"]))
            if best is None or rank < best:
                best = rank
        if best is not None:
            predecessors[str(current["tmc"])] = (best[3], float(best[0]))
        prior_rows.append(current)
    return predecessors


def corridor_transition(
    predecessor_route: list[int] | None,
    candidate_route: list[int],
    links_by_id: pd.DataFrame,
    adjacency: dict[int, list[tuple[int, int, float]]],
) -> dict[str, object]:
    """Score whether two source-adjacent TMC routes stay on one model branch."""
    if not predecessor_route:
        return {
            "corridor_transition_status": "predecessor_no_path",
            "corridor_transition_distance_mi": "",
            "corridor_transition_penalty": 0.0,
        }
    if set(predecessor_route) & set(candidate_route):
        return {
            "corridor_transition_status": "shared_link",
            "corridor_transition_distance_mi": 0.0,
            "corridor_transition_penalty": 0.0,
        }

    predecessor_end = links_by_id.loc[int(predecessor_route[-1])]
    candidate_start = links_by_id.loc[int(candidate_route[0])]
    predecessor_node = int(predecessor_end["to_node_id"])
    candidate_node = int(candidate_start["from_node_id"])
    if predecessor_node == candidate_node:
        return {
            "corridor_transition_status": "node_connected",
            "corridor_transition_distance_mi": 0.0,
            "corridor_transition_penalty": 0.0,
        }

    connector_links, connector_miles = shortest_link_path(
        adjacency,
        predecessor_node,
        candidate_node,
        cutoff_mi=TRANSITION_CONNECTOR_CUTOFF_MI,
    )
    if not math.isinf(connector_miles):
        return {
            "corridor_transition_status": "short_connector",
            "corridor_transition_distance_mi": round(float(connector_miles), 6),
            "corridor_transition_penalty": round(
                SHORT_CONNECTOR_BASE_PENALTY + SHORT_CONNECTOR_PER_MILE_PENALTY * float(connector_miles),
                6,
            ),
            "corridor_transition_link_ids": ";".join(map(str, connector_links)),
        }
    return {
        "corridor_transition_status": "disconnected",
        "corridor_transition_distance_mi": "",
        "corridor_transition_penalty": DISCONNECTED_BRANCH_PENALTY,
    }


def corridor_lane_class_mask(config: MatchConfig, links: pd.DataFrame) -> pd.Series:
    """Return links eligible under the Base_2025 LIMIT/facility classification."""
    physical_active = links.get("physical_active", pd.Series(False, index=links.index)).fillna(False).astype(bool)
    physical_gp = links.get("physical_gp", pd.Series(False, index=links.index)).fillna(False).astype(bool)
    physical_managed = links.get("physical_managed", pd.Series(False, index=links.index)).fillna(False).astype(bool)
    if config.lane_class == "gp":
        lane_class = physical_gp
    elif config.lane_class == "managed":
        lane_class = physical_managed
    else:
        lane_class = physical_active & (physical_gp | physical_managed)
    return lane_class.fillna(False).astype(bool)


def corridor_road_name_mask(config: MatchConfig, links: pd.DataFrame) -> pd.Series:
    road_name = pd.Series(False, index=links.index)
    for term in road_name_terms(config.road):
        road_name = road_name | links["street_upper"].str.contains(term, regex=False)
    return road_name


def select_transition_links_full(config: MatchConfig, tmc: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Select the expanded directed topology used only for inter-TMC transitions.

    Transition searches admit road-name matches and unnamed/ramp links of the
    same lane class so physical connectors such as I-95 Express Lanes link
    39101 (``STREETNAME=0``) are not lost.
    """
    corridor_geom = unary_union(tmc["geometry_obj"].tolist()).buffer(config.corridor_buffer_ft)
    intersects_buffer = links["geometry_obj"].map(lambda geom: geom.intersects(corridor_geom))
    lane_class = corridor_lane_class_mask(config, links)
    road_name = corridor_road_name_mask(config, links)
    street = links["street_upper"].fillna("").astype(str).str.strip()
    connector_name = street.isin({"", "0", "NAN", "NONE", "RAMP"}) | street.str.contains("RAMP", regex=False)
    selected = links[intersects_buffer & lane_class & (road_name | connector_name)].copy()
    return selected.sort_values(["link_id"]).reset_index(drop=True)


def select_corridor_links_full(config: MatchConfig, tmc: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    corridor_geom = unary_union(tmc["geometry_obj"].tolist()).buffer(config.corridor_buffer_ft)
    intersects_buffer = links["geometry_obj"].map(lambda geom: geom.intersects(corridor_geom))
    lane_class = corridor_lane_class_mask(config, links)

    base_mask = intersects_buffer & lane_class
    selected = links[base_mask].copy()
    if selected.empty:
        return selected.sort_values(["link_id"]).reset_index(drop=True)
    selected["road_name_match"] = corridor_road_name_mask(config, selected).astype(bool)

    tmc_bearings = tmc["tmc_bearing"].dropna().astype(float).tolist()
    if not tmc_bearings:
        return selected.sort_values(["link_id"]).reset_index(drop=True)

    selected = selected[
        selected["model_bearing"].map(
            lambda link_bearing: min(directed_angle_diff(float(link_bearing), b) for b in tmc_bearings)
            <= config.corridor_angle_tolerance_deg
        )
    ].copy()
    if selected.empty:
        return selected.sort_values(["link_id"]).reset_index(drop=True)
    return selected.sort_values(["link_id"]).reset_index(drop=True)


def _link_search_cost(link: pd.Series, tmc_line: LineString, tmc_bearing: float) -> float:
    """Return a non-negative edge cost used to generate plausible path alternatives."""
    length_mi = max(float(link.get("length_mi", 0.0)), 0.001)
    offset_ft = min(float(link["geometry_obj"].distance(tmc_line)), 2600.0)
    angle = directed_angle_diff(float(link.get("model_bearing", tmc_bearing)), float(tmc_bearing))
    street = str(link.get("street_upper", "")).strip()
    unnamed = street in {"", "0", "NAN", "NONE", "RAMP"} or "RAMP" in street
    road_name_match = bool(link.get("road_name_match", False))
    name_cost = 0.0 if road_name_match else (0.02 if unnamed else 0.04)
    ftype = pd.to_numeric(pd.Series([link.get("FTYPE")]), errors="coerce").iloc[0]
    generic_connector_cost = 0.12 if pd.notna(ftype) and float(ftype) == 0 else 0.0
    return (
        length_mi * (1.0 + offset_ft / 1200.0 + angle / 90.0)
        + name_cost
        + generic_connector_cost
    )


def build_local_link_graphs(
    config: MatchConfig,
    tmc_row: pd.Series,
    selected_links: pd.DataFrame,
    required_link_ids: set[int] | None = None,
) -> tuple[dict[str, nx.DiGraph], pd.DataFrame]:
    """Build small directed link graphs near one TMC, separated by facility class."""
    tmc_line = tmc_row["geometry_obj"]
    tmc_bearing = float(tmc_row["tmc_bearing"])
    local_mask = (
        selected_links["geometry_obj"].map(lambda geom: geom.intersects(tmc_line.buffer(config.route_search_buffer_ft)))
        & selected_links["model_bearing"].map(
            lambda value: directed_angle_diff(float(value), tmc_bearing) <= config.corridor_angle_tolerance_deg
        )
    )
    if required_link_ids:
        local_mask = local_mask | selected_links["link_id"].astype(int).isin(required_link_ids)
    local = selected_links[local_mask].copy()
    if local.empty:
        return {}, local

    graphs: dict[str, nx.DiGraph] = {}
    for facility_class, group in local.groupby("facility_class", sort=True):
        if facility_class not in {"gp", "managed"}:
            continue
        graph = nx.DiGraph()
        records = {int(row["link_id"]): row for _, row in group.iterrows()}
        outgoing: dict[int, list[int]] = {}
        for link_id, row in records.items():
            graph.add_node(link_id)
            outgoing.setdefault(int(row["from_node_id"]), []).append(link_id)
        for link_id, row in records.items():
            for next_link_id in outgoing.get(int(row["to_node_id"]), []):
                next_link = records[next_link_id]
                graph.add_edge(link_id, next_link_id, weight=_link_search_cost(next_link, tmc_line, tmc_bearing))
        graphs[str(facility_class)] = graph
    return graphs, local


def candidate_link_paths(
    start_link: pd.Series,
    end_link: pd.Series,
    graphs: dict[str, nx.DiGraph],
    links_by_id: pd.DataFrame,
    cutoff_mi: float,
    max_paths: int,
) -> list[list[int]]:
    """Return up to k distinct directed paths without mixing GP and managed links."""
    start_id = int(start_link["link_id"])
    end_id = int(end_link["link_id"])
    start_class = str(start_link.get("facility_class", ""))
    end_class = str(end_link.get("facility_class", ""))
    if start_class != end_class or start_class not in graphs:
        return []
    if start_id == end_id:
        return [[start_id]]
    graph = graphs[start_class]
    if start_id not in graph or end_id not in graph:
        return []
    paths: list[list[int]] = []
    try:
        generated = nx.shortest_simple_paths(graph, start_id, end_id, weight="weight")
        for route in itertools.islice(generated, max_paths * 4):
            route_ids = [int(link_id) for link_id in route]
            route_miles = float(links_by_id.loc[route_ids, "length_mi"].sum())
            if route_miles <= cutoff_mi:
                paths.append(route_ids)
                if len(paths) >= max_paths:
                    break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return paths


def no_path_row(tmc_row: pd.Series, reason: str) -> dict[str, object]:
    row = {
        "tmc": str(tmc_row["tmc"]),
        "road": tmc_row["road"],
        "direction": tmc_row["direction"],
        "intersection": tmc_row["intersection"],
        "road_order": tmc_row["road_order"],
        "tmc_miles": float(tmc_row["miles"]),
        "tmc_bearing": round(float(tmc_row["tmc_bearing"]), 3),
        "route_link_count": 0,
        "route_link_ids": "",
        "o_node_id": "",
        "d_node_id": "",
        "start_link_id": "",
        "end_link_id": "",
        "route_length_mi": "",
        "length_ratio": "",
        "avg_offset_ft": "",
        "max_offset_ft": "",
        "avg_bearing_diff_deg": "",
        "start_endpoint_distance_ft": "",
        "end_endpoint_distance_ft": "",
        "start_link_distance_ft": "",
        "end_link_distance_ft": "",
        "duplicate_link_count": 0,
        "road_name_mismatch_share": "",
        "unnamed_link_share": "",
        "attribute_discontinuity_count": 0,
        "facility_class_switch_count": 0,
        "candidate_route_count": 0,
        "same_endpoint_alternative_count": 0,
        "candidate_score_margin": "",
        "confidence": 0.0,
        "source_predecessor_tmc": "",
        "source_predecessor_gap_ft": "",
        "corridor_transition_status": "no_path",
        "corridor_transition_distance_mi": "",
        "corridor_transition_link_ids": "",
        "corridor_transition_penalty": "",
        "status": reason,
    }
    for period in PERIODS:
        prefix = period.lower()
        row.update(
            {
                f"{prefix}_open_link_count": 0,
                f"{prefix}_path_status": "no_path",
                f"{prefix}_open_share": "",
            }
        )
    return row


def route_period_status(route_link_ids: list[int], links_by_id: pd.DataFrame, period: str) -> dict[str, object]:
    prefix = period.lower()
    if not route_link_ids:
        return {
            f"{prefix}_open_link_count": 0,
            f"{prefix}_path_status": "no_path",
            f"{prefix}_open_share": "",
        }
    route_links = links_by_id.loc[route_link_ids]
    open_col = f"{prefix}_is_open"
    if open_col not in route_links.columns:
        open_count = len(route_links)
    else:
        open_count = int(route_links[open_col].fillna(False).astype(bool).sum())
    total = len(route_links)
    if open_count == total:
        status = "open"
    elif open_count == 0:
        status = "closed"
    else:
        status = "partial"
    return {
        f"{prefix}_open_link_count": open_count,
        f"{prefix}_path_status": status,
        f"{prefix}_open_share": round(open_count / max(total, 1), 4),
    }


def parse_link_ids(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        parts = value
    elif isinstance(value, str) and value.strip():
        parts = value.split(";")
    else:
        return []
    link_ids: list[int] = []
    for part in parts:
        try:
            link_ids.append(int(float(part)))
        except (TypeError, ValueError):
            continue
    return link_ids


def period_count_fields(group: pd.DataFrame, period: str) -> dict[str, object]:
    prefix = period.lower()
    column = f"{prefix}_path_status"
    values = group.get(column, pd.Series("no_path", index=group.index)).fillna("no_path").astype(str)
    counts = values.value_counts().to_dict()
    total = len(group)
    return {
        f"{prefix}_open_tmc_count": int(counts.get("open", 0)),
        f"{prefix}_closed_tmc_count": int(counts.get("closed", 0)),
        f"{prefix}_partial_tmc_count": int(counts.get("partial", 0)),
        f"{prefix}_no_path_tmc_count": int(counts.get("no_path", 0)),
        f"{prefix}_open_tmc_share_pct": round(float(counts.get("open", 0)) / max(total, 1) * 100, 1),
        f"{prefix}_path_status_counts_json": json.dumps(counts, sort_keys=True),
    }


def build_corridor_link_outputs(
    match_summary: pd.DataFrame,
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ordered corridor link sequences, inserting transition links before each TMC route."""
    if match_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    links_by_id = links.drop_duplicates("link_id").set_index(links["link_id"].astype(int), drop=False)
    summary_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for (road, direction), corridor_group in match_summary.groupby(["road", "direction"], sort=True, dropna=False):
        ordered_group = corridor_group.copy()
        ordered_group["_road_order_num"] = pd.to_numeric(ordered_group.get("road_order"), errors="coerce")
        ordered_group = ordered_group.sort_values(["_road_order_num", "tmc"], na_position="last")
        corridor_names = ordered_group.get("corridor_name", pd.Series(dtype="object")).dropna().astype(str)
        corridor_name = corridor_names.iloc[0] if not corridor_names.empty else str(road)

        entries: dict[int, dict[str, object]] = {}
        for _, row in ordered_group.iterrows():
            appearances = [
                ("transition", parse_link_ids(row.get("corridor_transition_link_ids"))),
                ("route", parse_link_ids(row.get("route_link_ids"))),
            ]
            for role, link_ids in appearances:
                for link_id in link_ids:
                    if link_id not in entries:
                        entries[link_id] = {
                            "roles": [],
                            "source_tmcs": [],
                            "source_road_orders": [],
                        }
                    entry = entries[link_id]
                    if role not in entry["roles"]:
                        entry["roles"].append(role)
                    tmc_id = str(row.get("tmc"))
                    if tmc_id not in entry["source_tmcs"]:
                        entry["source_tmcs"].append(tmc_id)
                    road_order = row.get("road_order")
                    if pd.notna(road_order):
                        road_order_text = str(int(float(road_order))) if float(road_order).is_integer() else str(float(road_order))
                        if road_order_text not in entry["source_road_orders"]:
                            entry["source_road_orders"].append(road_order_text)

        corridor_link_ids = list(entries)
        known_link_ids = [link_id for link_id in corridor_link_ids if link_id in links_by_id.index]
        known_links = links_by_id.loc[known_link_ids] if known_link_ids else pd.DataFrame()
        if isinstance(known_links, pd.Series):
            known_links = known_links.to_frame().T

        period_link_fields: dict[str, object] = {}
        for period in PERIODS:
            prefix = period.lower()
            if known_links.empty:
                open_count = 0
            else:
                open_count = int(known_links[f"{prefix}_is_open"].fillna(False).astype(bool).sum())
            total = len(corridor_link_ids)
            closed_count = max(total - open_count, 0)
            if total == 0:
                link_status = "no_path"
            elif open_count == total:
                link_status = "open"
            elif open_count == 0:
                link_status = "closed"
            else:
                link_status = "partial"
            period_link_fields.update(
                {
                    f"{prefix}_corridor_link_status": link_status,
                    f"{prefix}_open_corridor_link_count": open_count,
                    f"{prefix}_closed_corridor_link_count": closed_count,
                    f"{prefix}_open_corridor_link_share_pct": round(open_count / max(total, 1) * 100, 1),
                }
            )

        summary_rows.append(
            {
                "road": road,
                "direction": direction,
                "corridor_links": ";".join(map(str, corridor_link_ids)),
                "corridor_link_count": len(corridor_link_ids),
                "route_corridor_link_count": sum("route" in entry["roles"] for entry in entries.values()),
                "transition_corridor_link_count": sum("transition" in entry["roles"] for entry in entries.values()),
                "corridor_link_miles": round(
                    float(pd.to_numeric(known_links.get("length_mi", pd.Series(dtype=float)), errors="coerce").sum()), 3
                ),
                **{key: value for period in PERIODS for key, value in period_count_fields(ordered_group, period).items()},
                **period_link_fields,
            }
        )

        for sequence, link_id in enumerate(corridor_link_ids, start=1):
            entry = entries[link_id]
            link = links_by_id.loc[link_id] if link_id in links_by_id.index else pd.Series(dtype=object)
            roles = entry["roles"]
            role = "route_and_transition" if len(roles) > 1 else roles[0]
            long_rows.append(
                {
                    "road": road,
                    "direction": direction,
                    "corridor_name": corridor_name,
                    "sequence": sequence,
                    "link_id": link_id,
                    "link_role": role,
                    "source_tmcs": ";".join(entry["source_tmcs"]),
                    "source_road_orders": ";".join(entry["source_road_orders"]),
                    "from_node_id": int(link["from_node_id"]) if pd.notna(link.get("from_node_id")) else "",
                    "to_node_id": int(link["to_node_id"]) if pd.notna(link.get("to_node_id")) else "",
                    "length_mi": round(float(link["length_mi"]), 6) if pd.notna(link.get("length_mi")) else "",
                    "STREETNAME": link.get("STREETNAME", ""),
                    "am_allowed_use": link.get("am_allowed_use", ""),
                    "md_allowed_use": link.get("md_allowed_use", ""),
                    "pm_allowed_use": link.get("pm_allowed_use", ""),
                    "am_is_open": bool(link.get("am_is_open", False)) if pd.notna(link.get("am_is_open")) else False,
                    "md_is_open": bool(link.get("md_is_open", False)) if pd.notna(link.get("md_is_open")) else False,
                    "pm_is_open": bool(link.get("pm_is_open", False)) if pd.notna(link.get("pm_is_open")) else False,
                    "am_limit": link.get("am_limit", ""),
                    "md_limit": link.get("md_limit", ""),
                    "pm_limit": link.get("pm_limit", ""),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def order_corridor_summary_columns(corridors: pd.DataFrame) -> pd.DataFrame:
    corridors = corridors.drop(columns=["lane_class"], errors="ignore")
    leading_columns = [
        "road",
        "direction",
        "corridor_links",
        "corridor_link_count",
        "am_corridor_link_status",
        "md_corridor_link_status",
        "pm_corridor_link_status",
    ]
    leading_columns = [column for column in leading_columns if column in corridors.columns]
    remaining_columns = [column for column in corridors.columns if column not in leading_columns]
    return corridors[leading_columns + remaining_columns]


def match_corridor(
    config: MatchConfig,
    tmc_group: pd.DataFrame,
    links: pd.DataFrame,
    nodes: pd.DataFrame,
    write_candidates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmc_group = tmc_group.sort_values(["road_order", "tmc"]).reset_index(drop=True)
    selected_links = select_corridor_links_full(config, tmc_group, links)
    if selected_links.empty:
        summary = pd.DataFrame([no_path_row(row, "no_selected_links") for _, row in tmc_group.iterrows()])
        summary.insert(min(3, len(summary.columns)), "corridor_name", config.road)
        corridor_summary = build_corridor_summary(config, tmc_group, selected_links, summary)
        corridor_summary.insert(min(2, len(corridor_summary.columns)), "corridor_name", config.road)
        return summary, pd.DataFrame(), pd.DataFrame(), corridor_summary

    selected_node_ids = set(selected_links["from_node_id"].astype(int)) | set(selected_links["to_node_id"].astype(int))
    selected_nodes = nodes[nodes["node_id"].astype(int).isin(selected_node_ids)].copy()
    nodes_by_id = selected_nodes.set_index(selected_nodes["node_id"].astype(int), drop=False)
    link_tree = STRtree(selected_links["geometry_obj"].tolist())
    links_by_id = selected_links.set_index(selected_links["link_id"].astype(int), drop=False)
    transition_links = select_transition_links_full(config, tmc_group, links)
    transition_adjacency = build_adjacency(transition_links)

    summary_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    predecessors = source_chain_predecessors(tmc_group)
    selected_routes: dict[str, list[int]] = {}

    for _, tmc_row in tmc_group.iterrows():
        tmc_id = str(tmc_row["tmc"])
        tmc_line = tmc_row["geometry_obj"]
        start_pt = Point(tmc_row["start_x"], tmc_row["start_y"])
        end_pt = Point(tmc_row["end_x"], tmc_row["end_y"])
        tmc_bearing = float(tmc_row["tmc_bearing"])
        tmc_miles = float(tmc_row["miles"])
        predecessor_info = predecessors.get(tmc_id)
        predecessor_tmc = predecessor_info[0] if predecessor_info else ""
        predecessor_gap_ft = predecessor_info[1] if predecessor_info else ""
        predecessor_route = selected_routes.get(predecessor_tmc) if predecessor_tmc else None

        starts = endpoint_candidates(start_pt, tmc_line, tmc_bearing, "origin", selected_links, nodes_by_id, link_tree, config)
        ends = endpoint_candidates(end_pt, tmc_line, tmc_bearing, "destination", selected_links, nodes_by_id, link_tree, config)
        if not starts or not ends:
            summary_rows.append(no_path_row(tmc_row, "no_endpoint_candidates"))
            continue

        required_link_ids = {
            int(candidate["link"]["link_id"])
            for candidate in [*starts, *ends]
        }
        graphs, local_links = build_local_link_graphs(config, tmc_row, selected_links, required_link_ids)
        local_links_by_id = local_links.set_index(local_links["link_id"].astype(int), drop=False)
        route_candidates: dict[tuple[int, ...], dict[str, object]] = {}
        cutoff_mi = max(1.0, tmc_miles * config.max_route_ratio_for_search)
        for s_rank, start in enumerate(starts, start=1):
            for e_rank, end in enumerate(ends, start=1):
                routes = candidate_link_paths(
                    start["link"],
                    end["link"],
                    graphs,
                    local_links_by_id,
                    cutoff_mi,
                    config.max_candidate_paths,
                )
                if not routes:
                    if write_candidates:
                        candidate_rows.append(
                            {
                                "tmc": tmc_id,
                                "road": tmc_row["road"],
                                "direction": tmc_row["direction"],
                                "start_rank": s_rank,
                                "end_rank": e_rank,
                                "start_link_id": int(start["link"]["link_id"]),
                                "end_link_id": int(end["link"]["link_id"]),
                                "path_found": False,
                            }
                        )
                    continue
                for path_rank, route in enumerate(routes, start=1):
                    metrics = route_metrics(route, links_by_id, tmc_line, tmc_bearing, tmc_miles, start, end)
                    endpoint_name_selection_penalty = config.endpoint_name_selection_penalty * (
                        int(not bool(start.get("road_name_match", False)))
                        + int(not bool(end.get("road_name_match", False)))
                    )
                    local_score = (
                        float(metrics["score"])
                        + float(start["score"]) / 900.0
                        + float(end["score"]) / 900.0
                        + endpoint_name_selection_penalty
                    )
                    if predecessor_tmc:
                        transition = corridor_transition(predecessor_route, route, links_by_id, transition_adjacency)
                    else:
                        transition = {
                            "corridor_transition_status": "chain_start",
                            "corridor_transition_distance_mi": "",
                            "corridor_transition_penalty": 0.0,
                        }
                    selection_score = local_score + float(transition["corridor_transition_penalty"])
                    candidate = {
                        "tmc": tmc_id,
                        "road": tmc_row["road"],
                        "direction": tmc_row["direction"],
                        "start_rank": s_rank,
                        "end_rank": e_rank,
                        "path_rank": path_rank,
                        "start_link_id": int(start["link"]["link_id"]),
                        "end_link_id": int(end["link"]["link_id"]),
                        "o_node_id": int(start["link"]["from_node_id"]),
                        "d_node_id": int(end["link"]["to_node_id"]),
                        "path_found": True,
                        "route_link_ids": route,
                        "route_link_count": len(route),
                        "source_predecessor_tmc": predecessor_tmc,
                        "source_predecessor_gap_ft": predecessor_gap_ft,
                        "local_score": local_score,
                        "endpoint_name_selection_penalty": endpoint_name_selection_penalty,
                        "selection_score": selection_score,
                        "total_score": selection_score,
                        **transition,
                        **metrics,
                    }
                    if write_candidates:
                        candidate_rows.append({k: (";".join(map(str, v)) if k == "route_link_ids" else v) for k, v in candidate.items()})
                    route_key = tuple(route)
                    current = route_candidates.get(route_key)
                    if current is None or selection_score < float(current["selection_score"]):
                        route_candidates[route_key] = {
                            **candidate,
                            "start_candidate": start,
                            "end_candidate": end,
                        }

        ranked_candidates = sorted(route_candidates.values(), key=lambda item: (float(item["selection_score"]), tuple(item["route_link_ids"])))
        if not ranked_candidates:
            summary_rows.append(no_path_row(tmc_row, "no_path"))
            continue
        best = ranked_candidates[0]
        same_endpoint_alternatives = [
            candidate
            for candidate in ranked_candidates[1:]
            if int(candidate["start_link_id"]) == int(best["start_link_id"])
            and int(candidate["end_link_id"]) == int(best["end_link_id"])
        ]
        candidate_score_margin = (
            float(same_endpoint_alternatives[0]["selection_score"]) - float(best["selection_score"])
            if same_endpoint_alternatives
            else ""
        )

        selected_routes[tmc_id] = list(best["route_link_ids"])

        start = best["start_candidate"]
        end = best["end_candidate"]
        summary = {
            "tmc": tmc_id,
            "road": tmc_row["road"],
            "direction": tmc_row["direction"],
            "intersection": tmc_row["intersection"],
            "road_order": tmc_row["road_order"],
            "tmc_miles": round(tmc_miles, 6),
            "tmc_bearing": round(tmc_bearing, 3),
            "route_link_count": int(best["route_link_count"]),
            "route_link_ids": ";".join(map(str, best["route_link_ids"])),
            "o_node_id": int(best["o_node_id"]),
            "d_node_id": int(best["d_node_id"]),
            "start_link_id": int(best["start_link_id"]),
            "end_link_id": int(best["end_link_id"]),
            "route_length_mi": round(float(best["route_length_mi"]), 6),
            "length_ratio": round(float(best["length_ratio"]), 4),
            "avg_offset_ft": round(float(best["avg_offset_ft"]), 2),
            "max_offset_ft": round(float(best["max_offset_ft"]), 2),
            "avg_bearing_diff_deg": round(float(best["avg_bearing_diff_deg"]), 2),
            "start_endpoint_distance_ft": round(float(start["point_to_endpoint_ft"]), 2),
            "end_endpoint_distance_ft": round(float(end["point_to_endpoint_ft"]), 2),
            "start_link_distance_ft": round(float(start["point_to_link_ft"]), 2),
            "end_link_distance_ft": round(float(end["point_to_link_ft"]), 2),
            "duplicate_link_count": int(best["duplicate_link_count"]),
            "road_name_mismatch_share": round(float(best["road_name_mismatch_share"]), 4),
            "unnamed_link_share": round(float(best["unnamed_link_share"]), 4),
            "attribute_discontinuity_count": int(best["attribute_discontinuity_count"]),
            "facility_class_switch_count": int(best["facility_class_switch_count"]),
            "candidate_route_count": len(ranked_candidates),
            "same_endpoint_alternative_count": len(same_endpoint_alternatives),
            "candidate_score_margin": round(float(candidate_score_margin), 3) if candidate_score_margin != "" else "",
            "confidence": round(float(best["confidence"]), 2),
            "source_predecessor_tmc": best["source_predecessor_tmc"],
            "source_predecessor_gap_ft": (
                round(float(best["source_predecessor_gap_ft"]), 2)
                if best["source_predecessor_gap_ft"] != ""
                else ""
            ),
            "corridor_transition_status": best["corridor_transition_status"],
            "corridor_transition_distance_mi": best["corridor_transition_distance_mi"],
            "corridor_transition_link_ids": best.get("corridor_transition_link_ids", ""),
            "corridor_transition_penalty": round(float(best["corridor_transition_penalty"]), 3),
        }
        for period in PERIODS:
            summary.update(route_period_status(best["route_link_ids"], links_by_id, period))
        summary["status"] = classify_match(summary)
        summary_rows.append(summary)

        cumulative_mi = 0.0
        for seq, link_id in enumerate(best["route_link_ids"], start=1):
            link = links_by_id.loc[int(link_id)]
            cumulative_mi += float(link["length_mi"])
            long_rows.append(
                {
                    "tmc": tmc_id,
                    "road": tmc_row["road"],
                    "direction": tmc_row["direction"],
                    "road_order": tmc_row["road_order"],
                    "tmc_miles": round(tmc_miles, 6),
                    "route_link_count": int(best["route_link_count"]),
                    "route_length_mi": round(float(best["route_length_mi"]), 6),
                    "length_ratio": round(float(best["length_ratio"]), 4),
                    "match_confidence": round(float(best["confidence"]), 2),
                    "match_status": summary["status"],
                    "sequence": seq,
                    "link_id": int(link_id),
                    "from_node_id": int(link["from_node_id"]),
                    "to_node_id": int(link["to_node_id"]),
                    "length_mi": round(float(link["length_mi"]), 6),
                    "cumulative_mi": round(cumulative_mi, 6),
                    "distance_to_tmc_ft": round(float(link["geometry_obj"].distance(tmc_line)), 2),
                    "geometry_overlap_pct": round(
                        geometry_overlap_pct(link["geometry_obj"], tmc_line), 2
                    ),
                    "bearing_diff_deg": round(directed_angle_diff(float(link["model_bearing"]), tmc_bearing), 2),
                    "STREETNAME": link.get("STREETNAME", ""),
                    "allowed_use": link.get("allowed_use", ""),
                    "am_allowed_use": link.get("am_allowed_use", ""),
                    "md_allowed_use": link.get("md_allowed_use", ""),
                    "pm_allowed_use": link.get("pm_allowed_use", ""),
                    "am_is_open": bool(link.get("am_is_open", False)),
                    "md_is_open": bool(link.get("md_is_open", False)),
                    "pm_is_open": bool(link.get("pm_is_open", False)),
                    "am_limit": link.get("am_limit", ""),
                    "md_limit": link.get("md_limit", ""),
                    "pm_limit": link.get("pm_limit", ""),
                    "am_lanes": link.get("am_lanes", ""),
                    "md_lanes": link.get("md_lanes", ""),
                    "pm_lanes": link.get("pm_lanes", ""),
                    "am_period_lanes": link.get("am_period_lanes", ""),
                    "md_period_lanes": link.get("md_period_lanes", ""),
                    "pm_period_lanes": link.get("pm_period_lanes", ""),
                    "am_toll": link.get("am_toll", ""),
                    "md_toll": link.get("md_toll", ""),
                    "pm_toll": link.get("pm_toll", ""),
                    "am_period_toll": link.get("am_period_toll", ""),
                    "md_period_toll": link.get("md_period_toll", ""),
                    "pm_period_toll": link.get("pm_period_toll", ""),
                    "AMLIMIT": link.get("AMLIMIT", ""),
                    "MDLIMIT": link.get("MDLIMIT", ""),
                    "PMLIMIT": link.get("PMLIMIT", ""),
                    "lanes": link.get("lanes", ""),
                    "capacity": link.get("capacity", ""),
                    "free_speed": link.get("free_speed", ""),
                    "link_type": link.get("link_type", ""),
                    "FTYPE": link.get("FTYPE", ""),
                    "vdf_free_speed_mph": link.get("vdf_free_speed_mph", ""),
                    "facility_class": link.get("facility_class", ""),
                    "road_name_match": bool(link.get("road_name_match", False)),
                    "PROJECTID": link.get("PROJECTID", ""),
                    "LINKID": link.get("LINKID", ""),
                    "geometry": link.get("geometry", ""),
                    "geometry_wgs84": link.get("geometry_wgs84", ""),
                }
            )

    match_summary = pd.DataFrame(summary_rows)
    match_long = pd.DataFrame(long_rows)
    corridor_name = derive_corridor_name(config, match_long)
    for frame in (match_summary, match_long):
        if not frame.empty:
            frame.insert(min(3, len(frame.columns)), "corridor_name", corridor_name)
    candidates = pd.DataFrame(candidate_rows)
    corridor_summary = build_corridor_summary(config, tmc_group, selected_links, match_summary)
    corridor_summary.insert(min(2, len(corridor_summary.columns)), "corridor_name", corridor_name)
    return match_summary, match_long, candidates, corridor_summary


def _open_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Normalize generated or CSV-round-tripped open-status values."""

    if column not in frame:
        raise ValueError(f"Cannot build period product; missing {column}")
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def write_period_products(
    union_output_dir: Path,
    full_summary: pd.DataFrame,
    full_long: pd.DataFrame,
    full_corridors: pd.DataFrame,
    full_corridor_links: pd.DataFrame,
    full_candidates: pd.DataFrame | None = None,
    product_names: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Create AM/MD/PM mapping products from the authoritative union match.

    Matching is run once on the updated period-independent topology. Each
    period product then retains only route-link rows whose schedule-derived
    ``<period>_is_open`` flag is true. This preserves one authoritative spatial
    match while maintaining the legacy three-product contract consumed by CBI.
    """

    run_root = union_output_dir.parent
    names = product_names or {period: period.lower() for period in PERIODS}
    products: dict[str, dict[str, object]] = {}
    for period in PERIODS:
        prefix = period.lower()
        product_name = names[period]
        product_dir = run_root / product_name
        product_dir.mkdir(parents=True, exist_ok=False)

        period_long = full_long.loc[_open_mask(full_long, f"{prefix}_is_open")].copy()
        if full_corridor_links.empty:
            period_corridor_links = full_corridor_links.copy()
        else:
            period_corridor_links = full_corridor_links.loc[
                _open_mask(full_corridor_links, f"{prefix}_is_open")
            ].copy()

        full_summary.to_csv(product_dir / "full_route_match_summary.csv", index=False)
        period_long.to_csv(product_dir / "full_tmc_to_link.csv", index=False)
        full_corridors.to_csv(product_dir / "full_corridor_route_summary.csv", index=False)
        period_corridor_links.to_csv(product_dir / "full_corridor_links.csv", index=False)
        if full_candidates is not None:
            full_candidates.to_csv(product_dir / "full_route_match_candidates.csv", index=False)

        mapped_tmc_count = int(period_long["tmc"].nunique()) if "tmc" in period_long else 0
        details = {
            "product": product_name,
            "period": period,
            "source_product": union_output_dir.name,
            "route_link_rows": int(len(period_long)),
            "mapped_tmcs": mapped_tmc_count,
            "unique_links": int(period_long["link_id"].nunique()) if "link_id" in period_long else 0,
            "selection_rule": f"retain rows where {prefix}_is_open is true",
        }
        products[period] = details
        summary_lines = [
            f"# {period} TMC Mapping Product",
            "",
            f"- Source union product: `{union_output_dir.name}`",
            f"- Open route-link rows: {details['route_link_rows']:,}",
            f"- TMCs with at least one open mapped link: {mapped_tmc_count:,}",
            f"- Unique open links: {details['unique_links']:,}",
            f"- Rule: `{details['selection_rule']}`",
            "",
            "The updated matcher runs once on the complete AM/MD/PM topology. ",
            "This product applies the authoritative period schedule without rerunning spatial matching.",
        ]
        (product_dir / "full_route_matching_summary.md").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

    manifest = {
        "status": "PASS",
        "authoritative_union_product": union_output_dir.name,
        "period_products": products,
    }
    (run_root / "period_product_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return products


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run route/path TMC matching for the full RITIS inventory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--combined-product-name", default="combined")
    parser.add_argument("--period-product-template", default="{period}")
    parser.add_argument("--tmc-file", type=Path, required=True, help="Input TMC inventory CSV.")
    parser.add_argument("--am-link-file", type=Path, required=True)
    parser.add_argument("--md-link-file", type=Path, required=True)
    parser.add_argument("--pm-link-file", type=Path, required=True)
    parser.add_argument("--am-node-file", type=Path, required=True)
    parser.add_argument("--md-node-file", type=Path, required=True)
    parser.add_argument("--pm-node-file", type=Path, required=True)
    parser.add_argument("--source-crs-epsg", type=int, default=BASE2025_CRS_EPSG, help="EPSG code used by GMNS link/node coordinates.")
    parser.add_argument("--model-crs-epsg", type=int, default=MODEL_CRS_EPSG, help="Working EPSG code used for matching distances.")
    parser.add_argument("--network-label", default="gmns_network_am_md_pm period-independent AM/MD/PM network", help="Label written to the route matching summary.")
    parser.add_argument("--lane-class", default="auto", choices=["auto", "all_open", "gp", "managed"])
    parser.add_argument("--road", action="append", help="Optional road filter. Can be repeated.")
    parser.add_argument("--max-corridors", type=int, default=0, help="Optional maximum number of corridors for testing.")
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument(
        "--no-period-products",
        action="store_true",
        help="Write only the AM/MD/PM union product.",
    )
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    output_dir = output_root / args.combined_product_name
    output_dir.mkdir(parents=True, exist_ok=False)

    start_time = time.time()
    print("Loading full RITIS TMC inventory and model network...")
    tmc = load_all_tmc(Path(args.tmc_file).resolve(), args.model_crs_epsg)
    available_roads = set(tmc["road"].dropna().astype(str))
    matching_network_label = args.network_label
    links = load_base2025_physical_links(
        Path(args.am_link_file).resolve(),
        Path(args.pm_link_file).resolve(),
        Path(args.md_link_file).resolve(),
        source_crs_epsg=args.source_crs_epsg,
        working_crs_epsg=args.model_crs_epsg,
    )
    nodes = load_base2025_physical_nodes(
        Path(args.am_node_file).resolve(),
        Path(args.pm_node_file).resolve(),
        Path(args.md_node_file).resolve(),
        source_crs_epsg=args.source_crs_epsg,
        working_crs_epsg=args.model_crs_epsg,
    )

    if args.road:
        wanted_roads = {road.upper() for road in args.road}
        tmc = tmc[tmc["road"].astype(str).str.upper().isin(wanted_roads)].copy()

    corridors = (
        tmc[["road", "direction"]]
        .drop_duplicates()
        .sort_values(["road", "direction"])
        .itertuples(index=False, name=None)
    )
    corridors = list(corridors)
    if args.max_corridors > 0:
        corridors = corridors[: args.max_corridors]

    all_summary: list[pd.DataFrame] = []
    all_long: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []
    all_corridor_summary: list[pd.DataFrame] = []

    for idx, (road, direction) in enumerate(corridors, start=1):
        lane_class = infer_corridor_lane_class(str(road), args.lane_class, available_roads)
        config = MatchConfig(road=str(road), direction=str(direction), lane_class=lane_class)
        group = tmc[(tmc["road"] == road) & (tmc["direction"] == direction)].copy()
        print(f"[{idx}/{len(corridors)}] Matching {road} {direction} [{lane_class}]: {len(group)} TMCs")
        try:
            summary, long, candidates, corridor_summary = match_corridor(config, group, links, nodes, args.write_candidates)
        except Exception as exc:
            print(f"  WARNING: failed {road} {direction}: {exc}")
            summary = pd.DataFrame([no_path_row(row, f"error:{type(exc).__name__}") for _, row in group.iterrows()])
            summary.insert(min(3, len(summary.columns)), "corridor_name", str(road))
            long = pd.DataFrame()
            candidates = pd.DataFrame()
            corridor_summary = pd.DataFrame(
                [
                    {
                        "road": road,
                        "direction": direction,
                        "corridor_name": str(road),
                        "tmc_count": len(group),
                        "tmc_miles": round(float(pd.to_numeric(group["miles"], errors="coerce").sum()), 3),
                        "selected_model_links": 0,
                        "selected_model_link_miles": 0,
                        "matched_tmc_count": 0,
                        "route_path_tmc_count": 0,
                        "review_tmc_count": len(group),
                        "matched_share_pct": 0,
                        "route_path_coverage_pct": 0,
                        "avg_confidence": 0,
                        "avg_length_ratio": 0,
                        "avg_offset_ft": 0,
                        "status_counts_json": json.dumps({f"error:{type(exc).__name__}": len(group)}),
                    }
                ]
            )
        all_summary.append(summary)
        if not long.empty:
            all_long.append(long)
        if args.write_candidates and not candidates.empty:
            all_candidates.append(candidates)
        all_corridor_summary.append(corridor_summary)

    full_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    full_long = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    full_corridors = pd.concat(all_corridor_summary, ignore_index=True) if all_corridor_summary else pd.DataFrame()
    corridor_link_summary, full_corridor_links = build_corridor_link_outputs(full_summary, links)
    if not corridor_link_summary.empty:
        full_corridors = full_corridors.merge(corridor_link_summary, on=["road", "direction"], how="left")
    full_corridors = order_corridor_summary_columns(full_corridors)
    full_summary.to_csv(output_dir / "full_route_match_summary.csv", index=False)
    full_long.to_csv(output_dir / "full_tmc_to_link.csv", index=False)
    full_corridors.to_csv(output_dir / "full_corridor_route_summary.csv", index=False)
    full_corridor_links.to_csv(output_dir / "full_corridor_links.csv", index=False)
    full_candidates = None
    if args.write_candidates:
        full_candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
        full_candidates.to_csv(output_dir / "full_route_match_candidates.csv", index=False)

    status_counts = full_summary["status"].value_counts(dropna=False).to_dict() if "status" in full_summary else {}
    route_counts = pd.to_numeric(full_summary.get("route_link_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    route_path_count = int(route_counts.gt(0).sum())
    clean_match_count = int(full_summary["status"].eq("matched").sum()) if "status" in full_summary else 0
    total_tmcs = max(len(full_summary), 1)
    summary_lines = [
        "# Full Route/Path TMC Matching Summary",
        "",
        f"- Matching network: {matching_network_label}",
        f"- Physical network links used as candidates: {links['link_id'].nunique():,}",
        f"- TMCs processed: {len(full_summary):,}",
        f"- TMC/network matched coverage: {route_path_count:,} ({route_path_count / total_tmcs * 100:.1f}%)",
        f"- QA-passing route matches: {clean_match_count:,} ({clean_match_count / total_tmcs * 100:.1f}%)",
        f"- Route-link rows: {len(full_long):,}",
        f"- Corridors processed: {len(full_corridors):,}",
        f"- Unique route links: {full_long['link_id'].nunique() if 'link_id' in full_long else 0:,}",
        f"- Elapsed minutes: {(time.time() - start_time) / 60:.1f}",
        "",
        "## Match Rate Definitions",
        "",
        "- `TMC/network matched coverage` is the broad coverage measure. A TMC is counted when the matcher found at least one continuous path on the model network (`route_link_count > 0`). This includes paths that still have QA review flags.",
        "- `QA-passing route matches` is the stricter automatic-QA measure. A TMC is counted only when `status = matched`, meaning the path passed the current length-ratio, endpoint-distance, geometry-offset, bearing, loop, and confidence checks.",
        "- The gap between the two rates is therefore expected. It represents TMCs with a network path that should be reviewed, not TMCs that failed to match to the network completely.",
        "",
        "## Status Counts",
        "",
        *[f"- {key}: {value:,}" for key, value in status_counts.items()],
        "",
        "## Outputs",
        "",
        "- `full_route_match_summary.csv`: one row per TMC with continuous route/path fields.",
        "- `full_tmc_to_link.csv`: one row per TMC-route model link.",
        "- `full_corridor_route_summary.csv`: corridor-level route matching QA, ordered corridor links, and AM/MD/PM status summaries.",
        "- `full_corridor_links.csv`: one ordered row per unique corridor link, including transition links.",
    ]
    (output_dir / "full_route_matching_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if not args.no_period_products:
        products = write_period_products(
            output_dir,
            full_summary,
            full_long,
            full_corridors,
            full_corridor_links,
            full_candidates,
            product_names={
                period: args.period_product_template.format(
                    period=period.lower(), PERIOD=period
                )
                for period in PERIODS
            },
        )
        print(
            "Wrote period products: "
            + ", ".join(f"{period}={item['route_link_rows']:,} rows" for period, item in products.items())
        )
    print(f"Wrote full route matching outputs to {output_dir}")


if __name__ == "__main__":
    main()
