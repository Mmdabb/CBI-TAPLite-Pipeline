from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .run_tmc_mapmatching import infer_corridor_lane_class, source_chain_predecessors
from .tmc_line_matcher import (
    MODEL_CRS_EPSG,
    MatchConfig,
    bearing,
    directed_angle_diff,
    load_base2025_physical_links,
)


MATCHER_BUFFER_FT = float(MatchConfig().corridor_buffer_ft)
MATCHER_ANGLE_DEG = float(MatchConfig().corridor_angle_tolerance_deg)
FT_PER_MILE = 5280.0


def route_pattern(road: str) -> re.Pattern[str]:
    tokens = re.findall(r"[A-Z0-9]+", str(road).upper())
    if not tokens:
        raise ValueError(f"Road name does not contain a searchable token: {road!r}")
    # TMC road labels use several equivalent separators and can also be a
    # spelled-out facility (for example, BROADLANDS BLVD). Match the complete
    # ordered token sequence while allowing only non-alphanumeric separators.
    body = r"[^A-Z0-9]*".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![A-Z0-9]){body}(?![A-Z0-9])")


def facility_role(ftype: object, street_name: object) -> str:
    value = pd.to_numeric(pd.Series([ftype]), errors="coerce").iloc[0]
    street = str(street_name or "").upper()
    if (pd.notna(value) and int(value) == 0) or street.strip() in {"", "0", "NONE", "NAN"}:
        return "model_connector"
    if (pd.notna(value) and int(value) == 6) or "RAMP" in street:
        return "ramp"
    mapping_by_ftype = {
        1: "freeway_mainline",
        2: "principal_arterial",
        3: "minor_arterial",
        4: "collector_or_local",
        5: "expressway_or_parkway",
    }
    return mapping_by_ftype.get(int(value), "other") if pd.notna(value) else "other"


def multi_source_reachable(adjacency: dict[int, set[int]], sources: set[int]) -> set[int]:
    reached = set(sources)
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def safe_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def to_native(value: object):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def build_corridor_catalog(
    dashboard: Path | None,
    corridor_inputs: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dashboard is None:
        reports = sorted(
            path
            for path in corridor_inputs.iterdir()
            if path.is_dir() and (path / "TMC_Identification.csv").is_file()
        )
        if not reports:
            raise FileNotFoundError(
                f"No corridor inputs were found under {corridor_inputs}"
            )
    else:
        report_root = dashboard / "reports/corridors"
        if not report_root.is_dir():
            raise FileNotFoundError(
                f"Dashboard corridor reports not found: {report_root}"
            )
        reports = sorted(path for path in report_root.iterdir() if path.is_dir())
    transformer = Transformer.from_crs(4326, MODEL_CRS_EPSG, always_xy=True)
    catalog_rows: list[dict] = []
    tmc_frames: list[pd.DataFrame] = []
    for report in reports:
        source = corridor_inputs / report.name / "TMC_Identification.csv"
        tmc = pd.read_csv(source, dtype={"tmc": "string"}, low_memory=False)
        pairs = tmc[["road", "direction"]].dropna().drop_duplicates()
        if len(pairs) != 1:
            raise ValueError(f"Expected one road/direction in {source}, found {len(pairs)}")
        road = str(pairs.iloc[0]["road"]).strip()
        direction = str(pairs.iloc[0]["direction"]).strip()
        sx, sy = transformer.transform(tmc["start_longitude"].to_numpy(), tmc["start_latitude"].to_numpy())
        ex, ey = transformer.transform(tmc["end_longitude"].to_numpy(), tmc["end_latitude"].to_numpy())
        tmc = tmc.copy()
        tmc["dashboard_corridor"] = report.name
        tmc["start_x"] = sx
        tmc["start_y"] = sy
        tmc["end_x"] = ex
        tmc["end_y"] = ey
        tmc["geometry_obj"] = [
            LineString([(x1, y1), (x2, y2)])
            for x1, y1, x2, y2 in zip(sx, sy, ex, ey)
        ]
        tmc["tmc_bearing"] = tmc["geometry_obj"].map(bearing)
        tmc = tmc.sort_values(["road_order", "tmc"]).reset_index(drop=True)
        tmc["tmc_rank"] = range(1, len(tmc) + 1)
        tmc["tmc_rank_pct"] = (
            (tmc["tmc_rank"] - 1) / max(len(tmc) - 1, 1)
        )
        tmc_frames.append(tmc)
        catalog_rows.append(
            {
                "dashboard_corridor": report.name,
                "dashboard_road": road,
                "dashboard_direction": direction,
                "tmc_count": len(tmc),
                "first_road_order": safe_float(tmc["road_order"].min()),
                "last_road_order": safe_float(tmc["road_order"].max()),
            }
        )
    return pd.DataFrame(catalog_rows), pd.concat(tmc_frames, ignore_index=True)


def corridor_object(
    catalog_row: pd.Series,
    tmc: pd.DataFrame,
    full_tmc: pd.DataFrame,
    full_corridor: pd.DataFrame,
    links_by_id: pd.DataFrame,
    lane_class: str,
) -> dict:
    key = str(catalog_row["dashboard_corridor"])
    road = str(catalog_row["dashboard_road"])
    direction = str(catalog_row["dashboard_direction"])
    local_tmc = tmc[tmc["dashboard_corridor"].eq(key)].copy().reset_index(drop=True)
    geoms = local_tmc["geometry_obj"].tolist()
    tmc_ids = set(local_tmc["tmc"].astype(str))
    route_rows = full_tmc[full_tmc["tmc"].astype(str).isin(tmc_ids)].copy()
    route_ids = set(route_rows["link_id"].astype(int))
    route_link_rows = links_by_id.loc[links_by_id.index.intersection(route_ids)].copy()
    route_geoms = route_link_rows["geometry_obj"].tolist()
    route_geom_ids = route_link_rows.index.astype(int).tolist()

    corridor_rows = full_corridor[
        full_corridor["road"].astype(str).eq(road)
        & full_corridor["direction"].astype(str).eq(direction)
    ].copy()
    corridor_ids = set(corridor_rows["link_id"].astype(int))
    transition_ids = set(
        corridor_rows.loc[corridor_rows["link_role"].astype(str).str.contains("transition"), "link_id"].astype(int)
    )

    predecessors = source_chain_predecessors(local_tmc)
    predecessor_ids = set(predecessors)
    predecessor_values = {value[0] for value in predecessors.values()}
    chain_start_rows = local_tmc[~local_tmc["tmc"].astype(str).isin(predecessor_ids)].copy()
    chain_end_rows = local_tmc[~local_tmc["tmc"].astype(str).isin(predecessor_values)].copy()
    if chain_start_rows.empty:
        chain_start_rows = local_tmc.head(1)
    if chain_end_rows.empty:
        chain_end_rows = local_tmc.tail(1)

    return {
        "key": key,
        "road": road,
        "direction": direction,
        "lane_class": lane_class,
        "tmc": local_tmc,
        "tmc_geoms": geoms,
        "tmc_tree": STRtree(geoms),
        "tmc_union": unary_union(geoms),
        "tmc_bearings": local_tmc["tmc_bearing"].astype(float).tolist(),
        "route_ids": route_ids,
        "route_tree": STRtree(route_geoms) if route_geoms else None,
        "route_geom_ids": route_geom_ids,
        "route_union": unary_union(route_geoms) if route_geoms else None,
        "corridor_ids": corridor_ids,
        "transition_ids": transition_ids,
        "chain_starts": chain_start_rows,
        "chain_ends": chain_end_rows,
    }


def score_direction(link: pd.Series, corridor: dict) -> dict:
    geom = link["geometry_obj"]
    nearest_idx = int(corridor["tmc_tree"].nearest(geom))
    nearest = corridor["tmc"].iloc[nearest_idx]
    distance_ft = float(geom.distance(corridor["tmc_union"]))
    nearest_angle = directed_angle_diff(float(link["model_bearing"]), float(nearest["tmc_bearing"]))
    min_angle = min(
        directed_angle_diff(float(link["model_bearing"]), float(value))
        for value in corridor["tmc_bearings"]
    )
    route_distance = (
        float(geom.distance(corridor["route_union"]))
        if corridor["route_union"] is not None
        else float("nan")
    )
    nearest_route_id = None
    if corridor["route_tree"] is not None:
        route_idx = int(corridor["route_tree"].nearest(geom))
        nearest_route_id = corridor["route_geom_ids"][route_idx]
    score = distance_ft + 22.0 * nearest_angle + (0.10 * route_distance if math.isfinite(route_distance) else 0.0)
    return {
        "dashboard_corridor": corridor["key"],
        "dashboard_road": corridor["road"],
        "dashboard_direction": corridor["direction"],
        "inferred_matcher_lane_class": corridor["lane_class"],
        "direction_score": score,
        "distance_to_tmc_coverage_ft": distance_ft,
        "distance_to_selected_route_ft": route_distance,
        "bearing_diff_to_nearest_tmc_deg": nearest_angle,
        "minimum_corridor_bearing_diff_deg": min_angle,
        "nearest_tmc": str(nearest["tmc"]),
        "nearest_tmc_road_order": safe_float(nearest.get("road_order")),
        "nearest_tmc_rank": int(nearest["tmc_rank"]),
        "nearest_tmc_rank_pct": float(nearest["tmc_rank_pct"]),
        "nearest_selected_route_link_id": nearest_route_id,
    }


def terminal_ray_relation(midpoint: Point, corridor: dict) -> tuple[str, float, float]:
    candidates: list[tuple[str, float, float]] = []
    for row in corridor["chain_starts"].itertuples(index=False):
        point = Point(float(row.start_x), float(row.start_y))
        theta = math.radians(float(row.tmc_bearing))
        ux, uy = math.cos(theta), math.sin(theta)
        qx, qy = midpoint.x - point.x, midpoint.y - point.y
        along = qx * ux + qy * uy
        lateral = abs(qx * uy - qy * ux)
        if along < 0:
            candidates.append(("before_first_tmc", lateral + 0.10 * abs(along), lateral))
    for row in corridor["chain_ends"].itertuples(index=False):
        point = Point(float(row.end_x), float(row.end_y))
        theta = math.radians(float(row.tmc_bearing))
        ux, uy = math.cos(theta), math.sin(theta)
        qx, qy = midpoint.x - point.x, midpoint.y - point.y
        along = qx * ux + qy * uy
        lateral = abs(qx * uy - qy * ux)
        if along > 0:
            candidates.append(("after_last_tmc", lateral + 0.10 * abs(along), lateral))
    if not candidates:
        return "lateral_or_disconnected", float("nan"), float("nan")
    return min(candidates, key=lambda item: item[1])


def lane_class_pass(link: pd.Series, lane_class: str) -> bool:
    if lane_class == "gp":
        return bool(link["physical_gp"])
    if lane_class == "managed":
        return bool(link["physical_managed"])
    return bool(link["physical_active"] and (link["physical_gp"] or link["physical_managed"]))


def reason_for(row: dict) -> tuple[str, str, str, str, str]:
    if row["transition_only"]:
        return (
            "already_recovered_transition_link",
            "already represented",
            "none",
            "none",
            "The link is absent from TMC route rows but is already present in full_corridor_links as an inter-TMC transition.",
        )
    if not row["candidate_lane_class_pass"]:
        return (
            "facility_class_filtered_by_matcher",
            "facility eligibility",
            "none",
            "high",
            "Keep separate from the GP corridor; only map it through an explicit managed/restricted corridor inventory.",
        )
    if not row["candidate_spatial_buffer_pass"]:
        relation = row["coverage_relation"]
        if relation == "before_first_tmc":
            if row["distance_to_tmc_coverage_mi"] <= 1.0:
                treatment, priority = "soft terminal extrapolation", "medium"
                guidance = "A short upstream extrapolation is plausible because the link is within one mile of observed coverage; retain topology and direction safeguards."
            elif row["distance_to_tmc_coverage_mi"] <= 5.0:
                treatment, priority = "limited terminal extrapolation review", "low"
                guidance = "The upstream gap is one to five miles; use only a tightly distance-decayed extrapolation after manual continuity review."
            else:
                treatment, priority = "no automatic mapping outside coverage", "none"
                guidance = "The upstream link is more than five miles outside observed coverage; add observations or extend the corridor inventory instead of interpolating."
            return (
                "outside_tmc_coverage_before_start",
                "outside TMC coverage",
                treatment,
                priority,
                guidance,
            )
        if relation == "after_last_tmc":
            if row["distance_to_tmc_coverage_mi"] <= 1.0:
                treatment, priority = "soft terminal extrapolation", "medium"
                guidance = "A short downstream extrapolation is plausible because the link is within one mile of observed coverage; retain topology and direction safeguards."
            elif row["distance_to_tmc_coverage_mi"] <= 5.0:
                treatment, priority = "limited terminal extrapolation review", "low"
                guidance = "The downstream gap is one to five miles; use only a tightly distance-decayed extrapolation after manual continuity review."
            else:
                treatment, priority = "no automatic mapping outside coverage", "none"
                guidance = "The downstream link is more than five miles outside observed coverage; add observations or extend the corridor inventory instead of interpolating."
            return (
                "outside_tmc_coverage_after_end",
                "outside TMC coverage",
                treatment,
                priority,
                guidance,
            )
        return (
            "same_route_name_outside_lateral_tmc_buffer",
            "outside TMC coverage",
            "manual geographic review",
            "low",
            "Do not interpolate automatically; the link is outside the matcher's 3,500-ft corridor buffer and is not a clear terminal continuation.",
        )
    if not row["candidate_direction_pass"]:
        return (
            "direction_alignment_filter",
            "direction/alignment",
            "manual direction review",
            "medium",
            "Review geometry direction or one-way coding; the link failed the matcher's 75-degree direction filter.",
        )
    if row["facility_role"] == "model_connector":
        return (
            "model_connector_not_used_by_selected_tmc_path",
            "connector/ramp",
            "none",
            "high",
            "No observed-speed interpolation is recommended for centroid/model connectors.",
        )
    if row["facility_role"] == "ramp":
        return (
            "ramp_not_used_by_selected_tmc_path",
            "connector/ramp",
            "none",
            "high",
            "Keep ramp treatment separate from mainline TMC interpolation unless a ramp-specific observation is available.",
        )
    if row["physical_facility_class"] == "managed":
        return (
            "eligible_managed_parallel_facility_not_selected",
            "parallel managed facility",
            "separate managed mapping",
            "high",
            "Do not transfer GP observations automatically; use a managed-lane TMC inventory or explicit facility crosswalk.",
        )
    if row["coverage_relation"] == "between_covered_tmc_anchors":
        if row["distance_to_selected_route_ft"] <= 500:
            return (
                "internal_parallel_link_between_tmc_anchors",
                "internal eligible alternative",
                "spatial interpolation candidate",
                "high",
                "High-priority review: the GP link passed all matcher filters, is topologically bounded by selected-route anchors, and is within 500 ft of the selected path.",
            )
        return (
            "internal_alternative_branch_between_tmc_anchors",
            "internal eligible alternative",
            "manual branch review",
            "medium",
            "The link is bounded by selected-route anchors but is a more distant alternative branch; confirm carriageway continuity before interpolation.",
        )
    if row["coverage_relation"] == "within_tmc_spatial_coverage":
        if row["distance_to_selected_route_ft"] <= 500:
            return (
                "internal_parallel_link_within_tmc_coverage",
                "internal eligible alternative",
                "spatial interpolation candidate",
                "high",
                "High-priority review: the GP link passed all filters and closely parallels the selected route inside TMC coverage.",
            )
        return (
            "internal_lateral_branch_within_tmc_coverage",
            "internal eligible alternative",
            "manual branch review",
            "medium",
            "The link passed the candidate filters but is laterally separated from the selected route; do not interpolate until the branch is confirmed.",
        )
    if row["coverage_relation"] in {"before_first_tmc", "after_last_tmc"}:
        return (
            "eligible_terminal_extension_not_required_by_tmc_paths",
            "terminal extension",
            "soft terminal extrapolation",
            "medium",
            "The link passed matcher filters but lies beyond the observed TMC terminal; soft extrapolation is possible with a distance limit.",
        )
    return (
        "eligible_disconnected_or_alternative_named_segment",
        "eligible disconnected alternative",
        "manual geographic review",
        "low",
        "The link passed the matcher filters but is not on a selected path and has no clear bounded or terminal topology relation.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--dashboard-dir", type=Path)
    scope.add_argument(
        "--all-corridor-inputs",
        action="store_true",
        help="Audit every producer-named corridor under --corridor-inputs.",
    )
    parser.add_argument("--corridor-inputs", type=Path, required=True)
    parser.add_argument("--mapmatch-product", type=Path, required=True)
    parser.add_argument("--network-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    mapmatch_product = args.mapmatch_product.resolve()
    output_dir = (
        args.output_dir
        or mapmatch_product / "outputs" / "unmatched-link-audit"
    ).resolve()
    table_dir = output_dir / "tables"
    geo_dir = output_dir / "geospatial"
    table_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    dashboard = args.dashboard_dir.resolve() if args.dashboard_dir else None
    catalog, dashboard_tmc = build_corridor_catalog(
        dashboard,
        args.corridor_inputs.resolve(),
    )
    full_tmc = pd.read_csv(mapmatch_product / "full_tmc_to_link.csv", dtype={"tmc": "string"}, low_memory=False)
    full_tmc["link_id"] = pd.to_numeric(full_tmc["link_id"], errors="raise").astype(int)
    full_corridor = pd.read_csv(mapmatch_product / "full_corridor_links.csv", low_memory=False)
    full_corridor["link_id"] = pd.to_numeric(full_corridor["link_id"], errors="raise").astype(int)
    route_link_ids = set(full_tmc["link_id"])

    links = load_base2025_physical_links(
        args.network_root / "am" / "link.csv",
        args.network_root / "pm" / "link.csv",
        args.network_root / "md" / "link.csv",
    )
    links["link_id"] = pd.to_numeric(links["link_id"], errors="raise").astype(int)
    links = links.drop_duplicates("link_id").set_index("link_id", drop=False)
    street_upper = links["STREETNAME"].fillna("").astype(str).str.upper()

    shared_roads = set(
        pd.read_csv(
            args.metadata,
            usecols=["road"],
            low_memory=False,
        )["road"].dropna().astype(str)
    )
    catalog["inferred_matcher_lane_class"] = catalog["dashboard_road"].map(
        lambda road: infer_corridor_lane_class(road, "auto", shared_roads)
    )

    corridors: dict[str, dict] = {}
    for row in catalog.itertuples(index=False):
        series = pd.Series(row._asdict())
        corridors[row.dashboard_corridor] = corridor_object(
            series,
            dashboard_tmc,
            full_tmc,
            full_corridor,
            links,
            row.inferred_matcher_lane_class,
        )

    road_to_corridors: dict[str, list[dict]] = defaultdict(list)
    for corridor in corridors.values():
        road_to_corridors[corridor["road"]].append(corridor)

    road_named_ids: dict[str, set[int]] = {}
    membership_rows: list[dict] = []
    road_assignments: dict[tuple[str, int], dict] = {}
    for road, candidates in sorted(road_to_corridors.items()):
        mask = street_upper.str.contains(route_pattern(road), regex=True)
        ids = set(links.loc[mask, "link_id"].astype(int))
        road_named_ids[road] = ids
        for link_id in sorted(ids):
            link = links.loc[link_id]
            scored = [score_direction(link, corridor) for corridor in candidates]
            scored.sort(key=lambda item: (item["direction_score"], item["dashboard_corridor"]))
            margin = scored[1]["direction_score"] - scored[0]["direction_score"] if len(scored) > 1 else float("inf")
            for rank, item in enumerate(scored, start=1):
                membership_rows.append(
                    {
                        "link_id": link_id,
                        "STREETNAME": link.get("STREETNAME", ""),
                        "road_membership": road,
                        "direction_candidate_rank": rank,
                        "direction_score_margin_from_best": item["direction_score"] - scored[0]["direction_score"],
                        **item,
                    }
                )
            road_assignments[(road, link_id)] = {**scored[0], "direction_score_margin": margin}

    unmatched_memberships = [
        (road, link_id)
        for road, ids in road_named_ids.items()
        for link_id in ids
        if link_id not in route_link_ids
    ]
    unmatched_ids = sorted({link_id for _, link_id in unmatched_memberships})
    link_candidate_roads: dict[int, list[str]] = defaultdict(list)
    for road, link_id in unmatched_memberships:
        link_candidate_roads[link_id].append(road)

    primary_assignments: dict[int, dict] = {}
    for link_id in unmatched_ids:
        choices = [
            {**road_assignments[(road, link_id)], "road_membership": road}
            for road in link_candidate_roads[link_id]
        ]
        choices.sort(key=lambda item: (item["direction_score"], item["dashboard_corridor"], item["road_membership"]))
        global_margin = choices[1]["direction_score"] - choices[0]["direction_score"] if len(choices) > 1 else float("inf")
        primary_assignments[link_id] = {**choices[0], "global_assignment_margin": global_margin}

    topology: dict[tuple[str, str], dict] = {}
    for corridor in corridors.values():
        road = corridor["road"]
        assigned_named = {
            link_id
            for link_id in road_named_ids[road]
            if road_assignments[(road, link_id)]["dashboard_corridor"] == corridor["key"]
        }
        graph_ids = assigned_named | corridor["corridor_ids"] | corridor["route_ids"]
        for facility in ("gp", "managed", "inactive"):
            adjacency: dict[int, set[int]] = defaultdict(set)
            reverse: dict[int, set[int]] = defaultdict(set)
            for link_id in graph_ids:
                if link_id not in links.index:
                    continue
                link = links.loc[link_id]
                if str(link["facility_class"]) != facility:
                    continue
                u, v = int(link["from_node_id"]), int(link["to_node_id"])
                adjacency[u].add(v)
                reverse[v].add(u)
            anchor_ids = [
                link_id
                for link_id in corridor["route_ids"]
                if link_id in links.index and str(links.loc[link_id, "facility_class"]) == facility
            ]
            anchor_nodes = {
                node
                for link_id in anchor_ids
                for node in (int(links.loc[link_id, "from_node_id"]), int(links.loc[link_id, "to_node_id"]))
            }
            topology[(corridor["key"], facility)] = {
                "anchor_nodes": anchor_nodes,
                "forward": multi_source_reachable(adjacency, anchor_nodes),
                "reverse": multi_source_reachable(reverse, anchor_nodes),
            }

    detailed_rows: list[dict] = []
    features: list[dict] = []
    for link_id in unmatched_ids:
        link = links.loc[link_id]
        assignment = primary_assignments[link_id]
        corridor = corridors[assignment["dashboard_corridor"]]
        facility = str(link["facility_class"])
        topo = topology.get((corridor["key"], facility), {"anchor_nodes": set(), "forward": set(), "reverse": set()})
        u, v = int(link["from_node_id"]), int(link["to_node_id"])
        upstream_reachable = u in topo["forward"]
        downstream_reachable = v in topo["reverse"]
        if upstream_reachable and downstream_reachable:
            topology_relation = "bounded_between_anchors"
        elif upstream_reachable:
            topology_relation = "reachable_downstream_from_anchor"
        elif downstream_reachable:
            topology_relation = "can_reach_upstream_anchor"
        else:
            topology_relation = "disconnected_from_anchor_graph"

        spatial_pass = assignment["distance_to_tmc_coverage_ft"] <= MATCHER_BUFFER_FT
        direction_pass = assignment["minimum_corridor_bearing_diff_deg"] <= MATCHER_ANGLE_DEG
        lane_pass = lane_class_pass(link, assignment["inferred_matcher_lane_class"])
        candidate_pass = spatial_pass and direction_pass and lane_pass
        terminal_relation, terminal_ray_score, terminal_lateral = terminal_ray_relation(
            link["geometry_obj"].interpolate(0.5, normalized=True), corridor
        )
        rank_pct = assignment["nearest_tmc_rank_pct"]
        if spatial_pass:
            coverage_relation = (
                "between_covered_tmc_anchors"
                if topology_relation == "bounded_between_anchors"
                else "within_tmc_spatial_coverage"
            )
        elif topology_relation == "can_reach_upstream_anchor" and rank_pct <= 0.25:
            coverage_relation = "before_first_tmc"
        elif topology_relation == "reachable_downstream_from_anchor" and rank_pct >= 0.75:
            coverage_relation = "after_last_tmc"
        elif terminal_relation == "before_first_tmc" and rank_pct <= 0.25:
            coverage_relation = "before_first_tmc"
        elif terminal_relation == "after_last_tmc" and rank_pct >= 0.75:
            coverage_relation = "after_last_tmc"
        else:
            coverage_relation = "lateral_or_disconnected_outside_coverage"

        role = facility_role(link.get("FTYPE"), link.get("STREETNAME"))
        transition_only = link_id in corridor["transition_ids"] or (
            link_id in set(full_corridor.loc[full_corridor["link_role"].astype(str).str.contains("transition"), "link_id"])
        )
        direction_margin = assignment["direction_score_margin"]
        if math.isinf(direction_margin) or direction_margin >= 1500:
            direction_confidence = "high"
        elif direction_margin >= 400:
            direction_confidence = "medium"
        else:
            direction_confidence = "low"

        filter_failures = []
        if not spatial_pass:
            filter_failures.append("outside_3500ft_spatial_buffer")
        if not lane_pass:
            filter_failures.append("lane_class_filter")
        if not direction_pass:
            filter_failures.append("over_75deg_direction_filter")

        row = {
            "link_id": link_id,
            "from_node_id": u,
            "to_node_id": v,
            "STREETNAME": link.get("STREETNAME", ""),
            "dashboard_road_candidates": "; ".join(sorted(link_candidate_roads[link_id])),
            "assigned_dashboard_road": assignment["dashboard_road"],
            "assigned_dashboard_corridor": assignment["dashboard_corridor"],
            "assigned_dashboard_direction": assignment["dashboard_direction"],
            "direction_assignment_confidence": direction_confidence,
            "direction_score": assignment["direction_score"],
            "direction_score_margin": None if math.isinf(direction_margin) else direction_margin,
            "physical_facility_class": facility,
            "facility_role": role,
            "facility_category": f"{facility}_{role}",
            "inferred_matcher_lane_class": assignment["inferred_matcher_lane_class"],
            "FTYPE": safe_float(link.get("FTYPE")),
            "link_type": safe_float(link.get("link_type")),
            "lanes": safe_float(link.get("lanes")),
            "capacity": safe_float(link.get("capacity")),
            "free_speed_mph": safe_float(link.get("vdf_free_speed_mph", link.get("free_speed"))),
            "length_mi": safe_float(link.get("length_mi")),
            "allowed_use": link.get("allowed_use", ""),
            "AMLIMIT": safe_float(link.get("AMLIMIT")),
            "MDLIMIT": safe_float(link.get("MDLIMIT")),
            "PMLIMIT": safe_float(link.get("PMLIMIT")),
            "candidate_spatial_buffer_pass": spatial_pass,
            "candidate_lane_class_pass": lane_pass,
            "candidate_direction_pass": direction_pass,
            "candidate_pool_eligible_but_not_selected": candidate_pass,
            "candidate_filter_failures": "; ".join(filter_failures),
            "distance_to_tmc_coverage_ft": assignment["distance_to_tmc_coverage_ft"],
            "distance_to_tmc_coverage_mi": assignment["distance_to_tmc_coverage_ft"] / FT_PER_MILE,
            "distance_to_selected_route_ft": assignment["distance_to_selected_route_ft"],
            "bearing_diff_to_nearest_tmc_deg": assignment["bearing_diff_to_nearest_tmc_deg"],
            "minimum_corridor_bearing_diff_deg": assignment["minimum_corridor_bearing_diff_deg"],
            "nearest_tmc": assignment["nearest_tmc"],
            "nearest_tmc_road_order": assignment["nearest_tmc_road_order"],
            "nearest_tmc_rank": assignment["nearest_tmc_rank"],
            "nearest_tmc_rank_pct": assignment["nearest_tmc_rank_pct"],
            "nearest_selected_route_link_id": assignment["nearest_selected_route_link_id"],
            "topology_relation": topology_relation,
            "shares_selected_route_node": u in topo["anchor_nodes"] or v in topo["anchor_nodes"],
            "coverage_relation": coverage_relation,
            "terminal_ray_relation": terminal_relation,
            "terminal_ray_score_ft": None if not math.isfinite(terminal_ray_score) else terminal_ray_score,
            "terminal_ray_lateral_ft": None if not math.isfinite(terminal_lateral) else terminal_lateral,
            "transition_only": transition_only,
            "in_full_corridor_links": link_id in set(full_corridor["link_id"]),
        }
        distance_mi = row["distance_to_tmc_coverage_mi"]
        if spatial_pass:
            row["terminal_distance_class"] = "within_matcher_buffer"
        elif distance_mi <= 1.0:
            row["terminal_distance_class"] = "0.66_to_1_mile"
        elif distance_mi <= 5.0:
            row["terminal_distance_class"] = "1_to_5_miles"
        elif distance_mi <= 10.0:
            row["terminal_distance_class"] = "5_to_10_miles"
        else:
            row["terminal_distance_class"] = "over_10_miles"
        reason, group, treatment, priority, recommendation = reason_for(row)
        row.update(
            {
                "reason_tag": reason,
                "reason_group": group,
                "recommended_treatment": treatment,
                "review_priority": priority,
                "recommendation": recommendation,
                "reason_evidence": (
                    f"spatial={spatial_pass} ({row['distance_to_tmc_coverage_ft']:.0f} ft vs {MATCHER_BUFFER_FT:.0f}); "
                    f"lane={lane_pass} ({row['inferred_matcher_lane_class']} vs {facility}); "
                    f"direction={direction_pass} ({row['minimum_corridor_bearing_diff_deg']:.1f} deg vs {MATCHER_ANGLE_DEG:.0f}); "
                    f"route distance={row['distance_to_selected_route_ft']:.0f} ft; topology={topology_relation}; "
                    f"coverage={coverage_relation}; nearest TMC={row['nearest_tmc']} order={row['nearest_tmc_road_order']}"
                ),
                "geometry_wgs84": link.get("geometry_wgs84", ""),
                "geometry_projected": link["geometry_obj"].wkt,
            }
        )
        detailed_rows.append(row)
        feature_properties = {key: to_native(value) for key, value in row.items() if not key.startswith("geometry_")}
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(wkt.loads(str(link["geometry_wgs84"]))),
                "properties": feature_properties,
            }
        )

    detail = pd.DataFrame(detailed_rows).sort_values(
        ["assigned_dashboard_corridor", "reason_group", "facility_category", "link_id"]
    )
    direction_facility = (
        detail.groupby(
            ["assigned_dashboard_corridor", "assigned_dashboard_road", "assigned_dashboard_direction", "facility_category"],
            dropna=False,
            as_index=False,
        )
        .agg(link_count=("link_id", "size"), total_length_mi=("length_mi", "sum"))
        .sort_values(["assigned_dashboard_corridor", "link_count", "facility_category"], ascending=[True, False, True])
    )
    direction_reason = (
        detail.groupby(
            ["assigned_dashboard_corridor", "assigned_dashboard_road", "assigned_dashboard_direction", "reason_tag"],
            dropna=False,
            as_index=False,
        )
        .agg(
            link_count=("link_id", "size"),
            total_length_mi=("length_mi", "sum"),
            median_tmc_distance_ft=("distance_to_tmc_coverage_ft", "median"),
            median_route_distance_ft=("distance_to_selected_route_ft", "median"),
        )
        .sort_values(["assigned_dashboard_corridor", "link_count", "reason_tag"], ascending=[True, False, True])
    )
    reason_summary = (
        detail.groupby(["reason_group", "reason_tag", "recommended_treatment", "review_priority"], as_index=False)
        .agg(
            link_count=("link_id", "size"),
            total_length_mi=("length_mi", "sum"),
            corridor_count=("assigned_dashboard_corridor", "nunique"),
            median_tmc_distance_ft=("distance_to_tmc_coverage_ft", "median"),
            median_route_distance_ft=("distance_to_selected_route_ft", "median"),
        )
        .sort_values(["link_count", "reason_tag"], ascending=[False, True])
    )
    actionable = detail[
        detail["recommended_treatment"].isin(
            [
                "spatial interpolation candidate",
                "soft terminal extrapolation",
                "limited terminal extrapolation review",
                "manual branch review",
            ]
        )
    ].copy()
    high_priority_internal = detail[
        detail["reason_tag"].isin(
            ["internal_parallel_link_between_tmc_anchors", "internal_parallel_link_within_tmc_coverage"]
        )
    ].copy()

    detail.to_csv(table_dir / "unmatched_links_link_by_link_reasons.csv", index=False)
    direction_facility.to_csv(table_dir / "direction_facility_summary.csv", index=False)
    direction_reason.to_csv(table_dir / "direction_reason_summary.csv", index=False)
    reason_summary.to_csv(table_dir / "reason_summary.csv", index=False)
    actionable.to_csv(table_dir / "actionable_interpolation_and_extrapolation_candidates.csv", index=False)
    high_priority_internal.to_csv(table_dir / "high_priority_internal_spatial_interpolation_candidates.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(table_dir / "all_direction_candidate_scores.csv", index=False)
    catalog.to_csv(table_dir / "dashboard_corridor_catalog.csv", index=False)
    (geo_dir / "unmatched_links_link_by_link_reasons.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )

    method = {
        "dashboard": str(dashboard) if dashboard is not None else None,
        "corridor_scope": (
            "dashboard reports" if dashboard is not None else "all corridor inputs"
        ),
        "mapmatch_product": str(mapmatch_product),
        "network": str(args.network_root.resolve()),
        "matcher_candidate_rules": {
            "spatial_buffer_ft": MATCHER_BUFFER_FT,
            "direction_tolerance_deg": MATCHER_ANGLE_DEG,
            "lane_class": "The same infer_corridor_lane_class(auto) and GP/managed physical classification used by the matcher.",
        },
        "direction_assignment": "Lowest link-specific score across dashboard directions for the same road: distance to TMC geometry + 22*bearing difference + 0.10*distance to selected route.",
        "facility_roles": {
            "FTYPE 0": "model connector",
            "FTYPE 1": "freeway mainline",
            "FTYPE 2": "principal arterial",
            "FTYPE 3": "minor arterial",
            "FTYPE 4": "collector/local",
            "FTYPE 5": "expressway/parkway",
            "FTYPE 6": "ramp",
        },
        "coverage_relation": "Directed topology against selected-route anchor nodes plus the exact 3,500-ft TMC spatial buffer and terminal-ray evidence.",
        "limitations": [
            "The original run did not retain every losing path candidate, so eligible-but-not-selected reasons are reconstructed from the same candidate filters, final selected topology, and geometry rather than an archived per-path rejection log.",
            "STREETNAME is not directional; direction is inferred link by link and low-margin assignments are explicitly flagged.",
            "A spatial interpolation candidate still requires a carriageway/branch review before attributes are transferred.",
        ],
    }
    (output_dir / "methodology.json").write_text(json.dumps(method, indent=2), encoding="utf-8")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "unique_unmatched_links": len(detail),
        "dashboard_corridors": int(catalog["dashboard_corridor"].nunique()),
        "dashboard_roads": int(catalog["dashboard_road"].nunique()),
        "candidate_pool_eligible_but_not_selected": int(detail["candidate_pool_eligible_but_not_selected"].sum()),
        "high_priority_internal_spatial_interpolation_candidates": len(high_priority_internal),
        "actionable_candidate_count": len(actionable),
        "direction_low_confidence_count": int(detail["direction_assignment_confidence"].eq("low").sum()),
        "reason_counts": detail["reason_tag"].value_counts().to_dict(),
        "facility_counts": detail["facility_category"].value_counts().to_dict(),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
