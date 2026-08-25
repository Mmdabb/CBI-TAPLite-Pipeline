from __future__ import annotations

import heapq
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree


MODEL_CRS_EPSG = 2248
BASE2025_CRS_EPSG = 4326
PERIODS = ("AM", "MD", "PM")
VALID_LIMIT_CODES = {0, 2, 4, 5, 9}
GP_LIMIT_CODES = {0}
RESTRICTED_LIMIT_CODES = {2, 4, 5}
CLOSED_LIMIT_CODE = 9
LIMIT_ALLOWED_USE = {
    0: "sov;hov2;hov3;trk;apv;com",
    2: "hov2;hov3",
    4: "sov;hov2;hov3;com;apv",
    5: "apv",
    9: "closed",
}


@dataclass(frozen=True)
class MatchConfig:
    road: str = "I-66"
    direction: str = "EASTBOUND"
    lane_class: str = "all_open"
    corridor_buffer_ft: float = 3500.0
    endpoint_buffer_ft: float = 3500.0
    candidate_angle_tolerance_deg: float = 75.0
    corridor_angle_tolerance_deg: float = 75.0
    top_endpoint_candidates: int = 8
    max_route_ratio_for_search: float = 4.5
    endpoint_name_mismatch_penalty_ft: float = 1400.0
    endpoint_name_selection_penalty: float = 300.0
    route_search_buffer_ft: float = 2600.0
    max_candidate_paths: int = 6
    ambiguous_route_margin: float = 2.0

    @property
    def slug(self) -> str:
        road = re.sub(r"[^a-z0-9]+", "_", self.road.lower()).strip("_")
        direction = re.sub(r"[^a-z0-9]+", "_", self.direction.lower()).strip("_")
        lane = re.sub(r"[^a-z0-9]+", "_", self.lane_class.lower()).strip("_")
        return f"{road}_{direction}_{lane}"


def directed_angle_diff(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return abs((a - b + 180) % 360 - 180)


def bearing(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return float("nan")
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    out = df.head(max_rows).copy() if max_rows else df.copy()
    out = out.fillna("")
    lines = [
        "| " + " | ".join(map(str, out.columns)) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in out.columns) + " |")
    return "\n".join(lines)


def load_tmc(config: MatchConfig, tmc_file: Path, model_crs_epsg: int = MODEL_CRS_EPSG) -> pd.DataFrame:
    tmc = pd.read_csv(tmc_file, dtype={"tmc": "string"})
    tmc = tmc[(tmc["road"] == config.road) & (tmc["direction"] == config.direction)].copy()
    tmc = tmc.sort_values(["road_order", "tmc"]).reset_index(drop=True)
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


def load_model_links(link_file: Path) -> pd.DataFrame:
    wanted = [
        "link_id",
        "from_node_id",
        "to_node_id",
        "length",
        "length_in_mile",
        "lanes",
        "capacity",
        "free_speed",
        "link_type",
        "FTYPE",
        "vdf_free_speed_mph",
        "allowed_use",
        "STREETNAME",
        "PMLIMIT",
        "PROJECTID",
        "LINKID",
        "geometry",
    ]
    links = pd.read_csv(link_file, usecols=lambda c: c in wanted, low_memory=False)
    links["geometry_obj"] = links["geometry"].map(wkt.loads)
    links["model_bearing"] = links["geometry_obj"].map(bearing)
    links["street_upper"] = links["STREETNAME"].fillna("").astype(str).str.upper()
    links["allowed_upper"] = links["allowed_use"].fillna("").astype(str).str.upper()
    links["length_mi"] = pd.to_numeric(links["length_in_mile"], errors="coerce")
    missing_len = links["length_mi"].isna() | (links["length_mi"] <= 0)
    if missing_len.any():
        links.loc[missing_len, "length_mi"] = links.loc[missing_len, "geometry_obj"].map(lambda g: g.length / 5280.0)
    links["length_ft_geom"] = links["geometry_obj"].map(lambda g: float(g.length))
    return links


def load_model_nodes(node_file: Path) -> pd.DataFrame:
    nodes = pd.read_csv(node_file, usecols=["node_id", "zone_id", "x_coord", "y_coord"])
    nodes["geometry_obj"] = [Point(x, y) for x, y in zip(nodes["x_coord"], nodes["y_coord"])]
    return nodes


def _period_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


def _read_link_subset(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    wanted = set(columns)
    return pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)


def _is_open_allowed(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().upper()
    return text not in {"", "0", "CLOSED", "NAN", "NONE"}


def _union_allowed_use(row: pd.Series) -> str:
    modes: list[str] = []
    seen: set[str] = set()
    for period in PERIODS:
        col = f"{period.lower()}_allowed_use"
        value = row.get(col)
        if not _is_open_allowed(value):
            continue
        for part in str(value).split(";"):
            mode = part.strip()
            key = mode.lower()
            if mode and key not in seen:
                modes.append(mode)
                seen.add(key)
    return ";".join(modes) if modes else "closed"


def _period_status(link_file: Path, period: str) -> pd.DataFrame:
    period_lower = period.lower()
    limit_col = f"{period.upper()}LIMIT"
    lane_col = f"{period.upper()}LANE"
    toll_col = f"{period.upper()}TOLL"
    available = _period_columns(link_file)
    wanted = {"link_id", "allowed_use", "lanes", "toll", limit_col, lane_col, toll_col}
    df = _read_link_subset(link_file, wanted & available)
    if limit_col not in df.columns:
        df[limit_col] = df.get("allowed_use", pd.Series(index=df.index, dtype="object")).map(
            lambda value: 0 if _is_open_allowed(value) else CLOSED_LIMIT_CODE
        )
    if lane_col not in df.columns:
        df[lane_col] = df["lanes"] if "lanes" in df.columns else pd.NA
    if toll_col not in df.columns:
        df[toll_col] = df["toll"] if "toll" in df.columns else pd.NA
    keep = ["link_id", "allowed_use", "lanes", "toll", limit_col, lane_col, toll_col]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[keep].copy()
    out = out.rename(
        columns={
            "allowed_use": f"{period_lower}_allowed_use",
            "lanes": f"{period_lower}_lanes",
            "toll": f"{period_lower}_toll",
            limit_col: f"{period_lower}_limit",
            lane_col: f"{period_lower}_period_lanes",
            toll_col: f"{period_lower}_period_toll",
        }
    )
    out[f"{period_lower}_limit"] = pd.to_numeric(out[f"{period_lower}_limit"], errors="coerce").fillna(0)
    out[f"{period_lower}_is_open"] = out[f"{period_lower}_allowed_use"].map(_is_open_allowed)
    return out


def load_base2025_physical_links(
    am_link_file: Path,
    pm_link_file: Path,
    md_link_file: Path,
    source_crs_epsg: int = BASE2025_CRS_EPSG,
    working_crs_epsg: int = MODEL_CRS_EPSG,
) -> pd.DataFrame:
    """Load AM/MD/PM links as one period-independent physical network.

    The MD network is the complete physical topology in the current input, and
    the AM/MD/PM restrictions are retained as attributes. Closed reversible
    lanes are still allowed to participate in geometric/path matching.
    """
    base_cols = {
        "link_id",
        "from_node_id",
        "to_node_id",
        "length",
        "length_in_mile",
        "lanes",
        "capacity",
        "free_speed",
        "link_type",
        "FTYPE",
        "vdf_free_speed_mph",
        "STREETNAME",
        "AMLIMIT",
        "MDLIMIT",
        "PMLIMIT",
        "PROJECTID",
        "LINKID",
        "geometry",
        "crs",
    }
    am_base = _read_link_subset(am_link_file, base_cols)
    md_base = _read_link_subset(md_link_file, base_cols)
    pm_base = _read_link_subset(pm_link_file, base_cols)
    links = (
        pd.concat([md_base, am_base, pm_base], ignore_index=True)
        .sort_values(["link_id"])
        .drop_duplicates("link_id", keep="first")
        .reset_index(drop=True)
    )

    period_files = {"AM": am_link_file, "MD": md_link_file, "PM": pm_link_file}
    for period, link_file in period_files.items():
        links = links.merge(_period_status(link_file, period), on="link_id", how="left")
        prefix = period.lower()
        limit_col = f"{prefix}_limit"
        raw_limit_col = f"{period}LIMIT"
        raw_limits = pd.to_numeric(links.get(raw_limit_col), errors="coerce")
        links[limit_col] = pd.to_numeric(links[limit_col], errors="coerce").combine_first(raw_limits).fillna(
            CLOSED_LIMIT_CODE
        )
        allowed_col = f"{prefix}_allowed_use"
        derived_allowed = links[limit_col].map(lambda value: LIMIT_ALLOWED_USE.get(int(value), "closed"))
        links[allowed_col] = links[allowed_col].combine_first(derived_allowed).fillna("closed")
        links[f"{prefix}_is_open"] = links[allowed_col].map(_is_open_allowed)

    links["allowed_use"] = links.apply(_union_allowed_use, axis=1)
    links["allowed_upper"] = links["allowed_use"].fillna("").astype(str).str.upper()
    links["street_upper"] = links["STREETNAME"].fillna("").astype(str).str.upper()
    links["length_mi"] = pd.to_numeric(links.get("length_in_mile"), errors="coerce")
    links["geometry_wgs84"] = links["geometry"].astype(str)

    transformer = None
    if int(source_crs_epsg) != int(working_crs_epsg):
        transformer = Transformer.from_crs(source_crs_epsg, working_crs_epsg, always_xy=True)

    geometries = []
    for geom_text in links["geometry_wgs84"]:
        geom = wkt.loads(geom_text)
        if transformer is not None:
            geom = shapely_transform(transformer.transform, geom)
        geometries.append(geom)
    links["geometry_obj"] = geometries
    links["geometry"] = links["geometry_obj"].map(lambda geom: geom.wkt)

    missing_len = links["length_mi"].isna() | (links["length_mi"] <= 0)
    if missing_len.any():
        links.loc[missing_len, "length_mi"] = links.loc[missing_len, "geometry_obj"].map(lambda g: g.length / 5280.0)
    links["length_ft_geom"] = links["geometry_obj"].map(lambda g: float(g.length))
    links["model_bearing"] = links["geometry_obj"].map(bearing)
    for period in PERIODS:
        links[f"{period}LIMIT"] = pd.to_numeric(links.get(f"{period.lower()}_limit"), errors="coerce").fillna(
            CLOSED_LIMIT_CODE
        )
    period_limits = pd.DataFrame(
        {period: links[f"{period}LIMIT"].astype(int) for period in PERIODS},
        index=links.index,
    )
    valid_limits = period_limits.isin(VALID_LIMIT_CODES).all(axis=1)
    active_in_either_period = period_limits.ne(CLOSED_LIMIT_CODE).any(axis=1)
    managed_name = links["street_upper"].str.contains("HOV|EXPRESS|HOT", regex=True)
    restricted_period = period_limits.isin(RESTRICTED_LIMIT_CODES).any(axis=1)
    reversible_period = period_limits.eq(CLOSED_LIMIT_CODE).any(axis=1) & period_limits.nunique(axis=1).gt(1)
    links["limit_codes_valid"] = valid_limits
    links["physical_active"] = valid_limits & active_in_either_period
    links["physical_managed"] = links["physical_active"] & (
        managed_name | restricted_period | reversible_period
    )
    links["physical_gp"] = (
        links["physical_active"]
        & ~managed_name
        & period_limits.isin(GP_LIMIT_CODES).all(axis=1)
    )
    links["facility_class"] = "inactive"
    links.loc[links["physical_gp"], "facility_class"] = "gp"
    links.loc[links["physical_managed"], "facility_class"] = "managed"
    return links


def load_base2025_physical_nodes(
    am_node_file: Path,
    pm_node_file: Path,
    md_node_file: Path,
    source_crs_epsg: int = BASE2025_CRS_EPSG,
    working_crs_epsg: int = MODEL_CRS_EPSG,
) -> pd.DataFrame:
    am_nodes = pd.read_csv(am_node_file, usecols=["node_id", "zone_id", "x_coord", "y_coord"])
    md_nodes = pd.read_csv(md_node_file, usecols=["node_id", "zone_id", "x_coord", "y_coord"])
    pm_nodes = pd.read_csv(pm_node_file, usecols=["node_id", "zone_id", "x_coord", "y_coord"])
    nodes = (
        pd.concat([md_nodes, am_nodes, pm_nodes], ignore_index=True)
        .sort_values("node_id")
        .drop_duplicates("node_id", keep="first")
        .reset_index(drop=True)
    )
    transformer = None
    if int(source_crs_epsg) != int(working_crs_epsg):
        transformer = Transformer.from_crs(source_crs_epsg, working_crs_epsg, always_xy=True)
    if transformer is not None:
        x, y = transformer.transform(nodes["x_coord"].to_numpy(), nodes["y_coord"].to_numpy())
        nodes["x_coord"] = x
        nodes["y_coord"] = y
    nodes["geometry_obj"] = [Point(x, y) for x, y in zip(nodes["x_coord"], nodes["y_coord"])]
    return nodes


def select_corridor_links(config: MatchConfig, tmc: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    corridor_geom = unary_union(tmc["geometry_obj"].tolist()).buffer(config.corridor_buffer_ft)
    intersects_buffer = links["geometry_obj"].map(lambda geom: geom.intersects(corridor_geom))
    road_name = links["street_upper"].str.contains(config.road.upper(), regex=False)
    physical_active = links.get("physical_active", pd.Series(True, index=links.index)).fillna(False).astype(bool)
    physical_gp = links.get("physical_gp", pd.Series(False, index=links.index)).fillna(False).astype(bool)
    physical_managed = links.get("physical_managed", pd.Series(False, index=links.index)).fillna(False).astype(bool)

    if config.lane_class == "gp":
        lane_class = physical_gp
    elif config.lane_class == "managed":
        lane_class = physical_managed
    else:
        lane_class = physical_active

    selected = links[intersects_buffer & lane_class].copy()
    selected["road_name_match"] = road_name.loc[selected.index].astype(bool)
    if selected.empty:
        return selected.sort_values(["link_id"]).reset_index(drop=True)
    tmc_bearings = tmc["tmc_bearing"].dropna().astype(float).tolist()
    selected = selected[
        selected["model_bearing"].map(
            lambda link_bearing: min(directed_angle_diff(float(link_bearing), b) for b in tmc_bearings)
            <= config.corridor_angle_tolerance_deg
        )
    ].copy()
    if selected.empty:
        return selected.sort_values(["link_id"]).reset_index(drop=True)
    selected = selected.sort_values(["link_id"]).reset_index(drop=True)
    return selected


def build_adjacency(links: pd.DataFrame) -> dict[int, list[tuple[int, int, float]]]:
    adjacency: dict[int, list[tuple[int, int, float]]] = {}
    for row in links.itertuples(index=False):
        u = int(row.from_node_id)
        v = int(row.to_node_id)
        lid = int(row.link_id)
        length_mi = float(row.length_mi)
        adjacency.setdefault(u, []).append((v, lid, length_mi))
    for rows in adjacency.values():
        rows.sort(key=lambda item: (item[2], item[1]))
    return adjacency


def shortest_link_path(
    adjacency: dict[int, list[tuple[int, int, float]]],
    origin: int,
    destination: int,
    cutoff_mi: float | None = None,
) -> tuple[list[int], float]:
    if origin == destination:
        return [], 0.0
    heap: list[tuple[float, int]] = [(0.0, origin)]
    best = {origin: 0.0}
    prev: dict[int, tuple[int, int, float]] = {}
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > best.get(node, float("inf")):
            continue
        if cutoff_mi is not None and dist > cutoff_mi:
            continue
        if node == destination:
            links: list[int] = []
            cur = destination
            while cur != origin:
                pnode, plink, plen = prev[cur]
                links.append(plink)
                cur = pnode
            links.reverse()
            return links, dist
        for nxt, link_id, length_mi in adjacency.get(node, []):
            ndist = dist + length_mi
            if cutoff_mi is not None and ndist > cutoff_mi:
                continue
            if ndist < best.get(nxt, float("inf")):
                best[nxt] = ndist
                prev[nxt] = (node, link_id, length_mi)
                heapq.heappush(heap, (ndist, nxt))
    return [], float("inf")


def make_route(start_link: pd.Series, end_link: pd.Series, adjacency: dict[int, list[tuple[int, int, float]]], cutoff_mi: float) -> list[int] | None:
    start_lid = int(start_link["link_id"])
    end_lid = int(end_link["link_id"])
    if start_lid == end_lid:
        return [start_lid]
    mid_links, mid_len = shortest_link_path(
        adjacency,
        int(start_link["to_node_id"]),
        int(end_link["from_node_id"]),
        cutoff_mi=cutoff_mi,
    )
    if math.isinf(mid_len):
        return None
    route = [start_lid] + mid_links + [end_lid]
    deduped: list[int] = []
    for lid in route:
        if not deduped or deduped[-1] != lid:
            deduped.append(lid)
    return deduped


def endpoint_candidates(
    point: Point,
    tmc_line: LineString,
    tmc_bearing: float,
    role: str,
    selected_links: pd.DataFrame,
    nodes_by_id: pd.DataFrame,
    tree: STRtree,
    config: MatchConfig,
) -> list[dict[str, object]]:
    idxs = list(tree.query(point.buffer(config.endpoint_buffer_ft)))
    if not idxs:
        idxs = [int(tree.nearest(point))]
    rows: list[dict[str, object]] = []
    for idx in idxs:
        link = selected_links.iloc[int(idx)]
        angle = directed_angle_diff(float(link["model_bearing"]), float(tmc_bearing))
        if angle > config.candidate_angle_tolerance_deg:
            continue
        endpoint_node_id = int(link["from_node_id"] if role == "origin" else link["to_node_id"])
        if endpoint_node_id not in nodes_by_id.index:
            continue
        endpoint = nodes_by_id.loc[endpoint_node_id]
        point_to_link_ft = float(point.distance(link["geometry_obj"]))
        point_to_endpoint_ft = float(point.distance(endpoint["geometry_obj"]))
        line_to_link_ft = float(tmc_line.distance(link["geometry_obj"]))
        road_name_match = bool(link.get("road_name_match", True))
        name_penalty = 0.0 if road_name_match else config.endpoint_name_mismatch_penalty_ft
        score = (
            point_to_link_ft
            + 0.18 * point_to_endpoint_ft
            + 0.20 * line_to_link_ft
            + 22.0 * angle
            + name_penalty
        )
        rows.append(
            {
                "link": link,
                "node_id": endpoint_node_id,
                "score": score,
                "point_to_link_ft": point_to_link_ft,
                "point_to_endpoint_ft": point_to_endpoint_ft,
                "line_to_link_ft": line_to_link_ft,
                "angle_diff": angle,
                "road_name_match": road_name_match,
                "name_penalty": name_penalty,
            }
        )
    rows.sort(key=lambda r: (float(r["score"]), int(r["link"]["link_id"])))
    return rows[: config.top_endpoint_candidates]


def route_metrics(
    route_link_ids: list[int],
    links_by_id: pd.DataFrame,
    tmc_line: LineString,
    tmc_bearing: float,
    tmc_miles: float,
    start_candidate: dict[str, object],
    end_candidate: dict[str, object],
) -> dict[str, object]:
    route_links = links_by_id.loc[route_link_ids]
    length_mi = float(route_links["length_mi"].sum())
    length_ft = route_links["length_ft_geom"].clip(lower=1).to_numpy()
    offsets = route_links["geometry_obj"].map(lambda geom: float(geom.distance(tmc_line))).to_numpy()
    bearings = route_links["model_bearing"].map(lambda b: directed_angle_diff(float(b), float(tmc_bearing))).to_numpy()
    avg_offset = float((offsets * length_ft).sum() / max(length_ft.sum(), 1.0))
    max_offset = float(offsets.max()) if len(offsets) else float("nan")
    avg_bearing = float((bearings * length_ft).sum() / max(length_ft.sum(), 1.0))
    length_ratio = length_mi / max(float(tmc_miles), 1e-6)
    duplicate_links = len(route_link_ids) - len(set(route_link_ids))

    road_name_match = route_links.get("road_name_match", pd.Series(True, index=route_links.index)).fillna(False).astype(bool)
    street = route_links.get("street_upper", pd.Series("", index=route_links.index)).fillna("").astype(str).str.strip()
    unnamed = street.isin({"", "0", "NAN", "NONE", "RAMP"}) | street.str.contains("RAMP", regex=False)
    total_route_ft = max(float(length_ft.sum()), 1.0)
    named_mismatch_share = float((length_ft * (~road_name_match & ~unnamed).to_numpy()).sum() / total_route_ft)
    unnamed_share = float((length_ft * unnamed.to_numpy()).sum() / total_route_ft)

    ftype = pd.to_numeric(route_links.get("FTYPE", pd.Series(float("nan"), index=route_links.index)), errors="coerce")
    speed = pd.to_numeric(
        route_links.get("vdf_free_speed_mph", route_links.get("free_speed", pd.Series(float("nan"), index=route_links.index))),
        errors="coerce",
    )
    capacity = pd.to_numeric(route_links.get("capacity", pd.Series(float("nan"), index=route_links.index)), errors="coerce")
    facility_class = route_links.get("facility_class", pd.Series("", index=route_links.index)).fillna("").astype(str)
    attribute_discontinuity_count = 0
    facility_class_switch_count = 0
    for position in range(1, len(route_links)):
        prev_ftype, curr_ftype = ftype.iloc[position - 1], ftype.iloc[position]
        if pd.notna(prev_ftype) and pd.notna(curr_ftype) and prev_ftype > 0 and curr_ftype > 0 and prev_ftype != curr_ftype:
            attribute_discontinuity_count += 1
        prev_speed, curr_speed = speed.iloc[position - 1], speed.iloc[position]
        if pd.notna(prev_speed) and pd.notna(curr_speed) and min(prev_speed, curr_speed) > 0:
            if max(prev_speed, curr_speed) / min(prev_speed, curr_speed) > 1.6:
                attribute_discontinuity_count += 1
        prev_capacity, curr_capacity = capacity.iloc[position - 1], capacity.iloc[position]
        if pd.notna(prev_capacity) and pd.notna(curr_capacity) and min(prev_capacity, curr_capacity) > 0:
            if max(prev_capacity, curr_capacity) / min(prev_capacity, curr_capacity) > 2.0:
                attribute_discontinuity_count += 1
        prev_class, curr_class = facility_class.iloc[position - 1], facility_class.iloc[position]
        if prev_class and curr_class and prev_class != curr_class:
            facility_class_switch_count += 1

    ratio_penalty = abs(math.log(max(length_ratio, 1e-6))) * 28.0
    if 0.45 <= length_ratio <= 2.30:
        ratio_penalty *= 0.25
    score = (
        avg_offset / 45.0
        + max_offset / 120.0
        + avg_bearing * 1.4
        + ratio_penalty
        + float(start_candidate["point_to_endpoint_ft"]) / 850.0
        + float(end_candidate["point_to_endpoint_ft"]) / 850.0
        + duplicate_links * 25.0
        + named_mismatch_share * 3.0
        + unnamed_share
        + attribute_discontinuity_count * 5.0
        + facility_class_switch_count * 25.0
    )
    confidence = max(0.0, min(100.0, 100.0 - score))
    return {
        "route_length_mi": length_mi,
        "length_ratio": length_ratio,
        "avg_offset_ft": avg_offset,
        "max_offset_ft": max_offset,
        "avg_bearing_diff_deg": avg_bearing,
        "duplicate_link_count": duplicate_links,
        "road_name_mismatch_share": named_mismatch_share,
        "unnamed_link_share": unnamed_share,
        "attribute_discontinuity_count": attribute_discontinuity_count,
        "facility_class_switch_count": facility_class_switch_count,
        "score": score,
        "confidence": confidence,
    }


def classify_match(row: dict[str, object]) -> str:
    if row.get("route_link_count", 0) == 0:
        return "no_path"
    if row["duplicate_link_count"] > 0:
        return "review_loop"
    if row.get("facility_class_switch_count", 0) > 0:
        return "review_facility_class"
    if row.get("corridor_transition_status") == "disconnected":
        return "review_corridor_continuity"
    if row["start_endpoint_distance_ft"] > 3200 or row["end_endpoint_distance_ft"] > 3200:
        return "review_od_distance"
    if row["length_ratio"] < 0.35 or row["length_ratio"] > 3.0:
        return "review_length"
    if row["avg_offset_ft"] > 900 or row["max_offset_ft"] > 2600:
        return "review_geometry"
    if row["avg_bearing_diff_deg"] > 55:
        return "review_direction"
    margin = row.get("candidate_score_margin", "")
    if margin != "" and pd.notna(margin) and float(margin) < 2.0:
        return "review_ambiguous_route"
    if row["confidence"] >= 55:
        return "matched"
    return "review_low_confidence"


def match_tmc_rows(
    config: MatchConfig,
    *,
    tmc_file: Path,
    am_link_file: Path,
    md_link_file: Path,
    pm_link_file: Path,
    am_node_file: Path,
    md_node_file: Path,
    pm_node_file: Path,
    source_crs_epsg: int = BASE2025_CRS_EPSG,
    working_crs_epsg: int = MODEL_CRS_EPSG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmc = load_tmc(config, tmc_file, working_crs_epsg)
    links = load_base2025_physical_links(
        am_link_file,
        pm_link_file,
        md_link_file,
        source_crs_epsg,
        working_crs_epsg,
    )
    nodes = load_base2025_physical_nodes(
        am_node_file,
        pm_node_file,
        md_node_file,
        source_crs_epsg,
        working_crs_epsg,
    )
    selected_links = select_corridor_links(config, tmc, links)
    if selected_links.empty:
        raise RuntimeError("No selected corridor links. Check corridor filters.")

    selected_node_ids = set(selected_links["from_node_id"].astype(int)) | set(selected_links["to_node_id"].astype(int))
    selected_nodes = nodes[nodes["node_id"].astype(int).isin(selected_node_ids)].copy()
    nodes_by_id = selected_nodes.set_index(selected_nodes["node_id"].astype(int), drop=False)
    link_tree = STRtree(selected_links["geometry_obj"].tolist())
    links_by_id = selected_links.set_index(selected_links["link_id"].astype(int), drop=False)
    adjacency = build_adjacency(selected_links)

    summary_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for _, tmc_row in tmc.iterrows():
        tmc_id = str(tmc_row["tmc"])
        tmc_line = tmc_row["geometry_obj"]
        start_pt = Point(tmc_row["start_x"], tmc_row["start_y"])
        end_pt = Point(tmc_row["end_x"], tmc_row["end_y"])
        tmc_bearing = float(tmc_row["tmc_bearing"])
        tmc_miles = float(tmc_row["miles"])

        starts = endpoint_candidates(start_pt, tmc_line, tmc_bearing, "origin", selected_links, nodes_by_id, link_tree, config)
        ends = endpoint_candidates(end_pt, tmc_line, tmc_bearing, "destination", selected_links, nodes_by_id, link_tree, config)

        best: dict[str, object] | None = None
        cutoff_mi = max(1.0, tmc_miles * config.max_route_ratio_for_search)
        for s_rank, start in enumerate(starts, start=1):
            for e_rank, end in enumerate(ends, start=1):
                route = make_route(start["link"], end["link"], adjacency, cutoff_mi=cutoff_mi)
                if route is None:
                    candidate_rows.append(
                        {
                            "tmc": tmc_id,
                            "start_rank": s_rank,
                            "end_rank": e_rank,
                            "start_link_id": int(start["link"]["link_id"]),
                            "end_link_id": int(end["link"]["link_id"]),
                            "path_found": False,
                        }
                    )
                    continue
                metrics = route_metrics(route, links_by_id, tmc_line, tmc_bearing, tmc_miles, start, end)
                total_score = float(metrics["score"]) + float(start["score"]) / 900.0 + float(end["score"]) / 900.0
                candidate = {
                    "tmc": tmc_id,
                    "start_rank": s_rank,
                    "end_rank": e_rank,
                    "start_link_id": int(start["link"]["link_id"]),
                    "end_link_id": int(end["link"]["link_id"]),
                    "o_node_id": int(start["link"]["from_node_id"]),
                    "d_node_id": int(end["link"]["to_node_id"]),
                    "path_found": True,
                    "route_link_ids": route,
                    "route_link_count": len(route),
                    "total_score": total_score,
                    **metrics,
                }
                candidate_rows.append({k: (";".join(map(str, v)) if k == "route_link_ids" else v) for k, v in candidate.items()})
                if best is None or total_score < float(best["total_score"]):
                    best = {
                        **candidate,
                        "start_candidate": start,
                        "end_candidate": end,
                    }

        if best is None:
            row = {
                "tmc": tmc_id,
                "road": tmc_row["road"],
                "direction": tmc_row["direction"],
                "intersection": tmc_row["intersection"],
                "road_order": tmc_row["road_order"],
                "tmc_miles": tmc_miles,
                "tmc_bearing": round(tmc_bearing, 3),
                "status": "no_path",
                "confidence": 0.0,
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
            }
            summary_rows.append(row)
            continue

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
            "confidence": round(float(best["confidence"]), 2),
        }
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
                    "sequence": seq,
                    "link_id": int(link_id),
                    "from_node_id": int(link["from_node_id"]),
                    "to_node_id": int(link["to_node_id"]),
                    "length_mi": round(float(link["length_mi"]), 6),
                    "cumulative_mi": round(cumulative_mi, 6),
                    "distance_to_tmc_ft": round(float(link["geometry_obj"].distance(tmc_line)), 2),
                    "bearing_diff_deg": round(directed_angle_diff(float(link["model_bearing"]), tmc_bearing), 2),
                    "STREETNAME": link.get("STREETNAME", ""),
                    "allowed_use": link.get("allowed_use", ""),
                    "PMLIMIT": link.get("PMLIMIT", ""),
                    "lanes": link.get("lanes", ""),
                    "capacity": link.get("capacity", ""),
                    "free_speed": link.get("free_speed", ""),
                    "link_type": link.get("link_type", ""),
                    "PROJECTID": link.get("PROJECTID", ""),
                    "LINKID": link.get("LINKID", ""),
                    "geometry": link.get("geometry", ""),
                }
            )

    match_summary = pd.DataFrame(summary_rows)
    match_long = pd.DataFrame(long_rows)
    candidates = pd.DataFrame(candidate_rows)
    corridor_summary = build_corridor_summary(config, tmc, selected_links, match_summary)
    return match_summary, match_long, candidates, corridor_summary


def build_corridor_summary(
    config: MatchConfig, tmc: pd.DataFrame, selected_links: pd.DataFrame, match_summary: pd.DataFrame
) -> pd.DataFrame:
    status_counts = match_summary["status"].value_counts().to_dict() if "status" in match_summary else {}
    matched_mask = match_summary["status"].eq("matched") if "status" in match_summary else pd.Series(False)
    route_counts = pd.to_numeric(match_summary.get("route_link_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    route_path_mask = route_counts.gt(0)
    review_mask = ~matched_mask
    return pd.DataFrame(
        [
            {
                "road": config.road,
                "direction": config.direction,
                "tmc_count": len(tmc),
                "tmc_miles": round(float(pd.to_numeric(tmc["miles"], errors="coerce").sum()), 3),
                "selected_model_links": len(selected_links),
                "selected_model_link_miles": round(float(selected_links["length_mi"].sum()), 3),
                "matched_tmc_count": int(matched_mask.sum()),
                "route_path_tmc_count": int(route_path_mask.sum()),
                "review_tmc_count": int(review_mask.sum()),
                "matched_share_pct": round(float(matched_mask.mean() * 100), 1) if len(match_summary) else 0.0,
                "route_path_coverage_pct": round(float(route_path_mask.mean() * 100), 1) if len(match_summary) else 0.0,
                "avg_confidence": round(float(pd.to_numeric(match_summary["confidence"], errors="coerce").mean()), 2),
                "avg_length_ratio": round(float(pd.to_numeric(match_summary["length_ratio"], errors="coerce").mean()), 3),
                "avg_offset_ft": round(float(pd.to_numeric(match_summary["avg_offset_ft"], errors="coerce").mean()), 2),
                "status_counts_json": json.dumps(status_counts, sort_keys=True),
            }
        ]
    )


def write_geo_layers(
    output_dir: Path,
    config: MatchConfig,
    match_summary: pd.DataFrame,
    match_long: pd.DataFrame,
    *,
    tmc_file: Path,
    working_crs_epsg: int = MODEL_CRS_EPSG,
) -> None:
    to_wgs84 = Transformer.from_crs(working_crs_epsg, 4326, always_xy=True)

    tmc = load_tmc(config, tmc_file, working_crs_epsg)
    tmc = tmc[tmc["tmc"].isin(set(match_summary["tmc"].astype(str)))].copy()
    tmc = tmc.merge(match_summary[["tmc", "status", "confidence", "route_link_ids"]], on="tmc", how="left")
    tmc_rows = []
    for row in tmc.itertuples(index=False):
        geom = LineString([(row.start_longitude, row.start_latitude), (row.end_longitude, row.end_latitude)])
        tmc_rows.append(
            {
                "tmc": row.tmc,
                "road": row.road,
                "direction": row.direction,
                "intersection": row.intersection,
                "miles": row.miles,
                "road_order": row.road_order,
                "status": row.status,
                "confidence": row.confidence,
                "geometry": geom.wkt,
            }
        )
    pd.DataFrame(tmc_rows).to_csv(output_dir / "tmc_lines_wgs84.csv", index=False)

    if not match_long.empty:
        link_rows = match_long.copy()
        link_rows["geometry_wgs84"] = link_rows["geometry"].map(
            lambda geom: shapely_transform(to_wgs84.transform, wkt.loads(geom)).wkt if isinstance(geom, str) and geom else ""
        )
        link_rows.to_csv(output_dir / "tmc_to_link_wgs84.csv", index=False)


def write_summary_md(output_dir: Path, config: MatchConfig, corridor_summary: pd.DataFrame, match_summary: pd.DataFrame) -> None:
    status_counts = match_summary["status"].value_counts().rename_axis("status").reset_index(name="tmcs")
    review = match_summary[~match_summary["status"].eq("matched")][
        [
            "tmc",
            "status",
            "intersection",
            "tmc_miles",
            "route_length_mi",
            "length_ratio",
            "avg_offset_ft",
            "avg_bearing_diff_deg",
            "confidence",
        ]
    ].copy()
    lines = [
        "# TMC Line Matching Summary",
        "",
        f"Corridor: `{config.road}` `{config.direction}` `{config.lane_class}`",
        "",
        "## Corridor Summary",
        "",
        md_table(corridor_summary),
        "",
        "## Status Counts",
        "",
        md_table(status_counts),
        "",
        "## Review Required",
        "",
        md_table(review, max_rows=40),
        "",
        "## Output Files",
        "",
        "- `tmc_match_summary.csv`: one row per TMC with QA status.",
        "- `tmc_to_link.csv`: one row per matched model link.",
        "- `tmc_match_candidates.csv`: candidate O/D path attempts.",
        "- `corridor_match_summary.csv`: corridor-level matching metrics.",
        "- `tmc_lines_wgs84.csv` and `tmc_to_link_wgs84.csv`: dashboard/QGIS-ready WKT layers.",
    ]
    (output_dir / "tmc_line_matching_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
