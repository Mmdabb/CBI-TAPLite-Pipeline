from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .config import ExpansionConfig
from .detector import detect_profile_t2, interpolate_normalized_profiles


SNAPSHOT_FILES = {
    "representatives": "tmc_period_representatives.csv",
    "profiles": "tmc_profiles.csv",
    "mappings": "map_matches.csv",
    "routes": "route_summary.csv",
    "network": "regional_network.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _parse_bool(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def load_snapshot(module_root: Path) -> Dict[str, pd.DataFrame]:
    snapshot_dir = Path(module_root) / "input-snapshot"
    missing = [
        snapshot_dir / name
        for name in SNAPSHOT_FILES.values()
        if not (snapshot_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Run prepare before expansion; missing: "
            + ", ".join(str(path) for path in missing)
        )
    frames = {
        key: pd.read_csv(
            snapshot_dir / filename,
            dtype={"tmc": str},
            low_memory=False,
        )
        for key, filename in SNAPSHOT_FILES.items()
    }
    if "period_is_open" in frames["mappings"]:
        frames["mappings"]["period_is_open"] = _parse_bool(
            frames["mappings"]["period_is_open"]
        )
    if "anchor_source_reliable" in frames["representatives"]:
        frames["representatives"]["anchor_source_reliable"] = _parse_bool(
            frames["representatives"]["anchor_source_reliable"]
        )
    if "is_clean_valid_episode" in frames["representatives"]:
        frames["representatives"]["is_clean_valid_episode"] = _parse_bool(
            frames["representatives"]["is_clean_valid_episode"]
        )
    return frames


def add_corridor_positions(
    routes: pd.DataFrame, mappings: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    routes = _numeric(
        routes,
        (
            "road_order",
            "tmc_miles",
            "confidence",
            "o_node_id",
            "d_node_id",
        ),
    )
    routes["tmc"] = routes["tmc"].astype(str).str.strip()
    routes["road"] = routes["road"].astype(str).str.strip()
    routes["direction"] = routes["direction"].astype(str).str.strip().str.upper()
    routes = routes.sort_values(
        ["period", "road", "direction", "road_order", "tmc"],
        kind="mergesort",
    ).reset_index(drop=True)
    routes["tmc_miles"] = routes["tmc_miles"].fillna(0.0).clip(lower=0.0)
    groups = routes.groupby(
        ["period", "road", "direction"], sort=False, dropna=False
    )
    routes["tmc_start_mi"] = groups["tmc_miles"].cumsum() - routes["tmc_miles"]
    routes["tmc_end_mi"] = routes["tmc_start_mi"] + routes["tmc_miles"]
    routes["tmc_position_mi"] = (
        routes["tmc_start_mi"] + 0.5 * routes["tmc_miles"]
    )

    mappings = _numeric(
        mappings,
        (
            "road_order",
            "sequence",
            "link_id",
            "length_mi",
            "cumulative_mi",
            "distance_to_tmc_ft",
            "map_confidence",
        ),
    )
    mappings["tmc"] = mappings["tmc"].astype(str).str.strip()
    mappings["road"] = mappings["road"].astype(str).str.strip()
    mappings["direction"] = (
        mappings["direction"].astype(str).str.strip().str.upper()
    )
    route_position_columns = routes[
        [
            "period",
            "tmc",
            "road",
            "direction",
            "tmc_start_mi",
            "tmc_end_mi",
            "tmc_position_mi",
            "tmc_miles",
        ]
    ].drop_duplicates(["period", "tmc"], keep="first")
    mappings = mappings.merge(
        route_position_columns.drop(columns=["road", "direction"]),
        on=["period", "tmc"],
        how="left",
        validate="many_to_one",
    )
    route_lengths = (
        mappings.groupby(["period", "tmc"], as_index=False)["cumulative_mi"]
        .max()
        .rename(columns={"cumulative_mi": "mapped_route_length_mi"})
    )
    mappings = mappings.merge(
        route_lengths,
        on=["period", "tmc"],
        how="left",
        validate="many_to_one",
    )
    center_on_route = (
        mappings["cumulative_mi"].fillna(0.0)
        - 0.5 * mappings["length_mi"].fillna(0.0)
    )
    fraction = center_on_route / mappings["mapped_route_length_mi"].where(
        mappings["mapped_route_length_mi"] > 0
    )
    fraction = fraction.clip(lower=0.0, upper=1.0).fillna(0.5)
    mappings["corridor_position_mi"] = mappings["tmc_start_mi"] + (
        fraction * mappings["tmc_miles"]
    )
    mappings["corridor_position_mi"] = mappings[
        "corridor_position_mi"
    ].fillna(mappings["tmc_position_mi"])
    return routes, mappings


def build_mapped_targets(
    mappings: pd.DataFrame, network: pd.DataFrame
) -> pd.DataFrame:
    open_maps = mappings[mappings["period_is_open"]].copy()
    open_maps["_distance"] = open_maps["distance_to_tmc_ft"].fillna(np.inf)
    open_maps["_occurrence"] = pd.to_numeric(
        open_maps.get("map_row_number"), errors="coerce"
    ).fillna(np.inf)
    open_maps = open_maps.sort_values(
        ["period", "link_id", "_distance", "_occurrence"],
        kind="mergesort",
    )
    counts = (
        open_maps.groupby(["period", "link_id"], as_index=False)
        .agg(
            mapped_tmc_count=("tmc", "nunique"),
            mapped_row_count=("tmc", "size"),
        )
    )
    primary = open_maps.drop_duplicates(
        ["period", "link_id"], keep="first"
    ).merge(
        counts,
        on=["period", "link_id"],
        how="left",
        validate="one_to_one",
    )
    primary["target_origin"] = "mapped_route_link"
    primary["primary_mapped_tmc"] = primary["tmc"]
    primary["target_position_mi"] = primary["corridor_position_mi"]
    primary["target_id"] = (
        primary["period"].astype(str)
        + "__"
        + primary["link_id"].astype("Int64").astype(str)
    )
    network = _normalize_network(network)
    attributes = [
        "period",
        "link_id",
        "from_node_id",
        "to_node_id",
        "network_length_mi",
        "lanes",
        "capacity",
        "free_speed",
        "link_type",
        "allowed_use",
        "STREETNAME",
        "period_volume",
    ]
    attributes = [column for column in attributes if column in network]
    primary = primary.merge(
        network[attributes].drop_duplicates(["period", "link_id"]),
        on=["period", "link_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_network"),
    )
    return primary.drop(columns=["_distance", "_occurrence"], errors="ignore")


def _normalize_network(network: pd.DataFrame) -> pd.DataFrame:
    network = _numeric(
        network,
        (
            "link_id",
            "from_node_id",
            "to_node_id",
            "length",
            "length_in_mile",
            "lanes",
            "capacity",
            "free_speed",
            "link_type",
            "period_volume",
        ),
    )
    if "length_in_mile" in network:
        length = network["length_in_mile"]
    else:
        length = pd.Series(np.nan, index=network.index)
    if "length" in network:
        length = length.fillna(network["length"])
    network["network_length_mi"] = length.clip(lower=0.0001)
    return network


def _normalize_street(value: object) -> str:
    text = str(value).strip().upper()
    for token in (" ", "-", ".", "_"):
        text = text.replace(token, "")
    return text


def _parse_route_ids(value: object) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    result: List[int] = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(float(item)))
        except ValueError:
            continue
    return result


def _build_period_graph(
    network: pd.DataFrame,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in network.itertuples(index=False):
        if not (
            np.isfinite(row.from_node_id)
            and np.isfinite(row.to_node_id)
            and np.isfinite(row.link_id)
        ):
            continue
        length = float(row.network_length_mi)
        street = _normalize_street(getattr(row, "STREETNAME", ""))
        raw_link_type = getattr(row, "link_type", np.nan)
        link_type = (
            int(raw_link_type) if np.isfinite(raw_link_type) else -1
        )
        payload = {
            "weight": max(length, 0.0001),
            "length_mi": max(length, 0.0001),
            "link_id": int(row.link_id),
            "street": street,
            "is_ramp": bool("RAMP" in street or link_type in {306, 307}),
        }
        source = int(row.from_node_id)
        target = int(row.to_node_id)
        existing = graph.get_edge_data(source, target)
        if existing is None or payload["length_mi"] < existing["length_mi"]:
            graph.add_edge(source, target, **payload)
    return graph


def build_gap_bridge_targets(
    routes: pd.DataFrame,
    mappings: pd.DataFrame,
    network: pd.DataFrame,
    mapped_targets: pd.DataFrame,
    config: ExpansionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "target_id",
        "period",
        "link_id",
        "road",
        "direction",
        "target_position_mi",
        "target_origin",
        "bridge_from_tmc",
        "bridge_to_tmc",
        "bridge_path_miles",
    ]
    if not config.enable_gap_bridging:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    network = _normalize_network(network)
    mapped_keys = set(
        zip(
            mapped_targets["period"].astype(str),
            mapped_targets["link_id"].astype(int),
        )
    )
    target_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    allowed_statuses = set(config.map_anchor_statuses)
    graph_by_period: Dict[str, nx.DiGraph] = {}
    network_index_by_period: Dict[str, pd.DataFrame] = {}
    for period, period_network in network.groupby("period", sort=False):
        period_network = period_network.copy()
        graph_by_period[str(period)] = _build_period_graph(period_network)
        network_index_by_period[str(period)] = period_network.set_index(
            "link_id", drop=False
        )

    for (period, road, direction), corridor_routes in routes.groupby(
        ["period", "road", "direction"], sort=False
    ):
        usable = corridor_routes[
            corridor_routes["status"].astype(str).isin(allowed_statuses)
            & (
                pd.to_numeric(
                    corridor_routes["confidence"], errors="coerce"
                ).fillna(-np.inf)
                >= config.minimum_map_confidence
            )
            & corridor_routes["route_link_ids"].fillna("").astype(str).ne("")
        ].copy()
        if len(usable) < 2:
            continue
        usable = usable.sort_values(["road_order", "tmc"], kind="mergesort")
        corridor_map = mappings[
            mappings["period"].eq(period)
            & mappings["road"].eq(road)
            & mappings["direction"].eq(direction)
        ]
        allowed_streets = {
            _normalize_street(value)
            for value in corridor_map.get(
                "STREETNAME", pd.Series(dtype=str)
            )
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if str(value).strip() and str(value).lower() != "nan"
        }
        graph = graph_by_period[str(period)]
        network_index = network_index_by_period[str(period)]

        def corridor_weight(
            _source: int, _target: int, edge: Dict[str, object]
        ) -> float:
            penalty = 1.0
            if (
                allowed_streets
                and str(edge.get("street", "")) not in allowed_streets
            ):
                penalty *= float(config.bridge_off_corridor_penalty)
            if bool(edge.get("is_ramp", False)):
                penalty *= float(config.bridge_ramp_penalty)
            return float(edge["length_mi"]) * penalty

        route_records = list(usable.itertuples(index=False))
        for left, right in zip(route_records[:-1], route_records[1:]):
            left_ids = _parse_route_ids(left.route_link_ids)
            right_ids = _parse_route_ids(right.route_link_ids)
            if not left_ids or not right_ids or set(left_ids) & set(right_ids):
                continue
            if not (
                np.isfinite(float(left.d_node_id))
                and np.isfinite(float(right.o_node_id))
            ):
                continue
            source = int(float(left.d_node_id))
            target = int(float(right.o_node_id))
            audit = {
                "period": period,
                "road": road,
                "direction": direction,
                "from_tmc": str(left.tmc),
                "to_tmc": str(right.tmc),
                "source_node": source,
                "target_node": target,
            }
            if source == target:
                audit.update({"status": "already_connected", "path_miles": 0.0})
                audit_rows.append(audit)
                continue
            try:
                _, node_path = nx.single_source_dijkstra(
                    graph,
                    source,
                    target=target,
                    cutoff=float(config.maximum_bridge_path_miles)
                    * max(
                        1.0,
                        float(config.bridge_off_corridor_penalty),
                    ),
                    weight=corridor_weight,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                audit.update({"status": "no_path", "path_miles": np.nan})
                audit_rows.append(audit)
                continue
            path_edges = list(zip(node_path[:-1], node_path[1:]))
            link_ids = [
                int(graph.get_edge_data(u, v)["link_id"]) for u, v in path_edges
            ]
            path_lengths = [
                float(graph.get_edge_data(u, v)["length_mi"])
                for u, v in path_edges
            ]
            path_miles = float(sum(path_lengths))
            if path_miles > float(config.maximum_bridge_path_miles):
                audit.update(
                    {"status": "rejected_too_long", "path_miles": path_miles}
                )
                audit_rows.append(audit)
                continue
            start_position = float(left.tmc_position_mi)
            end_position = float(right.tmc_position_mi)
            if not np.isfinite(start_position + end_position) or (
                end_position <= start_position
            ):
                audit.update(
                    {
                        "status": "rejected_position_order",
                        "path_miles": path_miles,
                    }
                )
                audit_rows.append(audit)
                continue
            cumulative = 0.0
            added = 0
            for link_id, link_length in zip(link_ids, path_lengths):
                center = cumulative + 0.5 * link_length
                cumulative += link_length
                key = (str(period), int(link_id))
                if key in mapped_keys or link_id not in network_index.index:
                    continue
                fraction = center / max(path_miles, 1e-6)
                position = start_position + fraction * (
                    end_position - start_position
                )
                net = network_index.loc[link_id]
                if isinstance(net, pd.DataFrame):
                    net = net.iloc[0]
                target_rows.append(
                    {
                        "target_id": str(period) + "__" + str(int(link_id)),
                        "period": period,
                        "link_id": int(link_id),
                        "road": road,
                        "direction": direction,
                        "target_position_mi": position,
                        "target_origin": "corridor_gap_bridge",
                        "primary_mapped_tmc": "",
                        "mapped_tmc_count": 0,
                        "mapped_row_count": 0,
                        "bridge_from_tmc": str(left.tmc),
                        "bridge_to_tmc": str(right.tmc),
                        "bridge_path_miles": path_miles,
                        "from_node_id": net.get("from_node_id", np.nan),
                        "to_node_id": net.get("to_node_id", np.nan),
                        "network_length_mi": net.get(
                            "network_length_mi", np.nan
                        ),
                        "lanes": net.get("lanes", np.nan),
                        "capacity": net.get("capacity", np.nan),
                        "free_speed": net.get("free_speed", np.nan),
                        "link_type": net.get("link_type", np.nan),
                        "allowed_use": net.get("allowed_use", ""),
                        "STREETNAME": net.get("STREETNAME", ""),
                        "period_volume": net.get("period_volume", np.nan),
                    }
                )
                added += 1
            audit.update(
                {
                    "status": "accepted" if added else "path_already_mapped",
                    "path_miles": path_miles,
                    "added_links": added,
                }
            )
            audit_rows.append(audit)
    targets = pd.DataFrame(target_rows)
    if not targets.empty:
        targets = targets.sort_values(
            ["period", "link_id", "bridge_path_miles"],
            kind="mergesort",
        ).drop_duplicates(["period", "link_id"], keep="first")
    return targets, pd.DataFrame(audit_rows)


def build_anchors(
    representatives: pd.DataFrame,
    routes: pd.DataFrame,
    config: ExpansionConfig,
) -> pd.DataFrame:
    representatives = _numeric(
        representatives,
        (
            "t2_hour",
            "anchor_t2_hour",
            "anchor_daily_probe_day_count",
            "anchor_daily_t2_std_hour",
        ),
    )
    representatives = representatives.drop(
        columns=["road", "direction", "status", "confidence"],
        errors="ignore",
    )
    route_qa = routes[
        [
            "period",
            "tmc",
            "road",
            "direction",
            "tmc_position_mi",
            "tmc_start_mi",
            "tmc_end_mi",
            "status",
            "confidence",
        ]
    ].drop_duplicates(["period", "tmc"], keep="first")
    anchors = representatives.merge(
        route_qa,
        on=["period", "tmc"],
        how="inner",
        validate="one_to_one",
    )
    anchors["map_anchor_eligible"] = (
        anchors["status"].astype(str).isin(set(config.map_anchor_statuses))
        & (
            pd.to_numeric(anchors["confidence"], errors="coerce").fillna(
                -np.inf
            )
            >= config.minimum_map_confidence
        )
    )
    anchors["anchor_eligible"] = (
        anchors["anchor_source_reliable"].fillna(False).astype(bool)
        & anchors["map_anchor_eligible"]
        & anchors["anchor_t2_hour"].notna()
    )
    anchors = anchors[anchors["anchor_eligible"]].copy()
    anchors["anchor_position_mi"] = anchors["tmc_position_mi"]
    return anchors.sort_values(
        ["period", "road", "direction", "anchor_position_mi"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_direct_assignments(
    mappings: pd.DataFrame,
    representatives: pd.DataFrame,
    config: ExpansionConfig,
) -> pd.DataFrame:
    representatives = _numeric(
        representatives,
        (
            "t2_hour",
            "daily_probe_day_count",
            "daily_probe_t2_std_hour",
            "anchor_t2_hour",
        ),
    )
    open_maps = mappings[mappings["period_is_open"]].copy()
    joined = open_maps.merge(
        representatives,
        on=["period", "tmc"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_episode"),
    )
    available = joined[joined["t2_hour"].notna()].copy()
    if available.empty:
        return pd.DataFrame()
    available["_distance"] = available["distance_to_tmc_ft"].fillna(np.inf)
    available["_occurrence"] = pd.to_numeric(
        available.get("map_row_number"), errors="coerce"
    ).fillna(np.inf)
    available = available.sort_values(
        ["period", "link_id", "_distance", "_occurrence"],
        kind="mergesort",
    ).drop_duplicates(["period", "link_id"], keep="first")
    source_reliable = available["anchor_t2_hour"].notna()
    map_reliable = (
        available["map_status"].astype(str).isin(
            set(config.map_anchor_statuses)
        )
        & (
            pd.to_numeric(available["map_confidence"], errors="coerce").fillna(
                -np.inf
            )
            >= config.minimum_map_confidence
        )
    )
    available["assignment_confidence"] = np.select(
        [source_reliable & map_reliable, source_reliable | map_reliable],
        ["high", "medium"],
        default="low",
    )
    available["assignment_tier"] = "A_direct"
    available["assignment_method"] = "direct_mapped_tmc"
    available["selected_tmc"] = available["tmc"]
    available["left_anchor_tmc"] = available["tmc"]
    available["right_anchor_tmc"] = available["tmc"]
    available["left_anchor_distance_mi"] = 0.0
    available["right_anchor_distance_mi"] = 0.0
    available["anchor_gap_mi"] = 0.0
    available["propagation_minutes_per_mile"] = 0.0
    available["profile_detection_succeeded"] = False
    return available


def make_profile_lookup(
    profiles: pd.DataFrame,
) -> Dict[str, pd.Series]:
    profiles = _numeric(profiles, ("t_min", "normalized_speed"))
    result: Dict[str, pd.Series] = {}
    for tmc, group in profiles.groupby("tmc", sort=False):
        series = (
            group.dropna(subset=["t_min"])
            .sort_values("t_min")
            .drop_duplicates("t_min", keep="first")
            .set_index("t_min")["normalized_speed"]
        )
        result[str(tmc)] = series
    return result


def profile_array(
    lookup: Dict[str, pd.Series], tmc: str, axis: np.ndarray
) -> np.ndarray:
    series = lookup.get(str(tmc))
    if series is None:
        return np.full(len(axis), np.nan)
    return series.reindex(axis).to_numpy(dtype=float)


def fit_propagation_slope(
    anchors: pd.DataFrame, config: ExpansionConfig
) -> Tuple[float, int]:
    valid = anchors.dropna(
        subset=["anchor_position_mi", "anchor_t2_hour"]
    ).drop_duplicates("anchor_position_mi")
    if len(valid) < 3:
        return 0.0, int(len(valid))
    x = valid["anchor_position_mi"].to_numpy(dtype=float)
    y = valid["anchor_t2_hour"].to_numpy(dtype=float)
    slope = float(theilslopes(y, x)[0])
    limit = float(config.maximum_abs_propagation_minutes_per_mile) / 60.0
    return float(np.clip(slope, -limit, limit)), int(len(valid))


def _period_contains(period: Tuple[int, int], t2_hour: float) -> bool:
    minute = float(t2_hour) * 60.0
    return float(period[0]) <= minute < float(period[1])


def predict_spatial_t2(
    target_position: float,
    anchors: pd.DataFrame,
    period_name: str,
    profile_lookup: Dict[str, pd.Series],
    config: ExpansionConfig,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "t2_hour": np.nan,
        "assignment_tier": "",
        "assignment_method": "",
        "assignment_confidence": "",
        "selected_tmc": "",
        "left_anchor_tmc": "",
        "right_anchor_tmc": "",
        "left_anchor_distance_mi": np.nan,
        "right_anchor_distance_mi": np.nan,
        "anchor_gap_mi": np.nan,
        "propagation_minutes_per_mile": np.nan,
        "propagation_anchor_count": 0,
        "profile_detection_succeeded": False,
        "profile_minimum_ratio": np.nan,
        "profile_coverage": np.nan,
        "linear_t2_candidate": np.nan,
        "profile_t2_candidate": np.nan,
    }
    if anchors.empty or not np.isfinite(target_position):
        return result
    ordered = anchors.sort_values("anchor_position_mi", kind="mergesort")
    left = ordered[ordered["anchor_position_mi"] <= target_position]
    right = ordered[ordered["anchor_position_mi"] >= target_position]
    left_row = left.iloc[-1] if not left.empty else None
    right_row = right.iloc[0] if not right.empty else None
    period = config.periods[str(period_name)]
    if (
        left_row is not None
        and right_row is not None
        and str(left_row["tmc"]) != str(right_row["tmc"])
    ):
        left_position = float(left_row["anchor_position_mi"])
        right_position = float(right_row["anchor_position_mi"])
        gap = right_position - left_position
        if 0.0 < gap <= float(config.maximum_interpolation_gap_miles):
            weight_right = (target_position - left_position) / gap
            linear_t2 = (
                (1.0 - weight_right) * float(left_row["anchor_t2_hour"])
                + weight_right * float(right_row["anchor_t2_hour"])
            )
            result.update(
                {
                    "assignment_tier": "B_bracketed",
                    "assignment_method": "normalized_profile_interpolation",
                    "assignment_confidence": (
                        "medium_high"
                        if gap
                        <= 0.5 * float(config.maximum_interpolation_gap_miles)
                        else "medium"
                    ),
                    "left_anchor_tmc": str(left_row["tmc"]),
                    "right_anchor_tmc": str(right_row["tmc"]),
                    "left_anchor_distance_mi": target_position - left_position,
                    "right_anchor_distance_mi": right_position - target_position,
                    "anchor_gap_mi": gap,
                    "linear_t2_candidate": linear_t2,
                }
            )
            interval = 15
            axis = np.arange(period[0], period[1], interval, dtype=float)
            left_profile = profile_array(
                profile_lookup, str(left_row["tmc"]), axis
            )
            right_profile = profile_array(
                profile_lookup, str(right_row["tmc"]), axis
            )
            interpolated = interpolate_normalized_profiles(
                left_profile, right_profile, weight_right
            )
            detected = detect_profile_t2(
                axis,
                interpolated,
                period,
                config.profile_threshold_ratio,
                config.profile_minimum_episode_minutes,
                config.profile_merge_gap_minutes,
                config.profile_minimum_coverage,
            )
            if detected is not None:
                result.update(
                    {
                        "profile_t2_candidate": float(detected["t2_hour"]),
                        "profile_detection_succeeded": True,
                        "profile_minimum_ratio": float(
                            detected["minimum_ratio"]
                        ),
                        "profile_coverage": float(
                            detected["profile_coverage"]
                        ),
                    }
                )
            if (
                config.bracket_assignment_method == "linear_t2"
                and _period_contains(period, linear_t2)
            ):
                result.update(
                    {
                        "t2_hour": linear_t2,
                        "assignment_method": "linear_t2_interpolation",
                    }
                )
                return result
            if (
                config.bracket_assignment_method == "normalized_profile"
                and detected is not None
            ):
                result["t2_hour"] = float(detected["t2_hour"])
                return result
            if (
                config.bracket_assignment_method == "normalized_profile"
                and config.enable_linear_t2_fallback
                and _period_contains(period, linear_t2)
            ):
                result.update(
                    {
                        "t2_hour": linear_t2,
                        "assignment_method": "linear_t2_fallback",
                        "assignment_confidence": "medium_low",
                    }
                )
                return result

    nearest_candidates: List[pd.Series] = []
    if left_row is not None:
        nearest_candidates.append(left_row)
    if right_row is not None:
        nearest_candidates.append(right_row)
    if not nearest_candidates:
        return result
    nearest = min(
        nearest_candidates,
        key=lambda row: abs(float(row["anchor_position_mi"]) - target_position),
    )
    distance = abs(float(nearest["anchor_position_mi"]) - target_position)
    if distance > float(config.maximum_extrapolation_miles):
        return result
    slope, count = fit_propagation_slope(ordered, config)
    predicted = float(nearest["anchor_t2_hour"]) + slope * (
        target_position - float(nearest["anchor_position_mi"])
    )
    if not _period_contains(period, predicted):
        return result
    result.update(
        {
            "t2_hour": predicted,
            "assignment_tier": "C_one_sided",
            "assignment_method": "nearest_profile_propagation_shift",
            "assignment_confidence": (
                "medium"
                if distance
                <= 0.5 * float(config.maximum_extrapolation_miles)
                else "low"
            ),
            "selected_tmc": str(nearest["tmc"]),
            "left_anchor_tmc": (
                str(nearest["tmc"])
                if float(nearest["anchor_position_mi"]) <= target_position
                else ""
            ),
            "right_anchor_tmc": (
                str(nearest["tmc"])
                if float(nearest["anchor_position_mi"]) >= target_position
                else ""
            ),
            "left_anchor_distance_mi": (
                distance
                if float(nearest["anchor_position_mi"]) <= target_position
                else np.nan
            ),
            "right_anchor_distance_mi": (
                distance
                if float(nearest["anchor_position_mi"]) >= target_position
                else np.nan
            ),
            "anchor_gap_mi": np.nan,
            "propagation_minutes_per_mile": slope * 60.0,
            "propagation_anchor_count": count,
        }
    )
    return result


def apply_expansion(
    targets: pd.DataFrame,
    direct: pd.DataFrame,
    anchors: pd.DataFrame,
    profiles: pd.DataFrame,
    config: ExpansionConfig,
) -> pd.DataFrame:
    profile_lookup = make_profile_lookup(profiles)
    direct = direct.rename(
        columns={
            "map_status": "selected_map_status",
            "map_confidence": "selected_map_confidence",
            "distance_to_tmc_ft": "selected_map_distance_to_tmc_ft",
        }
    )
    direct_columns = [
        "period",
        "link_id",
        "t2_hour",
        "assignment_tier",
        "assignment_method",
        "assignment_confidence",
        "selected_tmc",
        "left_anchor_tmc",
        "right_anchor_tmc",
        "left_anchor_distance_mi",
        "right_anchor_distance_mi",
        "anchor_gap_mi",
        "propagation_minutes_per_mile",
        "profile_detection_succeeded",
        "t2_source_method",
        "episode_id",
        "is_clean_valid_episode",
        "daily_probe_day_count",
        "daily_probe_t2_std_hour",
        "selected_map_status",
        "selected_map_confidence",
        "selected_map_distance_to_tmc_ft",
    ]
    direct_columns = [column for column in direct_columns if column in direct]
    expanded = targets.merge(
        direct[direct_columns],
        on=["period", "link_id"],
        how="left",
        validate="one_to_one",
    )
    rows: List[Dict[str, object]] = []
    group_lookup = {
        key: group.copy()
        for key, group in anchors.groupby(
            ["period", "road", "direction"], sort=False
        )
    }
    for index, row in expanded.iterrows():
        if pd.notna(row.get("t2_hour")):
            continue
        key = (str(row["period"]), str(row["road"]), str(row["direction"]))
        prediction = predict_spatial_t2(
            float(row["target_position_mi"]),
            group_lookup.get(key, pd.DataFrame()),
            str(row["period"]),
            profile_lookup,
            config,
        )
        prediction["_index"] = index
        rows.append(prediction)
    if rows:
        predictions = pd.DataFrame(rows).set_index("_index")
        for column in predictions:
            if column not in expanded:
                expanded[column] = np.nan
            expanded.loc[predictions.index, column] = predictions[column]
    expanded["assignment_tier"] = expanded["assignment_tier"].fillna("")
    expanded["assignment_method"] = expanded["assignment_method"].fillna("")
    expanded["assignment_confidence"] = expanded[
        "assignment_confidence"
    ].fillna("")
    expanded["assignment_status"] = np.where(
        pd.to_numeric(expanded["t2_hour"], errors="coerce").notna(),
        "assigned",
        "unassigned",
    )
    expanded["t2_hour"] = pd.to_numeric(
        expanded["t2_hour"], errors="coerce"
    ).round(6)
    return expanded.sort_values(
        ["period", "link_id"], kind="mergesort"
    ).reset_index(drop=True)


def coverage_summary(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for period, group in expanded.groupby("period", sort=True):
        total = int(len(group))
        direct = int(group["assignment_tier"].eq("A_direct").sum())
        assigned = int(group["t2_hour"].notna().sum())
        length = pd.to_numeric(group["network_length_mi"], errors="coerce").fillna(
            0.0
        )
        volume = pd.to_numeric(group["period_volume"], errors="coerce").fillna(
            0.0
        )
        vmt = length * volume
        denominator = float(vmt.sum())
        rows.append(
            {
                "period": period,
                "scope": "all_targets",
                "target_links": total,
                "direct_links": direct,
                "assigned_links": assigned,
                "direct_coverage_pct": (
                    100.0 * direct / total if total else 0.0
                ),
                "expanded_coverage_pct": (
                    100.0 * assigned / total if total else 0.0
                ),
                "added_links": assigned - direct,
                "assigned_vmt_coverage_pct": (
                    100.0
                    * float(vmt[group["t2_hour"].notna()].sum())
                    / denominator
                    if denominator > 0
                    else np.nan
                ),
            }
        )
        for tier, tier_group in group[group["t2_hour"].notna()].groupby(
            "assignment_tier", sort=True
        ):
            rows.append(
                {
                    "period": period,
                    "scope": tier,
                    "target_links": total,
                    "direct_links": direct,
                    "assigned_links": int(len(tier_group)),
                    "direct_coverage_pct": np.nan,
                    "expanded_coverage_pct": (
                        100.0 * len(tier_group) / total if total else 0.0
                    ),
                    "added_links": (
                        0 if tier == "A_direct" else int(len(tier_group))
                    ),
                    "assigned_vmt_coverage_pct": (
                        100.0
                        * float(vmt.loc[tier_group.index].sum())
                        / denominator
                        if denominator > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def format_expanded_output(expanded: pd.DataFrame) -> pd.DataFrame:
    frame = expanded.copy()

    def coalesce(target: str, candidates: Sequence[str]) -> None:
        values = pd.Series(np.nan, index=frame.index)
        for column in candidates:
            if column in frame:
                values = values.fillna(frame[column])
        frame[target] = values

    coalesce("from_node_id_final", ["from_node_id_network", "from_node_id"])
    coalesce("to_node_id_final", ["to_node_id_network", "to_node_id"])
    coalesce("lanes_final", ["lanes_network", "lanes"])
    coalesce("capacity_final", ["capacity_network", "capacity"])
    coalesce("free_speed_final", ["free_speed_network", "free_speed"])
    coalesce("link_type_final", ["link_type_network", "link_type"])
    coalesce("street_name_final", ["STREETNAME_network", "STREETNAME"])
    coalesce("allowed_use_final", ["allowed_use_network", "allowed_use"])
    rename = {
        "from_node_id_final": "from_node_id",
        "to_node_id_final": "to_node_id",
        "lanes_final": "lanes",
        "capacity_final": "capacity",
        "free_speed_final": "free_speed",
        "link_type_final": "link_type",
        "street_name_final": "street_name",
        "allowed_use_final": "allowed_use",
        "map_status": "primary_map_status",
        "map_confidence": "primary_map_confidence",
        "distance_to_tmc_ft": "primary_map_distance_to_tmc_ft",
        "is_clean_valid_episode": "source_screened_accepted",
    }
    columns = [
        "period",
        "link_id",
        "from_node_id_final",
        "to_node_id_final",
        "road",
        "direction",
        "street_name_final",
        "target_origin",
        "target_position_mi",
        "network_length_mi",
        "lanes_final",
        "capacity_final",
        "free_speed_final",
        "link_type_final",
        "allowed_use_final",
        "period_volume",
        "primary_mapped_tmc",
        "mapped_tmc_count",
        "map_status",
        "map_confidence",
        "distance_to_tmc_ft",
        "bridge_from_tmc",
        "bridge_to_tmc",
        "bridge_path_miles",
        "t2_hour",
        "assignment_status",
        "assignment_tier",
        "assignment_method",
        "assignment_confidence",
        "selected_tmc",
        "left_anchor_tmc",
        "right_anchor_tmc",
        "left_anchor_distance_mi",
        "right_anchor_distance_mi",
        "anchor_gap_mi",
        "propagation_minutes_per_mile",
        "propagation_anchor_count",
        "profile_detection_succeeded",
        "profile_minimum_ratio",
        "profile_coverage",
        "linear_t2_candidate",
        "profile_t2_candidate",
        "t2_source_method",
        "episode_id",
        "is_clean_valid_episode",
        "daily_probe_day_count",
        "daily_probe_t2_std_hour",
        "selected_map_status",
        "selected_map_confidence",
        "selected_map_distance_to_tmc_ft",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    output = frame[columns].rename(columns=rename)
    for column in (
        "link_id",
        "from_node_id",
        "to_node_id",
        "mapped_tmc_count",
        "propagation_anchor_count",
        "daily_probe_day_count",
    ):
        if column in output:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).astype("Int64")
    return output


def write_prototype_results(
    module_root: Path,
    expanded: pd.DataFrame,
    summary: pd.DataFrame,
    generated_utc: str,
) -> Path:
    all_targets = summary[summary["scope"].eq("all_targets")].copy()
    lines = [
        "# T2 coverage expansion prototype results",
        "",
        f"- Generated UTC: `{generated_utc}`",
        f"- Target link-period rows: **{len(expanded):,}**",
        f"- Direct assignments: **{int(expanded['assignment_tier'].eq('A_direct').sum()):,}**",
        f"- Expanded assignments: **{int(expanded['t2_hour'].notna().sum()):,}**",
        f"- Added assignments: **{int(expanded['t2_hour'].notna().sum() - expanded['assignment_tier'].eq('A_direct').sum()):,}**",
        f"- Corridor-gap target rows: **{int(expanded['target_origin'].eq('corridor_gap_bridge').sum()):,}**",
        f"- Assigned corridor-gap rows: **{int((expanded['target_origin'].eq('corridor_gap_bridge') & expanded['t2_hour'].notna()).sum()):,}**",
        "",
        "## Period coverage",
        "",
        "| Period | Targets | Direct | Expanded | Added | Direct coverage | Expanded coverage | Assigned VMT coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_targets.itertuples(index=False):
        lines.append(
            "| {period} | {targets:,} | {direct:,} | {assigned:,} | "
            "{added:,} | {direct_pct:.1f}% | {expanded_pct:.1f}% | "
            "{vmt_pct:.1f}% |".format(
                period=row.period,
                targets=int(row.target_links),
                direct=int(row.direct_links),
                assigned=int(row.assigned_links),
                added=int(row.added_links),
                direct_pct=float(row.direct_coverage_pct),
                expanded_pct=float(row.expanded_coverage_pct),
                vmt_pct=float(row.assigned_vmt_coverage_pct),
            )
        )
    lines.extend(
        [
            "",
            "## Assignment tiers",
            "",
            "| Tier | Assigned link-periods |",
            "|---|---:|",
        ]
    )
    for tier, count in (
        expanded.loc[expanded["t2_hour"].notna(), "assignment_tier"]
        .value_counts()
        .items()
    ):
        lines.append(f"| {tier} | {int(count):,} |")

    validation_path = Path(module_root) / "outputs" / "validation_summary.csv"
    if validation_path.is_file():
        validation = pd.read_csv(validation_path, low_memory=False)
        lines.extend(
            [
                "",
                "## Spatial block holdout validation",
                "",
                "| Method | Prediction coverage | MAE | P90 absolute error | Within 30 minutes |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in validation.itertuples(index=False):
            lines.append(
                "| {method} | {coverage:.1f}% | {mae:.1f} min | "
                "{p90:.1f} min | {within30:.1f}% |".format(
                    method=str(row.method).replace("_", " "),
                    coverage=float(row.prediction_coverage_pct),
                    mae=float(row.mae_minutes),
                    p90=float(row.p90_absolute_error_minutes),
                    within30=float(row.within_30_minutes_pct),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Linear T2 interpolation is the default Tier B method because it had "
            "the lowest overall holdout MAE. Normalized-profile interpolation "
            "is retained as a diagnostic candidate rather than discarded. "
            "One-sided propagation remains lower confidence. MD errors are "
            "materially larger than AM and PM errors and should be reviewed "
            "before any main-package integration.",
            "",
            "The direct layer uses pre-filter episode representatives as "
            "requested. Only screening-accepted, map-QA-passing, sufficiently "
            "stable representatives are allowed to propagate spatially.",
            "",
        ]
    )
    path = Path(module_root) / "outputs" / "prototype_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_expansion(
    module_root: Path,
    config: ExpansionConfig,
) -> Dict[str, object]:
    module_root = Path(module_root).resolve()
    frames = load_snapshot(module_root)
    routes, mappings = add_corridor_positions(
        frames["routes"], frames["mappings"]
    )
    mapped_targets = build_mapped_targets(mappings, frames["network"])
    bridge_targets, bridge_audit = build_gap_bridge_targets(
        routes,
        mappings,
        frames["network"],
        mapped_targets,
        config,
    )
    if bridge_targets.empty:
        targets = mapped_targets.copy()
    else:
        shared = sorted(set(mapped_targets.columns) | set(bridge_targets.columns))
        for column in shared:
            if column not in mapped_targets:
                mapped_targets[column] = (
                    False
                    if column in bridge_targets
                    and bridge_targets[column].dtype == bool
                    else np.nan
                )
            if column not in bridge_targets:
                bridge_targets[column] = (
                    False
                    if column in mapped_targets
                    and mapped_targets[column].dtype == bool
                    else np.nan
                )
        targets = pd.concat(
            [mapped_targets[shared], bridge_targets[shared]],
            ignore_index=True,
            sort=False,
        ).sort_values(
            ["period", "link_id", "target_origin"], kind="mergesort"
        ).drop_duplicates(["period", "link_id"], keep="first")
    anchors = build_anchors(frames["representatives"], routes, config)
    direct = build_direct_assignments(
        mappings, frames["representatives"], config
    )
    expanded = apply_expansion(
        targets, direct, anchors, frames["profiles"], config
    )
    summary = coverage_summary(expanded)
    formatted = format_expanded_output(expanded)

    output_dir = module_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    period_root = output_dir / "period_link_files"
    formatted.to_csv(output_dir / "expanded_link_t2.csv", index=False)
    expanded.to_csv(
        output_dir / "expanded_link_t2_full_audit.csv", index=False
    )
    bridge_audit.to_csv(output_dir / "gap_bridge_audit.csv", index=False)
    anchors.to_csv(output_dir / "eligible_spatial_anchors.csv", index=False)
    summary.to_csv(output_dir / "coverage_summary.csv", index=False)
    summary_payload = summary.replace({np.nan: None}).to_dict(orient="records")
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    period_products: Dict[str, Dict[str, object]] = {}
    for period in ("AM", "MD", "PM"):
        folder = period_root / period.lower()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "link_t2_expansion.csv"
        period_frame = formatted[formatted["period"].eq(period)].copy()
        period_frame.to_csv(path, index=False)
        period_products[period] = {
            "path": str(path.relative_to(module_root)),
            "rows": int(len(period_frame)),
            "assigned": int(period_frame["t2_hour"].notna().sum()),
            "sha256": sha256(path),
        }
    generated_utc = datetime.now(timezone.utc).isoformat()
    results_report_path = write_prototype_results(
        module_root, expanded, summary, generated_utc
    )
    snapshot_dir = module_root / "input-snapshot"
    manifest = {
        "status": "PASS",
        "generated_utc": generated_utc,
        "mode": "isolated_t2_coverage_expansion",
        "config": config.to_dict(),
        "snapshot_files": {
            filename: sha256(snapshot_dir / filename)
            for filename in SNAPSHOT_FILES.values()
        },
        "targets": int(len(expanded)),
        "mapped_targets": int(
            expanded["target_origin"].eq("mapped_route_link").sum()
        ),
        "gap_bridge_targets": int(
            expanded["target_origin"].eq("corridor_gap_bridge").sum()
        ),
        "eligible_spatial_anchors": int(len(anchors)),
        "direct_assignments": int(
            expanded["assignment_tier"].eq("A_direct").sum()
        ),
        "expanded_assignments": int(expanded["t2_hour"].notna().sum()),
        "added_assignments": int(
            expanded["t2_hour"].notna().sum()
            - expanded["assignment_tier"].eq("A_direct").sum()
        ),
        "assignments_by_tier": {
            str(key): int(value)
            for key, value in expanded.loc[
                expanded["t2_hour"].notna(), "assignment_tier"
            ]
            .value_counts()
            .items()
        },
        "period_products": period_products,
        "outputs": {
            "expanded_link_t2": {
                "rows": int(len(formatted)),
                "sha256": sha256(output_dir / "expanded_link_t2.csv"),
            },
            "expanded_link_t2_full_audit": {
                "rows": int(len(expanded)),
                "sha256": sha256(
                    output_dir / "expanded_link_t2_full_audit.csv"
                ),
            },
            "coverage_summary": {
                "rows": int(len(summary)),
                "sha256": sha256(output_dir / "coverage_summary.csv"),
            },
            "gap_bridge_audit": {
                "rows": int(len(bridge_audit)),
                "sha256": sha256(output_dir / "gap_bridge_audit.csv"),
            },
            "prototype_results": {
                "sha256": sha256(results_report_path),
            },
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
