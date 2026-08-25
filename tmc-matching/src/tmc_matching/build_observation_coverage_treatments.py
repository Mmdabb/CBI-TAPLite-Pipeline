from __future__ import annotations

"""Build auditable observation treatments without changing canonical matches.

The frozen canonical node-pair winners remain the authoritative direct mapping.
This module only adds explicitly tiered coverage for corridor-named network links
that were not selected by the route matcher:

* actual managed-facility observations, when a compatible managed TMC exists;
* bounded spatial interpolation between observed TMCs;
* distance-damped terminal extrapolation within five miles; and
* a defensible exclusion record for every remaining audited link.

Virtual profiles are derived exclusively from observed weekday speeds.  Model
assignment speeds are deliberately excluded to avoid circular calibration.
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point

from .tmc_line_matcher import bearing, directed_angle_diff
PROFILE_COLUMNS = (
    "speed",
    "historical_average_speed",
    "reference_speed",
)
MANAGED_ROAD_PATTERN = re.compile(r"\(\s*(?:HOV|HOT|EXPRESS|MANAGED)\s*\)", re.I)
FT_PER_MILE = 5280.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def base_road(value: object) -> str:
    return normalized_text(MANAGED_ROAD_PATTERN.sub("", str(value or "")))


def is_managed_tmc_road(value: object) -> bool:
    return bool(MANAGED_ROAD_PATTERN.search(str(value or "")))


def truthy(values: pd.Series) -> pd.Series:
    return (
        values.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def weekday_average_profiles(
    readings_path: Path,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Return one observed weekday-average record per TMC and 15-minute bin."""

    header = set(pd.read_csv(readings_path, nrows=0).columns)
    tmc_column = "tmc_code" if "tmc_code" in header else "tmc"
    time_column = next(
        name
        for name in ("measurement_tstamp", "datetime", "timestamp")
        if name in header
    )
    numeric = [column for column in PROFILE_COLUMNS if column in header]
    if "speed" not in numeric:
        raise ValueError(f"{readings_path} does not contain speed")
    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        readings_path,
        usecols=[tmc_column, time_column, *numeric],
        dtype={tmc_column: "string"},
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["datetime"] = pd.to_datetime(chunk[time_column], errors="coerce")
        chunk = chunk[chunk["datetime"].notna()].copy()
        chunk = chunk[chunk["datetime"].dt.weekday < 5].copy()
        chunk["tmc"] = chunk[tmc_column].astype("string").str.strip().str.upper()
        chunk["time_minute"] = (
            chunk["datetime"].dt.hour * 60 + chunk["datetime"].dt.minute
        )
        for column in numeric:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            if column == "speed":
                chunk.loc[~chunk[column].between(1.0, 150.0), column] = np.nan
        keys = ["tmc", "time_minute"]
        sums = chunk.groupby(keys, as_index=False)[numeric].sum(min_count=1)
        counts = chunk.groupby(keys, as_index=False)[numeric].count()
        grouped = sums.rename(
            columns={column: f"{column}_sum" for column in numeric}
        ).merge(
            counts.rename(
                columns={column: f"{column}_count" for column in numeric}
            ),
            on=keys,
            how="outer",
            validate="one_to_one",
        )
        partials.append(grouped)
    if not partials:
        raise ValueError(f"No weekday observations found in {readings_path}")
    totals = pd.concat(partials, ignore_index=True).groupby(
        ["tmc", "time_minute"], as_index=False
    ).sum(numeric_only=True)
    for column in numeric:
        totals[column] = totals[f"{column}_sum"] / totals[f"{column}_count"].where(
            totals[f"{column}_count"] > 0
        )
    for column in PROFILE_COLUMNS:
        if column not in totals:
            totals[column] = np.nan
    return totals[["tmc", "time_minute", *PROFILE_COLUMNS]].sort_values(
        ["tmc", "time_minute"]
    )


def profile_lookup(profiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(tmc): frame.set_index("time_minute").sort_index()
        for tmc, frame in profiles.groupby("tmc", sort=False)
    }


def geometry_from_metadata(row: Mapping[str, object]) -> LineString:
    return LineString(
        [
            (float(row["start_longitude"]), float(row["start_latitude"])),
            (float(row["end_longitude"]), float(row["end_latitude"])),
        ]
    )


def managed_inventory(metadata: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    work = metadata[metadata["road"].map(is_managed_tmc_road)].copy()
    work["base_road"] = work["road"].map(base_road)
    work["direction_key"] = work["direction"].map(normalized_text)
    return {
        key: frame.sort_values(["road_order", "tmc"]).copy()
        for key, frame in work.groupby(["base_road", "direction_key"], sort=False)
    }


def choose_managed_match(
    row: pd.Series,
    inventory: Mapping[tuple[str, str], pd.DataFrame],
) -> dict[str, object] | None:
    key = (base_road(row["assigned_dashboard_road"]), normalized_text(row["assigned_dashboard_direction"]))
    candidates = inventory.get(key)
    if candidates is None or candidates.empty:
        return None
    link_geom = wkt.loads(str(row["geometry_wgs84"]))
    link_bearing = bearing(link_geom)
    scored: list[tuple[float, float, pd.Series]] = []
    for candidate in candidates.itertuples(index=False):
        candidate_row = pd.Series(candidate._asdict())
        geom = geometry_from_metadata(candidate_row)
        # Audit geometries are also available in projected feet.  WGS84
        # distance is used only to rank managed candidates; the audit's
        # projected coverage distance remains the eligibility gate.
        distance = float(link_geom.distance(geom))
        angle = directed_angle_diff(link_bearing, bearing(geom))
        scored.append((distance, angle, candidate_row))
    scored.sort(key=lambda item: (item[0], item[1], str(item[2]["tmc"])))
    _, angle, winner = scored[0]
    if angle > 75.0 or float(row["distance_to_tmc_coverage_ft"]) > 3500.0:
        return None
    return {
        "source_tmc_upstream": str(winner["tmc"]),
        "source_tmc_downstream": "",
        "source_tmc_primary": str(winner["tmc"]),
        "source_road": str(winner["road"]),
        "source_road_order": float(winner["road_order"]),
        "weight_upstream": 1.0,
        "weight_downstream": 0.0,
        "managed_direction_difference_deg": float(angle),
    }


def corridor_geometry_catalog(metadata: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    catalog: dict[tuple[str, str], dict[str, object]] = {}
    actual = metadata[~metadata["road"].map(is_managed_tmc_road)].copy()
    actual["road_key"] = actual["road"].map(base_road)
    actual["direction_key"] = actual["direction"].map(normalized_text)
    for key, frame in actual.groupby(["road_key", "direction_key"], sort=False):
        frame = frame.sort_values(["road_order", "tmc"]).reset_index(drop=True)
        midpoints = [geometry_from_metadata(row).interpolate(0.5, normalized=True) for _, row in frame.iterrows()]
        if len(midpoints) == 1:
            line = LineString([midpoints[0], Point(midpoints[0].x + 1e-9, midpoints[0].y + 1e-9)])
        else:
            line = LineString([(point.x, point.y) for point in midpoints])
        positions = np.array([line.project(point) for point in midpoints], dtype=float)
        catalog[key] = {"tmcs": frame, "line": line, "positions": positions}
    return catalog


def interpolation_sources(
    row: pd.Series,
    catalog: Mapping[tuple[str, str], dict[str, object]],
) -> dict[str, object] | None:
    key = (base_road(row["assigned_dashboard_road"]), normalized_text(row["assigned_dashboard_direction"]))
    corridor = catalog.get(key)
    if corridor is None or len(corridor["tmcs"]) < 2:
        return None
    link_geom = wkt.loads(str(row["geometry_wgs84"]))
    position = float(corridor["line"].project(link_geom.interpolate(0.5, normalized=True)))
    positions = np.asarray(corridor["positions"], dtype=float)
    lower = np.where(positions <= position)[0]
    upper = np.where(positions >= position)[0]
    if not len(lower) or not len(upper):
        return None
    lower_index = int(lower[-1])
    upper_index = int(upper[0])
    if lower_index == upper_index:
        if upper_index + 1 < len(positions):
            upper_index += 1
        elif lower_index > 0:
            lower_index -= 1
        else:
            return None
    upstream = corridor["tmcs"].iloc[lower_index]
    downstream = corridor["tmcs"].iloc[upper_index]
    d_up = max(0.0, position - float(positions[lower_index]))
    d_down = max(0.0, float(positions[upper_index]) - position)
    total = d_up + d_down
    if total <= 0.0:
        weight_up = weight_down = 0.5
    else:
        weight_up = d_down / total
        weight_down = d_up / total
    return {
        "source_tmc_upstream": str(upstream["tmc"]),
        "source_tmc_downstream": str(downstream["tmc"]),
        "source_tmc_primary": str(upstream["tmc"] if weight_up >= weight_down else downstream["tmc"]),
        "source_road": str(upstream["road"]),
        "source_road_order": float(weight_up * float(upstream["road_order"]) + weight_down * float(downstream["road_order"])),
        "distance_upstream_mi": d_up * 69.0,
        "distance_downstream_mi": d_down * 69.0,
        "weight_upstream": weight_up,
        "weight_downstream": weight_down,
    }


def terminal_source(
    row: pd.Series,
    catalog: Mapping[tuple[str, str], dict[str, object]],
) -> dict[str, object] | None:
    key = (base_road(row["assigned_dashboard_road"]), normalized_text(row["assigned_dashboard_direction"]))
    corridor = catalog.get(key)
    if corridor is None or corridor["tmcs"].empty:
        return None
    source = (
        corridor["tmcs"].iloc[0]
        if row["coverage_relation"] == "before_first_tmc"
        else corridor["tmcs"].iloc[-1]
    )
    return {
        "source_tmc_upstream": str(source["tmc"]),
        "source_tmc_downstream": "",
        "source_tmc_primary": str(source["tmc"]),
        "source_road": str(source["road"]),
        "source_road_order": float(source["road_order"]),
        "weight_upstream": 1.0,
        "weight_downstream": 0.0,
    }


def treatment_for_audit_row(
    row: pd.Series,
    *,
    managed: Mapping[tuple[str, str], pd.DataFrame],
    catalog: Mapping[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    common = {
        "link_id": int(row["link_id"]),
        "from_node_id": int(row["from_node_id"]),
        "to_node_id": int(row["to_node_id"]),
        "corridor": str(row["assigned_dashboard_corridor"]),
        "road": str(row["assigned_dashboard_road"]),
        "direction": str(row["assigned_dashboard_direction"]),
        "facility_class": str(row["physical_facility_class"]),
        "facility_role": str(row["facility_role"]),
        "source_reason_tag": str(row["reason_tag"]),
        "source_recommendation": str(row["recommended_treatment"]),
        "distance_to_observation_mi": float(row["distance_to_tmc_coverage_mi"]),
        "direction_confidence": str(row["direction_assignment_confidence"]),
        "topology_relation": str(row["topology_relation"]),
        "coverage_relation": str(row["coverage_relation"]),
        "STREETNAME": str(row.get("STREETNAME", "")),
        "length_mi": float(row.get("length_mi", np.nan)),
        "lanes": float(row.get("lanes", np.nan)),
        "capacity": float(row.get("capacity", np.nan)),
        "free_speed": float(row.get("free_speed_mph", np.nan)),
        "link_type": float(row.get("link_type", np.nan)),
        "geometry_wgs84": str(row.get("geometry_wgs84", "")),
        "weight_upstream": np.nan,
        "weight_downstream": np.nan,
        "distance_decay_weight": np.nan,
        "decay_scale_mi": np.nan,
        "source_tmc_upstream": "",
        "source_tmc_downstream": "",
        "source_tmc_primary": "",
        "source_road": "",
        "source_road_order": np.nan,
    }

    if (
        row["physical_facility_class"] == "managed"
        and row["facility_role"] not in {"ramp", "model_connector"}
    ):
        source = choose_managed_match(row, managed)
        if source is not None:
            return {
                **common,
                **source,
                "treatment_tier": 1,
                "treatment": "actual_managed_canonical",
                "observation_class": "actual",
                "confidence": "high",
                "decision": "included",
                "decision_reason": "compatible managed TMC inventory and same-direction facility match",
            }

    if (
        row["reason_tag"]
        in {
            "internal_parallel_link_between_tmc_anchors",
            "internal_parallel_link_within_tmc_coverage",
        }
        and row["physical_facility_class"] == "gp"
        and row["facility_role"] not in {"ramp", "model_connector"}
        and bool(row["candidate_pool_eligible_but_not_selected"])
        and float(row["distance_to_selected_route_ft"]) <= 500.0
        and row["direction_assignment_confidence"] != "low"
    ):
        source = interpolation_sources(row, catalog)
        if source is not None:
            return {
                **common,
                **source,
                "treatment_tier": 2,
                "treatment": "virtual_internal_interpolation",
                "observation_class": "virtual",
                "confidence": "high",
                "decision": "included",
                "decision_reason": "same-direction GP link bounded within observed corridor coverage and within 500 ft of selected path",
            }

    distance = float(row["distance_to_tmc_coverage_mi"])
    terminal = row["coverage_relation"] in {"before_first_tmc", "after_last_tmc"}
    eligible_terminal = (
        terminal
        and distance <= 5.0
        and row["physical_facility_class"] == "gp"
        and row["facility_role"] not in {"ramp", "model_connector"}
        and bool(row["candidate_direction_pass"])
        and bool(row["candidate_lane_class_pass"])
    )
    if eligible_terminal and distance <= 1.0 and row["direction_assignment_confidence"] in {"high", "medium"}:
        source = terminal_source(row, catalog)
        if source is not None:
            scale = 1.0
            return {
                **common,
                **source,
                "treatment_tier": 3,
                "treatment": "virtual_terminal_extrapolation_0_1mi",
                "observation_class": "virtual",
                "confidence": "medium",
                "decision": "included",
                "distance_decay_weight": math.exp(-distance / scale),
                "decay_scale_mi": scale,
                "decision_reason": "same-facility directional terminal continuation within one mile",
            }
    if eligible_terminal and 1.0 < distance <= 5.0 and row["direction_assignment_confidence"] == "high":
        source = terminal_source(row, catalog)
        if source is not None:
            scale = 1.5
            return {
                **common,
                **source,
                "treatment_tier": 4,
                "treatment": "virtual_terminal_extrapolation_1_5mi",
                "observation_class": "virtual",
                "confidence": "low",
                "decision": "included",
                "distance_decay_weight": math.exp(-distance / scale),
                "decay_scale_mi": scale,
                "decision_reason": "same-facility high-confidence terminal continuation one to five miles from observations",
            }

    exclusion = "no defensible automatic treatment"
    if distance > 5.0:
        exclusion = "more than five miles outside observed coverage"
    elif row["facility_role"] in {"ramp", "model_connector"}:
        exclusion = "ramp or model connector requires facility-specific observation"
    elif row["direction_assignment_confidence"] == "low" or not bool(row["candidate_direction_pass"]):
        exclusion = "direction assignment is not sufficiently reliable"
    elif row["physical_facility_class"] == "managed":
        exclusion = "no compatible managed-facility TMC inventory"
    elif row["coverage_relation"] == "lateral_or_disconnected_outside_coverage":
        exclusion = "lateral or disconnected same-name segment"
    elif "branch" in str(row["reason_tag"]):
        exclusion = "alternative branch lacks confirmed carriageway continuity"
    return {
        **common,
        "treatment_tier": 9,
        "treatment": "excluded",
        "observation_class": "excluded",
        "confidence": "none",
        "decision": "excluded",
        "decision_reason": exclusion,
    }


def combine_profiles(
    treatment: Mapping[str, object],
    profiles: Mapping[str, pd.DataFrame],
    corridor_speed_envelope: tuple[float, float],
) -> pd.DataFrame:
    upstream = profiles.get(str(treatment["source_tmc_upstream"]).upper())
    if upstream is None or upstream.empty:
        raise KeyError(f"Missing profile for {treatment['source_tmc_upstream']}")
    result = upstream.copy()
    if treatment["treatment"] == "virtual_internal_interpolation":
        downstream = profiles.get(str(treatment["source_tmc_downstream"]).upper())
        if downstream is None or downstream.empty:
            raise KeyError(f"Missing profile for {treatment['source_tmc_downstream']}")
        joined = upstream.join(downstream, how="inner", lsuffix="_up", rsuffix="_down")
        w_up = float(treatment["weight_upstream"])
        w_down = float(treatment["weight_downstream"])
        result = pd.DataFrame(index=joined.index)
        for column in PROFILE_COLUMNS:
            result[column] = w_up * joined[f"{column}_up"] + w_down * joined[f"{column}_down"]
    else:
        weight = float(treatment["distance_decay_weight"])
        reference = result["reference_speed"].where(
            result["reference_speed"].between(1.0, 150.0),
            result["speed"].max(),
        )
        result["speed"] = reference + weight * (result["speed"] - reference)
        result["historical_average_speed"] = reference + weight * (
            result["historical_average_speed"] - reference
        )
    lower, upper = corridor_speed_envelope
    result["speed"] = result["speed"].clip(lower=lower, upper=upper)
    result.index.name = "time_minute"
    return result.reset_index()


def virtual_identification_row(treatment: Mapping[str, object]) -> dict[str, object]:
    geometry = wkt.loads(str(treatment["geometry_wgs84"]))
    start = geometry.coords[0]
    end = geometry.coords[-1]
    return {
        "tmc": treatment["virtual_tmc"],
        "road": treatment["road"],
        "direction": treatment["direction"],
        "intersection": f"VIRTUAL {treatment['treatment']} LINK {treatment['link_id']}",
        "state": "VA",
        "county": "",
        "zip": "",
        "start_latitude": start[1],
        "start_longitude": start[0],
        "end_latitude": end[1],
        "end_longitude": end[0],
        "miles": treatment["length_mi"],
        "road_order": treatment["source_road_order"],
        "timezone_name": "America/New_York",
        "type": "VIRTUAL",
        "country": "USA",
        "active_start_date": "",
        "active_end_date": "",
    }


def virtual_mapping_row(treatment: Mapping[str, object]) -> dict[str, object]:
    return {
        "tmc": treatment["virtual_tmc"],
        "road": treatment["road"],
        "direction": treatment["direction"],
        "corridor_name": treatment["corridor"],
        "road_order": treatment["source_road_order"],
        "tmc_miles": treatment["length_mi"],
        "route_link_count": 1,
        "route_length_mi": treatment["length_mi"],
        "length_ratio": 1.0,
        "match_confidence": 100.0 if treatment["confidence"] == "high" else 80.0,
        "match_status": treatment["treatment"],
        "sequence": 1,
        "link_id": treatment["link_id"],
        "from_node_id": treatment["from_node_id"],
        "to_node_id": treatment["to_node_id"],
        "length_mi": treatment["length_mi"],
        "cumulative_mi": treatment["length_mi"],
        "distance_to_tmc_ft": 0.0,
        "geometry_overlap_pct": 100.0,
        "bearing_diff_deg": 0.0,
        "STREETNAME": treatment["STREETNAME"],
        "lanes": treatment["lanes"],
        "capacity": treatment["capacity"],
        "free_speed": treatment["free_speed"],
        "link_type": treatment["link_type"],
        "treatment": treatment["treatment"],
        "source_tmc_upstream": treatment["source_tmc_upstream"],
        "source_tmc_downstream": treatment["source_tmc_downstream"],
        "weight_upstream": treatment["weight_upstream"],
        "weight_downstream": treatment["weight_downstream"],
        "distance_decay_weight": treatment["distance_decay_weight"],
    }


def write_virtual_products(
    treatments: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: pd.DataFrame,
    output: Path,
) -> dict[str, object]:
    virtual = treatments[treatments["observation_class"].eq("virtual")].copy()
    lookup = profile_lookup(profiles)
    corridor_envelopes = {
        key: (
            max(1.0, float(frame["speed"].quantile(0.01))),
            min(150.0, float(frame["speed"].quantile(0.99))),
        )
        for key, frame in profiles.merge(
            metadata[["tmc", "road", "direction"]], on="tmc", how="left"
        ).groupby(["road", "direction"], sort=False)
    }
    profile_rows: list[pd.DataFrame] = []
    identification_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    representative_date = pd.Timestamp("2025-10-01")
    for index, row in enumerate(virtual.to_dict("records"), start=1):
        row["virtual_tmc"] = f"VIRTUAL-{int(row['link_id'])}-{index:04d}"
        virtual.loc[virtual["link_id"].eq(row["link_id"]), "virtual_tmc"] = row["virtual_tmc"]
        envelope = corridor_envelopes.get(
            (row["road"], row["direction"]), (1.0, 150.0)
        )
        generated = combine_profiles(row, lookup, envelope)
        generated["tmc_code"] = row["virtual_tmc"]
        generated["measurement_tstamp"] = [
            (representative_date + pd.Timedelta(minutes=int(minute))).strftime("%Y-%m-%d %H:%M:%S")
            for minute in generated["time_minute"]
        ]
        generated["travel_time_minutes"] = (
            float(row["length_mi"]) / generated["speed"].where(generated["speed"] > 0) * 60.0
        )
        generated["treatment"] = row["treatment"]
        generated["source_tmc_upstream"] = row["source_tmc_upstream"]
        generated["source_tmc_downstream"] = row["source_tmc_downstream"]
        generated["weight_upstream"] = row["weight_upstream"]
        generated["weight_downstream"] = row["weight_downstream"]
        generated["distance_decay_weight"] = row["distance_decay_weight"]
        profile_rows.append(generated)
        identification_rows.append(virtual_identification_row(row))
        mapping_rows.append(virtual_mapping_row(row))
    if virtual.empty:
        return {"virtual_links": 0, "virtual_tmcs": 0, "virtual_profile_rows": 0}
    all_profiles = pd.concat(profile_rows, ignore_index=True)
    ident = pd.DataFrame(identification_rows)
    mapping = pd.DataFrame(mapping_rows)
    virtual_root = output / "virtual"
    virtual_root.mkdir(parents=True, exist_ok=True)
    virtual.to_csv(virtual_root / "virtual_link_treatments.csv", index=False)
    all_profiles.to_csv(virtual_root / "virtual_weekday_profiles.csv", index=False)
    ident.to_csv(virtual_root / "virtual_tmc_identification.csv", index=False)
    mapping.to_csv(virtual_root / "virtual_tmc_to_link.csv", index=False)
    cbi_root = virtual_root / "cbi-corridors"
    for corridor, group in virtual.groupby("corridor", sort=True):
        folder = cbi_root / f"VIRTUAL_{corridor}"
        folder.mkdir(parents=True, exist_ok=True)
        tmc_ids = set(group["virtual_tmc"])
        ident[ident["tmc"].isin(tmc_ids)].to_csv(folder / "TMC_Identification.csv", index=False)
        all_profiles[all_profiles["tmc_code"].isin(tmc_ids)][
            [
                "tmc_code",
                "measurement_tstamp",
                "speed",
                "historical_average_speed",
                "reference_speed",
                "travel_time_minutes",
            ]
        ].to_csv(folder / "Readings.csv", index=False)
    return {
        "virtual_links": int(len(virtual)),
        "virtual_tmcs": int(virtual["virtual_tmc"].nunique()),
        "virtual_profile_rows": int(len(all_profiles)),
        "virtual_mapping": str(virtual_root / "virtual_tmc_to_link.csv"),
        "virtual_cbi_input_root": str(cbi_root),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = (
        args.output_dir
        or args.readings.resolve().parent / "outputs" / "tmc-observation-coverage"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for folder in (
        output / "actual/gp-canonical",
        output / "actual/managed-canonical",
        output / "virtual/internal-interpolation",
        output / "virtual/terminal-extrapolation-0-1mi",
        output / "virtual/terminal-extrapolation-1-5mi",
        output / "excluded",
        output / "profiles",
        output / "manifests",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    canonical = pd.read_csv(args.canonical, dtype={"tmc": "string"}, low_memory=False)
    if "selected_for_node_pair_lookup" in canonical:
        canonical = canonical[truthy(canonical["selected_for_node_pair_lookup"])].copy()
    if canonical.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError("Canonical node-pair winners are not unique")
    metadata = pd.read_csv(args.metadata, dtype={"tmc": "string"}, low_memory=False)
    metadata["tmc"] = metadata["tmc"].astype("string").str.strip().str.upper()
    profiles = weekday_average_profiles(args.readings)
    observed_tmcs = set(profiles.loc[profiles["speed"].notna(), "tmc"])
    if not set(canonical["tmc"].astype(str).str.upper()).issubset(observed_tmcs):
        missing = sorted(set(canonical["tmc"].astype(str).str.upper()) - observed_tmcs)
        raise ValueError(f"Canonical winners lack observed weekday profiles: {missing[:10]}")

    canonical["observation_class"] = "actual"
    canonical["treatment_tier"] = 0
    canonical["treatment"] = np.where(
        canonical["road"].map(is_managed_tmc_road),
        "canonical_managed_actual",
        "canonical_gp_actual",
    )
    canonical["provenance"] = "frozen canonical node-pair winner; unchanged"
    canonical_gp = canonical[canonical["treatment"].eq("canonical_gp_actual")]
    canonical_managed = canonical[canonical["treatment"].eq("canonical_managed_actual")]
    canonical_gp.to_csv(output / "actual/gp-canonical/canonical_gp_actual.csv", index=False)
    canonical_managed.to_csv(output / "actual/managed-canonical/canonical_managed_actual.csv", index=False)
    profiles.to_csv(output / "profiles/actual_tmc_weekday_profiles.csv", index=False)

    audit = pd.read_csv(args.audit, low_memory=False)
    for column in (
        "candidate_pool_eligible_but_not_selected",
        "candidate_direction_pass",
        "candidate_lane_class_pass",
    ):
        audit[column] = truthy(audit[column])
    managed = managed_inventory(metadata)
    catalog = corridor_geometry_catalog(metadata)
    treatment_rows = [
        treatment_for_audit_row(row, managed=managed, catalog=catalog)
        for _, row in audit.iterrows()
    ]
    treatments = pd.DataFrame(treatment_rows).sort_values(
        ["treatment_tier", "corridor", "link_id"]
    )
    if treatments["link_id"].duplicated().any():
        raise ValueError("An audited link received more than one treatment")
    if set(treatments["link_id"]) & set(canonical["link_id"]):
        raise ValueError("A supplemental treatment conflicts with a canonical link")

    treatment_paths = {
        "actual_managed_canonical": output / "actual/managed-canonical/supplemental_managed_actual.csv",
        "virtual_internal_interpolation": output / "virtual/internal-interpolation/link_treatments.csv",
        "virtual_terminal_extrapolation_0_1mi": output / "virtual/terminal-extrapolation-0-1mi/link_treatments.csv",
        "virtual_terminal_extrapolation_1_5mi": output / "virtual/terminal-extrapolation-1-5mi/link_treatments.csv",
        "excluded": output / "excluded/excluded_links.csv",
    }
    for treatment, path in treatment_paths.items():
        treatments[treatments["treatment"].eq(treatment)].to_csv(path, index=False)
    treatments.to_csv(output / "link_treatment_decisions.csv", index=False)

    virtual_products = write_virtual_products(treatments, profiles, metadata, output)
    actual_managed_supplemental = treatments[
        treatments["treatment"].eq("actual_managed_canonical")
    ].copy()
    actual_managed_supplemental.to_csv(
        output / "actual/managed-canonical/supplemental_managed_actual.csv",
        index=False,
    )

    counts = treatments["treatment"].value_counts().astype(int).to_dict()
    summary = (
        treatments.groupby(
            ["treatment_tier", "treatment", "observation_class", "decision", "confidence"],
            dropna=False,
            as_index=False,
        )
        .agg(link_count=("link_id", "size"), total_length_mi=("length_mi", "sum"))
        .sort_values(["treatment_tier", "treatment"])
    )
    summary.to_csv(output / "treatment_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "canonical_source": str(args.canonical.resolve()),
        "canonical_source_sha256": sha256(args.canonical),
        "canonical_node_pairs_unchanged": int(len(canonical)),
        "canonical_gp_actual": int(len(canonical_gp)),
        "canonical_managed_actual": int(len(canonical_managed)),
        "audit_source": str(args.audit.resolve()),
        "audit_source_sha256": sha256(args.audit),
        "audited_unmatched_links": int(len(audit)),
        "treatment_counts": counts,
        "treatment_precedence": [
            "frozen canonical actual",
            "supplemental managed actual",
            "virtual internal interpolation",
            "virtual terminal extrapolation 0-1 mile",
            "virtual terminal extrapolation 1-5 miles",
            "excluded",
        ],
        "virtual_profile_rule": "observed weekday profiles only; assignment speeds prohibited",
        "internal_interpolation_rule": "along-corridor barycentric distance weights between bracketing actual TMCs",
        "terminal_0_1mi_decay": "exp(-distance_mi / 1.0)",
        "terminal_1_5mi_decay": "exp(-distance_mi / 1.5)",
        "beyond_5mi_rule": "excluded",
        "actual_tmc_weekday_profile_rows": int(len(profiles)),
        "virtual_products": virtual_products,
        "output_dir": str(output),
    }
    (output / "manifests/treatment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
