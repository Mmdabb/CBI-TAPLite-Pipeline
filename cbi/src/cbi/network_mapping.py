from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


COMPOSITE_SCORE_WEIGHTS = {
    "geometry_overlap_score": 0.25,
    "road_name_agreement_score": 0.15,
    "direction_compatibility_score": 0.20,
    "functional_class_compatibility_score": 0.10,
    "relative_position_score": 0.10,
    "observation_quality_score": 0.10,
    "length_compatibility_score": 0.10,
}

TEXT_SOURCE_COLUMNS = {
    "road",
    "direction",
    "STREETNAME",
    "match_status",
    "geometry_overlap_source",
}

NUMERIC_SOURCE_COLUMNS = {
    "link_id",
    "from_node_id",
    "to_node_id",
    "length_mi",
    "lanes",
    "capacity",
    "free_speed",
    "link_type",
    "road_order",
    "sequence",
    "cumulative_mi",
    "distance_to_tmc_ft",
    "bearing_diff_deg",
    "first_map_occurrence",
    "geometry_overlap_pct",
    "route_link_count",
    "tmc_miles",
    "route_length_mi",
    "length_ratio",
    "match_confidence",
    "observation_valid_speed_share",
    "observation_interval_completeness_share",
    "observation_weekday_coverage_share",
    "observation_profile_bin_coverage_share",
    "observation_quality_score",
    *COMPOSITE_SCORE_WEIGHTS,
    "composite_match_score",
    "composite_available_weight",
}

CANONICAL_PAIR_COLUMNS = [
    "tmc",
    "road",
    "direction",
    "road_order",
    "link_id",
    "from_node_id",
    "to_node_id",
    "STREETNAME",
    "length_mi",
    "lanes",
    "capacity",
    "free_speed",
    "link_type",
    "sequence",
    "cumulative_mi",
    "distance_to_tmc_ft",
    "bearing_diff_deg",
    "route_link_count",
    "tmc_miles",
    "route_length_mi",
    "length_ratio",
    "match_confidence",
    "match_status",
    "geometry_overlap_pct",
    "geometry_overlap_source",
    "geometry_overlap_score",
    "road_name_agreement_score",
    "direction_compatibility_score",
    "functional_class_compatibility_score",
    "relative_position_score",
    "observation_valid_speed_share",
    "observation_interval_completeness_share",
    "observation_weekday_coverage_share",
    "observation_profile_bin_coverage_share",
    "observation_quality_score",
    "length_compatibility_score",
    "composite_match_score",
    "composite_available_weight",
    "first_map_occurrence",
    "link_tmc_rank",
    "link_tmc_ranking_basis",
    "tmc_link_rank",
    "tmc_link_ranking_basis",
    "node_pair_tmc_rank",
    "node_pair_tmc_ranking_basis",
    "selected_for_node_pair_lookup",
    "node_pair_has_observed_candidate",
    "node_pair_winner_has_observation",
]


def normalize_tmc(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper()


def _unit_interval(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(lower=0.0, upper=1.0)


def _road_tokens(value: object) -> list[str]:
    text = str(value or "").upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    replacements = {
        "INTERSTATE": "I",
        "UNITED STATES": "US",
        "HIGHWAY": "HWY",
        "BOULEVARD": "BLVD",
        "AVENUE": "AVE",
        "PARKWAY": "PKWY",
        "TURNPIKE": "TPKE",
        "ROAD": "RD",
        "STREET": "ST",
    }
    for source, target in replacements.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    ignored = {
        "N",
        "S",
        "E",
        "W",
        "NB",
        "SB",
        "EB",
        "WB",
        "NORTHBOUND",
        "SOUTHBOUND",
        "EASTBOUND",
        "WESTBOUND",
        "RD",
        "ST",
        "AVE",
        "BLVD",
        "PKWY",
        "HWY",
        "TPKE",
    }
    return [token for token in text.split() if token and token not in ignored]


def _route_identifiers(tokens: list[str]) -> set[str]:
    identifiers: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if token in {"I", "US", "VA", "SR", "ROUTE"} and tokens[index + 1].isdigit():
            prefix = "VA" if token in {"SR", "ROUTE"} else token
            identifiers.add(f"{prefix}-{int(tokens[index + 1])}")
    for token in tokens:
        match = re.fullmatch(r"(I|US|VA|SR)(\d+)", token)
        if match:
            prefix = "VA" if match.group(1) == "SR" else match.group(1)
            identifiers.add(f"{prefix}-{int(match.group(2))}")
    return identifiers


def _road_name_score(road: object, street: object) -> float:
    road_tokens = _road_tokens(road)
    street_tokens = _road_tokens(street)
    if not road_tokens or not street_tokens:
        return float("nan")
    road_routes = _route_identifiers(road_tokens)
    street_routes = _route_identifiers(street_tokens)
    if road_routes and street_routes:
        return 1.0 if road_routes & street_routes else 0.0
    road_set = set(road_tokens)
    street_set = set(street_tokens)
    shared = road_set & street_set
    if shared:
        jaccard = len(shared) / len(road_set | street_set)
        containment = len(shared) / min(len(road_set), len(street_set))
        return float(max(jaccard, containment))
    similarity = SequenceMatcher(
        None, " ".join(road_tokens), " ".join(street_tokens)
    ).ratio()
    # A numbered road often has a named-highway alias in the network. Treat a
    # no-token match as neutral unless both sides provide conflicting numbers.
    return float(max(0.5, similarity))


def _inferred_tmc_facility_class(road: object) -> float:
    tokens = _road_tokens(road)
    identifiers = _route_identifiers(tokens)
    if any(value.startswith("I-") for value in identifiers):
        return 1.0
    if any(value.startswith("US-") for value in identifiers):
        return 2.0
    if any(value.startswith("VA-") for value in identifiers):
        return 3.0
    return 3.5 if tokens else float("nan")


def _attach_route_summary(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    summary_path = source.with_name("full_route_match_summary.csv")
    if not summary_path.is_file() or "tmc" not in frame:
        return frame
    header = set(pd.read_csv(summary_path, nrows=0).columns)
    wanted = {
        "tmc",
        "tmc_miles",
        "route_link_count",
        "route_length_mi",
        "length_ratio",
        "confidence",
        "status",
    }
    if "tmc" not in header:
        return frame
    summary = pd.read_csv(
        summary_path,
        usecols=lambda column: column in wanted,
        dtype={"tmc": "string"},
        low_memory=False,
    )
    summary["tmc"] = normalize_tmc(summary["tmc"])
    summary = summary.drop_duplicates("tmc", keep="first").rename(
        columns={"confidence": "match_confidence", "status": "match_status"}
    )
    value_columns = [column for column in summary.columns if column != "tmc"]
    merged = frame.merge(
        summary,
        on="tmc",
        how="left",
        validate="many_to_one",
        suffixes=("", "_route_summary"),
    )
    for column in value_columns:
        summary_column = f"{column}_route_summary"
        if summary_column in merged:
            current_available = merged[column].notna()
            if column in TEXT_SOURCE_COLUMNS:
                current_available &= merged[column].astype("string").str.strip().ne("")
            merged[column] = merged[column].where(current_available, merged[summary_column])
            merged = merged.drop(columns=summary_column)
    return merged


def _attach_observation_quality(
    frame: pd.DataFrame,
    observation_quality: pd.DataFrame | None,
) -> pd.DataFrame:
    if observation_quality is None or observation_quality.empty:
        return frame
    quality = observation_quality.copy()
    tmc_column = "tmc" if "tmc" in quality else "tmc_code"
    if tmc_column not in quality:
        raise ValueError("Observation-quality table must contain tmc or tmc_code")
    quality = quality.rename(columns={tmc_column: "tmc"})
    quality["tmc"] = normalize_tmc(quality["tmc"])
    quality = quality.drop_duplicates("tmc", keep="last")
    quality_columns = [
        column
        for column in quality.columns
        if column == "tmc" or column.startswith("observation_")
    ]
    existing = [column for column in quality_columns if column != "tmc" and column in frame]
    if existing:
        frame = frame.drop(columns=existing)
    return frame.merge(
        quality[quality_columns], on="tmc", how="left", validate="many_to_one"
    )


def _score_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()

    overlap = pd.to_numeric(scored.get("geometry_overlap_pct"), errors="coerce")
    distance = pd.to_numeric(scored.get("distance_to_tmc_ft"), errors="coerce")
    scored["geometry_overlap_score"] = _unit_interval(overlap / 100.0)
    fallback = np.exp(-distance.clip(lower=0.0) / 300.0)
    scored["geometry_overlap_score"] = scored["geometry_overlap_score"].fillna(fallback)
    scored["geometry_overlap_source"] = np.where(
        overlap.notna(), "link_share_within_tmc_buffer", "lateral_distance_proxy_300ft_decay"
    )

    scored["road_name_agreement_score"] = [
        _road_name_score(road, street)
        for road, street in zip(scored.get("road", ""), scored.get("STREETNAME", ""))
    ]

    bearing = pd.to_numeric(scored.get("bearing_diff_deg"), errors="coerce")
    scored["direction_compatibility_score"] = np.cos(
        np.deg2rad(bearing.clip(lower=0.0, upper=90.0))
    )
    scored.loc[bearing > 90.0, "direction_compatibility_score"] = 0.0

    link_type = pd.to_numeric(scored.get("link_type"), errors="coerce")
    link_facility = np.mod(link_type, 10.0)
    link_facility = link_facility.where(link_facility.between(1.0, 9.0))
    tmc_facility = pd.Series(
        [_inferred_tmc_facility_class(value) for value in scored.get("road", "")],
        index=scored.index,
        dtype=float,
    )
    scored["functional_class_compatibility_score"] = np.exp(
        -0.7 * (link_facility - tmc_facility).abs()
    )

    route_length = pd.to_numeric(scored.get("route_length_mi"), errors="coerce")
    route_length = route_length.fillna(
        scored.groupby("tmc")["length_mi"].transform("sum")
    )
    midpoint = (
        pd.to_numeric(scored.get("cumulative_mi"), errors="coerce")
        - pd.to_numeric(scored.get("length_mi"), errors="coerce") / 2.0
    )
    route_fraction = midpoint / route_length.where(route_length > 0.0)
    scored["relative_position_score"] = (
        1.0 - 2.0 * (route_fraction - 0.5).abs()
    ).clip(lower=0.0, upper=1.0)

    length_ratio = pd.to_numeric(scored.get("length_ratio"), errors="coerce")
    tmc_miles = pd.to_numeric(scored.get("tmc_miles"), errors="coerce")
    length_ratio = length_ratio.fillna(route_length / tmc_miles.where(tmc_miles > 0.0))
    valid_ratio = length_ratio.where(length_ratio > 0.0)
    scored["length_compatibility_score"] = np.exp(-np.log(valid_ratio).abs())

    for column in COMPOSITE_SCORE_WEIGHTS:
        if column not in scored:
            scored[column] = np.nan
        scored[column] = _unit_interval(scored[column])
    weighted_sum = pd.Series(0.0, index=scored.index)
    available_weight = pd.Series(0.0, index=scored.index)
    for column, weight in COMPOSITE_SCORE_WEIGHTS.items():
        available = scored[column].notna()
        weighted_sum += scored[column].fillna(0.0) * weight
        available_weight += available.astype(float) * weight
    scored["composite_available_weight"] = available_weight
    scored["composite_match_score"] = (
        weighted_sum / available_weight.where(available_weight > 0.0) * 100.0
    )
    return scored


def _rank_pairs(
    pairs: pd.DataFrame,
    *,
    group_column: str,
    rank_column: str,
    basis_column: str,
    tie_column: str,
) -> pd.DataFrame:
    ranked = pairs.copy()
    ranked["_missing_score"] = ranked["composite_match_score"].isna()
    ranked["_missing_distance"] = ranked["distance_to_tmc_ft"].isna()
    ranked = ranked.sort_values(
        [
            group_column,
            "_missing_score",
            "composite_match_score",
            "_missing_distance",
            "distance_to_tmc_ft",
            "first_map_occurrence",
            tie_column,
        ],
        ascending=[True, True, False, True, True, True, True],
        kind="mergesort",
        na_position="last",
    )
    ranked[rank_column] = ranked.groupby(group_column, sort=False).cumcount() + 1
    ranked[basis_column] = np.select(
        [
            ranked["composite_match_score"].notna(),
            ranked["distance_to_tmc_ft"].notna(),
        ],
        ["highest_composite_match_score", "closest_distance_to_tmc_ft"],
        default="first_mapping_occurrence",
    )
    return ranked.drop(columns=["_missing_score", "_missing_distance"])


def _rank_node_pair_candidates(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select one immutable TMC winner for each directed network node pair.

    This is the authoritative link-to-observation ranking.  It is computed
    directly from every candidate in ``full_tmc_to_link.csv`` and must be
    reused downstream without filtering through a TMC-primary view or
    reranking by assignment period.
    """

    ranked = pairs.copy()
    ranked["node_pair_tmc_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked["node_pair_tmc_ranking_basis"] = pd.Series(
        pd.NA, index=ranked.index, dtype="string"
    )
    ranked["selected_for_node_pair_lookup"] = False
    valid = ranked["from_node_id"].notna() & ranked["to_node_id"].notna()
    candidates = ranked.loc[valid].copy()
    candidates["_has_observation"] = candidates[
        "observation_quality_score"
    ].notna()
    candidates["node_pair_has_observed_candidate"] = candidates.groupby(
        ["from_node_id", "to_node_id"], sort=False
    )["_has_observation"].transform("any")
    if candidates.empty:
        return ranked
    candidates["_missing_score"] = candidates["composite_match_score"].isna()
    candidates["_missing_distance"] = candidates["distance_to_tmc_ft"].isna()
    candidates = candidates.sort_values(
        [
            "from_node_id",
            "to_node_id",
            "_missing_score",
            "composite_match_score",
            "_missing_distance",
            "distance_to_tmc_ft",
            "first_map_occurrence",
            "tmc",
            "link_id",
        ],
        ascending=[True, True, True, False, True, True, True, True, True],
        kind="mergesort",
        na_position="last",
    )
    candidates["node_pair_tmc_rank"] = (
        candidates.groupby(["from_node_id", "to_node_id"], sort=False).cumcount()
        + 1
    )
    candidates["node_pair_tmc_ranking_basis"] = np.select(
        [
            candidates["composite_match_score"].notna(),
            candidates["distance_to_tmc_ft"].notna(),
        ],
        ["highest_composite_match_score", "closest_distance_to_tmc_ft"],
        default="first_mapping_occurrence",
    )
    candidates["selected_for_node_pair_lookup"] = candidates[
        "node_pair_tmc_rank"
    ].eq(1)
    candidates["node_pair_winner_has_observation"] = candidates.groupby(
        ["from_node_id", "to_node_id"], sort=False
    )["_has_observation"].transform("first")
    update_columns = [
        "node_pair_tmc_rank",
        "node_pair_tmc_ranking_basis",
        "selected_for_node_pair_lookup",
        "node_pair_has_observed_candidate",
        "node_pair_winner_has_observation",
    ]
    ranked.loc[candidates.index, update_columns] = candidates[update_columns]
    ranked["node_pair_tmc_rank"] = pd.to_numeric(
        ranked["node_pair_tmc_rank"], errors="coerce"
    ).astype("Int64")
    ranked["selected_for_node_pair_lookup"] = ranked[
        "selected_for_node_pair_lookup"
    ].fillna(False).astype(bool)
    selected = ranked[ranked["selected_for_node_pair_lookup"]]
    if selected.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError("Canonical node-pair winners are not unique")
    return ranked


def build_observation_quality(input_root: Path) -> pd.DataFrame:
    """Summarize valid weekday speed availability for winner selection."""

    rows: list[pd.DataFrame] = []
    root = Path(input_root).resolve()
    for corridor in sorted(root.iterdir()):
        readings_path = corridor / "Readings.csv"
        metadata_path = corridor / "TMC_Identification.csv"
        if not readings_path.is_file() or not metadata_path.is_file():
            continue
        header = set(pd.read_csv(readings_path, nrows=0).columns)
        tmc_column = "tmc_code" if "tmc_code" in header else "tmc"
        time_column = next(
            (name for name in ("measurement_tstamp", "datetime", "timestamp") if name in header),
            None,
        )
        speed_column = next(
            (name for name in ("speed", "speed_mph", "avg_speed") if name in header),
            None,
        )
        if tmc_column not in header or time_column is None or speed_column is None:
            continue
        frame = pd.read_csv(
            readings_path,
            usecols=[tmc_column, time_column, speed_column],
            dtype={tmc_column: "string"},
            low_memory=False,
        ).rename(columns={tmc_column: "tmc"})
        frame["tmc"] = normalize_tmc(frame["tmc"])
        frame["datetime"] = pd.to_datetime(frame[time_column], errors="coerce")
        frame["speed"] = pd.to_numeric(frame[speed_column], errors="coerce")
        frame = frame[frame["datetime"].notna() & frame["tmc"].notna()].copy()
        frame = frame[frame["datetime"].dt.weekday < 5].copy()
        if frame.empty:
            continue
        frame["valid_speed"] = frame["speed"].between(1.0, 150.0)
        frame["date"] = frame["datetime"].dt.normalize()
        frame["profile_bin"] = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
        expected_intervals = max(1, int(frame["datetime"].nunique()))
        expected_days = max(1, int(frame["date"].nunique()))
        expected_bins = max(1, int(frame["profile_bin"].nunique()))
        valid = frame[frame["valid_speed"]]
        total_by_tmc = frame.groupby("tmc", sort=False).size().rename("total_rows")
        quality = valid.groupby("tmc", sort=False).agg(
            valid_rows=("speed", "size"),
            valid_intervals=("datetime", "nunique"),
            covered_weekdays=("date", "nunique"),
            covered_profile_bins=("profile_bin", "nunique"),
        )
        quality = total_by_tmc.to_frame().join(quality, how="left").fillna(0.0)
        quality["observation_valid_speed_share"] = (
            quality["valid_rows"] / quality["total_rows"].where(quality["total_rows"] > 0)
        ).clip(0.0, 1.0)
        quality["observation_interval_completeness_share"] = (
            quality["valid_intervals"] / expected_intervals
        ).clip(0.0, 1.0)
        quality["observation_weekday_coverage_share"] = (
            quality["covered_weekdays"] / expected_days
        ).clip(0.0, 1.0)
        quality["observation_profile_bin_coverage_share"] = (
            quality["covered_profile_bins"] / expected_bins
        ).clip(0.0, 1.0)
        quality["observation_quality_score"] = (
            0.20 * quality["observation_valid_speed_share"]
            + 0.50 * quality["observation_interval_completeness_share"]
            + 0.15 * quality["observation_weekday_coverage_share"]
            + 0.15 * quality["observation_profile_bin_coverage_share"]
        )
        quality = quality.reset_index()
        quality["observation_corridor"] = corridor.name
        rows.append(quality)
    if not rows:
        return pd.DataFrame(
            columns=["tmc", "observation_quality_score", "observation_corridor"]
        )
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return combined.sort_values(
        ["tmc", "observation_quality_score"], ascending=[True, False], kind="mergesort"
    ).drop_duplicates("tmc", keep="first").reset_index(drop=True)


def load_canonical_mapping(
    path: Path,
    observation_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize, score, and rank every distinct TMC/network-link pair once."""

    source = Path(path).resolve()
    header = set(pd.read_csv(source, nrows=0).columns)
    tmc_column = "tmc" if "tmc" in header else "tmc_code"
    required = {tmc_column, "link_id"}
    missing = sorted(required - header)
    if missing:
        raise ValueError(f"{source} is missing mapping columns: {missing}")
    wanted = {tmc_column, *TEXT_SOURCE_COLUMNS, *NUMERIC_SOURCE_COLUMNS}
    frame = pd.read_csv(
        source,
        usecols=lambda column: column in wanted,
        dtype={tmc_column: "string"},
        low_memory=False,
    ).rename(columns={tmc_column: "tmc"})
    frame["tmc"] = normalize_tmc(frame["tmc"])
    for column in TEXT_SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype("string")
    for column in NUMERIC_SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["first_map_occurrence"] = frame["first_map_occurrence"].where(
        frame["first_map_occurrence"].notna(),
        pd.Series(np.arange(1, len(frame) + 1), index=frame.index),
    )
    frame = frame.dropna(subset=["tmc", "link_id"]).copy()
    frame["link_id"] = frame["link_id"].astype(np.int64)
    frame = _attach_route_summary(frame, source)
    frame = _attach_observation_quality(frame, observation_quality)
    frame = _score_pairs(frame)

    frame["_missing_score"] = frame["composite_match_score"].isna()
    frame["_missing_distance"] = frame["distance_to_tmc_ft"].isna()
    pairs = (
        frame.sort_values(
            [
                "tmc",
                "link_id",
                "_missing_score",
                "composite_match_score",
                "_missing_distance",
                "distance_to_tmc_ft",
                "first_map_occurrence",
            ],
            ascending=[True, True, True, False, True, True, True],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(["tmc", "link_id"], keep="first")
        .drop(columns=["_missing_score", "_missing_distance"])
    )
    link_ranked = _rank_pairs(
        pairs,
        group_column="link_id",
        rank_column="link_tmc_rank",
        basis_column="link_tmc_ranking_basis",
        tie_column="tmc",
    )
    link_fields = link_ranked[
        ["tmc", "link_id", "link_tmc_rank", "link_tmc_ranking_basis"]
    ]
    tmc_ranked = _rank_pairs(
        pairs,
        group_column="tmc",
        rank_column="tmc_link_rank",
        basis_column="tmc_link_ranking_basis",
        tie_column="link_id",
    )
    canonical = tmc_ranked.merge(
        link_fields,
        on=["tmc", "link_id"],
        how="left",
        validate="one_to_one",
    )
    canonical = _rank_node_pair_candidates(canonical)
    for column in CANONICAL_PAIR_COLUMNS:
        if column not in canonical:
            canonical[column] = np.nan
    return canonical[CANONICAL_PAIR_COLUMNS].sort_values(
        ["link_id", "link_tmc_rank", "tmc"],
        kind="mergesort",
    ).reset_index(drop=True)


def write_canonical_mapping_artifacts(
    source_path: Path,
    output_directory: Path,
    observation_quality: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Publish the scored all-pairs mapping and both deterministic primary views."""

    source = Path(source_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = load_canonical_mapping(source, observation_quality=observation_quality)
    all_pairs_path = output / "canonical_tmc_link_pairs.csv"
    link_primary_path = output / "canonical_link_tmc_primary.csv"
    tmc_primary_path = output / "canonical_tmc_link_primary.csv"
    node_pair_primary_path = output / "canonical_node_pair_tmc.csv"
    quality_path = output / "canonical_tmc_observation_quality.csv"
    pairs.to_csv(all_pairs_path, index=False)
    pairs[pairs["link_tmc_rank"].eq(1)].to_csv(link_primary_path, index=False)
    pairs[pairs["tmc_link_rank"].eq(1)].to_csv(tmc_primary_path, index=False)
    pairs[pairs["selected_for_node_pair_lookup"]].to_csv(
        node_pair_primary_path, index=False
    )
    if observation_quality is None:
        observation_quality = pd.DataFrame(columns=["tmc", "observation_quality_score"])
    observation_quality.to_csv(quality_path, index=False)
    manifest_path = output / "canonical_mapping_manifest.json"
    node_pair_winners = pairs[pairs["selected_for_node_pair_lookup"]].copy()
    observed_candidate_tmcs = set(
        pairs.loc[pairs["observation_quality_score"].notna(), "tmc"]
    )
    winning_tmcs = set(node_pair_winners["tmc"])
    manifest = {
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
        "all_pair_rows": int(len(pairs)),
        "unique_tmcs": int(pairs["tmc"].nunique()),
        "unique_links": int(pairs["link_id"].nunique()),
        "unique_directed_node_pairs": int(
            pairs[["from_node_id", "to_node_id"]].dropna().drop_duplicates().shape[0]
        ),
        "node_pair_winner_rows": int(len(node_pair_winners)),
        "node_pair_winners_with_observations": int(
            node_pair_winners["node_pair_winner_has_observation"].fillna(False).sum()
        ),
        "node_pair_winners_without_observations_despite_observed_candidate": int(
            (
                node_pair_winners["node_pair_has_observed_candidate"].fillna(False)
                & ~node_pair_winners["node_pair_winner_has_observation"].fillna(False)
            ).sum()
        ),
        "node_pair_winners_not_historical_tmc_primary": int(
            node_pair_winners["tmc_link_rank"].ne(1).sum()
        ),
        "observed_candidate_tmcs": int(len(observed_candidate_tmcs)),
        "observed_tmcs_winning_at_least_one_node_pair": int(
            len(observed_candidate_tmcs & winning_tmcs)
        ),
        "observed_tmcs_excluded_as_nonwinners": int(
            len(observed_candidate_tmcs - winning_tmcs)
        ),
        "all_pairs": str(all_pairs_path),
        "link_primary": str(link_primary_path),
        "tmc_primary": str(tmc_primary_path),
        "node_pair_primary": str(node_pair_primary_path),
        "observation_quality": str(quality_path),
        "selection_rule": (
            "within each directed node pair, highest availability-normalized "
            "composite match score; then closest distance_to_tmc_ft; then "
            "first stable mapping occurrence"
        ),
        "authoritative_link_to_tmc_rule": (
            "rank every full_tmc_to_link candidate once by directed "
            "from_node_id/to_node_id with the same composite score and "
            "deterministic tie-breakers; reuse "
            "selected_for_node_pair_lookup "
            "without period-specific or CBI-primary reranking"
        ),
        "composite_score_weights": COMPOSITE_SCORE_WEIGHTS,
        "missing_component_rule": (
            "omit unavailable components and renormalize by available weight"
        ),
        "geometry_fallback_rule": (
            "use exp(-distance_to_tmc_ft/300) when geometry_overlap_pct is absent"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "all_pairs": all_pairs_path,
        "link_primary": link_primary_path,
        "tmc_primary": tmc_primary_path,
        "node_pair_primary": node_pair_primary_path,
        "observation_quality": quality_path,
        "manifest": manifest_path,
    }
