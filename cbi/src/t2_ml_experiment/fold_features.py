from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


CLASS_PRIOR_SOURCES = {
    "expected_mu_class": "mu_obs_vphpl",
    "expected_vc_class": "vc_mph",
    "expected_vf_class": "vf_model_mph",
    "expected_critical_density_class": "critical_density_vpmpl",
}

CLASS_LEVELS = [
    ["period", "network_link_type", "network_ftype"],
    ["period", "network_link_type"],
    ["period"],
]


def _class_prior(
    training: pd.DataFrame,
    query: pd.DataFrame,
    source: str,
    minimum_count: int = 5,
) -> pd.Series:
    result = pd.Series(np.nan, index=query.index, dtype=float)
    numeric_source = pd.to_numeric(training[source], errors="coerce")
    source_frame = training.copy()
    source_frame[source] = numeric_source
    for level in CLASS_LEVELS:
        stats = source_frame.groupby(level, dropna=False)[source].agg(
            ["median", "count"]
        )
        stats = stats[stats["count"] >= minimum_count]["median"]
        if len(level) == 1:
            candidate = query[level[0]].map(stats)
        else:
            keys = pd.MultiIndex.from_frame(query[level])
            candidate = pd.Series(
                stats.reindex(keys).to_numpy(), index=query.index
            )
        result = result.fillna(candidate)
    return result.fillna(numeric_source.median())


def add_class_physics_features(
    training: pd.DataFrame,
    query: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_out = training.copy()
    query_out = query.copy()
    for output, source in CLASS_PRIOR_SOURCES.items():
        if source not in training:
            train_out[output] = np.nan
            query_out[output] = np.nan
            continue
        train_out[output] = _class_prior(training, training, source)
        query_out[output] = _class_prior(training, query, source)
    return train_out, query_out


SPATIAL_OUTPUT_COLUMNS = [
    "spatial_anchor_count",
    "spatial_nearest_distance_mi",
    "spatial_bracket_gap_mi",
    "spatial_has_bracket",
    "spatial_t2_relative_min",
    "spatial_log_span_prior",
    "spatial_logit_fraction_prior",
]


def _anchor_table(training: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "corridor",
        "direction",
        "period",
        "tmc_code",
        "corridor_position_mi",
        "target_t2_relative_min",
        "target_log_span_min",
        "target_logit_t2_fraction",
    ]
    available = [column for column in needed if column in training]
    anchors = training[available].copy()
    for column in [
        "corridor_position_mi",
        "target_t2_relative_min",
        "target_log_span_min",
        "target_logit_t2_fraction",
    ]:
        anchors[column] = pd.to_numeric(anchors[column], errors="coerce")
    anchors = anchors.dropna(
        subset=[
            "corridor",
            "direction",
            "period",
            "tmc_code",
            "corridor_position_mi",
            "target_t2_relative_min",
            "target_log_span_min",
            "target_logit_t2_fraction",
        ]
    )
    return (
        anchors.groupby(
            [
                "corridor",
                "direction",
                "period",
                "tmc_code",
            ],
            as_index=False,
            dropna=False,
        )
        .median(numeric_only=True)
        .sort_values(
            [
                "corridor",
                "direction",
                "period",
                "corridor_position_mi",
            ],
            kind="mergesort",
        )
    )


def _interpolate(left: pd.Series, right: pd.Series, position: float, column: str) -> float:
    left_position = float(left["corridor_position_mi"])
    right_position = float(right["corridor_position_mi"])
    if right_position <= left_position:
        return float(left[column])
    weight = (position - left_position) / (right_position - left_position)
    return float(left[column]) + weight * (
        float(right[column]) - float(left[column])
    )


def spatial_features_from_training(
    training: pd.DataFrame,
    query: pd.DataFrame,
    *,
    exclude_query_tmc: bool,
) -> pd.DataFrame:
    anchors = _anchor_table(training)
    grouped: Dict[tuple, pd.DataFrame] = {
        key: group.reset_index(drop=True)
        for key, group in anchors.groupby(
            ["corridor", "direction", "period"], dropna=False, sort=False
        )
    }
    rows = []
    for item in query.itertuples(index=False):
        key = (
            getattr(item, "corridor", np.nan),
            getattr(item, "direction", np.nan),
            getattr(item, "period", np.nan),
        )
        position = pd.to_numeric(
            pd.Series([getattr(item, "corridor_position_mi", np.nan)]),
            errors="coerce",
        ).iloc[0]
        group = grouped.get(key)
        if group is None or pd.isna(position):
            rows.append({column: np.nan for column in SPATIAL_OUTPUT_COLUMNS})
            continue
        candidates = group
        if exclude_query_tmc:
            candidates = candidates[
                candidates["tmc_code"].astype(str)
                != str(getattr(item, "tmc_code", ""))
            ]
        if candidates.empty:
            rows.append({column: np.nan for column in SPATIAL_OUTPUT_COLUMNS})
            continue
        candidates = candidates.sort_values(
            "corridor_position_mi", kind="mergesort"
        )
        distances = (
            candidates["corridor_position_mi"] - float(position)
        ).abs()
        nearest = candidates.loc[distances.idxmin()]
        left = candidates[
            candidates["corridor_position_mi"] <= float(position)
        ]
        right = candidates[
            candidates["corridor_position_mi"] >= float(position)
        ]
        has_bracket = not left.empty and not right.empty
        if has_bracket:
            left_anchor = left.iloc[-1]
            right_anchor = right.iloc[0]
            t2_prior = _interpolate(
                left_anchor,
                right_anchor,
                float(position),
                "target_t2_relative_min",
            )
            span_prior = _interpolate(
                left_anchor,
                right_anchor,
                float(position),
                "target_log_span_min",
            )
            fraction_prior = _interpolate(
                left_anchor,
                right_anchor,
                float(position),
                "target_logit_t2_fraction",
            )
            bracket_gap = (
                float(right_anchor["corridor_position_mi"])
                - float(left_anchor["corridor_position_mi"])
            )
        else:
            t2_prior = float(nearest["target_t2_relative_min"])
            span_prior = float(nearest["target_log_span_min"])
            fraction_prior = float(
                nearest["target_logit_t2_fraction"]
            )
            bracket_gap = np.nan
        rows.append(
            {
                "spatial_anchor_count": int(len(candidates)),
                "spatial_nearest_distance_mi": float(distances.min()),
                "spatial_bracket_gap_mi": bracket_gap,
                "spatial_has_bracket": int(has_bracket),
                "spatial_t2_relative_min": t2_prior,
                "spatial_log_span_prior": span_prior,
                "spatial_logit_fraction_prior": fraction_prior,
            }
        )
    return pd.DataFrame(rows, index=query.index)


def add_spatial_features(
    training: pd.DataFrame,
    query: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_out = training.copy()
    query_out = query.copy()
    train_features = spatial_features_from_training(
        training, training, exclude_query_tmc=True
    )
    query_features = spatial_features_from_training(
        training, query, exclude_query_tmc=True
    )
    for column in SPATIAL_OUTPUT_COLUMNS:
        train_out[column] = train_features[column]
        query_out[column] = query_features[column]
    return train_out, query_out


def prepare_fold_features(
    training: pd.DataFrame,
    query: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "none":
        return training.copy(), query.copy()
    if mode == "class_physics":
        return add_class_physics_features(training, query)
    if mode == "spatial":
        return add_spatial_features(training, query)
    if mode == "class_spatial":
        train_out, query_out = add_class_physics_features(training, query)
        return add_spatial_features(train_out, query_out)
    raise ValueError(f"Unknown fold feature mode: {mode}")
