from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NETWORK_CATEGORICAL = [
    "period",
    "network_link_type",
    "network_ftype",
    "network_allowed_use",
    "network_mode",
    "network_jurisdiction",
    "network_comp",
    "network_toll_group",
]

LOW_CARDINALITY_CATEGORICAL = [
    "period",
    "network_link_type",
    "network_ftype",
    "network_allowed_use",
    "network_mode",
]

DAILY_CATEGORICAL = [*NETWORK_CATEGORICAL, "weekday_name"]
DAILY_LOW_CARDINALITY_CATEGORICAL = [
    *LOW_CARDINALITY_CATEGORICAL,
    "weekday_name",
]

CORE_NUMERIC = [
    "network_lanes",
    "network_capacity_per_lane",
    "network_free_speed_mph",
    "network_length_mi",
    "network_toll",
    "network_vdf_alpha",
    "network_vdf_beta",
    "network_vdf_plf",
    "network_ref_volume",
    "network_ref_cost",
    "network_vdf_fftt",
    "period_volume",
    "hourly_link_capacity",
    "hourly_capacity_per_lane",
    "period_volume_per_lane",
    "period_demand_capacity_ratio",
]

TRAFFIC_MODEL_NUMERIC = [
    "reported_hourly_lane_capacity",
    "period_model_speed_mph",
    "period_model_freeflow_mph",
    "period_model_speed_ratio",
    "period_model_delay_index",
    "period_model_vc",
    "period_model_vdf",
    "period_model_vmt",
    "period_travel_time",
    "period_lane_limit",
    "period_toll_value",
    "period_truck_share",
    "period_hov_share",
    "period_commercial_share",
]

GRAPH_NUMERIC = [
    "upstream_link_count",
    "downstream_link_count",
    "upstream_max_dc",
    "downstream_max_dc",
    "upstream_mean_capacity",
    "downstream_min_capacity",
    "downstream_mean_capacity",
    "downstream_capacity_ratio",
    "downstream_bottleneck_strength",
    "upstream_max_model_vc",
    "downstream_max_model_vc",
    "downstream_min_speed_ratio",
    "is_merge_node",
    "is_diverge_node",
]

CLASS_PHYSICS_NUMERIC = [
    "expected_mu_class",
    "expected_vc_class",
    "expected_vf_class",
    "expected_critical_density_class",
]

SPATIAL_NUMERIC = [
    "spatial_anchor_count",
    "spatial_nearest_distance_mi",
    "spatial_bracket_gap_mi",
    "spatial_has_bracket",
    "spatial_t2_relative_min",
    "spatial_log_span_prior",
    "spatial_logit_fraction_prior",
]

PROFILE_NUMERIC = [
    "profile_speed_trough_relative_min",
    "profile_flow_peak_relative_min",
    "profile_min_speed_ratio",
    "profile_mean_speed_ratio",
    "profile_max_flow_capacity_ratio",
    "profile_mean_flow_capacity_ratio",
    "profile_first_capacity_cross_relative_min",
    "profile_share_bins_over_capacity",
    "profile_bin_coverage",
    "profile_day_count",
]

FD_NUMERIC = [
    "fd_capacity_vphpl",
    "vc_mph",
    "vf_model_mph",
    "critical_density_vpmpl",
]

EPISODE_DIAGNOSTIC_NUMERIC = [
    "P_hr",
    "demand_capacity_ratio",
    "mu_obs_vphpl",
    "min_speed_mph",
    "mean_speed_mph",
    "episode_demand",
    "magnitude",
    "severity",
]

LEAKAGE_OR_POST_DETECTION_FIELDS = set(EPISODE_DIAGNOSTIC_NUMERIC)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    feature_set: str
    categorical: Sequence[str]
    numeric: Sequence[str]
    deployment_scope: str
    description: str
    fold_features: str = "none"
    target_mode: str = "constrained"

    @property
    def eligible_for_full_network(self) -> bool:
        return self.deployment_scope == "full_network"


def model_specs(daily: bool = False) -> List[ModelSpec]:
    categorical = DAILY_CATEGORICAL if daily else NETWORK_CATEGORICAL
    low_cardinality = (
        DAILY_LOW_CARDINALITY_CATEGORICAL
        if daily
        else LOW_CARDINALITY_CATEGORICAL
    )
    traffic = [*CORE_NUMERIC, *TRAFFIC_MODEL_NUMERIC]
    graph = [*traffic, *GRAPH_NUMERIC]
    return [
        ModelSpec(
            "period_median",
            "period_baseline",
            "period_only",
            ["period"],
            [],
            "full_network",
            "Training-fold median by period.",
        ),
        ModelSpec(
            "facility_class_median",
            "class_baseline",
            "period_facility_class",
            ["period", "network_link_type", "network_ftype"],
            [],
            "full_network",
            "Training-fold median by period and facility class.",
        ),
        ModelSpec(
            "ridge_core",
            "ridge",
            "core_network",
            categorical,
            CORE_NUMERIC,
            "full_network",
            "Regularized linear model using static link and period-demand fields.",
        ),
        ModelSpec(
            "ridge_core_low_cardinality",
            "ridge",
            "core_network_low_cardinality",
            low_cardinality,
            CORE_NUMERIC,
            "full_network",
            "Ridge model excluding jurisdiction, component, and toll-group codes.",
        ),
        ModelSpec(
            "extra_trees_core",
            "extra_trees",
            "core_network",
            categorical,
            CORE_NUMERIC,
            "full_network",
            "Nonlinear ensemble using static link and period-demand fields.",
        ),
        ModelSpec(
            "hist_gradient_core",
            "hist_gradient",
            "core_network",
            categorical,
            CORE_NUMERIC,
            "full_network",
            "Boosted-tree model using static link and period-demand fields.",
        ),
        ModelSpec(
            "ridge_traffic_model",
            "ridge",
            "network_plus_assignment_traffic",
            categorical,
            traffic,
            "full_network",
            "Linear model adding assignment speed, V/C, VDF, travel time, and vehicle mix.",
        ),
        ModelSpec(
            "ridge_traffic_low_cardinality",
            "ridge",
            "network_assignment_low_cardinality",
            low_cardinality,
            traffic,
            "full_network",
            "Low-cardinality ridge model adding assignment traffic fields.",
        ),
        ModelSpec(
            "ridge_graph_low_cardinality",
            "ridge",
            "network_graph_low_cardinality",
            low_cardinality,
            graph,
            "full_network",
            "Low-cardinality ridge model adding graph and bottleneck fields.",
        ),
        ModelSpec(
            "extra_trees_traffic_model",
            "extra_trees",
            "network_plus_assignment_traffic",
            categorical,
            traffic,
            "full_network",
            "Tree ensemble adding assignment traffic fields.",
        ),
        ModelSpec(
            "hist_gradient_traffic_model",
            "hist_gradient",
            "network_plus_assignment_traffic",
            categorical,
            traffic,
            "full_network",
            "Boosted trees adding assignment traffic fields.",
        ),
        ModelSpec(
            "random_forest_graph",
            "random_forest",
            "network_traffic_graph",
            categorical,
            graph,
            "full_network",
            "Random forest adding upstream/downstream bottleneck structure.",
        ),
        ModelSpec(
            "extra_trees_graph",
            "extra_trees",
            "network_traffic_graph",
            categorical,
            graph,
            "full_network",
            "Extra Trees adding upstream/downstream bottleneck structure.",
        ),
        ModelSpec(
            "hist_gradient_graph",
            "hist_gradient",
            "network_traffic_graph",
            categorical,
            graph,
            "full_network",
            "Boosted trees adding upstream/downstream bottleneck structure.",
        ),
        ModelSpec(
            "extra_trees_graph_class_physics",
            "extra_trees",
            "network_graph_plus_class_physics",
            categorical,
            [*graph, *CLASS_PHYSICS_NUMERIC],
            "full_network",
            "Graph model with training-only class estimates of mu, vc, and vf.",
            fold_features="class_physics",
        ),
        ModelSpec(
            "hist_gradient_graph_class_physics",
            "hist_gradient",
            "network_graph_plus_class_physics",
            categorical,
            [*graph, *CLASS_PHYSICS_NUMERIC],
            "full_network",
            "Boosted graph model with training-only class physics priors.",
            fold_features="class_physics",
        ),
        ModelSpec(
            "extra_trees_graph_raw_boundaries",
            "extra_trees",
            "network_traffic_graph",
            categorical,
            graph,
            "diagnostic",
            "Independent raw t0/t2/t3 models used to test the target assumption.",
            target_mode="raw",
        ),
        ModelSpec(
            "spatial_anchor_baseline",
            "spatial_baseline",
            "fold_safe_corridor_anchors",
            ["period"],
            SPATIAL_NUMERIC,
            "mapped_corridor_only",
            "Fold-safe interpolation/nearest-anchor T2 with class duration priors.",
            fold_features="spatial",
        ),
        ModelSpec(
            "extra_trees_graph_spatial",
            "extra_trees",
            "network_graph_plus_fold_safe_spatial",
            [*categorical, "direction"],
            [*graph, *SPATIAL_NUMERIC],
            "mapped_corridor_only",
            "Graph model adding fold-safe same-corridor anchor features.",
            fold_features="spatial",
        ),
        ModelSpec(
            "hist_gradient_graph_spatial",
            "hist_gradient",
            "network_graph_plus_fold_safe_spatial",
            [*categorical, "direction"],
            [*graph, *SPATIAL_NUMERIC],
            "mapped_corridor_only",
            "Boosted graph model adding fold-safe same-corridor anchors.",
            fold_features="spatial",
        ),
        ModelSpec(
            "extra_trees_sensor_profile",
            "extra_trees",
            "network_graph_plus_sensor_profile",
            [*categorical, "direction"],
            [*graph, *PROFILE_NUMERIC, *FD_NUMERIC],
            "sensor_profile_only",
            "Conditional model using the full average-weekday speed/flow profile.",
        ),
        ModelSpec(
            "extra_trees_episode_diagnostic_upper_bound",
            "extra_trees",
            "episode_post_detection_upper_bound",
            [*categorical, "direction"],
            [
                *graph,
                *PROFILE_NUMERIC,
                *FD_NUMERIC,
                *EPISODE_DIAGNOSTIC_NUMERIC,
            ],
            "diagnostic",
            "Non-deployable model using post-detection episode variables.",
        ),
    ]


class HierarchicalMedianBaseline:
    def __init__(self, group_columns: Sequence[str]):
        self.group_columns = list(group_columns)
        self.group_values: Dict[str, pd.Series] = {}
        self.period_values: Dict[str, pd.Series] = {}
        self.global_values: Dict[str, float] = {}

    def fit(
        self, frame: pd.DataFrame, target_columns: Sequence[str]
    ):
        self.target_columns = list(target_columns)
        for target in self.target_columns:
            self.group_values[target] = frame.groupby(
                self.group_columns, dropna=False
            )[target].median()
            self.period_values[target] = frame.groupby(
                "period", dropna=False
            )[target].median()
            self.global_values[target] = float(frame[target].median())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = []
        for target in self.target_columns:
            group_series = self.group_values[target]
            if len(self.group_columns) == 1:
                values = frame[self.group_columns[0]].map(group_series)
            else:
                keys = pd.MultiIndex.from_frame(
                    frame[self.group_columns]
                )
                values = pd.Series(
                    group_series.reindex(keys).to_numpy(),
                    index=frame.index,
                )
            fallback = frame["period"].map(self.period_values[target])
            values = values.fillna(fallback).fillna(
                self.global_values[target]
            )
            predictions.append(values.to_numpy(dtype=float))
        return np.column_stack(predictions)


def _preprocessor(
    categorical: Sequence[str],
    numeric: Sequence[str],
    scale: bool,
) -> ColumnTransformer:
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(numeric)),
            (
                "categorical",
                categorical_pipeline,
                list(categorical),
            ),
        ],
        remainder="drop",
    )


def build_estimator(
    spec: ModelSpec,
    *,
    random_seed: int,
    n_jobs: int,
    forest_estimators: int,
) -> Pipeline:
    scale = spec.kind == "ridge"
    features = _preprocessor(spec.categorical, spec.numeric, scale)
    if spec.kind == "ridge":
        estimator = Ridge(alpha=10.0)
    elif spec.kind == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=forest_estimators,
            min_samples_leaf=4,
            max_features=0.8,
            n_jobs=n_jobs,
            random_state=random_seed,
        )
    elif spec.kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=forest_estimators,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=n_jobs,
            random_state=random_seed,
        )
    elif spec.kind == "hist_gradient":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unsupported estimator kind: {spec.kind}")
    return Pipeline([("features", features), ("model", estimator)])


def assert_operational_features_are_leakage_free(
    specs: Sequence[ModelSpec],
) -> None:
    for spec in specs:
        if not spec.eligible_for_full_network:
            continue
        overlap = set(spec.numeric).intersection(
            LEAKAGE_OR_POST_DETECTION_FIELDS
        )
        if overlap:
            raise ValueError(
                f"{spec.name} uses post-detection fields: "
                f"{sorted(overlap)}"
            )


def get_model_spec(name: str, daily: bool = False) -> ModelSpec:
    return next(spec for spec in model_specs(daily=daily) if spec.name == name)
