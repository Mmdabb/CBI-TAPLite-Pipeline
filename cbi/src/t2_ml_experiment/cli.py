from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from .config import load_config
from .data import build_experiment_tables
from .fold_features import prepare_fold_features
from .models import (
    ModelSpec,
    assert_operational_features_are_leakage_free,
    build_estimator,
    get_model_spec,
    model_specs,
)
from .reporting import create_figures, write_report
from .targets import (
    TARGET_COLUMNS,
    reconstruct_boundaries,
    reconstruct_with_fixed_t2,
)
from .validation import (
    cross_validate_models,
    make_temporal_split,
    summarize_by_link_type,
    summarize_by_period,
    summarize_metrics,
)


BASELINE_NAMES = {"period_median", "facility_class_median"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_manifest(specs: Sequence[ModelSpec]) -> Dict[str, object]:
    return {
        "models": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "feature_set": spec.feature_set,
                "deployment_scope": spec.deployment_scope,
                "eligible_for_full_network": spec.eligible_for_full_network,
                "categorical_features": list(spec.categorical),
                "numeric_features": list(spec.numeric),
                "fold_features": spec.fold_features,
                "target_mode": spec.target_mode,
                "description": spec.description,
            }
            for spec in specs
        ],
        "deployment_rule": (
            "Only deployment_scope=full_network models may be selected for the "
            "experimental network-wide file."
        ),
        "target_parameterization": {
            "t2": "minutes from period start",
            "duration": "log(t3 - t0 in minutes)",
            "t2_fraction": "logit((t2 - t0) / (t3 - t0))",
        },
    }


def _feature_coverage(
    aggregate: pd.DataFrame,
    daily: pd.DataFrame,
    network: pd.DataFrame,
    specs: Sequence[ModelSpec],
) -> pd.DataFrame:
    features = sorted(
        {
            column
            for spec in specs
            for column in [*spec.categorical, *spec.numeric]
            if not column.startswith(("expected_", "spatial_"))
        }
    )
    rows = []
    for dataset_name, frame in [
        ("aggregate_tmc_period", aggregate),
        ("daily_episode", daily),
        ("full_network_link_period", network),
    ]:
        for feature in features:
            available = feature in frame
            non_null = int(frame[feature].notna().sum()) if available else 0
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature": feature,
                    "available": available,
                    "non_null_rows": non_null,
                    "total_rows": len(frame),
                    "coverage_fraction": non_null / len(frame) if len(frame) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _run_validation_pair(
    frame: pd.DataFrame,
    specs: Sequence[ModelSpec],
    *,
    data_model: str,
    config,
) -> pd.DataFrame:
    outputs = []
    for validation_name, group_column in [
        ("tmc_held_out", "tmc_code"),
        ("corridor_held_out", "corridor"),
    ]:
        outputs.append(
            cross_validate_models(
                frame,
                specs,
                validation_name=validation_name,
                data_model=data_model,
                group_column=group_column,
                n_splits=config.cv_folds,
                random_seed=config.random_seed,
                n_jobs=config.n_jobs,
                forest_estimators=config.forest_estimators,
            )
        )
    return pd.concat(outputs, ignore_index=True)


def _best_model_name(
    leaderboard: pd.DataFrame,
    *,
    scope: str,
    exclude_baselines: bool,
) -> str:
    candidates = leaderboard[
        (leaderboard["data_model"] == "aggregate_all_days")
        & (leaderboard["validation"] == "corridor_held_out")
        & (leaderboard["deployment_scope"] == scope)
        & (leaderboard["target_mode"] == "constrained")
    ].copy()
    if exclude_baselines:
        candidates = candidates[~candidates["model"].isin(BASELINE_NAMES)]
    if candidates.empty:
        raise ValueError(f"No eligible model found for scope={scope}")
    return str(candidates.iloc[0]["model"])


def _fit_final_models(
    training: pd.DataFrame,
    network: pd.DataFrame,
    spec: ModelSpec,
    config,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    prepared_training, prepared_network = prepare_fold_features(
        training, network, spec.fold_features
    )
    feature_columns = list(dict.fromkeys([*spec.categorical, *spec.numeric]))
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    transformed_predictions = []
    fitted_models: Dict[str, object] = {}
    importance_rows = []
    def source_feature(transformed_name: str) -> str:
        name = transformed_name.split("__", 1)[-1]
        for column in sorted(spec.categorical, key=len, reverse=True):
            if name == column or name.startswith(f"{column}_"):
                return column
        return name

    for target in TARGET_COLUMNS:
        estimator = build_estimator(
            spec,
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
            forest_estimators=config.forest_estimators,
        )
        estimator.fit(
            prepared_training[feature_columns],
            prepared_training[target],
        )
        transformed_predictions.append(
            estimator.predict(prepared_network[feature_columns])
        )
        fitted_models[target] = estimator
        joblib.dump(estimator, model_dir / f"{spec.name}__{target}.joblib")
        model = estimator.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            names = estimator.named_steps["features"].get_feature_names_out()
            for name, importance in zip(names, model.feature_importances_):
                importance_rows.append(
                    {
                        "target": target,
                        "transformed_feature": name,
                        "source_feature": source_feature(name),
                        "importance": float(importance),
                        "signed_value": np.nan,
                        "importance_type": "tree_impurity",
                    }
                )
        elif hasattr(model, "coef_"):
            names = estimator.named_steps["features"].get_feature_names_out()
            coefficients = np.ravel(model.coef_).astype(float)
            scale = np.abs(coefficients).sum()
            normalized = (
                np.abs(coefficients) / scale
                if scale > 0
                else np.zeros_like(coefficients)
            )
            for name, coefficient, importance in zip(
                names, coefficients, normalized
            ):
                importance_rows.append(
                    {
                        "target": target,
                        "transformed_feature": name,
                        "source_feature": source_feature(name),
                        "importance": float(importance),
                        "signed_value": float(coefficient),
                        "importance_type": "normalized_absolute_coefficient",
                    }
                )
    prediction_matrix = np.column_stack(transformed_predictions)
    boundaries = reconstruct_boundaries(
        prepared_network["period"],
        prediction_matrix[:, 0],
        prediction_matrix[:, 1],
        prediction_matrix[:, 2],
    )
    transformed = pd.DataFrame(
        prediction_matrix,
        columns=[f"ml_pred_{column}" for column in TARGET_COLUMNS],
        index=network.index,
    )
    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        importance = importance.sort_values(
            ["target", "importance"], ascending=[True, False]
        ).reset_index(drop=True)
    return pd.concat([transformed, boundaries], axis=1), fitted_models, importance


def _experimental_network_file(
    network: pd.DataFrame,
    ml: pd.DataFrame,
    best_corridor_predictions: pd.DataFrame,
) -> pd.DataFrame:
    identity = [
        "period",
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_link_type",
        "network_ftype",
        "existing_t2_source",
        "existing_t2_detail",
        "existing_t0_hour",
        "existing_t2_hour",
        "existing_t3_hour",
    ]
    result = network[[column for column in identity if column in network]].copy()
    result["ml_pred_t0_hour"] = ml["pred_t0_hour"].to_numpy()
    result["ml_pred_t2_hour"] = ml["pred_t2_hour"].to_numpy()
    result["ml_pred_t3_hour"] = ml["pred_t3_hour"].to_numpy()
    source = result.get(
        "existing_t2_source", pd.Series("", index=result.index)
    ).fillna("").astype(str).str.lower()
    direct = source.str.contains("direct") & result["existing_t2_hour"].notna()
    hierarchy_t2 = result["existing_t2_hour"].notna() & ~direct
    fixed = reconstruct_with_fixed_t2(
        result["existing_t2_hour"],
        ml["ml_pred_target_log_span_min"].to_numpy(),
        ml["ml_pred_target_logit_t2_fraction"].to_numpy(),
    )
    result["experimental_t0_hour"] = result["ml_pred_t0_hour"]
    result["experimental_t2_hour"] = result["ml_pred_t2_hour"]
    result["experimental_t3_hour"] = result["ml_pred_t3_hour"]
    result.loc[hierarchy_t2, "experimental_t0_hour"] = fixed.loc[
        hierarchy_t2, "pred_t0_hour"
    ]
    result.loc[hierarchy_t2, "experimental_t2_hour"] = result.loc[
        hierarchy_t2, "existing_t2_hour"
    ]
    result.loc[hierarchy_t2, "experimental_t3_hour"] = fixed.loc[
        hierarchy_t2, "pred_t3_hour"
    ]
    direct_all = (
        direct
        & result["existing_t0_hour"].notna()
        & result["existing_t3_hour"].notna()
    )
    for boundary in ["t0", "t2", "t3"]:
        result.loc[direct_all, f"experimental_{boundary}_hour"] = result.loc[
            direct_all, f"existing_{boundary}_hour"
        ]
    result["experimental_boundary_source"] = np.select(
        [direct_all, hierarchy_t2],
        ["direct_observed", "existing_t2_plus_ml_shape"],
        default="ml_full_boundaries",
    )
    calibration = (
        best_corridor_predictions.groupby("period")[
            ["abs_error_t0_min", "abs_error_t2_min", "abs_error_t3_min"]
        ]
        .quantile(0.90)
        .rename(
            columns={
                "abs_error_t0_min": "estimated_p90_abs_error_t0_min",
                "abs_error_t2_min": "estimated_p90_abs_error_t2_min",
                "abs_error_t3_min": "estimated_p90_abs_error_t3_min",
            }
        )
    )
    result = result.merge(
        calibration, left_on="period", right_index=True, how="left"
    )
    ordered = (
        result["experimental_t0_hour"].le(result["experimental_t2_hour"])
        & result["experimental_t2_hour"].le(result["experimental_t3_hour"])
    )
    if not ordered.all():
        raise ValueError(
            f"Experimental network output has {(~ordered).sum()} unordered rows."
        )
    return result


def run(config_path: Path, output_run_name: Optional[str] = None) -> Path:
    config = load_config(config_path)
    specs = model_specs()
    if config.model_names:
        requested = set(config.model_names)
        available = {spec.name for spec in specs}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unknown configured model_names: {unknown}")
        retained = requested | BASELINE_NAMES
        specs = [spec for spec in specs if spec.name in retained]
    assert_operational_features_are_leakage_free(specs)
    aggregate, daily, network, input_files = build_experiment_tables(
        config.cbi_run_dir,
        config.boundary_mapping_run_dir,
        config.spatial_run_dir,
    )
    if output_run_name is None:
        output_run_name = f"t2-ml-experiment-{datetime.now():%Y-%m-%d-%H-%M}"
    output_dir = config.output_root / output_run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    data_dir = output_dir / "data"
    metrics_dir = output_dir / "metrics"
    data_dir.mkdir()
    metrics_dir.mkdir()
    aggregate.to_csv(data_dir / "training_tmc_period.csv", index=False)
    daily.to_csv(data_dir / "training_daily_episode.csv", index=False)
    _feature_coverage(aggregate, daily, network, specs).to_csv(
        data_dir / "feature_coverage.csv", index=False
    )
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(_feature_manifest(specs), indent=2), encoding="utf-8"
    )

    primary_predictions = _run_validation_pair(
        aggregate, specs, data_model="aggregate_all_days", config=config
    )
    primary_leaderboard = summarize_metrics(primary_predictions)
    best_full_name = _best_model_name(
        primary_leaderboard,
        scope="full_network",
        exclude_baselines=True,
    )
    conditional = primary_leaderboard[
        primary_leaderboard["deployment_scope"].isin(
            ["mapped_corridor_only", "sensor_profile_only"]
        )
    ]
    best_conditional_name = (
        str(
            conditional[
                conditional["validation"].eq("corridor_held_out")
            ].iloc[0]["model"]
        )
        if not conditional.empty
        else None
    )

    selected_names = [
        *sorted(BASELINE_NAMES),
        best_full_name,
        *([best_conditional_name] if best_conditional_name else []),
    ]
    selected_names = list(dict.fromkeys(selected_names))
    reliable = aggregate[
        (aggregate["observed_day_count"] >= config.reliable_minimum_days)
        & (
            aggregate["observed_t2_std_hour"]
            <= config.reliable_maximum_t2_std_hours
        )
    ].copy()
    selected_aggregate_specs = [
        get_model_spec(name) for name in selected_names
    ]
    assumption_predictions = []
    if reliable["tmc_code"].nunique() >= 2 and reliable["corridor"].nunique() >= 2:
        assumption_predictions.append(
            _run_validation_pair(
                reliable,
                selected_aggregate_specs,
                data_model="aggregate_reliable_subset",
                config=config,
            )
        )
    selected_daily_specs = [
        get_model_spec(name, daily=True) for name in selected_names
    ]
    assumption_predictions.append(
        _run_validation_pair(
            daily,
            selected_daily_specs,
            data_model="daily_weighted",
            config=config,
        )
    )
    assumption_predictions.append(
        cross_validate_models(
            daily,
            selected_daily_specs,
            validation_name="future_days_held_out",
            data_model="daily_weighted",
            splits=make_temporal_split(daily, config.temporal_holdout_days),
            n_splits=1,
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
            forest_estimators=config.forest_estimators,
        )
    )
    all_predictions = pd.concat(
        [primary_predictions, *assumption_predictions], ignore_index=True
    )
    leaderboard = summarize_metrics(all_predictions)
    by_period = summarize_by_period(all_predictions)
    by_link_type = summarize_by_link_type(all_predictions)
    all_predictions.to_csv(
        metrics_dir / "out_of_fold_predictions.csv", index=False
    )
    leaderboard.to_csv(metrics_dir / "leaderboard.csv", index=False)
    by_period.to_csv(metrics_dir / "metrics_by_period.csv", index=False)
    by_link_type.to_csv(metrics_dir / "metrics_by_link_type.csv", index=False)

    best_spec = get_model_spec(best_full_name)
    ml_network, _, importance = _fit_final_models(
        aggregate, network, best_spec, config, output_dir
    )
    if not importance.empty:
        importance.to_csv(metrics_dir / "feature_importance.csv", index=False)
    best_corridor_predictions = primary_predictions[
        (primary_predictions["validation"] == "corridor_held_out")
        & (primary_predictions["model"] == best_full_name)
    ]
    experimental_network = _experimental_network_file(
        network, ml_network, best_corridor_predictions
    )
    experimental_network.to_csv(
        output_dir / "experimental_network_boundaries.csv", index=False
    )
    source_summary = (
        experimental_network.groupby(
            ["period", "experimental_boundary_source"], dropna=False
        )
        .size()
        .rename("link_period_count")
        .reset_index()
    )
    source_summary.to_csv(
        metrics_dir / "network_prediction_source_summary.csv", index=False
    )

    create_figures(
        leaderboard,
        all_predictions,
        output_dir,
        best_full_name,
        importance,
    )
    write_report(
        output_dir,
        aggregate,
        daily,
        leaderboard,
        all_predictions,
        source_summary,
        best_full_name,
    )

    unique_inputs = list(dict.fromkeys(Path(path) for path in input_files))
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(config_path.resolve()),
        "cbi_run_dir": str(config.cbi_run_dir),
        "boundary_mapping_run_dir": str(config.boundary_mapping_run_dir),
        "spatial_run_dir": str(config.spatial_run_dir),
        "output_dir": str(output_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "random_seed": config.random_seed,
        "cv_folds": config.cv_folds,
        "n_jobs": config.n_jobs,
        "worker_fraction": config.worker_fraction,
        "max_workers": config.max_workers,
        "forest_estimators": config.forest_estimators,
        "aggregate_training_rows": len(aggregate),
        "daily_training_rows": len(daily),
        "reliable_training_rows": len(reliable),
        "network_link_period_rows": len(network),
        "unique_tmcs": int(aggregate["tmc_code"].nunique()),
        "unique_corridors": int(aggregate["corridor"].nunique()),
        "selected_full_network_model": best_full_name,
        "selected_conditional_model": best_conditional_name,
        "input_files": [
            {
                "path": str(path),
                "sha256": _hash_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in unique_inputs
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated NVTA t0/t2/t3 ML research tournament."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-run-name")
    args = parser.parse_args(argv)
    output_dir = run(args.config, args.output_run_name)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
