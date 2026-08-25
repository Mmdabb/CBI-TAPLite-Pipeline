from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .fold_features import prepare_fold_features
from .models import HierarchicalMedianBaseline, ModelSpec, build_estimator
from .targets import TARGET_COLUMNS, reconstruct_boundaries


def make_group_splits(
    frame: pd.DataFrame,
    group_column: str,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = frame[group_column].astype(str).fillna("UNKNOWN")
    actual_splits = min(n_splits, groups.nunique())
    if actual_splits < 2:
        raise ValueError(
            f"Need at least two unique groups in {group_column}"
        )
    splitter = GroupKFold(n_splits=actual_splits)
    return list(splitter.split(frame, groups=groups))


def make_temporal_split(
    frame: pd.DataFrame,
    holdout_days: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    unique_dates = sorted(dates.dropna().unique())
    if len(unique_dates) <= holdout_days:
        raise ValueError("Not enough dates for the requested temporal holdout.")
    test_dates = set(unique_dates[-holdout_days:])
    test_mask = dates.isin(test_dates).to_numpy()
    return [(np.flatnonzero(~test_mask), np.flatnonzero(test_mask))]


def _fit_kwargs(frame: pd.DataFrame) -> Dict[str, object]:
    if "sample_weight" not in frame:
        return {}
    weights = pd.to_numeric(
        frame["sample_weight"], errors="coerce"
    ).fillna(1.0)
    return {"model__sample_weight": weights.to_numpy(dtype=float)}


def _predict_transformed(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_seed: int,
    n_jobs: int,
    forest_estimators: int,
) -> np.ndarray:
    if spec.kind in {"period_baseline", "class_baseline"}:
        groups = (
            ["period"]
            if spec.kind == "period_baseline"
            else list(spec.categorical)
        )
        baseline = HierarchicalMedianBaseline(groups).fit(
            train, TARGET_COLUMNS
        )
        return baseline.predict(test)
    if spec.kind == "spatial_baseline":
        baseline = HierarchicalMedianBaseline(
            ["period", "network_link_type", "network_ftype"]
        ).fit(train, TARGET_COLUMNS)
        prediction = baseline.predict(test)
        replacements = [
            "spatial_t2_relative_min",
            "spatial_log_span_prior",
            "spatial_logit_fraction_prior",
        ]
        for index, column in enumerate(replacements):
            if column in test:
                values = pd.to_numeric(
                    test[column], errors="coerce"
                ).to_numpy(dtype=float)
                valid = np.isfinite(values)
                prediction[valid, index] = values[valid]
        return prediction
    estimator = build_estimator(
        spec,
        random_seed=random_seed,
        n_jobs=n_jobs,
        forest_estimators=forest_estimators,
    )
    predictions = []
    fit_kwargs = _fit_kwargs(train)
    for target in TARGET_COLUMNS:
        fitted = clone(estimator).fit(
            train, train[target], **fit_kwargs
        )
        predictions.append(fitted.predict(test))
    return np.column_stack(predictions)


def _predict_raw_boundaries(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_seed: int,
    n_jobs: int,
    forest_estimators: int,
) -> pd.DataFrame:
    estimator = build_estimator(
        spec,
        random_seed=random_seed,
        n_jobs=n_jobs,
        forest_estimators=forest_estimators,
    )
    fit_kwargs = _fit_kwargs(train)
    predictions = {}
    for target in ["t0_hour", "t2_hour", "t3_hour"]:
        fitted = clone(estimator).fit(
            train, train[target], **fit_kwargs
        )
        predictions[f"pred_{target}"] = fitted.predict(test)
    result = pd.DataFrame(predictions, index=test.index).rename(
        columns={
            "pred_t0_hour": "pred_t0_hour",
            "pred_t2_hour": "pred_t2_hour",
            "pred_t3_hour": "pred_t3_hour",
        }
    )
    result["pred_span_min"] = (
        result["pred_t3_hour"] - result["pred_t0_hour"]
    ) * 60.0
    result["pred_t2_fraction"] = (
        (result["pred_t2_hour"] - result["pred_t0_hour"])
        / (result["pred_t3_hour"] - result["pred_t0_hour"])
    )
    return result


def _identity_columns(frame: pd.DataFrame) -> List[str]:
    candidates = [
        "tmc_period_id",
        "tmc_code",
        "period",
        "corridor",
        "direction",
        "network_link_id",
        "network_link_type",
        "network_ftype",
        "date",
        "observed_day_count",
        "t0_hour",
        "t2_hour",
        "t3_hour",
    ]
    return [column for column in candidates if column in frame]


def cross_validate_models(
    frame: pd.DataFrame,
    specs: Sequence[ModelSpec],
    *,
    validation_name: str,
    data_model: str,
    n_splits: int,
    random_seed: int,
    n_jobs: int,
    forest_estimators: int,
    group_column: Optional[str] = None,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    outputs: List[pd.DataFrame] = []
    if splits is None:
        if group_column is None:
            raise ValueError("group_column or explicit splits is required.")
        splits = make_group_splits(frame, group_column, n_splits)
    identity_columns = _identity_columns(frame)

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        base_train = frame.iloc[train_index].copy()
        base_test = frame.iloc[test_index].copy()
        feature_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for spec in specs:
            if spec.fold_features not in feature_cache:
                feature_cache[spec.fold_features] = prepare_fold_features(
                    base_train, base_test, spec.fold_features
                )
            train, test = feature_cache[spec.fold_features]
            if spec.target_mode == "raw":
                boundaries = _predict_raw_boundaries(
                    spec,
                    train,
                    test,
                    random_seed=random_seed + fold,
                    n_jobs=n_jobs,
                    forest_estimators=forest_estimators,
                )
            else:
                transformed = _predict_transformed(
                    spec,
                    train,
                    test,
                    random_seed=random_seed + fold,
                    n_jobs=n_jobs,
                    forest_estimators=forest_estimators,
                )
                boundaries = reconstruct_boundaries(
                    test["period"],
                    transformed[:, 0],
                    transformed[:, 1],
                    transformed[:, 2],
                )
            result = test[identity_columns].copy().rename(
                columns={
                    "t0_hour": "observed_t0_hour",
                    "t2_hour": "observed_t2_hour",
                    "t3_hour": "observed_t3_hour",
                }
            )
            result = pd.concat([result, boundaries], axis=1)
            result["model"] = spec.name
            result["feature_set"] = spec.feature_set
            result["deployment_scope"] = spec.deployment_scope
            result["eligible_for_full_network"] = (
                spec.eligible_for_full_network
            )
            result["target_mode"] = spec.target_mode
            result["data_model"] = data_model
            result["validation"] = validation_name
            result["group_column"] = group_column or "explicit_split"
            result["fold"] = fold
            outputs.append(result)
    combined = pd.concat(outputs, ignore_index=True)
    for boundary in ["t0", "t2", "t3"]:
        combined[f"error_{boundary}_min"] = (
            combined[f"pred_{boundary}_hour"]
            - combined[f"observed_{boundary}_hour"]
        ) * 60.0
        combined[f"abs_error_{boundary}_min"] = combined[
            f"error_{boundary}_min"
        ].abs()
    combined["mean_abs_boundary_error_min"] = combined[
        [
            "abs_error_t0_min",
            "abs_error_t2_min",
            "abs_error_t3_min",
        ]
    ].mean(axis=1)
    combined["physical_violation"] = (
        (combined["pred_t0_hour"] > combined["pred_t2_hour"])
        | (combined["pred_t2_hour"] > combined["pred_t3_hour"])
    )
    return combined


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_columns = [
        "data_model",
        "validation",
        "model",
        "feature_set",
        "deployment_scope",
        "eligible_for_full_network",
        "target_mode",
    ]
    groups = predictions.groupby(group_columns, dropna=False)
    for keys, frame in groups:
        all_abs = frame[
            [
                "abs_error_t0_min",
                "abs_error_t2_min",
                "abs_error_t3_min",
            ]
        ].to_numpy()
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "row_count": len(frame),
                "mae_t0_min": frame["abs_error_t0_min"].mean(),
                "mae_t2_min": frame["abs_error_t2_min"].mean(),
                "mae_t3_min": frame["abs_error_t3_min"].mean(),
                "mean_boundary_mae_min": all_abs.mean(),
                "median_boundary_abs_error_min": np.median(all_abs),
                "p90_boundary_abs_error_min": np.quantile(
                    all_abs, 0.90
                ),
                "t2_bias_min": frame["error_t2_min"].mean(),
                "share_t2_within_15_min": (
                    frame["abs_error_t2_min"] <= 15.0
                ).mean(),
                "share_t2_within_30_min": (
                    frame["abs_error_t2_min"] <= 30.0
                ).mean(),
                "share_all_boundaries_within_30_min": (
                    frame[
                        [
                            "abs_error_t0_min",
                            "abs_error_t2_min",
                            "abs_error_t3_min",
                        ]
                    ].max(axis=1)
                    <= 30.0
                ).mean(),
                "physical_violation_rate": frame[
                    "physical_violation"
                ].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "data_model",
            "validation",
            "mean_boundary_mae_min",
            "mae_t2_min",
        ]
    ).reset_index(drop=True)


def summarize_by_period(predictions: pd.DataFrame) -> pd.DataFrame:
    return _summarize_stratum(predictions, "period")


def summarize_by_link_type(predictions: pd.DataFrame) -> pd.DataFrame:
    return _summarize_stratum(predictions, "network_link_type")


def _summarize_stratum(
    predictions: pd.DataFrame,
    stratum: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups = predictions.groupby(
        ["data_model", "validation", "model", stratum], dropna=False
    )
    for keys, frame in groups:
        data_model, validation, model, value = keys
        rows.append(
            {
                "data_model": data_model,
                "validation": validation,
                "model": model,
                stratum: value,
                "row_count": len(frame),
                "mae_t0_min": frame["abs_error_t0_min"].mean(),
                "mae_t2_min": frame["abs_error_t2_min"].mean(),
                "mae_t3_min": frame["abs_error_t3_min"].mean(),
                "mean_boundary_mae_min": frame[
                    [
                        "abs_error_t0_min",
                        "abs_error_t2_min",
                        "abs_error_t3_min",
                    ]
                ].to_numpy().mean(),
                "p90_boundary_abs_error_min": np.quantile(
                    frame[
                        [
                            "abs_error_t0_min",
                            "abs_error_t2_min",
                            "abs_error_t3_min",
                        ]
                    ].to_numpy(),
                    0.90,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["data_model", "validation", "model", stratum]
    ).reset_index(drop=True)
