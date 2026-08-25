from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExpansionConfig
from .detector import detect_profile_t2, interpolate_normalized_profiles
from .expansion import (
    add_corridor_positions,
    build_anchors,
    fit_propagation_slope,
    load_snapshot,
    make_profile_lookup,
    profile_array,
    sha256,
)


def _period_contains(period: Tuple[int, int], value: float) -> bool:
    minute = float(value) * 60.0
    return float(period[0]) <= minute < float(period[1])


def _brackets(
    train: pd.DataFrame, position: float
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    ordered = train.sort_values("anchor_position_mi", kind="mergesort")
    left = ordered[ordered["anchor_position_mi"] <= position]
    right = ordered[ordered["anchor_position_mi"] >= position]
    return (
        left.iloc[-1] if not left.empty else None,
        right.iloc[0] if not right.empty else None,
    )


def _prediction_row(
    method: str,
    observed: pd.Series,
    predicted: float,
    fold: int,
    holdout_type: str,
    gap_mi: float,
    left_tmc: str,
    right_tmc: str,
) -> Dict[str, object]:
    error = float(predicted) - float(observed["anchor_t2_hour"])
    return {
        "method": method,
        "period": str(observed["period"]),
        "road": str(observed["road"]),
        "direction": str(observed["direction"]),
        "tmc": str(observed["tmc"]),
        "fold": int(fold),
        "holdout_type": holdout_type,
        "position_mi": float(observed["anchor_position_mi"]),
        "observed_t2_hour": float(observed["anchor_t2_hour"]),
        "predicted_t2_hour": float(predicted),
        "error_hours": error,
        "absolute_error_minutes": abs(error) * 60.0,
        "gap_mi": gap_mi,
        "left_anchor_tmc": left_tmc,
        "right_anchor_tmc": right_tmc,
        "source_method": str(observed.get("anchor_source_method", "")),
        "map_confidence": float(observed.get("confidence", np.nan)),
    }


def validate_anchor_group(
    anchors: pd.DataFrame,
    profile_lookup: Dict[str, pd.Series],
    config: ExpansionConfig,
) -> Tuple[List[Dict[str, object]], int]:
    ordered = anchors.sort_values(
        ["anchor_position_mi", "tmc"], kind="mergesort"
    ).reset_index(drop=True)
    if len(ordered) < 3:
        return [], 0
    fold_count = min(int(config.validation_folds), len(ordered))
    blocks = np.array_split(np.arange(len(ordered)), fold_count)
    rows: List[Dict[str, object]] = []
    evaluated = 0
    period_name = str(ordered["period"].iloc[0])
    period = config.periods[period_name]
    axis = np.arange(period[0], period[1], 15, dtype=float)

    for fold, test_indices in enumerate(blocks, start=1):
        train = ordered.drop(index=test_indices).copy()
        test = ordered.iloc[test_indices]
        if train.empty:
            continue
        slope, _ = fit_propagation_slope(train, config)
        corridor_median = float(train["anchor_t2_hour"].median())
        for _, observed in test.iterrows():
            evaluated += 1
            position = float(observed["anchor_position_mi"])
            left, right = _brackets(train, position)
            interior = left is not None and right is not None
            holdout_type = "interior_block" if interior else "edge_block"
            rows.append(
                _prediction_row(
                    "corridor_median_copy",
                    observed,
                    corridor_median,
                    fold,
                    holdout_type,
                    np.nan,
                    "",
                    "",
                )
            )
            nearest = min(
                list(train.itertuples(index=False)),
                key=lambda row: abs(float(row.anchor_position_mi) - position),
            )
            nearest_position = float(nearest.anchor_position_mi)
            nearest_t2 = float(nearest.anchor_t2_hour)
            nearest_distance = abs(nearest_position - position)
            rows.append(
                _prediction_row(
                    "nearest_tmc",
                    observed,
                    nearest_t2,
                    fold,
                    holdout_type,
                    nearest_distance,
                    str(nearest.tmc) if nearest_position <= position else "",
                    str(nearest.tmc) if nearest_position >= position else "",
                )
            )
            shifted = nearest_t2 + slope * (position - nearest_position)
            if (
                nearest_distance <= config.maximum_extrapolation_miles
                and _period_contains(period, shifted)
            ):
                rows.append(
                    _prediction_row(
                        "one_sided_propagation_shift",
                        observed,
                        shifted,
                        fold,
                        holdout_type,
                        nearest_distance,
                        str(nearest.tmc)
                        if nearest_position <= position
                        else "",
                        str(nearest.tmc)
                        if nearest_position >= position
                        else "",
                    )
                )
            if (
                left is None
                or right is None
                or str(left["tmc"]) == str(right["tmc"])
            ):
                continue
            left_position = float(left["anchor_position_mi"])
            right_position = float(right["anchor_position_mi"])
            gap = right_position - left_position
            if (
                gap <= 0.0
                or gap > float(config.maximum_interpolation_gap_miles)
            ):
                continue
            weight_right = (position - left_position) / gap
            linear = (
                (1.0 - weight_right) * float(left["anchor_t2_hour"])
                + weight_right * float(right["anchor_t2_hour"])
            )
            if _period_contains(period, linear):
                rows.append(
                    _prediction_row(
                        "linear_t2_interpolation",
                        observed,
                        linear,
                        fold,
                        holdout_type,
                        gap,
                        str(left["tmc"]),
                        str(right["tmc"]),
                    )
                )
            left_profile = profile_array(
                profile_lookup, str(left["tmc"]), axis
            )
            right_profile = profile_array(
                profile_lookup, str(right["tmc"]), axis
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
                rows.append(
                    _prediction_row(
                        "normalized_profile_interpolation",
                        observed,
                        float(detected["t2_hour"]),
                        fold,
                        holdout_type,
                        gap,
                        str(left["tmc"]),
                        str(right["tmc"]),
                    )
                )
    return rows, evaluated


def summarize_validation(
    predictions: pd.DataFrame, benchmark_count: int
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method, group in predictions.groupby("method", sort=True):
        errors = pd.to_numeric(
            group["absolute_error_minutes"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "method": method,
                "benchmark_tmc_periods": int(benchmark_count),
                "predictions": int(len(errors)),
                "prediction_coverage_pct": (
                    100.0 * len(errors) / benchmark_count
                    if benchmark_count
                    else 0.0
                ),
                "mae_minutes": float(errors.mean()) if len(errors) else np.nan,
                "median_absolute_error_minutes": (
                    float(errors.median()) if len(errors) else np.nan
                ),
                "p90_absolute_error_minutes": (
                    float(errors.quantile(0.90)) if len(errors) else np.nan
                ),
                "within_15_minutes_pct": (
                    100.0 * float((errors <= 15.0).mean())
                    if len(errors)
                    else np.nan
                ),
                "within_30_minutes_pct": (
                    100.0 * float((errors <= 30.0).mean())
                    if len(errors)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mae_minutes", "prediction_coverage_pct"],
        ascending=[True, False],
        kind="mergesort",
    )


def summarize_validation_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    target_columns = ["period", "road", "direction", "tmc", "fold"]
    specifications = [
        ("period", ["period"]),
        ("holdout_type", ["holdout_type"]),
        (
            "corridor_direction_period",
            ["period", "road", "direction"],
        ),
    ]
    rows: List[Dict[str, object]] = []
    for stratum_type, columns in specifications:
        benchmark_columns = list(dict.fromkeys(target_columns + columns))
        benchmarks = (
            predictions[benchmark_columns]
            .drop_duplicates(target_columns)
            .groupby(columns, dropna=False)
            .size()
            .rename("benchmark_tmc_periods")
            .reset_index()
        )
        grouped = predictions.groupby(
            columns + ["method"], sort=True, dropna=False
        )
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            labels = dict(zip(columns + ["method"], keys))
            benchmark_match = benchmarks.copy()
            for column in columns:
                benchmark_match = benchmark_match[
                    benchmark_match[column].eq(labels[column])
                ]
            benchmark = (
                int(benchmark_match["benchmark_tmc_periods"].iloc[0])
                if not benchmark_match.empty
                else 0
            )
            errors = pd.to_numeric(
                group["absolute_error_minutes"], errors="coerce"
            ).dropna()
            row: Dict[str, object] = {
                "stratum_type": stratum_type,
                "period": "",
                "road": "",
                "direction": "",
                "holdout_type": "",
                "method": labels["method"],
                "benchmark_tmc_periods": benchmark,
                "predictions": int(len(errors)),
                "prediction_coverage_pct": (
                    100.0 * len(errors) / benchmark if benchmark else 0.0
                ),
                "mae_minutes": (
                    float(errors.mean()) if len(errors) else np.nan
                ),
                "median_absolute_error_minutes": (
                    float(errors.median()) if len(errors) else np.nan
                ),
                "p90_absolute_error_minutes": (
                    float(errors.quantile(0.90)) if len(errors) else np.nan
                ),
                "within_15_minutes_pct": (
                    100.0 * float((errors <= 15.0).mean())
                    if len(errors)
                    else np.nan
                ),
                "within_30_minutes_pct": (
                    100.0 * float((errors <= 30.0).mean())
                    if len(errors)
                    else np.nan
                ),
            }
            for column in columns:
                row[column] = labels[column]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["stratum_type", "period", "road", "direction", "method"],
        kind="mergesort",
    )


def _write_validation_figure(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values("mae_minutes", ascending=True)
    labels = ordered["method"].str.replace("_", " ", regex=False)
    values = ordered["mae_minutes"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.barh(labels, values, color="#4472C4")
    axis.set_xlabel("Mean absolute T2 error (minutes)")
    axis.set_title("Spatial block holdout validation")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            float(value) + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.1f}",
            va="center",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _report_markdown(
    summary: pd.DataFrame,
    benchmark_count: int,
    group_count: int,
    generated_utc: str,
) -> str:
    lines = [
        "# T2 spatial holdout validation",
        "",
        f"- Generated UTC: `{generated_utc}`",
        f"- Reliable benchmark TMC-periods: **{benchmark_count:,}**",
        f"- Directional corridor-period groups: **{group_count:,}**",
        "- Holdout design: contiguous spatial blocks; every representation of a "
        "withheld TMC is excluded from training.",
        "",
        "## Method summary",
        "",
        "| Method | Predictions | Coverage | MAE (min) | Median AE | P90 AE | Within 15 min | Within 30 min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| {method} | {predictions:,} | {coverage:.1f}% | {mae:.1f} | "
            "{median:.1f} | {p90:.1f} | {within15:.1f}% | {within30:.1f}% |".format(
                method=str(row.method).replace("_", " "),
                predictions=int(row.predictions),
                coverage=float(row.prediction_coverage_pct),
                mae=float(row.mae_minutes),
                median=float(row.median_absolute_error_minutes),
                p90=float(row.p90_absolute_error_minutes),
                within15=float(row.within_15_minutes_pct),
                within30=float(row.within_30_minutes_pct),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The method with the smallest MAE is not automatically the production "
            "choice. Coverage, P90 error, gap length, facility type, and the "
            "effect of T2 error on QVDF reconstruction should be reviewed "
            "together. Profile interpolation and linear T2 interpolation are "
            "reported separately; a failed profile detection is never counted "
            "as a successful profile prediction.",
            "",
        ]
    )
    return "\n".join(lines)


def run_validation(
    module_root: Path,
    config: ExpansionConfig,
) -> Dict[str, object]:
    module_root = Path(module_root).resolve()
    frames = load_snapshot(module_root)
    routes, _ = add_corridor_positions(
        frames["routes"], frames["mappings"]
    )
    anchors = build_anchors(frames["representatives"], routes, config)
    profile_lookup = make_profile_lookup(frames["profiles"])
    all_rows: List[Dict[str, object]] = []
    benchmark_count = 0
    group_count = 0
    for _, group in anchors.groupby(
        ["period", "road", "direction"], sort=True
    ):
        rows, evaluated = validate_anchor_group(
            group, profile_lookup, config
        )
        if evaluated:
            group_count += 1
            benchmark_count += evaluated
            all_rows.extend(rows)
    predictions = pd.DataFrame(all_rows)
    if predictions.empty:
        raise ValueError("No spatial validation predictions were produced")
    summary = summarize_validation(predictions, benchmark_count)
    strata = summarize_validation_strata(predictions)

    output_dir = module_root / "outputs"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "validation_predictions.csv"
    summary_path = output_dir / "validation_summary.csv"
    strata_path = output_dir / "validation_summary_by_stratum.csv"
    report_path = output_dir / "validation_report.md"
    figure_path = figure_dir / "validation_mae_by_method.png"
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    strata.to_csv(strata_path, index=False)
    generated_utc = datetime.now(timezone.utc).isoformat()
    report_path.write_text(
        _report_markdown(
            summary, benchmark_count, group_count, generated_utc
        ),
        encoding="utf-8",
    )
    _write_validation_figure(summary, figure_path)
    result = {
        "status": "PASS",
        "generated_utc": generated_utc,
        "benchmark_tmc_periods": int(benchmark_count),
        "corridor_period_groups": int(group_count),
        "predictions": int(len(predictions)),
        "best_mae_method": str(summary.iloc[0]["method"]),
        "best_mae_minutes": float(summary.iloc[0]["mae_minutes"]),
        "outputs": {
            "predictions": {
                "path": str(predictions_path.relative_to(module_root)),
                "sha256": sha256(predictions_path),
            },
            "summary": {
                "path": str(summary_path.relative_to(module_root)),
                "sha256": sha256(summary_path),
            },
            "summary_by_stratum": {
                "path": str(strata_path.relative_to(module_root)),
                "sha256": sha256(strata_path),
            },
            "report": {
                "path": str(report_path.relative_to(module_root)),
                "sha256": sha256(report_path),
            },
            "figure": {
                "path": str(figure_path.relative_to(module_root)),
                "sha256": sha256(figure_path),
            },
        },
    }
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
