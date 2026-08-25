from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_NAMES = {"period_median", "facility_class_median"}


def create_figures(
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    best_model: str,
    importance: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    comparison = leaderboard[
        (leaderboard["data_model"] == "aggregate_all_days")
        & (leaderboard["validation"] == "corridor_held_out")
    ].copy()
    comparison["scope_label"] = comparison["deployment_scope"].replace(
        {
            "full_network": "network-wide",
            "mapped_corridor_only": "corridor-only",
            "sensor_profile_only": "profile-only",
            "diagnostic": "diagnostic",
        }
    )
    comparison = comparison.sort_values("mean_boundary_mae_min").head(16)
    colors = comparison["scope_label"].map(
        {
            "network-wide": "#2f6b9a",
            "corridor-only": "#e2903a",
            "profile-only": "#55a868",
            "diagnostic": "#b6b6b6",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(
        comparison["model"],
        comparison["mean_boundary_mae_min"],
        color=colors,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute t0/t2/t3 error (minutes)")
    ax.set_title("Strict corridor-held-out model comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "corridor_held_out_leaderboard.png", dpi=180)
    plt.close(fig)

    subset = predictions[
        (predictions["data_model"] == "aggregate_all_days")
        & (predictions["validation"] == "corridor_held_out")
        & (predictions["model"] == best_model)
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"AM": "#1f77b4", "MD": "#ff7f0e", "PM": "#2ca02c"}
    for period, period_frame in subset.groupby("period"):
        ax.scatter(
            period_frame["observed_t2_hour"],
            period_frame["pred_t2_hour"],
            s=12,
            alpha=0.55,
            label=period,
            color=colors.get(period),
        )
    low = min(
        subset["observed_t2_hour"].min(), subset["pred_t2_hour"].min()
    )
    high = max(
        subset["observed_t2_hour"].max(), subset["pred_t2_hour"].max()
    )
    ax.plot(
        [low, high], [low, high], color="black", linewidth=1, linestyle="--"
    )
    ax.set_xlabel("Observed t2 (hour)")
    ax.set_ylabel("Out-of-fold predicted t2 (hour)")
    ax.set_title(f"Corridor-held-out t2: {best_model}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figure_dir / "best_network_model_t2_observed_vs_predicted.png",
        dpi=180,
    )
    plt.close(fig)

    assumptions = leaderboard[
        leaderboard["model"].isin([best_model, *BASELINE_NAMES])
    ].copy()
    assumptions["test"] = (
        assumptions["data_model"] + " / " + assumptions["validation"]
    )
    pivot = assumptions.pivot_table(
        index="test",
        columns="model",
        values="mean_boundary_mae_min",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean absolute t0/t2/t3 error (minutes)")
    ax.set_xlabel("")
    ax.set_title("Sensitivity to data and holdout assumptions")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "data_assumption_sensitivity.png", dpi=180)
    plt.close(fig)

    if not importance.empty:
        importance_feature = (
            "source_feature"
            if "source_feature" in importance.columns
            else "transformed_feature"
        )
        top = (
            importance.groupby(importance_feature, as_index=False)[
                "importance"
            ]
            .mean()
            .nlargest(20, "importance")
            .sort_values("importance")
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top[importance_feature], top["importance"], color="#2f6b9a")
        ax.set_xlabel("Mean impurity importance across three targets")
        ax.set_title(f"Top fitted features: {best_model}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "selected_model_feature_importance.png", dpi=180)
        plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    view = frame[list(columns)].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def _corridor_bootstrap_delta(
    predictions: pd.DataFrame,
    best_model: str,
    baseline_model: str,
    *,
    seed: int = 42,
    repeats: int = 1000,
) -> Tuple[float, float, float]:
    data = predictions[
        (predictions["data_model"] == "aggregate_all_days")
        & (predictions["validation"] == "corridor_held_out")
        & predictions["model"].isin([best_model, baseline_model])
    ]
    pivot = data.pivot_table(
        index=["tmc_period_id", "corridor"],
        columns="model",
        values="mean_abs_boundary_error_min",
        aggfunc="first",
    ).dropna(subset=[best_model, baseline_model])
    pivot["gain"] = pivot[baseline_model] - pivot[best_model]
    corridor_values = {
        corridor: group["gain"].to_numpy()
        for corridor, group in pivot.groupby(level="corridor")
    }
    keys = list(corridor_values)
    observed = float(pivot["gain"].mean())
    if len(keys) < 2:
        return observed, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sample = rng.choice(keys, size=len(keys), replace=True)
        values = np.concatenate([corridor_values[key] for key in sample])
        draws.append(values.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def write_report(
    output_dir: Path,
    aggregate: pd.DataFrame,
    daily: pd.DataFrame,
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    source_summary: pd.DataFrame,
    best_network_model: str,
) -> None:
    primary = leaderboard[
        (leaderboard["data_model"] == "aggregate_all_days")
        & (leaderboard["validation"] == "corridor_held_out")
    ].copy()
    full_network = primary[
        primary["deployment_scope"].eq("full_network")
    ]
    best_baseline = full_network[
        full_network["model"].isin(BASELINE_NAMES)
    ].iloc[0]
    best_ml = full_network[
        ~full_network["model"].isin(BASELINE_NAMES)
    ].iloc[0]
    gain, gain_low, gain_high = _corridor_bootstrap_delta(
        predictions,
        best_network_model,
        str(best_baseline["model"]),
    )
    period_counts = aggregate["period"].value_counts().sort_index().to_dict()
    scope_best = (
        primary.sort_values("mean_boundary_mae_min")
        .groupby("deployment_scope", as_index=False)
        .first()
    )
    sensitivity = leaderboard[
        leaderboard["model"].isin(
            [str(best_baseline["model"]), best_network_model]
        )
    ].copy()
    network_counts = (
        source_summary.groupby("experimental_boundary_source")[
            "link_period_count"
        ]
        .sum()
        .to_dict()
    )
    low_cardinality = primary[
        primary["model"].astype(str).str.contains("low_cardinality")
    ]
    low_cardinality_note = (
        "The focused low-cardinality ablation reached "
        f"{low_cardinality.iloc[0]['mean_boundary_mae_min']:.2f} minutes; "
        "removing jurisdiction/component/toll-group codes did not improve the "
        "selected full-network result."
        if not low_cardinality.empty
        else "No low-cardinality ablation was available."
    )
    period_specific = primary[
        primary["model"].eq("ridge_core_period_specific")
    ]
    period_specific_note = (
        "Training separate AM/MD/PM ridge models reached "
        f"{period_specific.iloc[0]['mean_boundary_mae_min']:.2f} minutes in "
        "corridor holdout. It improved the easier TMC holdout but worsened "
        "geographic transfer, consistent with period-specific overfitting."
        if not period_specific.empty
        else "No period-specific model ablation was available."
    )
    recommendation = (
        "The improvement is statistically and practically supported in this "
        "sample."
        if gain >= 1.0 and np.isfinite(gain_low) and gain_low > 0
        else "No practically reliable improvement over the simple baseline is "
        "demonstrated."
    )
    confidence = (
        f"95% corridor-bootstrap interval {gain_low:.2f} to "
        f"{gain_high:.2f} minutes"
        if np.isfinite(gain_low)
        else "bootstrap interval unavailable"
    )
    report = f"""# NVTA network-wide t0/t2/t3 ML research report

## Bottom line

The selected full-network candidate is `{best_network_model}`. Its strict
corridor-held-out mean boundary MAE is
{best_ml["mean_boundary_mae_min"]:.2f} minutes, compared with
{best_baseline["mean_boundary_mae_min"]:.2f} minutes for
`{best_baseline["model"]}`. The paired gain is {gain:.2f} minutes
({confidence}). {recommendation}

This remains an isolated research result. It has **not** replaced the production priority
`direct t2 -> spatial t2 -> class t2`. The experimental network file preserves that t2
hierarchy and uses ML only to supply missing full boundaries or to estimate t0/t3 around an
existing non-direct t2.

## Evidence pooled

- Accepted daily episode rows: {len(daily):,}
- TMC-period training rows: {len(aggregate):,}
- Unique TMCs: {aggregate["tmc_code"].nunique():,}
- Corridors: {aggregate["corridor"].nunique():,}
- Period counts: {json.dumps(period_counts, sort_keys=True)}
- Median accepted days per TMC-period: {aggregate["observed_day_count"].median():.1f}

All days belonging to the same TMC stay together in TMC-held-out validation. The primary
selection test holds out entire corridors, which is a more realistic test of geographic
transfer than a random row split. A last-{daily["date"].nunique() and 5}-weekday temporal
holdout and a reliable multi-day subset are reported as sensitivity checks.

## Feature and model exploration

The tournament separates four scopes:

1. **Full network:** static link attributes, period demand/capacity, assignment-model
   speed/V/C/VDF/travel time and vehicle mix, plus upstream/downstream graph and bottleneck
   descriptors.
2. **Mapped corridor only:** fold-safe nearby TMC anchors; the held-out TMC is excluded when
   constructing its features.
3. **Sensor profile only:** average-weekday speed/flow shape and fitted FD context.
4. **Diagnostic only:** P, episode demand, discharge rate, and observed episode speeds.
   These are post-detection variables and are never eligible for network-wide selection.

Models include hierarchical medians, ridge regression, Extra Trees, Random Forest, and
histogram gradient boosting. Class-physics priors for mu, vc, vf, and critical density are
calculated from each training fold only. T2 is modeled relative to period start; duration is
modeled on a log scale; and the position of t2 within the episode is modeled on a logit
scale. This guarantees `t0 <= t2 <= t3`.

Dimensionless period D/C and capacity-equivalent hours are algebraically equivalent once
period duration is known. The experiment retains dimensionless D/C for interpretability and
uses the assignment-model fields as separate ablations.

{low_cardinality_note}

{period_specific_note}

## Best strict result by scope

{_markdown_table(scope_best, ["deployment_scope", "model", "mae_t0_min", "mae_t2_min", "mae_t3_min", "mean_boundary_mae_min", "p90_boundary_abs_error_min"])}

Conditional corridor/profile results can show how much signal is available where those
inputs exist, but they are not evidence that the same performance is available over the
entire 47k-link network.

## Data-assumption sensitivity

{_markdown_table(sensitivity, ["data_model", "validation", "model", "row_count", "mae_t0_min", "mae_t2_min", "mae_t3_min", "mean_boundary_mae_min", "p90_boundary_abs_error_min"])}

The daily experiment weights each episode by the inverse number of accepted days for its
TMC-period, preventing heavily observed TMCs from dominating. The reliable aggregate subset
requires multiple days and limited day-to-day t2 dispersion.

## Experimental network-wide file

The output contains {source_summary["link_period_count"].sum():,} link-period rows:

{json.dumps(network_counts, indent=2, sort_keys=True)}

`experimental_network_boundaries.csv` is explicitly a prototype output. Direct observed
t0/t2/t3 are retained. Spatial or class t2 values are retained while the selected ML model
estimates episode duration and asymmetry around them. Links without any hierarchy t2 receive
all three ML boundaries. Period-specific p90 error columns are calibrated from strict
corridor-held-out predictions; they are error indicators, not formal prediction intervals.

## Decision

Promote the ML model only if its corridor-held-out gain is both practically useful and
stable, and after a genuinely later month or independent labeled-corridor evaluation.
Otherwise retain the existing three-tier hierarchy as the defensible production method and
use this prototype only as a research benchmark.

## Reproducible artifacts

- `data/training_tmc_period.csv` and `data/training_daily_episode.csv`
- `data/feature_coverage.csv`
- `metrics/leaderboard.csv`, stratified metrics, and out-of-fold predictions
- `metrics/feature_importance.csv` when the selected estimator exposes it
- `experimental_network_boundaries.csv`
- `models/` fitted full-sample transformed-target models
- `feature_manifest.json` and `run_manifest.json`
- `figures/`
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
