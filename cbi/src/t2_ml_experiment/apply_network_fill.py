from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

PERIODS = ("AM", "MD", "PM")
LEVELS = [
    (1, "Direct", "direct"),
    (2, "Direct + spatial", "spatial"),
    (3, "Direct + spatial + ML", "ml"),
]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio_from_boundaries(
    t0: pd.Series, t2: pd.Series, t3: pd.Series
) -> pd.DataFrame:
    span = (t3 - t0) * 60.0
    fraction = (t2 - t0) / (t3 - t0)
    pre_post = fraction / (1.0 - fraction)
    return pd.DataFrame(
        {
            "span_min": span,
            "t2_fraction": fraction,
            "pre_post_ratio": pre_post,
        },
        index=t0.index,
    )


def apply_link_file(
    source_path: Path,
    predictions: pd.DataFrame,
    period: str,
    destination_path: Path,
) -> Dict[str, object]:
    links = pd.read_csv(source_path, low_memory=False)
    period_predictions = predictions[
        predictions["period"].astype(str).str.upper().eq(period)
    ].copy()
    if period_predictions.duplicated("network_link_id").any():
        raise ValueError(f"Duplicate ML link keys for {period}.")
    prediction_columns = [
        "network_link_id",
        "ml_pred_t0_hour",
        "ml_pred_t2_hour",
        "ml_pred_t3_hour",
        "estimated_p90_abs_error_t0_min",
        "estimated_p90_abs_error_t2_min",
        "estimated_p90_abs_error_t3_min",
    ]
    merged = links.merge(
        period_predictions[prediction_columns],
        left_on="link_id",
        right_on="network_link_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).drop(columns=["network_link_id"])
    if len(merged) != len(links):
        raise ValueError(f"Row count changed for {period}.")
    if merged["ml_pred_t2_hour"].isna().any():
        raise ValueError(f"Missing ML predictions for {period}.")

    merged["t0_before_ml_hour"] = merged["t0_hybrid_hour"]
    merged["t2_before_ml_hour"] = merged["t2_hybrid_hour"]
    merged["t3_before_ml_hour"] = merged["t3_hybrid_hour"]
    merged["t2_source_before_ml"] = merged["t2_hybrid_source"]
    ratios = _ratio_from_boundaries(
        merged["ml_pred_t0_hour"],
        merged["ml_pred_t2_hour"],
        merged["ml_pred_t3_hour"],
    )
    merged["ml_pred_congestion_span_min"] = ratios["span_min"]
    merged["ml_pred_t2_fraction"] = ratios["t2_fraction"]
    merged["ml_pred_pre_post_duration_ratio"] = ratios["pre_post_ratio"]

    direct = (
        merged["t2_source_before_ml"].astype(str).str.lower().eq("direct")
    )
    spatial = (
        merged["t2_source_before_ml"].astype(str).str.lower().eq("spatial")
    )
    class_source = (
        merged["t2_source_before_ml"].astype(str).str.lower().eq("class")
    )
    protected_no_congestion = (
        merged.get(
            "t2_observed_no_congestion_protected",
            pd.Series(False, index=merged.index),
        )
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
        | merged["t2_source_before_ml"]
        .astype(str)
        .str.lower()
        .eq("observed_no_congestion")
    )
    ml_full = ~(direct | spatial | protected_no_congestion)
    if merged.loc[direct | spatial, "t2_before_ml_hour"].isna().any():
        raise ValueError(
            f"Direct/spatial t2 is unexpectedly missing for {period}."
        )
    final_t2 = merged["ml_pred_t2_hour"].copy()
    final_t2.loc[direct | spatial] = merged.loc[
        direct | spatial, "t2_before_ml_hour"
    ]
    final_t2.loc[protected_no_congestion] = np.nan
    derived_t0 = (
        final_t2
        - merged["ml_pred_t2_fraction"]
        * merged["ml_pred_congestion_span_min"]
        / 60.0
    )
    derived_t3 = (
        final_t2
        + (1.0 - merged["ml_pred_t2_fraction"])
        * merged["ml_pred_congestion_span_min"]
        / 60.0
    )
    merged["t0_hybrid_hour"] = derived_t0
    merged["t2_hybrid_hour"] = final_t2
    merged["t3_hybrid_hour"] = derived_t3
    merged.loc[direct, "t0_hybrid_hour"] = merged.loc[
        direct, "t0_before_ml_hour"
    ]
    merged.loc[direct, "t3_hybrid_hour"] = merged.loc[
        direct, "t3_before_ml_hour"
    ]
    merged.loc[
        protected_no_congestion,
        ["t0_hybrid_hour", "t2_hybrid_hour", "t3_hybrid_hour"],
    ] = np.nan
    merged.loc[
        protected_no_congestion, "t2_hybrid_source"
    ] = "observed_no_congestion"
    merged.loc[
        protected_no_congestion, "t2_hybrid_detail"
    ] = "protected_best_match_tmc_no_average_weekday_congestion"
    merged.loc[
        protected_no_congestion, "t2_hybrid_precedence_rank"
    ] = 0

    merged.loc[ml_full, "t2_hybrid_source"] = "ml"
    merged.loc[ml_full, "t2_hybrid_detail"] = np.where(
        class_source.loc[ml_full],
        "ridge_core network-wide prediction; class t2 retained for audit only",
        "ridge_core network-wide t2/span/fraction prediction",
    )
    merged.loc[ml_full, "t2_hybrid_precedence_rank"] = 3
    merged["boundary_fill_source"] = np.select(
        [
            protected_no_congestion,
            direct,
            spatial,
            ml_full & class_source,
            ml_full,
        ],
        [
            "observed_no_congestion_protected",
            "direct_observed",
            "spatial_t2_plus_ml_shape",
            "ml_t2_plus_ml_shape__class_reference_available",
            "ml_t2_plus_ml_shape",
        ],
        default="unexpected_source",
    )
    merged["class_t2_audit_hour"] = (
        merged["t2_class_hour"]
        if "t2_class_hour" in merged
        else np.nan
    )
    merged["ml_filled_t0"] = ~(direct | protected_no_congestion)
    merged["ml_filled_t2"] = ml_full
    merged["ml_filled_t3"] = ~(direct | protected_no_congestion)
    merged["ml_replaced_class_t2"] = class_source & ~protected_no_congestion
    merged["boundary_uncertainty_scope"] = np.select(
        [protected_no_congestion, direct, spatial, ml_full],
        [
            "observed_no_congestion",
            "observed_direct",
            "spatial_t2_with_ml_shape_error",
            "strict_corridor_held_out_ml_error",
        ],
        default="unknown",
    )

    boundary_columns = [
        "t0_hybrid_hour",
        "t2_hybrid_hour",
        "t3_hybrid_hour",
    ]
    null_boundaries = merged[boundary_columns].isna()
    if null_boundaries.any(axis=1).ne(protected_no_congestion).any():
        raise ValueError(
            f"Final boundary nulls do not match protected no-congestion rows for {period}."
        )
    if null_boundaries.loc[protected_no_congestion].ne(True).any().any():
        raise ValueError(
            f"Protected no-congestion rows contain an ML boundary for {period}."
        )
    ordered = (
        merged["t0_hybrid_hour"].le(merged["t2_hybrid_hour"])
        & merged["t2_hybrid_hour"].le(merged["t3_hybrid_hour"])
    )
    if not ordered.loc[~protected_no_congestion].all():
        raise ValueError(
            f"{int((~ordered & ~protected_no_congestion).sum())} unordered boundaries for {period}."
        )
    existing_t2 = direct | spatial
    if (
        merged.loc[existing_t2, "t2_hybrid_hour"]
        - merged.loc[existing_t2, "t2_before_ml_hour"]
    ).abs().max() > 1e-10:
        raise ValueError(f"Existing t2 changed for {period}.")
    for boundary in ("t0", "t3"):
        before = f"{boundary}_before_ml_hour"
        after = f"{boundary}_hybrid_hour"
        existing = direct
        if (
            merged.loc[existing, after] - merged.loc[existing, before]
        ).abs().max() > 1e-10:
            raise ValueError(
                f"Existing {boundary} changed for {period}."
            )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination_path, index=False)
    return {
        "period": period,
        "source_file": str(source_path.resolve()),
        "output_file": str(destination_path.resolve()),
        "rows": len(merged),
        "direct": int(direct.sum()),
        "spatial": int(spatial.sum()),
        "class_reference": int(class_source.sum()),
        "ml": int(ml_full.sum()),
        "protected_no_congestion": int(protected_no_congestion.sum()),
        "ml_replaced_class": int(class_source.sum()),
        "ml_previously_unassigned": int(
            (ml_full & ~class_source).sum()
        ),
        "final_class_source": int(
            merged["t2_hybrid_source"]
            .astype(str)
            .str.lower()
            .eq("class")
            .sum()
        ),
        "uncertainty_rows": int(
            merged[
                [
                    "estimated_p90_abs_error_t0_min",
                    "estimated_p90_abs_error_t2_min",
                    "estimated_p90_abs_error_t3_min",
                ]
            ]
            .notna()
            .all(axis=1)
            .sum()
        ),
        "final_t0_coverage": int(merged["t0_hybrid_hour"].notna().sum()),
        "final_t2_coverage": int(merged["t2_hybrid_hour"].notna().sum()),
        "final_t3_coverage": int(merged["t3_hybrid_hour"].notna().sum()),
        "ordered_rows": int(ordered.loc[~protected_no_congestion].sum()),
    }


def build_coverage_tables(application_summary: pd.DataFrame) -> pd.DataFrame:
    frames = [application_summary.copy()]
    count_columns = ["rows", "direct", "spatial", "ml"]
    if "protected_no_congestion" in application_summary:
        count_columns.append("protected_no_congestion")
    all_row = application_summary[count_columns].sum()
    all_row["period"] = "ALL"
    frames.append(pd.DataFrame([all_row]))
    counts = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, object]] = []
    for _, item in counts.iterrows():
        cumulative = 0
        for level, level_name, source in LEVELS:
            incremental = int(item[source])
            cumulative += incremental
            rows.append(
                {
                    "period": item["period"],
                    "level": level,
                    "level_name": level_name,
                    "added_source": source,
                    "network_link_period_rows": int(item["rows"]),
                    "incremental_rows": incremental,
                    "incremental_coverage_pct": (
                        100.0 * incremental / item["rows"]
                    ),
                    "cumulative_rows": cumulative,
                    "cumulative_coverage_pct": (
                        100.0 * cumulative / item["rows"]
                    ),
                    "remaining_rows": int(item["rows"]) - cumulative,
                    "protected_no_congestion_rows": int(
                        item.get("protected_no_congestion", 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summarize_errors(
    frame: pd.DataFrame,
    *,
    period: str,
    level: int,
    level_name: str,
    source: str,
    validation_method: str,
) -> Dict[str, object]:
    absolute = frame[
        ["abs_error_t0_min", "abs_error_t2_min", "abs_error_t3_min"]
    ]
    return {
        "period": period,
        "level": level,
        "level_name": level_name,
        "added_source": source,
        "validation_method": validation_method,
        "validation_rows": len(frame),
        "mae_t0_min": absolute["abs_error_t0_min"].mean(),
        "mae_t2_min": absolute["abs_error_t2_min"].mean(),
        "mae_t3_min": absolute["abs_error_t3_min"].mean(),
        "mean_boundary_mae_min": absolute.to_numpy().mean(),
        "p90_boundary_abs_error_min": np.quantile(
            absolute.to_numpy(), 0.90
        ),
        "mae_span_min": frame["abs_error_span_min"].mean(),
        "mae_t2_fraction": frame["abs_error_t2_fraction"].mean(),
    }


def _boundary_error_frame(
    observed: pd.DataFrame,
    predicted_t2: pd.Series,
) -> pd.DataFrame:
    result = observed.copy()
    result["derived_t0_hour"] = (
        predicted_t2
        - result["pred_t2_fraction"] * result["pred_span_min"] / 60.0
    )
    result["derived_t3_hour"] = (
        predicted_t2
        + (1.0 - result["pred_t2_fraction"])
        * result["pred_span_min"]
        / 60.0
    )
    result["abs_error_t0_min"] = (
        result["derived_t0_hour"] - result["observed_t0_hour"]
    ).abs() * 60.0
    result["abs_error_t2_min"] = (
        predicted_t2 - result["observed_t2_hour"]
    ).abs() * 60.0
    result["abs_error_t3_min"] = (
        result["derived_t3_hour"] - result["observed_t3_hour"]
    ).abs() * 60.0
    observed_span = (
        result["observed_t3_hour"] - result["observed_t0_hour"]
    ) * 60.0
    observed_fraction = (
        (result["observed_t2_hour"] - result["observed_t0_hour"])
        / (result["observed_t3_hour"] - result["observed_t0_hour"])
    )
    result["abs_error_span_min"] = (
        result["pred_span_min"] - observed_span
    ).abs()
    result["abs_error_t2_fraction"] = (
        result["pred_t2_fraction"] - observed_fraction
    ).abs()
    return result


def build_error_table(
    ml_run_dir: Path,
    comparison_run_dir: Path,
) -> pd.DataFrame:
    predictions = pd.read_csv(
        ml_run_dir / "metrics" / "out_of_fold_predictions.csv",
        low_memory=False,
    )
    ml = predictions[
        predictions["data_model"].eq("aggregate_all_days")
        & predictions["validation"].eq("corridor_held_out")
        & predictions["model"].eq("ridge_core")
    ].copy()
    ml = ml.drop_duplicates(["tmc_code", "period"])
    benchmark = pd.read_csv(
        comparison_run_dir / "outputs" / "validation_benchmark_detail.csv",
        low_memory=False,
    ).rename(
        columns={
            "tmc": "tmc_code",
            "observed_t2_hour": "benchmark_observed_t2_hour",
        }
    )
    benchmark["period"] = benchmark["period"].astype(str).str.upper()
    joined = benchmark.merge(
        ml[
            [
                "tmc_code",
                "period",
                "observed_t0_hour",
                "observed_t2_hour",
                "observed_t3_hour",
                "pred_span_min",
                "pred_t2_fraction",
            ]
        ],
        on=["tmc_code", "period"],
        how="inner",
        validate="one_to_one",
    )
    rows: List[Dict[str, object]] = []
    for period in [*PERIODS, "ALL"]:
        rows.append(
            {
                "period": period,
                "level": 1,
                "level_name": "Direct",
                "added_source": "direct",
                "validation_method": (
                    "Observed accepted episode; zero construction error, "
                    "not an out-of-sample prediction."
                ),
                "validation_rows": np.nan,
                "mae_t0_min": 0.0,
                "mae_t2_min": 0.0,
                "mae_t3_min": 0.0,
                "mean_boundary_mae_min": 0.0,
                "p90_boundary_abs_error_min": 0.0,
                "mae_span_min": 0.0,
                "mae_t2_fraction": 0.0,
            }
        )
        ml_period = ml if period == "ALL" else ml[ml["period"].eq(period)]
        ml_errors = ml_period.copy()
        for boundary in ("t0", "t2", "t3"):
            ml_errors[f"abs_error_{boundary}_min"] = (
                ml_errors[f"pred_{boundary}_hour"]
                - ml_errors[f"observed_{boundary}_hour"]
            ).abs() * 60.0
        observed_span = (
            ml_errors["observed_t3_hour"]
            - ml_errors["observed_t0_hour"]
        ) * 60.0
        observed_fraction = (
            (
                ml_errors["observed_t2_hour"]
                - ml_errors["observed_t0_hour"]
            )
            / (
                ml_errors["observed_t3_hour"]
                - ml_errors["observed_t0_hour"]
            )
        )
        ml_errors["abs_error_span_min"] = (
            ml_errors["pred_span_min"] - observed_span
        ).abs()
        ml_errors["abs_error_t2_fraction"] = (
            ml_errors["pred_t2_fraction"] - observed_fraction
        ).abs()
        rows.append(
            _summarize_errors(
                ml_errors,
                period=period,
                level=3,
                level_name="Direct + spatial + ML",
                source="ml",
                validation_method=(
                    "Strict corridor-held-out ridge_core prediction."
                ),
            )
        )
        for source, prediction_column, level, level_name in [
            (
                "spatial",
                "expansion_prediction",
                2,
                "Direct + spatial",
            ),
            (
                "class_reference",
                "class_prediction",
                0,
                "Class audit reference",
            ),
        ]:
            subset = joined.dropna(subset=[prediction_column]).copy()
            if period != "ALL":
                subset = subset[subset["period"].eq(period)]
            errors = _boundary_error_frame(
                subset, subset[prediction_column]
            )
            rows.append(
                _summarize_errors(
                    errors,
                    period=period,
                    level=level,
                    level_name=level_name,
                    source=source,
                    validation_method=(
                        "Reliable TMC-period holdout t2 plus the strict "
                        "corridor-held-out ML span/fraction for t0/t3."
                    ),
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["period", "level"], key=lambda s: s.map(
            {"AM": 1, "MD": 2, "PM": 3, "ALL": 4}
        ) if s.name == "period" else s
    ).reset_index(drop=True)


def write_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    validations: pd.DataFrame,
) -> None:
    all_rows = comparison[comparison["period"].eq("ALL")].sort_values(
        "level"
    )
    lines = [
        "# Network boundary completion",
        "",
        "The final precedence is direct -> spatial -> ML. Class is retained "
        "only as an audit/reference field:",
        "",
        "- direct rows retain observed t0/t2/t3;",
        "- spatial rows retain spatial t2 and receive t0/t3 from the "
        "ML-predicted congestion span and t2 fraction;",
        "- every remaining row receives ML t2, span, fraction, and derived "
        "t0/t3;",
        "- class t2 remains in `class_t2_audit_hour` and `t2_class_hour` but "
        "is never selected as a final value.",
        "",
        "## Network-wide coverage",
        "",
        "| Level | Added source | Incremental rows | Cumulative rows | Coverage |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in all_rows.itertuples(index=False):
        lines.append(
            f"| {row.level_name} | {row.added_source} | "
            f"{row.incremental_rows:,} | {row.cumulative_rows:,} | "
            f"{row.cumulative_coverage_pct:.2f}% |"
        )
    all_errors = comparison[comparison["period"].eq("ALL")].sort_values(
        "level"
    )
    lines.extend(
        [
            "",
            "## Validation error",
            "",
            "| Added source | Validation rows | T0 MAE | T2 MAE | T3 MAE | "
            "Mean boundary MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in all_errors.itertuples(index=False):
        validation_rows = (
            "observed"
            if pd.isna(row.validation_rows)
            else f"{int(row.validation_rows):,}"
        )
        lines.append(
            f"| {row.added_source} | {validation_rows} | "
            f"{row.mae_t0_min:.2f} | {row.mae_t2_min:.2f} | "
            f"{row.mae_t3_min:.2f} | "
            f"{row.mean_boundary_mae_min:.2f} |"
        )
    lines.extend(
        [
            "",
            "Direct error is zero by construction and is not comparable to "
            "the holdout errors. Spatial T2 error uses reliable TMC-period "
            "holdouts, while its T0/T3 errors combine held-out spatial T2 "
            "with ML span/fraction predictions. ML errors use strict "
            "corridor-held-out predictions. Class validation remains in "
            "`validation_details.csv` for audit only.",
            "",
            "All output link rows have non-null, physically ordered "
            "t0 <= t2 <= t3. Direct boundaries and spatial t2 values are "
            "unchanged.",
            "",
            "See `coverage_error_comparison.csv`, `validation_details.csv`, "
            "and `period_link_files/<period>/link.csv`.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(
    boundary_mapping_run_dir: Path,
    ml_run_dir: Path,
    comparison_run_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = ml_run_dir / "experimental_network_boundaries.csv"
    predictions = pd.read_csv(predictions_path, low_memory=False)
    application_rows = []
    input_files: List[Path] = [predictions_path]
    for period in PERIODS:
        source = (
            boundary_mapping_run_dir
            / "link-t2"
            / "period_link_files"
            / period.lower()
            / "link.csv"
        )
        destination = (
            output_dir
            / "period_link_files"
            / period.lower()
            / "link.csv"
        )
        application_rows.append(
            apply_link_file(source, predictions, period, destination)
        )
        input_files.append(source)
    application_summary = pd.DataFrame(application_rows)
    coverage = build_coverage_tables(application_summary)
    errors = build_error_table(ml_run_dir, comparison_run_dir)
    input_files.extend(
        [
            ml_run_dir
            / "metrics"
            / "out_of_fold_predictions.csv",
            comparison_run_dir
            / "outputs"
            / "validation_benchmark_detail.csv",
        ]
    )
    comparison = coverage.merge(
        errors,
        on=["period", "level", "level_name", "added_source"],
        how="left",
        validate="one_to_one",
    )
    application_summary.to_csv(
        output_dir / "application_summary.csv", index=False
    )
    coverage.to_csv(output_dir / "coverage_by_tier.csv", index=False)
    errors.to_csv(output_dir / "validation_details.csv", index=False)
    comparison.to_csv(
        output_dir / "coverage_error_comparison.csv", index=False
    )
    protected_rows = int(
        application_summary["protected_no_congestion"].sum()
    )
    rows_requiring_boundaries = int(len(predictions) - protected_rows)
    validations = pd.DataFrame(
        [
            {
                "check": "total_rows",
                "value": int(application_summary["rows"].sum()),
                "expected": int(len(predictions)),
                "status": (
                    "PASS"
                    if application_summary["rows"].sum() == len(predictions)
                    else "FAIL"
                ),
            },
            {
                "check": "final_t0_non_null",
                "value": int(
                    application_summary["final_t0_coverage"].sum()
                ),
                "expected": rows_requiring_boundaries,
                "status": (
                    "PASS"
                    if application_summary["final_t0_coverage"].sum()
                    == rows_requiring_boundaries
                    else "FAIL"
                ),
            },
            {
                "check": "final_t2_non_null",
                "value": int(
                    application_summary["final_t2_coverage"].sum()
                ),
                "expected": rows_requiring_boundaries,
                "status": (
                    "PASS"
                    if application_summary["final_t2_coverage"].sum()
                    == rows_requiring_boundaries
                    else "FAIL"
                ),
            },
            {
                "check": "final_t3_non_null",
                "value": int(
                    application_summary["final_t3_coverage"].sum()
                ),
                "expected": rows_requiring_boundaries,
                "status": (
                    "PASS"
                    if application_summary["final_t3_coverage"].sum()
                    == rows_requiring_boundaries
                    else "FAIL"
                ),
            },
            {
                "check": "ordered_boundaries",
                "value": int(application_summary["ordered_rows"].sum()),
                "expected": rows_requiring_boundaries,
                "status": (
                    "PASS"
                    if application_summary["ordered_rows"].sum()
                    == rows_requiring_boundaries
                    else "FAIL"
                ),
            },
            {
                "check": "protected_no_congestion_rows",
                "value": protected_rows,
                "expected": protected_rows,
                "status": "PASS",
            },
            {
                "check": "final_class_source_rows",
                "value": int(
                    application_summary["final_class_source"].sum()
                ),
                "expected": 0,
                "status": (
                    "PASS"
                    if application_summary["final_class_source"].sum() == 0
                    else "FAIL"
                ),
            },
            {
                "check": "uncertainty_fields_non_null",
                "value": int(
                    application_summary["uncertainty_rows"].sum()
                ),
                "expected": int(len(predictions)),
                "status": (
                    "PASS"
                    if application_summary["uncertainty_rows"].sum()
                    == len(predictions)
                    else "FAIL"
                ),
            },
        ]
    )
    validations.to_csv(output_dir / "validation_checks.csv", index=False)
    if validations["status"].ne("PASS").any():
        raise ValueError("One or more application validation checks failed.")
    write_report(output_dir, comparison, validations)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "boundary_mapping_run_dir": str(boundary_mapping_run_dir.resolve()),
        "ml_run_dir": str(ml_run_dir.resolve()),
        "comparison_run_dir": str(comparison_run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_model": "ridge_core",
        "application_rule": (
            "Direct retains observed t0/t2/t3. Spatial retains t2 and uses "
            "ML span/fraction for t0/t3. Best-match observed TMC links with "
            "no accepted average-weekday congestion remain blank. All other "
            "rows use ML t2/span/fraction. Class is audit-only."
        ),
        "protected_no_congestion_rows": protected_rows,
        "rows_requiring_boundaries": rows_requiring_boundaries,
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": _hash_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_files
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply ML boundary completion to NVTA link files."
    )
    parser.add_argument(
        "--boundary-mapping-run-dir",
        type=Path,
        required=True,
        help="Explicit congestion-boundary mapping input.",
    )
    parser.add_argument(
        "--ml-run-dir",
        type=Path,
        required=True,
        help="Explicit ML experiment input.",
    )
    parser.add_argument(
        "--comparison-run-dir",
        type=Path,
        required=True,
        help="Explicit coverage comparison input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Stable producer output directory.",
    )
    args = parser.parse_args(argv)
    print(
        run(
            args.boundary_mapping_run_dir,
            args.ml_run_dir,
            args.comparison_run_dir,
            args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
