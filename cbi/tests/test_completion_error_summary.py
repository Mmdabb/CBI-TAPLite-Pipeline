from __future__ import annotations

import math

import pandas as pd

from congestion_boundary_mapping.ridge_completion.apply_network_fill import (
    _resolve_benchmark_path,
    _summarize_errors,
)


def test_empty_optional_comparison_group_reports_nan_instead_of_crashing() -> None:
    empty = pd.DataFrame(
        columns=[
            "abs_error_t0_min",
            "abs_error_t2_min",
            "abs_error_t3_min",
            "abs_error_span_min",
            "abs_error_t2_fraction",
        ]
    )

    result = _summarize_errors(
        empty,
        period="AM",
        level=0,
        level_name="Class audit reference",
        source="class_reference",
        validation_method="optional comparison",
    )

    assert result["validation_rows"] == 0
    assert math.isnan(result["mean_boundary_mae_min"])
    assert math.isnan(result["p90_boundary_abs_error_min"])


def test_error_summary_ignores_nonfinite_optional_values() -> None:
    frame = pd.DataFrame(
        {
            "abs_error_t0_min": [10.0],
            "abs_error_t2_min": [float("nan")],
            "abs_error_t3_min": [20.0],
            "abs_error_span_min": [5.0],
            "abs_error_t2_fraction": [0.1],
        }
    )
    result = _summarize_errors(
        frame,
        period="AM",
        level=2,
        level_name="Direct + spatial",
        source="spatial",
        validation_method="test",
    )
    assert result["mean_boundary_mae_min"] == 15.0


def test_benchmark_path_prefers_detail_and_falls_back_to_predictions(
    tmp_path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    predictions = outputs / "validation_predictions.csv"
    predictions.write_text("tmc\n", encoding="utf-8")
    assert _resolve_benchmark_path(tmp_path) == predictions
    detail = outputs / "validation_benchmark_detail.csv"
    detail.write_text("tmc\n", encoding="utf-8")
    assert _resolve_benchmark_path(tmp_path) == detail
