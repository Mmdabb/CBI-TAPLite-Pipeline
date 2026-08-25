from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD_RANGES = {
    "AM": (6.0, 9.0),
    "MD": (9.0, 15.0),
    "PM": (15.0, 19.0),
}

TARGET_COLUMNS = [
    "target_t2_relative_min",
    "target_log_span_min",
    "target_logit_t2_fraction",
]


def transform_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    starts = out["period"].map(lambda p: PERIOD_RANGES[str(p).upper()][0]).astype(float)
    span_min = (out["t3_hour"] - out["t0_hour"]) * 60.0
    fraction = (out["t2_hour"] - out["t0_hour"]) / (out["t3_hour"] - out["t0_hour"])
    fraction = fraction.clip(1e-4, 1.0 - 1e-4)
    out["target_t2_relative_min"] = (out["t2_hour"] - starts) * 60.0
    out["target_log_span_min"] = np.log(span_min.clip(lower=1.0))
    out["target_logit_t2_fraction"] = np.log(fraction / (1.0 - fraction))
    return out


def reconstruct_boundaries(
    period: pd.Series,
    t2_relative_min: np.ndarray,
    log_span_min: np.ndarray,
    logit_fraction: np.ndarray,
) -> pd.DataFrame:
    periods = period.astype(str).str.upper()
    starts = periods.map(lambda p: PERIOD_RANGES[p][0]).to_numpy(dtype=float)
    ends = periods.map(lambda p: PERIOD_RANGES[p][1]).to_numpy(dtype=float)
    t2 = starts + np.asarray(t2_relative_min, dtype=float) / 60.0
    t2 = np.minimum(np.maximum(t2, starts), ends)
    span_min = np.exp(np.asarray(log_span_min, dtype=float))
    span_min = np.clip(span_min, 15.0, 900.0)
    logits = np.clip(np.asarray(logit_fraction, dtype=float), -20.0, 20.0)
    fraction = 1.0 / (1.0 + np.exp(-logits))
    t0 = t2 - fraction * span_min / 60.0
    t3 = t2 + (1.0 - fraction) * span_min / 60.0
    return pd.DataFrame(
        {
            "pred_t0_hour": t0,
            "pred_t2_hour": t2,
            "pred_t3_hour": t3,
            "pred_span_min": span_min,
            "pred_t2_fraction": fraction,
        },
        index=period.index,
    )


def reconstruct_with_fixed_t2(
    fixed_t2_hour: pd.Series,
    log_span_min: np.ndarray,
    logit_fraction: np.ndarray,
) -> pd.DataFrame:
    t2 = pd.to_numeric(fixed_t2_hour, errors="coerce").to_numpy(
        dtype=float
    )
    span_min = np.clip(
        np.exp(np.asarray(log_span_min, dtype=float)), 15.0, 900.0
    )
    logits = np.clip(
        np.asarray(logit_fraction, dtype=float), -20.0, 20.0
    )
    fraction = 1.0 / (1.0 + np.exp(-logits))
    return pd.DataFrame(
        {
            "pred_t0_hour": t2 - fraction * span_min / 60.0,
            "pred_t2_hour": t2,
            "pred_t3_hour": t2
            + (1.0 - fraction) * span_min / 60.0,
            "pred_span_min": span_min,
            "pred_t2_fraction": fraction,
        },
        index=fixed_t2_hour.index,
    )
