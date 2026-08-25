"""Numerical measures for aligned corridor speed profiles."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def weighted_harmonic_mean(
    values: Iterable[float], weights: Iterable[float]
) -> float:
    """Return a distance-weighted speed, equivalent to distance/travel time."""
    value_array = np.asarray(list(values), dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (value_array > 0)
        & (weight_array > 0)
    )
    if not valid.any():
        return float("nan")
    return float(
        weight_array[valid].sum()
        / np.sum(weight_array[valid] / value_array[valid])
    )


def speed_profile_metrics(
    observed: pd.Series,
    modeled: pd.Series,
    *,
    mape_minimum_observed_speed_mph: float,
) -> Dict[str, float]:
    """Calculate interval-aligned speed-profile accuracy measures."""
    observed_array = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    modeled_array = pd.to_numeric(modeled, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(observed_array) & np.isfinite(modeled_array)
    errors = modeled_array[valid] - observed_array[valid]

    result: Dict[str, float] = {
        "matched_interval_count": int(valid.sum()),
        "mae_mph": float("nan"),
        "mape_pct": float("nan"),
        "mape_interval_count": 0,
        "rmse_mph": float("nan"),
        "mean_bias_mph": float("nan"),
        "wmape_pct": float("nan"),
        "r_squared": float("nan"),
    }
    if not valid.any():
        return result

    absolute_errors = np.abs(errors)
    result["mae_mph"] = float(absolute_errors.mean())
    result["rmse_mph"] = float(np.sqrt(np.mean(np.square(errors))))
    result["mean_bias_mph"] = float(errors.mean())

    mape_valid = valid & (np.abs(observed_array) >= mape_minimum_observed_speed_mph)
    result["mape_interval_count"] = int(mape_valid.sum())
    if mape_valid.any():
        result["mape_pct"] = float(
            np.mean(
                np.abs(
                    (modeled_array[mape_valid] - observed_array[mape_valid])
                    / observed_array[mape_valid]
                )
            )
            * 100.0
        )

    observed_sum = float(np.abs(observed_array[valid]).sum())
    if observed_sum > 0:
        result["wmape_pct"] = float(absolute_errors.sum() / observed_sum * 100.0)

    if valid.sum() >= 2:
        observed_valid = observed_array[valid]
        modeled_valid = modeled_array[valid]
        if np.nanstd(observed_valid) > 0 and np.nanstd(modeled_valid) > 0:
            correlation = float(np.corrcoef(observed_valid, modeled_valid)[0, 1])
            result["r_squared"] = correlation * correlation
    return result

def congestion_episodes(
    frame: pd.DataFrame,
    flag_column: str,
    *,
    interval_minutes: int,
) -> List[Dict[str, float]]:
    """Convert a time-indexed congestion flag into contiguous episodes."""
    if frame.empty:
        return []
    ordered = frame.sort_values("t_min")
    rows = ordered[["t_min", flag_column]].to_dict("records")
    episodes: List[Dict[str, float]] = []
    start = None
    previous = None

    for row in rows:
        t_min = int(row["t_min"])
        flag = row[flag_column]
        congested = bool(flag) if pd.notna(flag) else False
        contiguous = previous is not None and t_min == previous + interval_minutes
        if congested and (start is None or not contiguous):
            if start is not None and previous is not None:
                episodes.append(
                    {
                        "start_min": start,
                        "end_min": previous + interval_minutes,
                        "duration_min": previous + interval_minutes - start,
                    }
                )
            start = t_min
        elif not congested and start is not None:
            episodes.append(
                {
                    "start_min": start,
                    "end_min": t_min,
                    "duration_min": t_min - start,
                }
            )
            start = None
        previous = t_min

    if start is not None and previous is not None:
        episodes.append(
            {
                "start_min": start,
                "end_min": previous + interval_minutes,
                "duration_min": previous + interval_minutes - start,
            }
        )
    return episodes


def congestion_fit_metrics(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
) -> Dict[str, float]:
    """Compare observed and modeled congestion on a shared interval grid."""
    required = ["observed_congested", "model_congested"]
    valid = frame[required].notna().all(axis=1)
    observed = frame.loc[valid, "observed_congested"].astype(bool)
    modeled = frame.loc[valid, "model_congested"].astype(bool)

    observed_count = int(observed.sum())
    modeled_count = int(modeled.sum())
    overlap_count = int((observed & modeled).sum())
    union_count = int((observed | modeled).sum())
    valid_count = int(valid.sum())

    observed_duration = observed_count * interval_minutes
    modeled_duration = modeled_count * interval_minutes
    signed_error = modeled_duration - observed_duration
    absolute_error = abs(signed_error)

    result: Dict[str, float] = {
        "congestion_valid_interval_count": valid_count,
        "observed_congestion_duration_min": observed_duration,
        "model_congestion_duration_min": modeled_duration,
        "congestion_duration_error_min": signed_error,
        "congestion_duration_absolute_error_min": absolute_error,
        "congestion_duration_ape_pct": float("nan"),
        "congestion_overlap_min": overlap_count * interval_minutes,
        "congestion_union_min": union_count * interval_minutes,
        "congestion_iou_pct": float("nan"),
        "congestion_precision_pct": float("nan"),
        "congestion_recall_pct": float("nan"),
        "congestion_f1_pct": float("nan"),
        "congestion_interval_accuracy_pct": float("nan"),
        "observed_congestion_episode_count": 0,
        "model_congestion_episode_count": 0,
        "observed_first_congestion_start_min": float("nan"),
        "model_first_congestion_start_min": float("nan"),
        "congestion_onset_error_min": float("nan"),
        "observed_last_congestion_end_min": float("nan"),
        "model_last_congestion_end_min": float("nan"),
        "congestion_clearance_error_min": float("nan"),
    }

    if observed_duration > 0:
        result["congestion_duration_ape_pct"] = (
            absolute_error / observed_duration * 100.0
        )
    if union_count > 0:
        result["congestion_iou_pct"] = overlap_count / union_count * 100.0
    if modeled_count > 0:
        result["congestion_precision_pct"] = overlap_count / modeled_count * 100.0
    if observed_count > 0:
        result["congestion_recall_pct"] = overlap_count / observed_count * 100.0
    precision = result["congestion_precision_pct"]
    recall = result["congestion_recall_pct"]
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
        result["congestion_f1_pct"] = 2.0 * precision * recall / (precision + recall)
    if valid_count > 0:
        result["congestion_interval_accuracy_pct"] = float(
            (observed.to_numpy() == modeled.to_numpy()).mean() * 100.0
        )

    observed_episodes = congestion_episodes(
        frame.loc[valid], "observed_congested", interval_minutes=interval_minutes
    )
    model_episodes = congestion_episodes(
        frame.loc[valid], "model_congested", interval_minutes=interval_minutes
    )
    result["observed_congestion_episode_count"] = len(observed_episodes)
    result["model_congestion_episode_count"] = len(model_episodes)

    if observed_episodes:
        result["observed_first_congestion_start_min"] = observed_episodes[0][
            "start_min"
        ]
        result["observed_last_congestion_end_min"] = observed_episodes[-1][
            "end_min"
        ]
    if model_episodes:
        result["model_first_congestion_start_min"] = model_episodes[0]["start_min"]
        result["model_last_congestion_end_min"] = model_episodes[-1]["end_min"]
    if observed_episodes and model_episodes:
        result["congestion_onset_error_min"] = (
            model_episodes[0]["start_min"] - observed_episodes[0]["start_min"]
        )
        result["congestion_clearance_error_min"] = (
            model_episodes[-1]["end_min"] - observed_episodes[-1]["end_min"]
        )
    return result
