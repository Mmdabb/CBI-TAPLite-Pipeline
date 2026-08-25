from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter


def _odd_window(target: int, available: int) -> int:
    value = min(max(3, int(target)), int(available))
    if value % 2 == 0:
        value -= 1
    return max(1, value)


def smooth_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return values.copy()
    x = np.arange(len(values), dtype=float)
    filled = values.copy()
    filled[~valid] = np.interp(x[~valid], x[valid], values[valid])
    window = _odd_window(3, len(values))
    if window < 3:
        return filled
    return savgol_filter(filled, window_length=window, polyorder=2, mode="interp")


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(mask) - 1))
    return result


def _merge_short_gaps(
    runs: List[Tuple[int, int]], max_gap_intervals: int
) -> List[Tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prior_start, prior_end = merged[-1]
        gap = start - prior_end - 1
        if gap <= max_gap_intervals:
            merged[-1] = (prior_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_profile_t2(
    t_minutes: np.ndarray,
    normalized_speed: np.ndarray,
    period: Tuple[int, int],
    threshold_ratio: float,
    minimum_episode_minutes: int,
    merge_gap_minutes: int,
    minimum_coverage: float,
) -> Optional[Dict[str, float]]:
    """Detect the worst sustained congested interval in one period.

    This is a copied, self-contained profile detector for the prototype. It
    retains the core CBI contract: smooth the profile, form sustained
    threshold-based congestion intervals, and define T2 at the interval
    minimum. It deliberately does not import the production detector.
    """

    times = np.asarray(t_minutes, dtype=float)
    ratios = np.asarray(normalized_speed, dtype=float)
    start_minute, end_minute = period
    keep = (times >= float(start_minute)) & (times < float(end_minute))
    times = times[keep]
    ratios = ratios[keep]
    if len(times) < 2:
        return None
    coverage = float(np.isfinite(ratios).mean())
    if coverage < float(minimum_coverage):
        return None
    order = np.argsort(times)
    times = times[order]
    ratios = ratios[order]
    interval = float(np.nanmedian(np.diff(times)))
    if not np.isfinite(interval) or interval <= 0:
        return None
    smoothed = smooth_profile(ratios)
    congested = np.isfinite(smoothed) & (smoothed < float(threshold_ratio))
    gap_intervals = max(0, int(round(float(merge_gap_minutes) / interval)))
    candidates = _merge_short_gaps(_runs(congested), gap_intervals)
    rows: List[Dict[str, float]] = []
    for left, right in candidates:
        duration = float((right - left + 1) * interval)
        if duration < float(minimum_episode_minutes):
            continue
        segment = smoothed[left : right + 1]
        if not np.isfinite(segment).any():
            continue
        local_min = int(np.nanargmin(segment))
        t2_index = left + local_min
        rows.append(
            {
                "t0_hour": float(times[left] / 60.0),
                "t2_hour": float(times[t2_index] / 60.0),
                "t3_hour": float((times[right] + interval) / 60.0),
                "duration_minutes": duration,
                "minimum_ratio": float(smoothed[t2_index]),
                "profile_coverage": coverage,
            }
        )
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            row["minimum_ratio"],
            -row["duration_minutes"],
            row["t2_hour"],
        )
    )
    return rows[0]


def interpolate_normalized_profiles(
    left: np.ndarray,
    right: np.ndarray,
    weight_right: float,
) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Profile arrays must have the same shape")
    weight = min(1.0, max(0.0, float(weight_right)))
    result = np.full(left.shape, np.nan, dtype=float)
    both = np.isfinite(left) & np.isfinite(right)
    result[both] = (1.0 - weight) * left[both] + weight * right[both]
    left_only = np.isfinite(left) & ~np.isfinite(right)
    right_only = ~np.isfinite(left) & np.isfinite(right)
    result[left_only] = left[left_only]
    result[right_only] = right[right_only]
    return result

