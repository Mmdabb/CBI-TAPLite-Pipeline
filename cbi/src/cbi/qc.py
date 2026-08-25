"""Stage 1 speed quality control for the refreshed CBI package."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .panels import build_corridor_panel
from .schemas import STAGE1_FLAG_COLUMNS

LOGGER = logging.getLogger("cbi")


def hard_physical_range(
    speed: np.ndarray,
    v_f_mph: float | np.ndarray = 75.0,
    v_max_factor: float = 1.25,
    v_max_abs: float = 90.0,
) -> np.ndarray:
    """Return integer pass flags for physically valid speed observations."""
    speed = np.asarray(speed, dtype=float)
    v_f = np.asarray(v_f_mph, dtype=float)
    upper = np.minimum(v_max_factor * v_f, v_max_abs)
    return (np.isfinite(speed) & (speed >= 0.0) & (speed <= upper)).astype(np.int8)


def hampel_detector(
    speed: np.ndarray,
    window: int = 11,
    n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pass_mask, rolling_median)`` for isolated local speed spikes."""
    speed = np.asarray(speed, dtype=float)
    if len(speed) == 0:
        return np.ones(0, dtype=np.int8), np.array([], dtype=float)

    if window < 1:
        raise ValueError("hampel window must be at least 1")
    if window % 2 == 0:
        window += 1

    s = pd.Series(speed)
    med = s.rolling(window=window, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
    scaled_mad = 1.4826 * mad
    fallback = scaled_mad.replace(0.0, np.nan).median()
    scaled_mad = scaled_mad.replace(0.0, np.nan).fillna(fallback).fillna(1.0)
    threshold = n_sigma * scaled_mad
    mask = (np.isfinite(speed) & (abs_dev <= threshold).to_numpy()).astype(np.int8)
    return mask, med.to_numpy(dtype=float)


def temporal_jump_check(speed: np.ndarray, dv_max: float = 30.0) -> np.ndarray:
    """Flag observations whose finite consecutive speed jump exceeds ``dv_max``."""
    speed = np.asarray(speed, dtype=float)
    mask = np.ones(len(speed), dtype=np.int8)
    if len(speed) <= 1:
        return mask

    prev = speed[:-1]
    curr = speed[1:]
    comparable = np.isfinite(prev) & np.isfinite(curr)
    mask[1:][comparable] = (np.abs(curr[comparable] - prev[comparable]) <= dv_max).astype(np.int8)
    return mask


def spatial_lag_consistency(
    speed_field: np.ndarray,
    road_order: Optional[np.ndarray] = None,
    time_axis: Optional[np.ndarray] = None,
    sensor_spacing_mi: Optional[np.ndarray] = None,
    wave_speed_mph: Optional[float] = None,
    max_gap_mph: float = 20.0,
) -> np.ndarray:
    """Conservative time-lag-aware spatial QC scaffold."""
    speed_field = np.asarray(speed_field, dtype=float)
    if speed_field.ndim != 2:
        raise ValueError("speed_field must be a 2-D [time, sensor] array")

    mask = np.ones(speed_field.shape, dtype=np.int8)
    if speed_field.shape[1] < 3:
        return mask
    if road_order is None or time_axis is None or wave_speed_mph is None:
        return mask
    if sensor_spacing_mi is None and not np.all(np.isfinite(np.asarray(road_order, dtype=float))):
        return mask
    if not np.isfinite(wave_speed_mph) or wave_speed_mph <= 0:
        return mask

    _ = max_gap_mph
    _ = sensor_spacing_mi
    return mask


def _interpolation_weight(values: pd.Series, i: int) -> Optional[float]:
    prev_value = values.iloc[i - 1]
    curr_value = values.iloc[i]
    next_value = values.iloc[i + 1]
    numeric = pd.to_numeric(pd.Series([prev_value, curr_value, next_value]), errors="coerce")
    if numeric.notna().all():
        denom = numeric.iloc[2] - numeric.iloc[0]
        if denom > 0:
            return float((numeric.iloc[1] - numeric.iloc[0]) / denom)

    times = pd.to_datetime(pd.Series([prev_value, curr_value, next_value]), errors="coerce")
    if times.notna().all():
        denom = (times.iloc[2] - times.iloc[0]).total_seconds()
        if denom > 0:
            return float((times.iloc[1] - times.iloc[0]).total_seconds() / denom)
    return None


def _repair_isolated_failures(grp: pd.DataFrame, time_col: str = "datetime") -> tuple[np.ndarray, np.ndarray]:
    clean = grp["speed_mph_clean"].to_numpy(dtype=float)
    repaired = clean.copy()
    methods = np.full(len(clean), "none", dtype=object)
    replacement = grp["hampel_replacement"].to_numpy(dtype=float)
    time_values = grp[time_col] if time_col in grp.columns else pd.Series([pd.NaT] * len(grp), index=grp.index)

    for i in range(1, len(clean) - 1):
        if np.isfinite(clean[i]):
            continue
        prev_ok = np.isfinite(clean[i - 1])
        next_ok = np.isfinite(clean[i + 1])
        if not (prev_ok and next_ok):
            continue
        weight = _interpolation_weight(time_values.reset_index(drop=True), i)
        if weight is None:
            weight = 0.5
        repaired[i] = clean[i - 1] + weight * (clean[i + 1] - clean[i - 1])
        methods[i] = "linear_interpolation"
        if not np.isfinite(repaired[i]) and np.isfinite(replacement[i]):
            repaired[i] = replacement[i]
            methods[i] = "hampel_fallback"
    return repaired, methods


def _spatial_lag_mask_long(
    df: pd.DataFrame,
    max_spatial_gap_mph: float,
    wave_speed_mph: Optional[float],
    time_col: str = "datetime",
) -> np.ndarray:
    spatial_mask_long = np.ones(len(df), dtype=np.int8)
    if time_col not in df.columns:
        return spatial_mask_long
    def time_key(value):
        if time_col == "datetime":
            return pd.Timestamp(value)
        return value

    for corr, grp in df.groupby("corridor", sort=False):
        panel = build_corridor_panel(grp, value_col="speed_mph_raw", time_col=time_col)
        speed_field = panel["speed_field"]
        if speed_field.size == 0:
            continue
        mask = spatial_lag_consistency(
            speed_field,
            road_order=panel.get("road_order"),
            time_axis=panel.get("time_axis"),
            wave_speed_mph=wave_speed_mph,
            max_gap_mph=max_spatial_gap_mph,
        )
        sensor_idx = {sid: i for i, sid in enumerate(panel["sensor_ids"])}
        time_idx = {time_key(t): i for i, t in enumerate(panel["time_axis"])}
        for row_idx, sid, ts in zip(grp.index, grp["sensor_uid"], grp[time_col]):
            si = sensor_idx.get(sid)
            ti = time_idx.get(time_key(ts))
            if si is not None and ti is not None:
                spatial_mask_long[row_idx] = mask[ti, si]
        LOGGER.info("Stage 1 spatial-lag QC for corridor %s: conservative all-pass scaffold", corr)
    return spatial_mask_long


def run_qc(
    df_raw: pd.DataFrame,
    v_f_mph: float = 75.0,
    v_max_factor: float = 1.25,
    v_max_abs: float = 90.0,
    dv_max: float = 30.0,
    hampel_window: int = 11,
    hampel_sigma: float = 3.0,
    max_spatial_gap_mph: float = 20.0,
    wave_speed_mph: Optional[float] = None,
    time_col: str = "datetime",
    dataset_kind: str = "raw",
) -> tuple[pd.DataFrame, dict]:
    """Apply refreshed Stage 1 QC and return ``(dataframe, summary)``."""
    required = {"sensor_uid", "speed_mph", time_col}
    missing = sorted(required - set(df_raw.columns))
    if missing:
        LOGGER.error("Stage 1 missing required input columns: %s", missing)
        raise ValueError(f"Stage 1 input is missing required columns: {missing}")

    LOGGER.info(
        "Stage 1 %s QC start: rows=%s sensors=%s time_col=%s",
        dataset_kind,
        len(df_raw),
        df_raw["sensor_uid"].nunique(),
        time_col,
    )
    df = df_raw.sort_values(["sensor_uid", time_col]).reset_index(drop=True).copy()
    df["stage1_dataset_kind"] = dataset_kind
    df["speed_mph_raw"] = df["speed_mph"].astype(float)
    df["qc_hard_range"] = 1
    df["qc_hampel"] = 1
    df["hampel_replacement"] = np.nan
    df["qc_jump"] = 1

    chunks = []
    for sid, grp in df.groupby("sensor_uid", sort=False):
        speed = grp["speed_mph_raw"].to_numpy(dtype=float)
        if "corridor_freeflow_speed_mph" in grp.columns:
            corridor_vf = pd.to_numeric(grp["corridor_freeflow_speed_mph"], errors="coerce").to_numpy(dtype=float)
            hard_range_vf = np.where(np.isfinite(corridor_vf) & (corridor_vf > 0), corridor_vf, v_f_mph)
        else:
            hard_range_vf = v_f_mph
        hampel_mask, hampel_replacement = hampel_detector(
            speed,
            window=hampel_window,
            n_sigma=hampel_sigma,
        )
        chunks.append(pd.DataFrame({
            "_idx": grp.index,
            "qc_hard_range": hard_physical_range(
                speed,
                v_f_mph=hard_range_vf,
                v_max_factor=v_max_factor,
                v_max_abs=v_max_abs,
            ),
            "qc_hampel": hampel_mask,
            "hampel_replacement": hampel_replacement,
            "qc_jump": temporal_jump_check(speed, dv_max=dv_max),
        }))
        LOGGER.debug("Stage 1 sensor %s: %s rows processed", sid, len(grp))

    if chunks:
        checks = pd.concat(chunks, ignore_index=True).set_index("_idx")
        df.loc[checks.index, checks.columns] = checks

    df["qc_spatial_lag"] = _spatial_lag_mask_long(
        df,
        max_spatial_gap_mph=max_spatial_gap_mph,
        wave_speed_mph=wave_speed_mph,
        time_col=time_col,
    )

    for col in STAGE1_FLAG_COLUMNS:
        df[col] = df[col].astype(np.int8)
    df["qc_pass"] = (
        df["qc_hard_range"]
        & df["qc_hampel"]
        & df["qc_jump"]
        & df["qc_spatial_lag"]
    ).astype(np.int8)

    df["speed_mph_clean"] = np.where(df["qc_pass"] == 1, df["speed_mph_raw"], np.nan)
    df["speed_mph_clean_repaired"] = np.nan
    df["qc_repair_method"] = "none"
    repaired_chunks = []
    method_chunks = []
    for _, grp in df.groupby("sensor_uid", sort=False):
        repaired, methods = _repair_isolated_failures(grp, time_col=time_col)
        repaired_chunks.append(pd.Series(repaired, index=grp.index))
        method_chunks.append(pd.Series(methods, index=grp.index))
    if repaired_chunks:
        repaired = pd.concat(repaired_chunks).sort_index()
        df.loc[repaired.index, "speed_mph_clean_repaired"] = repaired.to_numpy()
    if method_chunks:
        methods = pd.concat(method_chunks).sort_index()
        df.loc[methods.index, "qc_repair_method"] = methods.to_numpy()

    repaired_failed = df["qc_pass"].eq(0) & np.isfinite(df["speed_mph_clean_repaired"])
    df["qc_repaired_flag"] = repaired_failed.astype(np.int8)
    df.loc[df["qc_repaired_flag"].eq(0), "qc_repair_method"] = "none"
    df["qc_pass_repaired"] = df["qc_pass"].copy()
    df.loc[df["qc_repaired_flag"].eq(1), "qc_pass_repaired"] = 1
    df["qc_pass_repaired"] = df["qc_pass_repaired"].astype(np.int8)

    summary = _summarize_stage1(df)
    summary["dataset_kind"] = dataset_kind
    summary["time_col"] = time_col
    LOGGER.info(
        "Stage 1 %s QC end: rows=%s sensors=%s qc_pass_rate=%.3f qc_pass_repaired_rate=%.3f",
        dataset_kind,
        summary["n_rows"],
        summary["n_sensors"],
        summary["qc_pass_rate"],
        summary["qc_pass_repaired_rate"],
    )
    return df, summary


def _summarize_stage1(df: pd.DataFrame) -> dict:
    flag_cols = STAGE1_FLAG_COLUMNS + ["qc_pass"]
    failed = df["qc_pass"] == 0
    repaired = df["qc_repaired_flag"] == 1
    linear = df["qc_repair_method"].eq("linear_interpolation")
    hampel = df["qc_repair_method"].eq("hampel_fallback")
    consecutive_failed = 0
    for _, grp in df.groupby("sensor_uid", sort=False):
        is_nan = grp["speed_mph_clean"].isna().to_numpy()
        if not is_nan.any():
            continue
        starts = np.flatnonzero(is_nan & np.r_[True, ~is_nan[:-1]])
        ends = np.flatnonzero(is_nan & np.r_[~is_nan[1:], True])
        consecutive_failed += int(
            sum((end - start + 1) for start, end in zip(starts, ends) if end - start + 1 >= 2)
        )

    return {
        "n_rows": int(len(df)),
        "n_sensors": int(df["sensor_uid"].nunique()) if "sensor_uid" in df else 0,
        "n_corridors": int(df["corridor"].nunique()) if "corridor" in df else 0,
        "qc_pass_rate": float(df["qc_pass"].mean()) if len(df) else float("nan"),
        "qc_pass_repaired_rate": float(df["qc_pass_repaired"].mean()) if len(df) else float("nan"),
        "per_check_pass_rate": {col: float(df[col].mean()) if len(df) else float("nan") for col in flag_cols},
        "n_failed": int(failed.sum()),
        "n_isolated_repaired": int(repaired.sum()),
        "n_repaired_linear_interpolation": int(linear.sum()),
        "n_repaired_hampel_fallback": int(hampel.sum()),
        "n_consecutive_failed_left_nan": int(consecutive_failed),
        "required_stage1_columns": [
            "speed_mph_raw",
            "qc_hard_range",
            "qc_hampel",
            "hampel_replacement",
            "qc_jump",
            "qc_spatial_lag",
            "qc_pass",
            "speed_mph_clean",
            "speed_mph_clean_repaired",
            "qc_repaired_flag",
            "qc_repair_method",
            "qc_pass_repaired",
        ],
    }


def write_stage1(
    df_stage1: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    *,
    save_csv_outputs: bool = False,
) -> None:
    """Write Stage 1 outputs under the requested stage folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = out_dir / "per_sensor"
    parquet_dir.mkdir(exist_ok=True)

    for sid, grp in df_stage1.groupby("sensor_uid", sort=False):
        safe = str(sid).replace(":", "_").replace("/", "_").replace("\\", "_")
        grp.to_parquet(parquet_dir / f"{safe}.parquet", index=False)
        if save_csv_outputs:
            grp.to_csv(parquet_dir / f"{safe}.csv", index=False)

    tmp = df_stage1.assign(
        _isolated_repaired=df_stage1["qc_repaired_flag"].astype(int),
        _linear=df_stage1["qc_repair_method"].eq("linear_interpolation").astype(int),
        _hampel=df_stage1["qc_repair_method"].eq("hampel_fallback").astype(int),
        _failed=df_stage1["qc_pass"].eq(0).astype(int),
    )
    pass_by_sensor = (
        tmp.groupby("sensor_uid")
        .agg(
            n_rows=("qc_pass", "size"),
            qc_pass_rate=("qc_pass", "mean"),
            qc_pass_repaired_rate=("qc_pass_repaired", "mean"),
            n_failed=("_failed", "sum"),
            hard_range_pass_rate=("qc_hard_range", "mean"),
            hampel_pass_rate=("qc_hampel", "mean"),
            jump_pass_rate=("qc_jump", "mean"),
            spatial_lag_pass_rate=("qc_spatial_lag", "mean"),
            n_isolated_repaired=("_isolated_repaired", "sum"),
            n_repaired_linear_interpolation=("_linear", "sum"),
            n_repaired_hampel_fallback=("_hampel", "sum"),
        )
        .reset_index()
    )
    pass_by_sensor.to_csv(out_dir / "stage1_qc_summary_by_sensor.csv", index=False)
    df_stage1.to_parquet(out_dir / "stage1_qc_long.parquet", index=False)
    if save_csv_outputs:
        df_stage1.to_csv(out_dir / "stage1_qc_long.csv", index=False)
    with open(out_dir / "stage1_qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    LOGGER.info("Stage 1 output files written under %s", out_dir)


__all__ = [
    "hampel_detector",
    "hard_physical_range",
    "run_qc",
    "spatial_lag_consistency",
    "temporal_jump_check",
    "write_stage1",
]
