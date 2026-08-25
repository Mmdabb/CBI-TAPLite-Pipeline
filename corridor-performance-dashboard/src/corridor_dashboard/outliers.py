from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


# HARD rules: physical impossibility or out-of-range measurement.
# One hit excludes the episode outright. NaN-safe: nonfinite inputs are
# caught by the dedicated state_nonfinite rule instead of silently passing
# every ">" comparison as False.
PHYSICAL_RULES = {
    "state_nonfinite": lambda d: ~(
        np.isfinite(pd.to_numeric(d["duration_min"], errors="coerce"))
        & np.isfinite(pd.to_numeric(d["demand_capacity_ratio"], errors="coerce"))
        & np.isfinite(pd.to_numeric(d["min_speed_mph"], errors="coerce"))
    ),
    "duration_exceeds_day": lambda d: d["duration_min"] > 1440.0,
    "doc_above_physical": lambda d: d["demand_capacity_ratio"] > 8.0,
    "speed_out_of_range": lambda d: (d["min_speed_mph"] < 0.0) | (d["min_speed_mph"] > 90.0),
    "mu_nonpositive": lambda d: np.isfinite(
        pd.to_numeric(d["mu_obs_vphpl"], errors="coerce")
    ) & (d["mu_obs_vphpl"] <= 0.0),
    "doc_high_P_zero": lambda d: (d["demand_capacity_ratio"] > 1.0) & (d["duration_min"] < 15.0),
    "mu_above_capacity": lambda d: d["mu_obs_vphpl"] > 1.05 * d["per_lane_hourly_capacity"],
    "mu_starved": lambda d: (
        (d["demand_capacity_ratio"] > 1.5)
        & (d["mu_obs_vphpl"] < 0.30 * d["per_lane_hourly_capacity"])
    ),
    "vt2_above_threshold": lambda d: d["min_speed_mph"] >= d["threshold_used"],
    "vt2_too_low": lambda d: d["min_speed_mph"] < 5.0,
}

# SOFT rules (MAD z-scores and Huber residuals) are relational evidence only:
# a single 3-3.5 MAD flag over-excluded legitimate heavy-congestion days, so
# two independent soft flags must agree before an episode is dropped.
MIN_SOFT_FLAGS_TO_EXCLUDE = 2

SOFT_RULE_COLUMNS = (
    "flag_doc_low_P_high",
    "flag_duration_mad_outlier",
    "flag_demand_capacity_ratio_mad_outlier",
    "flag_huber_resid_doc_P",
    "flag_huber_resid_doc_mu",
    "flag_huber_resid_P_vt2",
)


def _robust_z(values: pd.Series, reference: pd.Series) -> pd.Series:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    if len(reference) < 10:
        return pd.Series(np.nan, index=values.index)
    median = reference.median()
    mad = (reference - median).abs().median() * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(np.nan, index=values.index)
    return (pd.to_numeric(values, errors="coerce") - median) / mad


def _huber_flags(x, y, *, log_x=False, log_y=False, threshold=3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Non-positive values under a log axis are excluded from the fit (hard
    # range rules catch them) instead of being clamped to log(1e-6) = -13.8,
    # which previously dragged the robust fit toward the clamp point.
    if log_x:
        tx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
    else:
        tx = x
    if log_y:
        ty = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
    else:
        ty = y
    valid = np.isfinite(tx) & np.isfinite(ty)
    flags = np.zeros(len(x), dtype=bool)
    if valid.sum() < 8:
        return flags
    model = HuberRegressor(max_iter=500).fit(tx[valid].reshape(-1, 1), ty[valid])
    residual = ty[valid] - model.predict(tx[valid].reshape(-1, 1))
    mad = np.median(np.abs(residual - np.median(residual))) * 1.4826
    if np.isfinite(mad) and mad > 0:
        flags[np.where(valid)[0][np.abs(residual) > threshold * mad]] = True
    return flags


def apply_outlier_screen(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes.copy()
    pieces = []
    for (_detector, _corridor), group in episodes.groupby(["detector", "corridor"], sort=False):
        out = group.copy()
        hard_columns = []
        for name, rule in PHYSICAL_RULES.items():
            col = f"flag_{name}"
            out[col] = rule(out).fillna(False)
            hard_columns.append(col)
        # Speed-only corridors have no measured mu; keep the flag semantics of
        # PHYSICAL_RULES (only finite non-positive mu is flagged).

        out["flag_doc_low_P_high"] = (
            (out["demand_capacity_ratio"] < 0.5) & (out["duration_min"] > 60.0)
        ).fillna(False)

        valid = out["is_valid_for_mu"].fillna(False).astype(bool)
        out["duration_mad_zscore"] = _robust_z(out["duration_min"], out.loc[valid, "duration_min"])
        out["demand_capacity_ratio_mad_zscore"] = _robust_z(
            out["demand_capacity_ratio"], out.loc[valid, "demand_capacity_ratio"]
        )
        out["flag_duration_mad_outlier"] = out["duration_mad_zscore"] > 3.5
        out["flag_demand_capacity_ratio_mad_outlier"] = out["demand_capacity_ratio_mad_zscore"] > 3.5

        out["flag_huber_resid_doc_P"] = _huber_flags(
            out["demand_capacity_ratio"], out["duration_min"] / 60.0, log_x=True, log_y=True
        )
        out["flag_huber_resid_doc_mu"] = _huber_flags(
            out["demand_capacity_ratio"], out["mu_obs_vphpl"]
        )
        out["flag_huber_resid_P_vt2"] = _huber_flags(
            out["duration_min"] / 60.0, out["min_speed_mph"], log_x=True, log_y=True
        )

        soft_columns = [c for c in SOFT_RULE_COLUMNS if c in out.columns]
        reason_columns = hard_columns + soft_columns
        out["n_hard_flags"] = out[hard_columns].fillna(False).sum(axis=1)
        out["n_soft_flags"] = out[soft_columns].fillna(False).sum(axis=1)
        # Two-tier decision: hard hit -> out; otherwise >= 2 agreeing soft flags.
        out["measured_outlier_flag"] = (
            (out["n_hard_flags"] > 0)
            | (out["n_soft_flags"] >= MIN_SOFT_FLAGS_TO_EXCLUDE)
        )
        out["measured_outlier_reasons"] = out.apply(
            lambda row: ";".join(col.replace("flag_", "") for col in reason_columns if bool(row[col])),
            axis=1,
        )
        out["is_clean_valid_episode"] = valid & ~out["measured_outlier_flag"]
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["detector", "corridor", "sensor_uid", "date", "t0_timestamp"]
    ).reset_index(drop=True)
