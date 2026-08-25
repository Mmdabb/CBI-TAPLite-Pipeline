from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


LOGGER = logging.getLogger("cbi")


def s3_speed(k, vf: float, kc: float, m: float):
    k = np.asarray(k, dtype=float)
    return vf / np.power(1.0 + np.power(np.maximum(k, 0.0) / kc, m), 2.0 / m)


def s3_flow(k, vf: float, kc: float, m: float):
    k = np.asarray(k, dtype=float)
    return k * s3_speed(k, vf, kc, m)


def inverse_s3_flow(speed, vf: float, kc: float, m: float, capacity: float):
    speed = np.asarray(speed, dtype=float)
    out = np.full_like(speed, np.nan)
    valid = np.isfinite(speed) & (speed >= 0)
    clipped = np.minimum(speed[valid], 0.99 * vf)
    ratio = np.maximum(vf / np.maximum(clipped, 1.0), 1.00001)
    density = kc * np.power(np.maximum(np.power(ratio, m / 2.0) - 1.0, 1e-8), 1.0 / m)
    out[valid] = np.minimum(clipped * density, capacity)
    return out


def _diagnostics(q_obs, q_pred) -> tuple[float, float]:
    q_obs = np.asarray(q_obs, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((q_pred - q_obs) ** 2)))
    denominator = float(np.sum((q_obs - q_obs.mean()) ** 2))
    r2 = float(1.0 - np.sum((q_pred - q_obs) ** 2) / denominator) if denominator > 0 else np.nan
    return r2, rmse


def fit_measured_s3(group: pd.DataFrame) -> dict:
    valid = group[
        group["qc_pass_repaired"].astype(bool)
        & pd.to_numeric(group["density_vpm"], errors="coerce").gt(0)
        & pd.to_numeric(group["flow_vph"], errors="coerce").gt(0)
    ].copy()
    k = pd.to_numeric(valid["density_vpm"], errors="coerce").to_numpy(dtype=float)
    q = pd.to_numeric(valid["flow_vph"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(k) & np.isfinite(q) & (k > 0) & (q > 0)
    k, q = k[mask], q[mask]
    if len(k) < 50:
        return {"fit_success": False, "fit_error": "fewer than 50 valid q-k observations"}

    speed = pd.to_numeric(valid["speed_mph_clean_repaired"], errors="coerce").to_numpy(dtype=float)[mask]
    vf0 = float(np.clip(np.nanpercentile(speed, 95), 45.0, 85.0))
    cap0 = float(np.nanpercentile(q, 99))
    kc0 = float(np.clip(cap0 / max(0.70 * vf0, 1.0), 10.0, 100.0))
    q_scale = max(float(np.nanmedian(q)), 100.0)
    v_scale = 10.0
    vf_lo = max(40.0, 0.90 * vf0)
    vf_hi = min(90.0, 1.10 * vf0)
    fit = least_squares(
        lambda par: np.concatenate([
            (s3_flow(k, *par) - q) / q_scale,
            (s3_speed(k, *par) - speed) / v_scale,
        ]),
        x0=np.array([vf0, kc0, 4.0]),
        bounds=(np.array([vf_lo, 5.0, 0.5]), np.array([vf_hi, 150.0, 15.0])),
        loss="soft_l1",
        f_scale=0.20,
        max_nfev=3000,
    )
    vf, kc, m = map(float, fit.x)
    vc = float(vf / (2.0 ** (2.0 / m)))
    capacity = float(kc * vc)
    pred = s3_flow(k, vf, kc, m)
    r2, rmse = _diagnostics(q, pred)
    bound_active = bool(
        np.isclose(vf, vf_lo, atol=0.05) or np.isclose(vf, vf_hi, atol=0.05)
        or np.isclose(kc, 5.0, atol=0.05) or np.isclose(kc, 150.0, atol=0.05)
        or np.isclose(m, 0.5, atol=0.01) or np.isclose(m, 15.0, atol=0.01)
    )
    return {
        "fit_success": bool(fit.success),
        "fit_error": "" if fit.success else str(fit.message),
        "vf_mph": vf,
        "kc_vpmpl": kc,
        "s3_m": m,
        "vc_mph": vc,
        "capacity_vphpl": capacity,
        "n_obs": int(len(k)),
        "r2": r2,
        "rmse_vphpl": rmse,
        "bound_active": bound_active,
        "fd_source": "robust_measured_s3",
    }


def synthetic_s3_context(
    group: pd.DataFrame,
    capacity_vphpl: float | None = None,
    default_free_flow_mph: float | None = None,
) -> dict:
    """Build speed-only S3 context after QC, using per-link physical inputs."""

    speed_column = (
        "speed_mph_clean_repaired"
        if "speed_mph_clean_repaired" in group
        else "speed_mph"
    )
    speed = pd.to_numeric(group[speed_column], errors="coerce")
    p95 = float(speed.quantile(0.95))
    reference = pd.to_numeric(group.get("reference_speed_mph"), errors="coerce").median()
    candidates = [p95, reference]
    if default_free_flow_mph is not None:
        candidates.append(float(default_free_flow_mph))
    vf = float(np.clip(np.nanmax(candidates), 35.0, 85.0))
    vc = 0.70 * vf
    m = float(2.0 * np.log(2.0) / np.log(vf / vc))
    if capacity_vphpl is None:
        capacity = pd.to_numeric(
            group.get("capacity_prior_vphpl"), errors="coerce"
        ).median()
    else:
        capacity = float(capacity_vphpl)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("Speed-only S3 context requires a positive per-lane capacity")
    kc = float(capacity / vc)
    return {
        "fit_success": True,
        "fit_error": "",
        "vf_mph": vf,
        "kc_vpmpl": kc,
        "s3_m": m,
        "vc_mph": vc,
        "capacity_vphpl": float(capacity),
        "n_obs": int(speed.notna().sum()),
        "r2": np.nan,
        "rmse_vphpl": np.nan,
        "bound_active": False,
        "fd_source": "reference_plus_observed_inverse_s3",
        "reference_speed_median_mph": float(reference) if np.isfinite(reference) else np.nan,
        "observed_speed_p95_mph": p95,
    }


def prepare_synthetic_flow(
    raw: pd.DataFrame,
    capacity_vphpl: float | None = None,
    default_free_flow_mph: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = raw.copy()
    rows = []
    for sid, group in out.groupby("sensor_uid", sort=False):
        context = synthetic_s3_context(
            group,
            capacity_vphpl=capacity_vphpl,
            default_free_flow_mph=default_free_flow_mph,
        )
        rows.append({"sensor_uid": sid, "corridor": group["corridor"].iloc[0], **context})
        idx = group.index
        speed_column = (
            "speed_mph_clean_repaired"
            if "speed_mph_clean_repaired" in group
            else "speed_mph"
        )
        speed = pd.to_numeric(group[speed_column], errors="coerce")
        flow = inverse_s3_flow(
            speed.to_numpy(dtype=float),
            context["vf_mph"], context["kc_vpmpl"], context["s3_m"], context["capacity_vphpl"],
        )
        out.loc[idx, "flow_vph"] = flow
        out.loc[idx, "density_vpm"] = flow / speed.where(speed > 1.0).to_numpy()
        out.loc[idx, "corridor_freeflow_speed_mph"] = context["vf_mph"]
        out.loc[idx, "fd_capacity_vphpl"] = context["capacity_vphpl"]
        out.loc[idx, "fd_vc_mph"] = context["vc_mph"]
    return out, pd.DataFrame(rows)


def calibrate_measured_fd(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, group in qc.groupby("sensor_uid", sort=False):
        result = fit_measured_s3(group)
        if not result.get("fit_success"):
            vf = float(pd.to_numeric(group["speed_mph_clean_repaired"], errors="coerce").quantile(0.95))
            capacity = float(pd.to_numeric(group["flow_vph"], errors="coerce").quantile(0.99))
            result.update({
                "vf_mph": vf, "vc_mph": 0.70 * vf, "capacity_vphpl": capacity,
                "kc_vpmpl": capacity / max(0.70 * vf, 1.0), "s3_m": np.nan,
                "fd_source": "measured_quantile_estimator", "bound_active": False,
            })
        rows.append({"sensor_uid": sid, "corridor": group["corridor"].iloc[0], **result})
    return pd.DataFrame(rows)


def attach_fd_context(df: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    columns = ["sensor_uid", "vf_mph", "vc_mph", "capacity_vphpl"]
    mapped = fd[columns].rename(columns={
        "vf_mph": "corridor_freeflow_speed_mph",
        "vc_mph": "fd_vc_mph",
        "capacity_vphpl": "fd_capacity_vphpl",
    })
    return df.drop(
        columns=["corridor_freeflow_speed_mph", "fd_vc_mph", "fd_capacity_vphpl"],
        errors="ignore",
    ).merge(mapped, on="sensor_uid", how="left")
