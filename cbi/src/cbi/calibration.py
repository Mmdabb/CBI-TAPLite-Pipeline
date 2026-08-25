from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import PipelineSettings

FIT_CATALOG_COLUMNS = [
    "data_basis",
    "calibration_scope",
    "detector",
    "corridor",
    "sensor_uid",
    "calibration_period",
    "vc_mph",
    "n_candidate_episodes",
    "f_d",
    "n",
    "f_p",
    "s",
    "alpha",
    "beta",
    "n_episodes",
    "duration_r2",
    "speed_r2",
    "duration_rmse_hr",
    "speed_rmse_mph",
    "duration_bound_active",
    "speed_bound_active",
    "status",
]

SELECTED_CALIBRATION_COLUMNS = [
    "data_basis",
    "detector",
    "corridor",
    "sensor_uid",
    "link_id",
    "period",
    "calibration_scope",
    "f_d",
    "n",
    "f_p",
    "s",
    "alpha",
    "beta",
    "n_episodes",
    "duration_r2",
    "speed_r2",
    "duration_rmse_hr",
    "speed_rmse_mph",
    "duration_bound_active",
    "speed_bound_active",
    "reliability",
]


@dataclass(frozen=True)
class QVDFCalibration:
    f_d: float
    n: float
    f_p: float
    s: float
    alpha: float
    beta: float
    n_episodes: int
    duration_r2: float
    speed_r2: float
    duration_rmse_hr: float
    speed_rmse_mph: float
    duration_bound_active: bool
    speed_bound_active: bool
    status: str

    def to_dict(self):
        return asdict(self)


def _r2(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    denominator = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((pred - obs) ** 2) / denominator) if denominator > 0 else np.nan


def _rmse(obs, pred):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(obs)) ** 2)))


def fit_qvdf(z, duration_hr, vt2, vc, min_episodes=3) -> QVDFCalibration:
    z, duration_hr, vt2 = map(lambda x: np.asarray(x, dtype=float), (z, duration_hr, vt2))
    duration_mask = np.isfinite(z) & np.isfinite(duration_hr) & (z > 0) & (duration_hr > 0)
    reduction = float(vc) / np.maximum(vt2, 1e-6) - 1.0
    speed_mask = duration_mask & np.isfinite(reduction) & (reduction > 1e-6)
    n_fit = int(min(duration_mask.sum(), speed_mask.sum()))
    if n_fit < min_episodes:
        return QVDFCalibration(*([np.nan] * 6), n_fit, *([np.nan] * 4), False, False, "insufficient_sample")

    xd, yd = np.log(z[duration_mask]), np.log(duration_hr[duration_mask])
    slope = np.polyfit(xd, yd, 1)[0] if len(np.unique(xd)) > 1 else 1.1
    intercept = np.median(yd - max(slope, 1.0) * xd)
    fit_d = least_squares(
        lambda p: p[0] + p[1] * xd - yd,
        x0=[np.clip(intercept, -4, 2), np.clip(slope, 1.001, 4.5)],
        bounds=([-5.0, 1.0], [3.0, 5.0]),
        loss="soft_l1", f_scale=0.20,
    )
    f_d, n = float(np.exp(fit_d.x[0])), float(fit_d.x[1])

    xs, ys = np.log(duration_hr[speed_mask]), np.log(reduction[speed_mask])
    slope_s = np.polyfit(xs, ys, 1)[0] if len(np.unique(xs)) > 1 else 1.0
    intercept_s = np.median(ys - slope_s * xs)
    fit_s = least_squares(
        lambda p: p[0] + p[1] * xs - ys,
        x0=[np.clip(intercept_s, -8, 3), np.clip(slope_s, 0.1, 3.9)],
        bounds=([-10.0, 0.05], [5.0, 4.0]),
        loss="soft_l1", f_scale=0.25,
    )
    f_p, s = float(np.exp(fit_s.x[0])), float(fit_s.x[1])
    p_pred = f_d * z[duration_mask] ** n
    v_pred = float(vc) / (1.0 + f_p * duration_hr[speed_mask] ** s)
    return QVDFCalibration(
        f_d, n, f_p, s, float((8 / 15) * f_p * f_d**s), float(n * s), n_fit,
        _r2(duration_hr[duration_mask], p_pred), _r2(vt2[speed_mask], v_pred),
        _rmse(duration_hr[duration_mask], p_pred), _rmse(vt2[speed_mask], v_pred),
        bool(np.isclose(n, 1, atol=1e-3) or np.isclose(n, 5, atol=1e-3)),
        bool(np.isclose(s, 0.05, atol=1e-3) or np.isclose(s, 4, atol=1e-3)),
        "ok",
    )


def calibrate_groups(
    episodes: pd.DataFrame,
    min_episodes: int = 3,
    *,
    data_basis: str,
) -> pd.DataFrame:
    clean = episodes[episodes["is_clean_valid_episode"].fillna(False)].copy()
    clean["P_hr"] = clean["duration_min"] / 60.0
    clean["z"] = clean["demand_capacity_ratio"]
    rows = []
    groupings = {
        "sensor_period": ["detector", "corridor", "sensor_uid", "calibration_period"],
        "corridor_period": ["detector", "corridor", "calibration_period"],
    }
    for scope, columns in groupings.items():
        for key, group in clean.groupby(columns, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(columns, key))
            vc = float(group["threshold_used"].median())
            fit = fit_qvdf(group["z"], group["P_hr"], group["min_speed_mph"], vc, min_episodes)
            rows.append({
                "data_basis": data_basis,
                "calibration_scope": scope,
                **labels,
                "sensor_uid": labels.get("sensor_uid", "ALL"),
                "vc_mph": vc,
                "n_candidate_episodes": int(len(group)),
                **fit.to_dict(),
            })
    return pd.DataFrame(rows, columns=FIT_CATALOG_COLUMNS)


def calibrate_episodes(
    episodes: pd.DataFrame,
    settings: PipelineSettings,
    *,
    data_basis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the refreshed robust model and select one authoritative fit per link-period.

    Sensor-period fits are preferred.  When a sensor has too few clean episodes,
    the fit produced by the identical model at corridor-period scope is used.
    No legacy calibration or transferred default is used.
    """

    if episodes.empty:
        return (
            pd.DataFrame(columns=FIT_CATALOG_COLUMNS),
            pd.DataFrame(columns=SELECTED_CALIBRATION_COLUMNS),
        )
    fitted = calibrate_groups(
        episodes,
        settings.minimum_calibration_episodes,
        data_basis=data_basis,
    )
    if fitted.empty:
        return fitted, pd.DataFrame(columns=SELECTED_CALIBRATION_COLUMNS)
    clean = episodes[episodes["is_clean_valid_episode"].fillna(False)].copy()
    candidates = (
        clean[["detector", "corridor", "sensor_uid", "link_id", "calibration_period"]]
        .drop_duplicates()
        .sort_values(["link_id", "calibration_period"])
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        sensor_fit = fitted[
            fitted["calibration_scope"].eq("sensor_period")
            & fitted["detector"].eq(candidate.detector)
            & fitted["corridor"].eq(candidate.corridor)
            & fitted["sensor_uid"].eq(candidate.sensor_uid)
            & fitted["calibration_period"].eq(candidate.calibration_period)
            & fitted["status"].eq("ok")
        ]
        corridor_fit = fitted[
            fitted["calibration_scope"].eq("corridor_period")
            & fitted["detector"].eq(candidate.detector)
            & fitted["corridor"].eq(candidate.corridor)
            & fitted["calibration_period"].eq(candidate.calibration_period)
            & fitted["status"].eq("ok")
        ]
        if not sensor_fit.empty:
            selected = sensor_fit.iloc[0]
            scope = "sensor_period"
        elif not corridor_fit.empty:
            selected = corridor_fit.iloc[0]
            scope = "corridor_period"
        else:
            continue
        bound_active = bool(
            selected["duration_bound_active"] or selected["speed_bound_active"]
        )
        n_episodes = int(selected["n_episodes"])
        if bound_active or n_episodes < 5:
            reliability = "low"
        elif n_episodes >= 10:
            reliability = "high"
        else:
            reliability = "medium"
        rows.append(
            {
                "data_basis": data_basis,
                "detector": candidate.detector,
                "corridor": candidate.corridor,
                "sensor_uid": candidate.sensor_uid,
                "link_id": int(candidate.link_id),
                "period": candidate.calibration_period,
                "calibration_scope": scope,
                "f_d": float(selected["f_d"]),
                "n": float(selected["n"]),
                "f_p": float(selected["f_p"]),
                "s": float(selected["s"]),
                "alpha": float(selected["alpha"]),
                "beta": float(selected["beta"]),
                "n_episodes": n_episodes,
                "duration_r2": float(selected["duration_r2"]),
                "speed_r2": float(selected["speed_r2"]),
                "duration_rmse_hr": float(selected["duration_rmse_hr"]),
                "speed_rmse_mph": float(selected["speed_rmse_mph"]),
                "duration_bound_active": bool(selected["duration_bound_active"]),
                "speed_bound_active": bool(selected["speed_bound_active"]),
                "reliability": reliability,
            }
        )
    return fitted, pd.DataFrame(rows, columns=SELECTED_CALIBRATION_COLUMNS)


def calibration_lookup(applied: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    if applied.empty:
        return {}
    return {
        (int(row.link_id), str(row.period)): row._asdict()
        for row in applied.itertuples(index=False)
    }
