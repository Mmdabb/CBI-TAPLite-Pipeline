from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import calibration_lookup
from .config import PipelineSettings, WEEKDAY_NAME, period_duration_hours
from .reconstruction import (
    conserve_flow_from_speed,
    detection_smoothed_speed,
    predict_duration_hours,
    predict_minimum_speed,
    predicted_bounds_about_t2,
    reconstruct_daily_profile,
)


DAILY_PAQ_COLUMNS = [
    "link_id",
    "sensor_uid",
    "tmc_code",
    "network_link_id",
    "network_from_node_id",
    "network_to_node_id",
    "episode_id",
    "date",
    "weekday",
    "cutoff",
    "cap",
    "uc",
    "vf",
    "length",
    "period",
    "record_type",
    "P",
    "demand",
    "capacity_reference_hours",
    "capacity_volume_veh_per_lane",
    "DC",
    "demand_capacity_basis",
    "demand_is_proxy",
    "v_t2",
    "t2",
    "t0",
    "t3",
    "qdf",
    "cd_mean_speed",
    "magnitude",
    "m",
]

STAGE0_COLUMNS = [
    "link_id",
    "sensor_uid",
    "tmc_code",
    "network_link_id",
    "network_from_node_id",
    "network_to_node_id",
    "network_mapping_status",
    "capacity_vphpl",
    "capacity_source",
    "speed_at_capacity_mph",
    "critical_density_veh_per_mile_lane",
    "free_flow_speed_model_mph",
    "reference_speed_source",
    "observed_speed_p95_mph",
    "observed_speed_p99_mph",
    "congestion_threshold_mph",
    "s3_shape_m",
    "fd_source",
    "fd_fit_r2",
]

TABLE6_COLUMNS = [
    "link_id",
    "period",
    "data_basis",
    "observation_days",
    "congested_days",
    "pct_days_congested",
    "mean_episode_duration_hr",
]

TABLE7_COLUMNS = [
    "link_id",
    "period",
    "calibration_data_basis",
    "calibration_scope",
    "f_d",
    "n",
    "f_p",
    "s",
    "alpha",
    "beta",
    "fit_n_episodes",
    "link_accepted_episodes",
    "reliability",
    "observed_mean_min_speed_mph",
    "modeled_min_speed_mph",
    "min_speed_error_mph",
    "min_speed_error_pct",
]

TABLE8_COLUMNS = [
    "link_id",
    "period",
    "weekday",
    "mean_demand_capacity_ratio",
    "predicted_duration_hr",
    "predicted_min_speed_mph",
    "predicted_mean_speed_mph",
    "gamma",
]

HANDOFF_COLUMNS = [
    "link_id",
    "sensor_uid",
    "tmc_code",
    "network_link_id",
    "from_node_id",
    "to_node_id",
    "t_min",
    "period",
    "speed_raw",
    "speed_smoothed",
    "speed_qvdf_model",
    "count_per_lane_15min",
    "lanes",
    "lanes_source",
    "count_total_15min",
    "length_mi",
    "free_flow_speed_model_mph",
    "congestion_threshold_mph",
    "capacity_vphpl",
    "capacity_source",
    "flow_source",
    "vmt",
    "vht",
    "vdt",
    "vcdt",
    "emis_co2_g_obs",
    "emis_co2_g_model",
    "emissions_method",
]

CONSERVED_FLOW_COLUMNS = [
    "link_id",
    "sensor_uid",
    "tmc_code",
    "network_link_id",
    "t_min",
    "period",
    "qvdf_flow_vphpl",
    "qvdf_count_total_15min",
    "critical_density_veh_per_mile_lane",
    "additive_flow_adjustment_vphpl",
    "relative_flow_adjustment",
]


def stage0_table(
    fd: pd.DataFrame,
    observations: pd.DataFrame,
    settings: PipelineSettings,
) -> pd.DataFrame:
    metadata = observations.groupby("sensor_uid", as_index=False).agg(
        link_id=("link_id", "first"),
        tmc_code=("tmc_code", "first"),
        network_link_id=("network_link_id", "first"),
        network_from_node_id=("network_from_node_id", "first"),
        network_to_node_id=("network_to_node_id", "first"),
        network_mapping_status=("network_mapping_status", "first"),
        capacity_source=("capacity_source", "first"),
        reference_speed_source=("reference_speed_source", "first"),
        observed_p99=("speed_mph", lambda values: float(pd.to_numeric(values, errors="coerce").quantile(0.99))),
    )
    table = fd.merge(metadata, on="sensor_uid", how="left")
    return pd.DataFrame(
        {
            "link_id": table["link_id"].astype(int),
            "sensor_uid": table["sensor_uid"].astype(str),
            "tmc_code": table["tmc_code"].astype(str),
            "network_link_id": table["network_link_id"],
            "network_from_node_id": table["network_from_node_id"],
            "network_to_node_id": table["network_to_node_id"],
            "network_mapping_status": table["network_mapping_status"],
            "capacity_vphpl": table["capacity_vphpl"].round(1),
            "capacity_source": table["capacity_source"],
            "speed_at_capacity_mph": table["vc_mph"].round(1),
            "critical_density_veh_per_mile_lane": table[
                "kc_vpmpl"
            ].round(1),
            "free_flow_speed_model_mph": table["vf_mph"].round(1),
            "reference_speed_source": table["reference_speed_source"],
            "observed_speed_p95_mph": table.get(
                "observed_speed_p95_mph",
                pd.Series(np.nan, index=table.index),
            ).round(1),
            "observed_speed_p99_mph": table["observed_p99"].round(1),
            "congestion_threshold_mph": (
                settings.cutoff_ratio * table["vf_mph"]
            ).round(1),
            "s3_shape_m": table["s3_m"].round(2),
            "fd_source": table["fd_source"],
            "fd_fit_r2": table.get(
                "r2", pd.Series(np.nan, index=table.index)
            ).round(3),
        }
    )[STAGE0_COLUMNS].sort_values("link_id")


def _fd_by_link(fd: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    metadata = observations.groupby("sensor_uid", as_index=False).agg(
        link_id=("link_id", "first"),
        tmc_code=("tmc_code", "first"),
        network_link_id=("network_link_id", "first"),
        network_from_node_id=("network_from_node_id", "first"),
        network_to_node_id=("network_to_node_id", "first"),
        network_mapping_status=("network_mapping_status", "first"),
        length_mi=("length_mi", "first"),
        lanes=("lanes", "first"),
        lanes_source=("lanes_source", "first"),
        capacity_source=("capacity_source", "first"),
    )
    return fd.merge(metadata, on="sensor_uid", how="left").set_index("link_id")


def attach_link_identifiers(
    table: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Add durable TMC and mapped-network identifiers to a link-level table."""

    if "link_id" not in table:
        return table.copy()
    identifiers = (
        observations.groupby("link_id", as_index=False)
        .agg(
            sensor_uid=("sensor_uid", "first"),
            tmc_code=("tmc_code", "first"),
            network_link_id=("network_link_id", "first"),
            network_from_node_id=("network_from_node_id", "first"),
            network_to_node_id=("network_to_node_id", "first"),
            network_mapping_status=("network_mapping_status", "first"),
        )
    )
    missing = [
        column
        for column in identifiers.columns
        if column == "link_id" or column not in table.columns
    ]
    return table.merge(identifiers[missing], on="link_id", how="left")


def format_episode_table(
    episodes: pd.DataFrame,
    observations: pd.DataFrame,
    fd: pd.DataFrame,
    settings: PipelineSettings,
    *,
    average_weekday: bool = False,
) -> pd.DataFrame:
    """Translate core episodes to the established corridor daily/average schema.

    The historical ``daily_paq_all.csv`` is actually an average-weekday table:
    links with detected congestion contain only their detected periods, while
    links with no detected congestion contain zero rows for all three periods.
    Daily calibration still uses the complete multiday episode table internally.
    """

    contexts = _fd_by_link(fd, observations)
    rows: list[dict[str, object]] = []
    detected_keys: set[tuple[int, str, str]] = set()
    accepted = (
        episodes[episodes["is_clean_valid_episode"].fillna(False)]
        if not episodes.empty
        and "is_clean_valid_episode" in episodes
        else episodes
    )
    if not accepted.empty:
        for episode in accepted.itertuples(index=False):
            link_id = int(episode.link_id)
            date = str(episode.date)
            period = str(episode.period)
            detected_keys.add((link_id, date, period))
            context = contexts.loc[link_id]
            rows.append(
                {
                    "link_id": link_id,
                    "sensor_uid": str(episode.sensor_uid),
                    "tmc_code": str(episode.tmc_code),
                    "network_link_id": getattr(
                        episode, "network_link_id", np.nan
                    ),
                    "network_from_node_id": getattr(
                        episode, "network_from_node_id", np.nan
                    ),
                    "network_to_node_id": getattr(
                        episode, "network_to_node_id", np.nan
                    ),
                    "episode_id": str(episode.episode_id),
                    "date": date,
                    "weekday": int(episode.weekday),
                    "cutoff": float(episode.threshold_used),
                    "cap": float(context["capacity_vphpl"]),
                    "uc": float(context["vc_mph"]),
                    "vf": float(context["vf_mph"]),
                    "length": float(episode.length_mi),
                    "period": period,
                    "record_type": "detected_accepted_episode",
                    "P": float(episode.P_hr),
                    "demand": float(episode.episode_demand),
                    "capacity_reference_hours": float(
                        episode.capacity_reference_hours
                    ),
                    "capacity_volume_veh_per_lane": float(
                        episode.capacity_volume_veh_per_lane
                    ),
                    "DC": float(episode.demand_capacity_ratio),
                    "demand_capacity_basis": str(
                        getattr(
                            episode,
                            "demand_capacity_basis",
                            "episode_demand_over_period_capacity",
                        )
                    ),
                    "demand_is_proxy": bool(
                        getattr(episode, "demand_is_proxy", True)
                    ),
                    "v_t2": float(episode.min_speed_mph),
                    "t2": float(episode.t2_hour),
                    "t0": float(episode.t0_hour),
                    "t3": float(episode.t3_hour),
                    "qdf": float(episode.qdf),
                    "cd_mean_speed": float(episode.mean_speed_mph),
                    "magnitude": float(episode.magnitude),
                    "m": float(episode.m),
                }
            )

    work = observations.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    speed_column = (
        "speed_mph_clean_repaired"
        if "speed_mph_clean_repaired" in work
        else "speed_mph"
    )
    work["_speed"] = pd.to_numeric(work[speed_column], errors="coerce")
    work["_date"] = work["date"].astype(str)
    work["_weekday"] = work["weekday"].astype(int)
    group_columns = ["link_id"] if average_weekday else ["link_id", "_date"]
    detected_links = {key[0] for key in detected_keys}
    for group_key, day in work.groupby(group_columns, sort=False):
        if average_weekday:
            link_id = int(group_key[0] if isinstance(group_key, tuple) else group_key)
            date = "Weekday"
            if link_id in detected_links:
                continue
        else:
            link_id, date = group_key
        context = contexts.loc[int(link_id)]
        for period, (start, end) in settings.periods.items():
            key = (int(link_id), str(date), period)
            if key in detected_keys:
                continue
            panel = day[(day["t_min"] >= start) & (day["t_min"] < end)]
            if panel.empty:
                continue
            speed = panel["_speed"]
            minimum_index = speed.idxmin() if speed.notna().any() else panel.index[0]
            demand = float(
                np.nansum(
                    pd.to_numeric(panel["flow_vph"], errors="coerce")
                    * settings.interval_minutes
                    / 60.0
                )
            )
            capacity = float(context["capacity_vphpl"])
            capacity_reference_hours = period_duration_hours(
                period, settings.periods
            )
            capacity_volume = capacity * capacity_reference_hours
            rows.append(
                {
                    "link_id": int(link_id),
                    "sensor_uid": str(context["sensor_uid"]),
                    "tmc_code": str(context["tmc_code"]),
                    "network_link_id": context["network_link_id"],
                    "network_from_node_id": context[
                        "network_from_node_id"
                    ],
                    "network_to_node_id": context[
                        "network_to_node_id"
                    ],
                    "episode_id": "",
                    "date": str(date),
                    "weekday": int(panel["_weekday"].iloc[0]),
                    "cutoff": settings.cutoff_ratio * float(context["vf_mph"]),
                    "cap": capacity,
                    "uc": float(context["vc_mph"]),
                    "vf": float(context["vf_mph"]),
                    "length": float(context["length_mi"]),
                    "period": period,
                    "record_type": (
                        "no_detected_episode_period_summary"
                    ),
                    "P": 0.0,
                    "demand": demand,
                    "capacity_reference_hours": (
                        capacity_reference_hours
                    ),
                    "capacity_volume_veh_per_lane": capacity_volume,
                    "DC": (
                        demand / capacity_volume
                        if capacity_volume > 0
                        else np.nan
                    ),
                    "demand_capacity_basis": (
                        "period_demand_over_period_capacity"
                    ),
                    "demand_is_proxy": bool(
                        day["flow_synthetic"].iloc[0]
                        if "flow_synthetic" in day
                        else True
                    ),
                    "v_t2": float(speed.loc[minimum_index]),
                    "t2": float(panel.loc[minimum_index, "t_min"]) / 60.0,
                    "t0": 0.0,
                    "t3": 0.0,
                    "qdf": 1.0,
                    "cd_mean_speed": float(speed.mean()),
                    "magnitude": 0.0,
                    "m": np.nan,
                }
            )
    return (
        pd.DataFrame(rows, columns=DAILY_PAQ_COLUMNS)
        .sort_values(["link_id", "date", "period", "t0"])
        .reset_index(drop=True)
    )


def table6_congestion_stats(
    formatted_episodes: pd.DataFrame,
    scored_episodes: pd.DataFrame,
    settings: PipelineSettings,
    *,
    data_basis: str,
) -> pd.DataFrame:
    clean = (
        scored_episodes[scored_episodes["is_clean_valid_episode"].fillna(False)]
        if not scored_episodes.empty
        else scored_episodes
    )
    rows = []
    for link_id in sorted(formatted_episodes["link_id"].unique()):
        for period in settings.periods:
            base = formatted_episodes[
                formatted_episodes["link_id"].eq(link_id)
                & formatted_episodes["period"].eq(period)
            ]
            if base.empty:
                continue
            positive = (
                clean[
                    clean["link_id"].eq(link_id) & clean["period"].eq(period)
                ]
                if not clean.empty
                else clean
            )
            valid_days = int(base["date"].nunique())
            congested_days = int(positive["date"].nunique()) if not positive.empty else 0
            rows.append(
                [
                    int(link_id),
                    period,
                    data_basis,
                    valid_days,
                    congested_days,
                    round(100.0 * congested_days / valid_days) if valid_days else 0,
                    round(float(positive["P_hr"].mean()), 2)
                    if congested_days
                    else 0.0,
                ]
            )
    return pd.DataFrame(rows, columns=TABLE6_COLUMNS)


def table7_calibrated(
    episodes: pd.DataFrame,
    applied: pd.DataFrame,
) -> pd.DataFrame:
    if applied.empty:
        return pd.DataFrame(columns=TABLE7_COLUMNS)
    clean = episodes[episodes["is_clean_valid_episode"].fillna(False)]
    rows = []
    for fit in applied.itertuples(index=False):
        group = clean[
            clean["link_id"].eq(fit.link_id) & clean["period"].eq(fit.period)
        ]
        if group.empty:
            continue
        duration = float(group["P_hr"].mean())
        observed = float(group["min_speed_mph"].mean())
        modeled = predict_minimum_speed(
            float(group["threshold_used"].median()), duration, fit._asdict()
        )
        error = modeled - observed
        rows.append(
            [
                int(fit.link_id),
                fit.period,
                fit.data_basis,
                fit.calibration_scope,
                round(float(fit.f_d), 4),
                round(float(fit.n), 4),
                round(float(fit.f_p), 4),
                round(float(fit.s), 4),
                round(float(fit.alpha), 4),
                round(float(fit.beta), 4),
                int(fit.n_episodes),
                int(len(group)),
                fit.reliability,
                round(observed, 1),
                round(modeled, 1),
                round(error, 1),
                round(100.0 * error / max(observed, 1.0), 1),
            ]
        )
    return pd.DataFrame(rows, columns=TABLE7_COLUMNS)


def table8_gamma(
    episodes: pd.DataFrame,
    applied: pd.DataFrame,
    fd: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if applied.empty:
        return pd.DataFrame(columns=TABLE8_COLUMNS)
    clean = episodes[episodes["is_clean_valid_episode"].fillna(False)]
    contexts = _fd_by_link(fd, observations)
    rows = []
    for fit in applied.itertuples(index=False):
        group = clean[
            clean["link_id"].eq(fit.link_id) & clean["period"].eq(fit.period)
        ]
        if group.empty:
            continue
        context = contexts.loc[int(fit.link_id)]
        for weekday, weekday_group in group.groupby("weekday"):
            dc = float(weekday_group["demand_capacity_ratio"].mean())
            predicted_duration = predict_duration_hours(dc, fit._asdict())
            cutoff = float(weekday_group["threshold_used"].median())
            minimum = predict_minimum_speed(
                cutoff, predicted_duration, fit._asdict()
            )
            mean_speed = cutoff / (
                1.0 + float(fit.alpha) * max(dc, 1e-6) ** float(fit.beta)
            )
            mu = float(
                pd.to_numeric(
                    weekday_group["mu_obs_vphpl"], errors="coerce"
                ).median()
            )
            if not np.isfinite(mu) or mu <= 0:
                mu = min(
                    float(context["capacity_vphpl"]),
                    float(weekday_group["episode_demand"].mean())
                    / max(float(weekday_group["P_hr"].mean()), 1e-3),
                )
            gamma = (
                64.0
                * mu
                * (float(context["length_mi"]) / float(context["vc_mph"]))
                * float(fit.f_p)
                * max(predicted_duration, 1e-3) ** (float(fit.s) - 4.0)
            )
            label = (
                "Weekday"
                if str(weekday_group["date"].iloc[0]) == "Weekday"
                else WEEKDAY_NAME.get(int(weekday), str(weekday))
            )
            rows.append(
                [
                    int(fit.link_id),
                    fit.period,
                    label,
                    round(dc, 2),
                    round(predicted_duration, 2),
                    round(minimum, 2),
                    round(mean_speed, 2),
                    round(gamma, 2),
                ]
            )
    return pd.DataFrame(rows, columns=TABLE8_COLUMNS)


def _r2(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if len(observed) < 2:
        return np.nan
    total = float(np.sum((observed - observed.mean()) ** 2))
    return (
        float(1.0 - np.sum((observed - predicted) ** 2) / total)
        if total > 0
        else np.nan
    )


def _mae(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return float(np.mean(np.abs(observed[valid] - predicted[valid]))) if valid.any() else np.nan


def _rmse(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return (
        float(np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2)))
        if valid.any()
        else np.nan
    )


def _mape(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & (np.abs(observed) > 1e-6)
    )
    return (
        float(np.mean(np.abs((observed[valid] - predicted[valid]) / observed[valid])) * 100.0)
        if valid.any()
        else np.nan
    )


def calibration_quality(
    episodes: pd.DataFrame,
    applied: pd.DataFrame,
    settings: PipelineSettings,
) -> pd.DataFrame:
    lookup = calibration_lookup(applied)
    clean = (
        episodes[episodes["is_clean_valid_episode"].fillna(False)].copy()
        if not episodes.empty
        else episodes
    )
    rows = []
    for period in settings.periods:
        group = clean[clean["period"].eq(period)] if not clean.empty else clean
        records = []
        for episode in group.itertuples(index=False):
            parameters = lookup.get((int(episode.link_id), period))
            if parameters is None:
                continue
            predicted_duration = predict_duration_hours(
                episode.demand_capacity_ratio, parameters
            )
            predicted_minimum = predict_minimum_speed(
                episode.threshold_used, episode.P_hr, parameters
            )
            predicted_t0, _, predicted_t3 = predicted_bounds_about_t2(
                episode.t0_hour,
                episode.t2_hour,
                episode.t3_hour,
                predicted_duration,
            )
            records.append(
                {
                    "P": episode.P_hr,
                    "predicted_P": predicted_duration,
                    "magnitude": episode.magnitude,
                    "predicted_magnitude": parameters["f_p"]
                    * max(episode.P_hr, 1e-6) ** parameters["s"],
                    "vt2": episode.min_speed_mph,
                    "predicted_vt2": predicted_minimum,
                    "t0": episode.t0_hour,
                    "predicted_t0": predicted_t0,
                    "t3": episode.t3_hour,
                    "predicted_t3": predicted_t3,
                    "link_id": episode.link_id,
                }
            )
        values = pd.DataFrame(records)
        if len(values) < 3:
            continue
        rows.append(
            {
                "period": period,
                "calibration_data_basis": str(
                    applied["data_basis"].iloc[0]
                    if "data_basis" in applied and not applied.empty
                    else ""
                ),
                "demand_capacity_basis": ";".join(
                    sorted(
                        group["demand_capacity_basis"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                )
                if "demand_capacity_basis" in group
                else "",
                "demand_is_proxy": bool(
                    group["demand_is_proxy"].fillna(True).all()
                )
                if "demand_is_proxy" in group
                else True,
                "n_episodes": len(values),
                "n_links": int(values["link_id"].nunique()),
                "step1_DC_P_R2": round(float(np.clip(_r2(values["P"], values["predicted_P"]), -1, 1)), 3),
                "step2_P_mag_R2": round(float(np.clip(_r2(values["magnitude"], values["predicted_magnitude"]), -1, 1)), 3),
                "P_MAE_h": round(_mae(values["P"], values["predicted_P"]), 2),
                "P_MAPE_pct": round(_mape(values["P"], values["predicted_P"]), 1),
                "vt2_MAE_mph": round(_mae(values["vt2"], values["predicted_vt2"]), 1),
                "vt2_MAPE_pct": round(_mape(values["vt2"], values["predicted_vt2"]), 1),
                "t0_MAE_min": round(60.0 * _mae(values["t0"], values["predicted_t0"]), 1),
                "t3_MAE_min": round(60.0 * _mae(values["t3"], values["predicted_t3"]), 1),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "period",
            "calibration_data_basis",
            "demand_capacity_basis",
            "demand_is_proxy",
            "n_episodes",
            "n_links",
            "step1_DC_P_R2",
            "step2_P_mag_R2",
            "P_MAE_h",
            "P_MAPE_pct",
            "vt2_MAE_mph",
            "vt2_MAPE_pct",
            "t0_MAE_min",
            "t3_MAE_min",
        ],
    )


def timeseries_quality(
    observations: pd.DataFrame,
    average: pd.DataFrame,
    settings: PipelineSettings,
) -> pd.DataFrame:
    rows = []
    for link_id, profile in average.groupby("link_id"):
        profile = profile.sort_values("t_min")
        raw = pd.to_numeric(profile["speed_mph"], errors="coerce").to_numpy(dtype=float)
        smoothed = detection_smoothed_speed(raw, settings)
        average_map = dict(zip(profile["t_min"], smoothed))
        daily = observations[observations["link_id"].eq(link_id)].copy()
        daily["datetime"] = pd.to_datetime(daily["datetime"])
        daily["_t_min"] = daily["datetime"].dt.hour * 60 + daily["datetime"].dt.minute
        speed_column = (
            "speed_mph_clean_repaired"
            if "speed_mph_clean_repaired" in daily
            else "speed_mph"
        )
        pivot = daily.pivot_table(
            index="_t_min", columns="date", values=speed_column
        )
        daily_rmse, daily_r2 = [], []
        for date in pivot.columns:
            observed = pivot[date].dropna()
            baseline = np.array(
                [average_map.get(int(t), np.nan) for t in observed.index]
            )
            daily_rmse.append(_rmse(observed, baseline))
            daily_r2.append(_r2(observed, baseline))
        coefficient = (
            float(
                (
                    pivot.std(axis=1)
                    / pivot.mean(axis=1).clip(lower=1.0)
                ).mean()
            )
            if pivot.shape[1] > 1
            else np.nan
        )
        rows.append(
            {
                "link_id": int(link_id),
                "n_weekdays": int(pivot.shape[1]),
                "n_time_bins": int(profile["t_min"].nunique()),
                "expected_time_bins": int(
                    round(24 * 60 / settings.interval_minutes)
                ),
                "full_day_complete": bool(
                    profile["t_min"].nunique()
                    == int(round(24 * 60 / settings.interval_minutes))
                    and int(profile["t_min"].min()) == 0
                    and int(profile["t_min"].max())
                    == 24 * 60 - settings.interval_minutes
                ),
                "smooth_vs_raw_R2": round(_r2(raw, smoothed), 3),
                "smooth_vs_raw_RMSE": round(_rmse(raw, smoothed), 2),
                "day2day_RMSE_mph": round(float(np.nanmean(daily_rmse)), 2)
                if daily_rmse
                else np.nan,
                "day2day_R2": round(float(np.nanmean(daily_r2)), 3)
                if daily_r2
                else np.nan,
                "day2day_CV": round(coefficient, 3),
            }
        )
    return pd.DataFrame(rows)


def quality_gates(
    calibration: pd.DataFrame,
    timeseries: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = {
        "min_n_links": 5,
        "min_step1_R2": 0.50,
        "min_step2_R2": 0.30,
        "max_P_MAPE": 30.0,
        "max_vt2_MAPE": 30.0,
        "min_smooth_R2": 0.90,
        "min_full_day_profile_share": 1.0,
    }
    smooth = (
        float(timeseries["smooth_vs_raw_R2"].median())
        if not timeseries.empty
        else np.nan
    )
    full_day_share = (
        float(timeseries["full_day_complete"].fillna(False).mean())
        if not timeseries.empty and "full_day_complete" in timeseries
        else np.nan
    )
    rows = []
    for result in calibration.itertuples(index=False):
        checks = [
            ("n_links_fitted", result.n_links, thresholds["min_n_links"], result.n_links >= thresholds["min_n_links"], ">="),
            ("step1_DC_P_R2", result.step1_DC_P_R2, thresholds["min_step1_R2"], result.step1_DC_P_R2 >= thresholds["min_step1_R2"], ">="),
            ("step2_P_mag_R2", result.step2_P_mag_R2, thresholds["min_step2_R2"], result.step2_P_mag_R2 >= thresholds["min_step2_R2"], ">="),
            ("P_MAPE_pct", result.P_MAPE_pct, thresholds["max_P_MAPE"], result.P_MAPE_pct <= thresholds["max_P_MAPE"], "<="),
            ("vt2_MAPE_pct", result.vt2_MAPE_pct, thresholds["max_vt2_MAPE"], result.vt2_MAPE_pct <= thresholds["max_vt2_MAPE"], "<="),
            ("smooth_vs_raw_R2_median", round(smooth, 3), thresholds["min_smooth_R2"], smooth >= thresholds["min_smooth_R2"], ">="),
            (
                "full_day_profile_share",
                round(full_day_share, 3),
                thresholds["min_full_day_profile_share"],
                full_day_share
                >= thresholds["min_full_day_profile_share"],
                ">=",
            ),
        ]
        for name, value, threshold, passed, operator in checks:
            rows.append(
                [
                    result.period,
                    name,
                    value,
                    f"{operator} {threshold}",
                    "PASS" if passed else "FAIL",
                ]
            )
    return pd.DataFrame(
        rows, columns=["period", "gate", "value", "threshold", "status"]
    )


def co2_g_per_mile(speed_mph):
    speed = np.clip(np.asarray(speed_mph, dtype=float), 3.0, 80.0)
    fuel_gallons_per_mile = np.clip(
        0.080 - 0.0022 * speed + 0.0000235 * speed**2, 0.02, 0.20
    )
    return fuel_gallons_per_mile * 8887.0


def build_handoff(
    average: pd.DataFrame,
    average_episodes: pd.DataFrame,
    average_applied: pd.DataFrame,
    fd: pd.DataFrame,
    observations: pd.DataFrame,
    settings: PipelineSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contexts = _fd_by_link(fd, observations)
    parameter_map = calibration_lookup(average_applied)
    order = sorted(int(value) for value in average["link_id"].unique())
    rows: list[dict[str, object]] = []
    accounting: dict[str, dict[str, float]] = {}
    interval_hours = settings.interval_minutes / 60.0
    for link_id in order:
        profile = average[
            average["link_id"].eq(link_id)
            & average["t_min"].ge(settings.wide_window[0])
            & average["t_min"].lt(settings.wide_window[1])
        ].sort_values("t_min")
        if len(profile) < 3:
            continue
        context = contexts.loc[link_id]
        time = profile["t_min"].to_numpy(dtype=float)
        raw = pd.to_numeric(profile["speed_mph"], errors="coerce").to_numpy(dtype=float)
        smoothed = detection_smoothed_speed(raw, settings)
        model = reconstruct_daily_profile(
            profile, average_episodes, parameter_map, settings
        )
        flow = pd.to_numeric(profile["flow_vph"], errors="coerce").to_numpy(dtype=float)
        free_flow = float(context["vf_mph"])
        cutoff = settings.cutoff_ratio * free_flow
        capacity = float(context["capacity_vphpl"])
        speed_at_capacity = float(context["vc_mph"])
        length = float(context["length_mi"])
        lanes = float(context["lanes"])
        for index, minute in enumerate(time):
            period = settings.period_for_minute(minute) or "NT"
            per_lane_count = (
                float(flow[index]) * interval_hours
                if np.isfinite(flow[index])
                else np.nan
            )
            total_count = per_lane_count * lanes if np.isfinite(per_lane_count) else np.nan
            vmt = total_count * length if np.isfinite(total_count) else np.nan
            observed_speed = max(float(raw[index]), 1.0)
            vht = total_count * length / observed_speed if np.isfinite(total_count) else np.nan
            free_flow_time = (
                total_count * length / free_flow if np.isfinite(total_count) else np.nan
            )
            capacity_time = (
                total_count * length / speed_at_capacity
                if np.isfinite(total_count)
                else np.nan
            )
            vdt = (
                max(0.0, vht - free_flow_time) if np.isfinite(vht) else np.nan
            )
            vcdt = (
                max(0.0, vht - capacity_time) if np.isfinite(vht) else np.nan
            )
            if np.isfinite(total_count):
                totals = accounting.setdefault(
                    period,
                    {"VMT": 0.0, "VHT": 0.0, "VFFTT": 0.0, "VDT": 0.0, "VCDT": 0.0},
                )
                totals["VMT"] += vmt
                totals["VHT"] += vht
                totals["VFFTT"] += free_flow_time
                totals["VDT"] += vdt
                totals["VCDT"] += vcdt
            rows.append(
                {
                    "link_id": link_id,
                    "sensor_uid": str(context["sensor_uid"]),
                    "tmc_code": str(context["tmc_code"]),
                    "network_link_id": context["network_link_id"],
                    "from_node_id": context["network_from_node_id"],
                    "to_node_id": context["network_to_node_id"],
                    "t_min": int(minute),
                    "period": period,
                    "speed_raw": round(float(raw[index]), 2),
                    "speed_smoothed": round(float(smoothed[index]), 2),
                    "speed_qvdf_model": round(float(model[index]), 2),
                    "count_per_lane_15min": round(per_lane_count, 1)
                    if np.isfinite(per_lane_count)
                    else np.nan,
                    "lanes": lanes,
                    "lanes_source": str(context["lanes_source"]),
                    "count_total_15min": round(total_count, 1)
                    if np.isfinite(total_count)
                    else np.nan,
                    "length_mi": round(length, 3),
                    "free_flow_speed_model_mph": free_flow,
                    "congestion_threshold_mph": round(cutoff, 1),
                    "capacity_vphpl": round(capacity, 0),
                    "capacity_source": str(context["capacity_source"]),
                    "flow_source": (
                        "synthetic_inverse_s3"
                        if bool(profile["flow_synthetic"].iloc[0])
                        else "measured"
                    ),
                    "vmt": round(vmt, 2) if np.isfinite(vmt) else np.nan,
                    "vht": round(vht, 3) if np.isfinite(vht) else np.nan,
                    "vdt": round(vdt, 3) if np.isfinite(vdt) else np.nan,
                    "vcdt": round(vcdt, 3) if np.isfinite(vcdt) else np.nan,
                    "emis_co2_g_obs": round(float(co2_g_per_mile(raw[index]) * vmt), 1)
                    if np.isfinite(vmt)
                    else np.nan,
                    "emis_co2_g_model": round(float(co2_g_per_mile(model[index]) * vmt), 1)
                    if np.isfinite(vmt)
                    else np.nan,
                    "emissions_method": (
                        "speed_fuel_curve_times_8887_g_co2_per_gallon"
                    ),
                    "qvdf_flow_vphpl": np.nan,
                    "qvdf_count_total_15min": np.nan,
                    "critical_density_veh_per_mile_lane": np.nan,
                    "additive_flow_adjustment_vphpl": np.nan,
                    "relative_flow_adjustment": np.nan,
                }
            )
    handoff = pd.DataFrame(rows)
    for (link_id, period), group in handoff[
        handoff["speed_qvdf_model"]
        < handoff["congestion_threshold_mph"]
    ].groupby(["link_id", "period"]):
        demand = float(
            pd.to_numeric(group["count_total_15min"], errors="coerce").sum()
            / max(float(group["lanes"].iloc[0]), 1.0)
        )
        context = contexts.loc[int(link_id)]
        flow, multiplier, achieved = conserve_flow_from_speed(
            group["speed_qvdf_model"].to_numpy(dtype=float),
            vf_mph=float(context["vf_mph"]),
            kc_vpmpl=float(context["kc_vpmpl"]),
            s3_m=float(context["s3_m"]),
            capacity_vphpl=float(context["capacity_vphpl"]),
            demand_veh_per_lane=demand,
            interval_hours=interval_hours,
        )
        lanes = float(group["lanes"].iloc[0])
        mean_base = max(float(np.nanmean(flow)), 1e-6)
        handoff.loc[group.index, "qvdf_flow_vphpl"] = np.round(flow, 1)
        handoff.loc[group.index, "qvdf_count_total_15min"] = np.round(
            flow * interval_hours * lanes, 1
        )
        handoff.loc[
            group.index, "critical_density_veh_per_mile_lane"
        ] = round(
            float(context["kc_vpmpl"]), 3
        )
        handoff.loc[
            group.index, "additive_flow_adjustment_vphpl"
        ] = round(multiplier, 2)
        handoff.loc[group.index, "relative_flow_adjustment"] = round(
            abs(multiplier) / mean_base, 3
        )
        if not np.isclose(achieved, demand, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"Flow conservation failed for link {link_id}, period {period}"
            )

    accounting_rows = []
    for period, values in accounting.items():
        average_speed = (
            values["VMT"] / values["VHT"] if values["VHT"] > 0 else 0.0
        )
        accounting_rows.append(
            [
                period,
                round(values["VMT"], 1),
                round(values["VHT"], 2),
                round(values["VFFTT"], 2),
                round(values["VDT"], 2),
                round(values["VCDT"], 2),
                round(average_speed, 1),
                round(100.0 * values["VDT"] / values["VHT"], 1)
                if values["VHT"] > 0
                else 0.0,
            ]
        )
    corridor_accounting = pd.DataFrame(
        accounting_rows,
        columns=[
            "period",
            "VMT_mi",
            "VHT_hr",
            "VFFTT_hr",
            "VDT_hr",
            "VCDT_hr",
            "avg_speed_mph",
            "delay_share_pct",
        ],
    )

    parameter_rows = []
    clean_average = (
        average_episodes[
            average_episodes["is_clean_valid_episode"].fillna(False)
        ]
        if not average_episodes.empty
        else average_episodes
    )
    for link_id in order:
        context = contexts.loc[link_id]
        for period in settings.periods:
            episodes = (
                clean_average[
                    clean_average["link_id"].eq(link_id)
                    & clean_average["period"].eq(period)
                ]
                if not clean_average.empty
                else clean_average
            )
            parameters = parameter_map.get((link_id, period))
            parameter_rows.append(
                {
                    "link_id": link_id,
                    "sensor_uid": str(context["sensor_uid"]),
                    "tmc_code": str(context["tmc_code"]),
                    "network_link_id": context["network_link_id"],
                    "from_node_id": context["network_from_node_id"],
                    "to_node_id": context["network_to_node_id"],
                    "period": period,
                    "has_average_weekday_episode": bool(
                        not episodes.empty
                    ),
                    "calibration_available": bool(parameters is not None),
                    "f_d": round(float(parameters["f_d"]), 4)
                    if parameters
                    else np.nan,
                    "n": round(float(parameters["n"]), 4)
                    if parameters
                    else np.nan,
                    "f_p": round(float(parameters["f_p"]), 4)
                    if parameters
                    else np.nan,
                    "s": round(float(parameters["s"]), 4)
                    if parameters
                    else np.nan,
                    "alpha": round(float(parameters["alpha"]), 4)
                    if parameters
                    else np.nan,
                    "beta": round(float(parameters["beta"]), 4)
                    if parameters
                    else np.nan,
                    "free_flow_speed_model_mph": float(context["vf_mph"]),
                    "speed_at_capacity_mph": round(
                        float(context["vc_mph"]), 1
                    ),
                    "congestion_threshold_mph": round(
                        settings.cutoff_ratio * float(context["vf_mph"]),
                        1,
                    ),
                    "capacity_vphpl": round(
                        float(context["capacity_vphpl"]), 0
                    ),
                    "calibration_data_basis": (
                        str(parameters.get("data_basis", ""))
                        if parameters
                        else ""
                    ),
                    "calibration_scope": (
                        str(parameters.get("calibration_scope", ""))
                        if parameters
                        else ""
                    ),
                    "calibration_n_episodes": (
                        int(parameters["n_episodes"])
                        if parameters
                        else np.nan
                    ),
                    "calibration_reliability": (
                        str(parameters.get("reliability", ""))
                        if parameters
                        else ""
                    ),
                }
            )
    parameters = pd.DataFrame(parameter_rows)
    return handoff, parameters, corridor_accounting
