from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .targets import PERIOD_RANGES, transform_boundaries


PERIOD_NETWORK_FIELDS: Dict[str, Dict[str, str]] = {
    "AM": {
        "period_volume": "I4AMVOL",
        "hourly_link_capacity": "IAMHRLKCAP",
        "reported_hourly_lane_capacity": "IAMHRLNCAP",
        "period_model_speed_mph": "I4AMSPD",
        "period_model_freeflow_mph": "I4AMFFSPD",
        "period_model_vc": "I4AMVC",
        "period_model_vdf": "I4AMVDF",
        "period_model_vmt": "I4AMVMT",
        "period_sov_volume": "I4AMSOV",
        "period_hov2_volume": "I4AMHV2",
        "period_hov3_volume": "I4AMHV3",
        "period_truck_volume": "I4AMTRK",
        "period_commercial_volume": "I4AMCV",
        "period_apx_volume": "I4AMAPX",
        "period_travel_time": "AMHTIME",
        "period_lane_limit": "AMLIMIT",
        "period_toll_value": "AMTOLL_VP",
    },
    "MD": {
        "period_volume": "I4MDVOL",
        "hourly_link_capacity": "MDHRLKCAP",
        "reported_hourly_lane_capacity": "MDHRLNCAP",
        "period_model_speed_mph": "I4MDSPD",
        "period_model_freeflow_mph": "I4MDFFSPD",
        "period_model_vc": "I4MDVC",
        "period_model_vdf": "I4MDVDF",
        "period_model_vmt": "I4MDVMT",
        "period_sov_volume": "I4MDSOV",
        "period_hov2_volume": "I4MDHV2",
        "period_hov3_volume": "I4MDHV3",
        "period_truck_volume": "I4MDTRK",
        "period_commercial_volume": "I4MDCV",
        "period_apx_volume": "I4MDAPX",
        "period_travel_time": "MDHTIME",
        "period_lane_limit": "MDLIMIT",
        "period_toll_value": "MDTOLL_VP",
    },
    "PM": {
        "period_volume": "I4PMVOL",
        "hourly_link_capacity": "IPMHRLKCAP",
        "reported_hourly_lane_capacity": "IPMHRLNCAP",
        "period_model_speed_mph": "I4PMSPD",
        "period_model_freeflow_mph": "I4PMFFSPD",
        "period_model_vc": "I4PMVC",
        "period_model_vdf": "I4PMVDF",
        "period_model_vmt": "I4PMVMT",
        "period_sov_volume": "I4PMSOV",
        "period_hov2_volume": "I4PMHV2",
        "period_hov3_volume": "I4PMHV3",
        "period_truck_volume": "I4PMTRK",
        "period_commercial_volume": "I4PMCV",
        "period_apx_volume": "I4PMAPX",
        "period_travel_time": "PMHTIME",
        "period_lane_limit": "PMLIMIT",
        "period_toll_value": "PMTOLL_VP",
    },
}

NETWORK_BASE_RENAME = {
    "link_id": "network_link_id",
    "from_node_id": "network_from_node_id",
    "to_node_id": "network_to_node_id",
    "dir_flag": "network_dir_flag",
    "link_type": "network_link_type",
    "FTYPE": "network_ftype",
    "allowed_use": "network_allowed_use",
    "lanes": "network_lanes",
    "capacity": "network_capacity_per_lane",
    "free_speed": "network_free_speed_mph",
    "length_in_mile": "network_length_mi",
    "toll": "network_toll",
    "vdf_alpha": "network_vdf_alpha",
    "vdf_beta": "network_vdf_beta",
    "vdf_plf": "network_vdf_plf",
    "ref_volume": "network_ref_volume",
    "ref_cost": "network_ref_cost",
    "vdf_fftt": "network_vdf_fftt",
    "MODE": "network_mode",
    "JUR": "network_jurisdiction",
    "COMP": "network_comp",
    "TOLLGRP": "network_toll_group",
    "TAZ": "network_taz",
    "STREETNAME": "network_street_name",
    "t0_hybrid_hour": "existing_t0_hour",
    "t2_hybrid_hour": "existing_t2_hour",
    "t3_hybrid_hour": "existing_t3_hour",
    "t2_hybrid_source": "existing_t2_source",
    "t2_hybrid_detail": "existing_t2_detail",
    "t2_hybrid_precedence_rank": "existing_t2_precedence_rank",
}

NETWORK_BASE_COLUMNS = list(NETWORK_BASE_RENAME)

FD_COLUMNS = [
    "sensor_uid",
    "capacity_vphpl",
    "speed_at_capacity_mph",
    "critical_density_veh_per_mile_lane",
    "free_flow_speed_model_mph",
]

PROFILE_COLUMNS = [
    "sensor_uid",
    "t_min",
    "avg_weekday_speed_mph",
    "avg_weekday_flow_veh_per_hr_lane",
    "reference_speed_mph",
    "capacity_vphpl",
    "n_days",
]

NETWORK_NUMERIC_COLUMNS = [
    "network_lanes",
    "network_capacity_per_lane",
    "network_free_speed_mph",
    "network_length_mi",
    "network_toll",
    "network_vdf_alpha",
    "network_vdf_beta",
    "network_vdf_plf",
    "network_ref_volume",
    "network_ref_cost",
    "network_vdf_fftt",
    "network_taz",
    "period_volume",
    "hourly_link_capacity",
    "reported_hourly_lane_capacity",
    "hourly_capacity_per_lane",
    "period_volume_per_lane",
    "period_demand_capacity_ratio",
    "capacity_equivalent_hours",
    "period_model_speed_mph",
    "period_model_freeflow_mph",
    "period_model_speed_ratio",
    "period_model_delay_index",
    "period_model_vc",
    "period_model_vdf",
    "period_model_vmt",
    "period_travel_time",
    "period_lane_limit",
    "period_toll_value",
    "period_truck_share",
    "period_hov_share",
    "period_commercial_share",
    "upstream_link_count",
    "downstream_link_count",
    "upstream_max_dc",
    "downstream_max_dc",
    "upstream_mean_capacity",
    "downstream_min_capacity",
    "downstream_mean_capacity",
    "downstream_capacity_ratio",
    "downstream_bottleneck_strength",
    "upstream_max_model_vc",
    "downstream_max_model_vc",
    "downstream_min_speed_ratio",
    "is_merge_node",
    "is_diverge_node",
]

PROFILE_FEATURE_COLUMNS = [
    "profile_speed_trough_relative_min",
    "profile_flow_peak_relative_min",
    "profile_min_speed_ratio",
    "profile_mean_speed_ratio",
    "profile_max_flow_capacity_ratio",
    "profile_mean_flow_capacity_ratio",
    "profile_first_capacity_cross_relative_min",
    "profile_share_bins_over_capacity",
    "profile_bin_coverage",
    "profile_day_count",
]

FD_FEATURE_COLUMNS = [
    "fd_capacity_vphpl",
    "vc_mph",
    "vf_model_mph",
    "critical_density_vpmpl",
]

EPISODE_DIAGNOSTIC_COLUMNS = [
    "P_hr",
    "demand_capacity_ratio",
    "mu_obs_vphpl",
    "min_speed_mph",
    "mean_speed_mph",
    "episode_demand",
    "magnitude",
    "severity",
]

AGGREGATED_NUMERIC_COLUMNS = [
    "t0_hour",
    "t2_hour",
    "t3_hour",
    "road_order",
    "corridor_position_mi",
    *NETWORK_NUMERIC_COLUMNS,
    *PROFILE_FEATURE_COLUMNS,
    *FD_FEATURE_COLUMNS,
    *EPISODE_DIAGNOSTIC_COLUMNS,
]

AGGREGATED_CATEGORICAL_COLUMNS = [
    "network_link_id",
    "corridor",
    "direction",
    "road",
    "network_link_type",
    "network_ftype",
    "network_allowed_use",
    "network_mode",
    "network_jurisdiction",
    "network_comp",
    "network_toll_group",
    "capacity_source",
    "reference_speed_source",
    "existing_t2_source",
]


def _numeric(frame: pd.DataFrame, columns: List[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def calculate_period_demand_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    def numeric_column(name: str) -> pd.Series:
        if name not in out:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(out[name], errors="coerce")

    period_hours = out["period"].map(
        lambda p: PERIOD_RANGES[str(p).upper()][1]
        - PERIOD_RANGES[str(p).upper()][0]
    ).astype(float)
    lanes = pd.to_numeric(
        out["network_lanes"], errors="coerce"
    ).replace(0, np.nan)
    hourly_capacity = pd.to_numeric(
        out["hourly_link_capacity"], errors="coerce"
    ).replace(0, np.nan)
    volume = pd.to_numeric(out["period_volume"], errors="coerce")
    out["hourly_capacity_per_lane"] = hourly_capacity / lanes
    out["period_volume_per_lane"] = volume / lanes
    out["period_demand_capacity_ratio"] = (
        volume / (hourly_capacity * period_hours)
    )
    out["capacity_equivalent_hours"] = volume / hourly_capacity
    model_speed = numeric_column("period_model_speed_mph").replace(0, np.nan)
    model_freeflow = numeric_column("period_model_freeflow_mph").replace(
        0, np.nan
    )
    out["period_model_speed_ratio"] = model_speed / model_freeflow
    out["period_model_delay_index"] = model_freeflow / model_speed
    out["period_truck_share"] = (
        numeric_column("period_truck_volume")
        / volume.replace(0, np.nan)
    )
    out["period_hov_share"] = (
        numeric_column("period_hov2_volume")
        + numeric_column("period_hov3_volume")
    ) / volume.replace(0, np.nan)
    out["period_commercial_share"] = (
        numeric_column("period_commercial_volume")
        / volume.replace(0, np.nan)
    )
    return out


def add_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for period, group in frame.groupby("period", sort=False):
        current = group.copy()
        downstream = (
            current.groupby("network_from_node_id", dropna=False)
            .agg(
                downstream_link_count=("network_link_id", "size"),
                downstream_min_capacity=("hourly_link_capacity", "min"),
                downstream_mean_capacity=("hourly_link_capacity", "mean"),
                downstream_max_dc=("period_demand_capacity_ratio", "max"),
                downstream_max_model_vc=("period_model_vc", "max"),
                downstream_min_speed_ratio=("period_model_speed_ratio", "min"),
            )
            .reset_index()
        )
        upstream = (
            current.groupby("network_to_node_id", dropna=False)
            .agg(
                upstream_link_count=("network_link_id", "size"),
                upstream_mean_capacity=("hourly_link_capacity", "mean"),
                upstream_max_dc=("period_demand_capacity_ratio", "max"),
                upstream_max_model_vc=("period_model_vc", "max"),
            )
            .reset_index()
        )
        current = current.merge(
            downstream,
            left_on="network_to_node_id",
            right_on="network_from_node_id",
            how="left",
            suffixes=("", "_downstream_key"),
            validate="many_to_one",
        ).drop(columns=["network_from_node_id_downstream_key"])
        current = current.merge(
            upstream,
            left_on="network_from_node_id",
            right_on="network_to_node_id",
            how="left",
            suffixes=("", "_upstream_key"),
            validate="many_to_one",
        ).drop(columns=["network_to_node_id_upstream_key"])
        current["downstream_capacity_ratio"] = (
            current["downstream_min_capacity"]
            / current["hourly_link_capacity"].replace(0, np.nan)
        )
        current["downstream_bottleneck_strength"] = (
            1.0 - current["downstream_capacity_ratio"]
        ).clip(lower=0.0)
        current["is_merge_node"] = (
            current["upstream_link_count"].fillna(0) >= 2
        ).astype(int)
        current["is_diverge_node"] = (
            current["downstream_link_count"].fillna(0) >= 2
        ).astype(int)
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True, sort=False)


def _mode_or_first(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else non_null.iloc[0]


def load_accepted_episodes(
    cbi_run_dir: Path,
) -> Tuple[pd.DataFrame, List[Path]]:
    files = sorted(
        (cbi_run_dir / "corridors").glob(
            "*/05-episode-filtering/daily_episodes_accepted.csv"
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No accepted daily episode files under {cbi_run_dir}"
        )
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    return pd.concat(frames, ignore_index=True), files


def load_network_periods(
    boundary_mapping_run_dir: Path,
) -> Tuple[pd.DataFrame, List[Path]]:
    base = boundary_mapping_run_dir / "link-t2" / "period_link_files"
    frames: List[pd.DataFrame] = []
    files: List[Path] = []
    for period, field_map in PERIOD_NETWORK_FIELDS.items():
        path = base / period.lower() / "link.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        source_columns = [
            column
            for column in NETWORK_BASE_COLUMNS + list(field_map.values())
            if column in header
        ]
        frame = pd.read_csv(
            path, usecols=source_columns, low_memory=False
        ).rename(columns=NETWORK_BASE_RENAME)
        frame = frame.rename(
            columns={source: target for target, source in field_map.items()}
        )
        for target in field_map:
            if target not in frame:
                frame[target] = np.nan
        frame["period"] = period
        frame = calculate_period_demand_features(frame)
        frames.append(frame)
        files.append(path)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    _numeric(
        combined,
        [
            "network_link_id",
            "network_from_node_id",
            "network_to_node_id",
            *NETWORK_NUMERIC_COLUMNS,
            "existing_t0_hour",
            "existing_t2_hour",
            "existing_t3_hour",
            "existing_t2_precedence_rank",
        ],
    )
    combined = add_graph_features(combined)
    if combined.duplicated(["period", "network_link_id"]).any():
        raise ValueError("Network period table contains duplicate link keys.")
    return combined, files


def load_fd_context(
    cbi_run_dir: Path,
) -> Tuple[pd.DataFrame, List[Path]]:
    files = sorted(
        (cbi_run_dir / "corridors").glob(
            "*/02-fundamental-diagram/link_fd_context.csv"
        )
    )
    frames: List[pd.DataFrame] = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in FD_COLUMNS if column in header]
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame["corridor"] = path.parent.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["corridor", *FD_COLUMNS]), files
    combined = pd.concat(frames, ignore_index=True).rename(
        columns={
            "capacity_vphpl": "fd_capacity_vphpl",
            "speed_at_capacity_mph": "vc_mph",
            "free_flow_speed_model_mph": "vf_model_mph",
            "critical_density_veh_per_mile_lane": "critical_density_vpmpl",
        }
    )
    combined = combined.drop_duplicates(
        subset=["corridor", "sensor_uid"], keep="first"
    )
    return combined, files


def _period_from_minute(value: float) -> str:
    for period, (start, end) in PERIOD_RANGES.items():
        if start * 60 <= value < end * 60:
            return period
    return ""


def load_profile_features(
    cbi_run_dir: Path,
) -> Tuple[pd.DataFrame, List[Path]]:
    files = sorted(
        (cbi_run_dir / "corridors").glob(
            "*/03-profiles/average_weekday_profile.csv"
        )
    )
    frames = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in PROFILE_COLUMNS if column in header]
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame["corridor"] = path.parent.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["corridor", "sensor_uid", "period", *PROFILE_FEATURE_COLUMNS]
        ), files
    profile = pd.concat(frames, ignore_index=True)
    _numeric(
        profile,
        [
            "t_min",
            "avg_weekday_speed_mph",
            "avg_weekday_flow_veh_per_hr_lane",
            "reference_speed_mph",
            "capacity_vphpl",
            "n_days",
        ],
    )
    profile["period"] = profile["t_min"].map(_period_from_minute)
    profile = profile[profile["period"].ne("")].copy()
    profile["speed_ratio"] = (
        profile["avg_weekday_speed_mph"]
        / profile["reference_speed_mph"].replace(0, np.nan)
    )
    profile["flow_capacity_ratio"] = (
        profile["avg_weekday_flow_veh_per_hr_lane"]
        / profile["capacity_vphpl"].replace(0, np.nan)
    )

    rows = []
    for (corridor, sensor_uid, period), group in profile.groupby(
        ["corridor", "sensor_uid", "period"], dropna=False
    ):
        start_min = PERIOD_RANGES[period][0] * 60
        expected_bins = int(
            (PERIOD_RANGES[period][1] - PERIOD_RANGES[period][0]) * 4
        )
        speed = group.dropna(subset=["avg_weekday_speed_mph"])
        flow = group.dropna(subset=["avg_weekday_flow_veh_per_hr_lane"])
        trough_min = (
            float(speed.loc[speed["avg_weekday_speed_mph"].idxmin(), "t_min"])
            if not speed.empty
            else np.nan
        )
        peak_min = (
            float(
                flow.loc[
                    flow["avg_weekday_flow_veh_per_hr_lane"].idxmax(), "t_min"
                ]
            )
            if not flow.empty
            else np.nan
        )
        over_capacity = group["flow_capacity_ratio"].ge(1.0)
        first_cross = (
            float(group.loc[over_capacity, "t_min"].min())
            if over_capacity.any()
            else np.nan
        )
        rows.append(
            {
                "corridor": corridor,
                "sensor_uid": sensor_uid,
                "period": period,
                "profile_speed_trough_relative_min": trough_min - start_min,
                "profile_flow_peak_relative_min": peak_min - start_min,
                "profile_min_speed_ratio": group["speed_ratio"].min(),
                "profile_mean_speed_ratio": group["speed_ratio"].mean(),
                "profile_max_flow_capacity_ratio": group[
                    "flow_capacity_ratio"
                ].max(),
                "profile_mean_flow_capacity_ratio": group[
                    "flow_capacity_ratio"
                ].mean(),
                "profile_first_capacity_cross_relative_min": (
                    first_cross - start_min
                ),
                "profile_share_bins_over_capacity": over_capacity.mean(),
                "profile_bin_coverage": min(1.0, len(group) / expected_bins),
                "profile_day_count": group["n_days"].median(),
            }
        )
    return pd.DataFrame(rows), files


def load_corridor_positions(
    spatial_run_dir: Path,
) -> Tuple[pd.DataFrame, List[Path]]:
    path = spatial_run_dir / "input-snapshot" / "route_summary.csv"
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "tmc_code",
                "period",
                "road",
                "direction",
                "corridor_position_mi",
            ]
        ), []
    routes = pd.read_csv(path, low_memory=False)
    routes = routes.rename(columns={"tmc": "tmc_code"})
    routes["period"] = routes["period"].astype(str).str.upper()
    _numeric(routes, ["road_order", "tmc_miles", "confidence"])
    routes = routes.sort_values(
        ["period", "road", "direction", "road_order", "tmc_code"],
        kind="mergesort",
    )
    routes["corridor_position_mi"] = (
        routes.groupby(["period", "road", "direction"], dropna=False)[
            "tmc_miles"
        ].cumsum()
        - 0.5 * routes["tmc_miles"].fillna(0.0)
    )
    routes = routes.sort_values(
        ["tmc_code", "period", "confidence"],
        ascending=[True, True, False],
        kind="mergesort",
    ).drop_duplicates(["tmc_code", "period"], keep="first")
    return routes[
        [
            "tmc_code",
            "period",
            "road",
            "direction",
            "corridor_position_mi",
        ]
    ], [path]


def prepare_daily_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["tmc_code", "period", "t0_hour", "t2_hour", "t3_hour"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Episode data is missing columns: {missing}")
    clean = frame.copy()
    clean["period"] = clean["period"].astype(str).str.upper()
    _numeric(clean, ["t0_hour", "t2_hour", "t3_hour"])
    clean = clean.dropna(subset=required)
    clean = clean[
        (clean["t0_hour"] <= clean["t2_hour"])
        & (clean["t2_hour"] <= clean["t3_hour"])
        & (clean["t3_hour"] > clean["t0_hour"])
        & clean["period"].isin(PERIOD_RANGES)
    ].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["weekday_name"] = clean["date"].dt.day_name().fillna("Unknown")
    clean["tmc_period_id"] = (
        clean["tmc_code"].astype(str) + "::" + clean["period"]
    )
    group_size = clean.groupby("tmc_period_id")["tmc_period_id"].transform(
        "size"
    )
    clean["sample_weight"] = 1.0 / group_size
    clean = transform_boundaries(clean)
    clean["observed_span_min"] = (
        clean["t3_hour"] - clean["t0_hour"]
    ) * 60.0
    clean["observed_t2_fraction"] = (
        (clean["t2_hour"] - clean["t0_hour"])
        / (clean["t3_hour"] - clean["t0_hour"])
    )
    return clean.sort_values(
        ["tmc_code", "period", "date"], kind="mergesort"
    ).reset_index(drop=True)


def aggregate_episode_rows(frame: pd.DataFrame) -> pd.DataFrame:
    clean = prepare_daily_rows(frame)
    numeric = [
        column
        for column in AGGREGATED_NUMERIC_COLUMNS
        if column in clean.columns
    ]
    categorical = [
        column
        for column in AGGREGATED_CATEGORICAL_COLUMNS
        if column in clean.columns
    ]
    aggregations = {column: "median" for column in numeric}
    aggregations.update(
        {column: _mode_or_first for column in categorical}
    )
    grouped = clean.groupby(
        ["tmc_code", "period"], as_index=False
    ).agg(aggregations)
    counts = (
        clean.groupby(["tmc_code", "period"], as_index=False)
        .agg(
            episode_count=("tmc_code", "size"),
            observed_day_count=("date", "nunique"),
            observed_t0_std_hour=("t0_hour", "std"),
            observed_t2_std_hour=("t2_hour", "std"),
            observed_t3_std_hour=("t3_hour", "std"),
            first_observed_date=("date", "min"),
            last_observed_date=("date", "max"),
        )
    )
    grouped = grouped.merge(
        counts, on=["tmc_code", "period"], how="left"
    )
    grouped["tmc_period_id"] = (
        grouped["tmc_code"].astype(str) + "::" + grouped["period"]
    )
    grouped = transform_boundaries(grouped)
    grouped["observed_span_min"] = (
        grouped["t3_hour"] - grouped["t0_hour"]
    ) * 60.0
    grouped["observed_t2_fraction"] = (
        (grouped["t2_hour"] - grouped["t0_hour"])
        / (grouped["t3_hour"] - grouped["t0_hour"])
    )
    return grouped.sort_values(
        ["tmc_code", "period"], kind="mergesort"
    ).reset_index(drop=True)


def build_experiment_tables(
    cbi_run_dir: Path,
    boundary_mapping_run_dir: Path,
    spatial_run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Path]]:
    episodes, episode_files = load_accepted_episodes(cbi_run_dir)
    network, network_files = load_network_periods(boundary_mapping_run_dir)
    fd, fd_files = load_fd_context(cbi_run_dir)
    profile, profile_files = load_profile_features(cbi_run_dir)
    positions, position_files = load_corridor_positions(spatial_run_dir)
    episodes["network_link_id"] = pd.to_numeric(
        episodes["network_link_id"], errors="coerce"
    )
    episodes["period"] = episodes["period"].astype(str).str.upper()
    joined = episodes.merge(
        network,
        on=["network_link_id", "period"],
        how="left",
        validate="many_to_one",
    )
    # Newer CBI handoff rows carry the selected link identity for lineage.
    # Reconcile those duplicate identity columns after the period-network join,
    # keeping the native network value authoritative while retaining the CBI
    # value as a fallback. Without this, pandas suffixes the fields and model
    # features such as ``network_link_type`` disappear from the training table.
    for column in (
        "network_from_node_id",
        "network_to_node_id",
        "network_link_type",
    ):
        left, right = f"{column}_x", f"{column}_y"
        if left in joined and right in joined:
            joined[column] = joined[right].combine_first(joined[left])
            joined = joined.drop(columns=[left, right])
    if not fd.empty:
        joined = joined.merge(
            fd,
            on=["corridor", "sensor_uid"],
            how="left",
            validate="many_to_one",
        )
    if not profile.empty:
        joined = joined.merge(
            profile,
            on=["corridor", "sensor_uid", "period"],
            how="left",
            validate="many_to_one",
        )
    if not positions.empty:
        joined = joined.merge(
            positions,
            on=["tmc_code", "period"],
            how="left",
            suffixes=("", "_route"),
            validate="many_to_one",
        )
        if "direction_route" in joined:
            joined["direction"] = joined["direction"].combine_first(
                joined["direction_route"]
            )
            joined = joined.drop(columns=["direction_route"])
    daily = prepare_daily_rows(joined)
    aggregate = aggregate_episode_rows(joined)
    inputs = (
        episode_files
        + network_files
        + fd_files
        + profile_files
        + position_files
    )
    return aggregate, daily, network, inputs


def build_training_table(
    cbi_run_dir: Path,
    boundary_mapping_run_dir: Path,
    spatial_run_dir: Path = None,
) -> Tuple[pd.DataFrame, List[Path]]:
    if spatial_run_dir is None:
        spatial_run_dir = (
            boundary_mapping_run_dir.parent.parent
            / "t2"
            / "coverage-expansion"
        )
    aggregate, _, _, files = build_experiment_tables(
        cbi_run_dir, boundary_mapping_run_dir, Path(spatial_run_dir)
    )
    return aggregate, files
