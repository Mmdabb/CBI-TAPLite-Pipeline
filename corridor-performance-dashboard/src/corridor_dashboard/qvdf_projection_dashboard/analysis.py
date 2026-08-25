from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..reconstruction import (
    predict_duration_hours,
    predict_minimum_speed,
    select_nonoverlapping_episodes,
)
from ..dashboard_filters import is_managed_corridor
from .settings import DashboardSettings


CALIBRATION_FIELDS = [
    "sensor_uid",
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
    "data_basis",
]


@dataclass
class CBIProducts:
    profiles: pd.DataFrame
    average_accepted: pd.DataFrame
    daily_accepted: pd.DataFrame
    parameters: pd.DataFrame
    coverage: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CBI dashboard input: {path}")
    return pd.read_csv(path, low_memory=False)


def _filter_tmc_rows(
    frame: pd.DataFrame,
    eligible_tmc_codes: set[str] | None,
) -> pd.DataFrame:
    if eligible_tmc_codes is None or "tmc_code" not in frame.columns:
        return frame
    tmc = frame["tmc_code"].astype("string").str.strip().str.upper()
    return frame.loc[tmc.isin(eligible_tmc_codes)].copy()


def collect_cbi_products(
    cbi_output_root: Path,
    eligible_tmc_codes: set[str] | None = None,
) -> CBIProducts:
    profiles: list[pd.DataFrame] = []
    average: list[pd.DataFrame] = []
    daily: list[pd.DataFrame] = []
    parameters: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for corridor_dir in sorted(cbi_output_root.iterdir()):
        if not corridor_dir.is_dir() or corridor_dir.name.startswith("_"):
            continue
        corridor = corridor_dir.name
        if is_managed_corridor(corridor):
            continue
        profile = _read_csv(
            corridor_dir / "03-profiles" / "average_weekday_profile.csv"
        )
        avg = _read_csv(
            corridor_dir
            / "05-episode-filtering"
            / "average_weekday_episodes_accepted.csv"
        )
        day = _read_csv(
            corridor_dir
            / "05-episode-filtering"
            / "daily_episodes_accepted.csv"
        )
        params = _read_csv(
            corridor_dir
            / "06-qvdf-calibration"
            / "qvdf_selected_parameters.csv"
        )
        profile = _filter_tmc_rows(profile, eligible_tmc_codes)
        avg = _filter_tmc_rows(avg, eligible_tmc_codes)
        day = _filter_tmc_rows(day, eligible_tmc_codes)
        params = _filter_tmc_rows(params, eligible_tmc_codes)
        if profile.empty:
            continue
        for frame in (profile, avg, day, params):
            if "corridor" not in frame:
                frame["corridor"] = corridor
        profiles.append(profile)
        average.append(avg)
        daily.append(day)
        parameters.append(params)
        mapped = (
            profile["network_link_id"].notna()
            if "network_link_id" in profile
            else pd.Series(False, index=profile.index)
        )
        coverage_rows.append(
            {
                "corridor": corridor,
                "profile_rows": int(len(profile)),
                "profile_tmc": int(profile["tmc_code"].nunique()),
                "mapped_tmc": int(profile.loc[mapped, "tmc_code"].nunique()),
                "average_accepted_episodes": int(len(avg)),
                "average_accepted_tmc": int(
                    avg["tmc_code"].nunique() if "tmc_code" in avg else 0
                ),
                "daily_accepted_episodes": int(len(day)),
                "calibrated_link_periods": int(len(params)),
            }
        )

    if not profiles:
        raise ValueError(f"No corridor CBI outputs found under {cbi_output_root}")
    return CBIProducts(
        profiles=pd.concat(profiles, ignore_index=True, sort=False),
        average_accepted=pd.concat(average, ignore_index=True, sort=False),
        daily_accepted=pd.concat(daily, ignore_index=True, sort=False),
        parameters=pd.concat(parameters, ignore_index=True, sort=False),
        coverage=pd.DataFrame(coverage_rows).sort_values("corridor"),
    )


def _representative_average_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes.copy()
    work = episodes.copy()
    work["min_speed_mph"] = pd.to_numeric(
        work["min_speed_mph"], errors="coerce"
    )
    work["P_hr"] = pd.to_numeric(work["P_hr"], errors="coerce")
    selected_rows: list[pd.Series] = []
    window_fields = [
        "episode_id",
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "P_hr",
        "threshold_used",
        "freeflow_speed_mph",
        "min_speed_mph",
        "length_mi",
        "mu_obs_vphpl",
        "per_lane_hourly_capacity",
    ]
    for _, group in work.groupby(
        ["corridor", "sensor_uid", "period"],
        sort=False,
        dropna=False,
    ):
        selected, dropped = select_nonoverlapping_episodes(group)
        if selected.empty:
            continue
        representative = selected.sort_values(
            ["min_speed_mph", "P_hr", "t2_hour"],
            ascending=[True, False, True],
            kind="mergesort",
        ).iloc[0].copy()
        windows = selected[
            [column for column in window_fields if column in selected]
        ].to_dict(orient="records")
        representative["accepted_episode_count"] = int(len(group))
        representative["reconstruction_episode_count"] = int(len(selected))
        representative["overlap_excluded_episode_count"] = int(len(dropped))
        representative["accepted_episode_windows_json"] = json.dumps(
            windows,
            separators=(",", ":"),
            default=str,
        )
        selected_rows.append(representative)
    return pd.DataFrame(selected_rows)


def _expanded_network_candidates(
    episodes: pd.DataFrame,
    model_link_map_path: Path,
) -> pd.DataFrame:
    source = Path(model_link_map_path)
    mapping = pd.read_csv(source, dtype={"tmc": "string"}, low_memory=False)
    required = {
        "tmc",
        "link_id",
        "from_node_id",
        "to_node_id",
        "node_pair_tmc_rank",
        "selected_for_node_pair_lookup",
    }
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"{source} is missing frozen mapping fields: {missing}")
    mapping["selected_for_node_pair_lookup"] = (
        mapping["selected_for_node_pair_lookup"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    mapping["node_pair_tmc_rank"] = pd.to_numeric(
        mapping["node_pair_tmc_rank"], errors="coerce"
    ).astype("Int64")
    mapping = mapping.loc[mapping["selected_for_node_pair_lookup"]].copy()
    if not mapping["node_pair_tmc_rank"].eq(1).all():
        raise ValueError("A frozen projection mapping winner does not have rank 1")
    if mapping.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError("Frozen projection mapping has duplicate node-pair winners")
    mapping["_map_occurrence"] = mapping["first_map_occurrence"]
    mapping["tmc_code"] = (
        mapping["tmc"].astype("string").str.strip().str.upper()
    )
    mapping = mapping.rename(
        columns={
            "road": "network_road",
            "direction": "network_direction",
            "road_order": "network_road_order",
            "link_id": "net_link_id",
            "from_node_id": "net_from_node_id",
            "to_node_id": "net_to_node_id",
            "length_mi": "network_length_mi",
            "lanes": "network_lanes",
            "capacity": "network_capacity",
            "sequence": "network_sequence",
            "distance_to_tmc_ft": "network_match_distance_ft",
            "bearing_diff_deg": "network_bearing_diff_deg",
        }
    ).drop(columns=["tmc"])
    for column in (
        "net_link_id",
        "net_from_node_id",
        "net_to_node_id",
        "network_sequence",
        "network_match_distance_ft",
        "network_bearing_diff_deg",
    ):
        mapping[column] = pd.to_numeric(mapping[column], errors="coerce")
    observed = episodes.rename(
        columns={"road_order": "observed_road_order"}
    )
    expanded = observed.drop(
        columns=[
            column
            for column in (
                "network_link_id",
                "network_from_node_id",
                "network_to_node_id",
            )
            if column in episodes
        ]
    ).merge(mapping, on="tmc_code", how="left")
    # The matcher owns corridor ordering.  Retain the CBI profile value only
    # as an audit field so pandas cannot silently suffix both columns during
    # the merge and hide the authoritative matcher value.
    expanded["road_order"] = expanded["network_road_order"]
    return expanded


def _select_tmc_period_candidates(expanded: pd.DataFrame) -> pd.DataFrame:
    """Retain only the frozen canonical node-pair winner in every period."""

    mapped = expanded[expanded["net_link_id"].notna()].copy()
    unmapped = expanded[expanded["net_link_id"].isna()].copy()
    boundary_columns = [
        "assignment_t0_hour",
        "assignment_t2_hour",
        "assignment_t3_hour",
        "assignment_vt2_mph",
    ]
    for column in boundary_columns:
        mapped[column] = pd.to_numeric(mapped.get(column), errors="coerce")
    mapped["_native_boundary_ready"] = (
        mapped[boundary_columns].notna().all(axis=1)
        & mapped["assignment_t0_hour"].lt(mapped["assignment_t2_hour"])
        & mapped["assignment_t2_hour"].lt(mapped["assignment_t3_hour"])
    )
    profile_json = mapped.get(
        "assignment_speed_profile_json", pd.Series("", index=mapped.index)
    ).fillna("").astype(str)
    mapped["_native_assignment_ready"] = (
        mapped["_native_boundary_ready"]
        | (
            mapped.get(
                "assignment_curve_source",
                pd.Series("", index=mapped.index),
            ).eq("taplite_spd_profile")
            & profile_json.ne("")
        )
    )
    mapped["node_pair_tmc_rank"] = pd.to_numeric(
        mapped.get("node_pair_tmc_rank"), errors="coerce"
    )
    mapped["tmc_link_rank"] = pd.to_numeric(
        mapped.get("tmc_link_rank"), errors="coerce"
    )
    mapped["_distance_missing"] = mapped["network_match_distance_ft"].isna()
    mapped = mapped.sort_values(
        [
            "corridor",
            "tmc_code",
            "period",
            "node_pair_tmc_rank",
            "tmc_link_rank",
            "_distance_missing",
            "network_match_distance_ft",
            "_map_occurrence",
            "net_link_id",
        ],
        ascending=[True, True, True, True, True, True, True, True, True],
        kind="mergesort",
    )
    mapped["candidate_link_count"] = mapped.groupby(
        ["corridor", "tmc_code", "period"]
    )["net_link_id"].transform("nunique")
    selected = mapped.drop_duplicates(
        ["corridor", "tmc_code", "period"], keep="first"
    ).copy()
    selected["link_selection_basis"] = np.where(
        selected["_native_assignment_ready"],
        "frozen_node_pair_winner_with_period_coverage",
        "frozen_node_pair_winner_without_period_coverage",
    )
    if not unmapped.empty:
        unmapped["candidate_link_count"] = 0
        unmapped["link_selection_basis"] = "unmapped"
        selected = pd.concat([selected, unmapped], ignore_index=True, sort=False)
    return selected.drop(
        columns=[
            "_distance_missing",
            "_native_boundary_ready",
            "_native_assignment_ready",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _finite(row: pd.Series, fields: list[str]) -> bool:
    return all(pd.notna(row.get(field)) for field in fields)


def build_projection_table(
    products: CBIProducts,
    assignment: pd.DataFrame,
    settings: DashboardSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one projection row per observed corridor TMC and period."""

    representative = _representative_average_episodes(
        products.average_accepted
    )
    profile_columns = [
        column
        for column in (
            "corridor",
            "sensor_uid",
            "tmc_code",
            "link_id",
            "direction",
            "road_order",
            "length_mi",
            "lanes",
            "capacity_vphpl",
            "free_flow_speed_model_mph",
            "speed_at_capacity_mph",
        )
        if column in products.profiles
    ]
    profile_links = products.profiles[profile_columns].drop_duplicates(
        ["corridor", "sensor_uid"]
    )
    profile_links = profile_links.assign(_join=1).merge(
        pd.DataFrame({"period": list(settings.periods), "_join": 1}),
        on="_join",
    ).drop(columns="_join")
    episode_columns = [
        column
        for column in representative.columns
        if column not in profile_links.columns
        or column in {"corridor", "sensor_uid", "period"}
    ]
    representative = profile_links.merge(
        representative[episode_columns],
        on=["corridor", "sensor_uid", "period"],
        how="left",
        validate="one_to_one",
    )
    parameters = products.parameters[
        [field for field in CALIBRATION_FIELDS if field in products.parameters]
    ].copy()
    parameters = parameters.sort_values(
        ["sensor_uid", "period", "calibration_scope"],
        kind="mergesort",
    ).drop_duplicates(["sensor_uid", "period"], keep="first")
    representative = representative.merge(
        parameters,
        on=["sensor_uid", "period"],
        how="left",
        suffixes=("", "_calibration"),
    )
    expanded = _expanded_network_candidates(
        representative, settings.model_link_map_path
    )
    expanded["net_link_id"] = pd.to_numeric(
        expanded["net_link_id"], errors="coerce"
    ).astype("Int64")
    assignment = assignment.copy()
    assignment["net_link_id"] = pd.to_numeric(
        assignment["net_link_id"], errors="coerce"
    ).astype("Int64")
    expanded = expanded.merge(
        assignment,
        on=["net_link_id", "period"],
        how="left",
        validate="many_to_one",
    )
    selected = _select_tmc_period_candidates(expanded)

    selected["P_A"] = pd.to_numeric(selected["P_hr"], errors="coerce")
    selected["DC_obs"] = pd.to_numeric(
        selected["demand_capacity_ratio"], errors="coerce"
    )
    selected["t2_obs"] = pd.to_numeric(selected["t2_hour"], errors="coerce")
    selected["vt2_obs"] = pd.to_numeric(
        selected["min_speed_mph"], errors="coerce"
    )
    selected["P_B"] = np.nan
    selected["P_C_vol"] = pd.to_numeric(
        selected["assignment_P_hour"], errors="coerce"
    )
    selected["vt2_B"] = np.nan
    selected["vt2_C_vol"] = pd.to_numeric(
        selected["assignment_vt2_mph"], errors="coerce"
    )
    selected["projected_t0_hour"] = pd.to_numeric(
        selected["assignment_t0_hour"], errors="coerce"
    )
    selected["projected_t2_hour"] = pd.to_numeric(
        selected["assignment_t2_hour"], errors="coerce"
    )
    selected["projected_t3_hour"] = pd.to_numeric(
        selected["assignment_t3_hour"], errors="coerce"
    )

    calibration_fields = ["f_d", "n", "f_p", "s"]
    for index, row in selected.iterrows():
        if not _finite(row, calibration_fields):
            continue
        parameters_dict = {
            field: float(row[field]) for field in calibration_fields
        }
        if pd.notna(row["DC_obs"]):
            duration = predict_duration_hours(
                float(row["DC_obs"]), parameters_dict
            )
            selected.at[index, "P_B"] = duration
            selected.at[index, "vt2_B"] = predict_minimum_speed(
                float(row["threshold_used"]), duration, parameters_dict
            )
    selected["has_network_mapping"] = selected["net_link_id"].notna()
    selected["has_calibration"] = selected[calibration_fields].notna().all(axis=1)
    selected["has_assignment"] = selected["dc_dta_vol"].notna()
    selected["has_accepted_episode"] = selected[
        ["t0_hour", "t2_hour", "t3_hour", "min_speed_mph"]
    ].notna().all(axis=1)
    selected["has_assignment_boundaries"] = (
        selected[
            [
                "projected_t0_hour",
                "projected_t2_hour",
                "projected_t3_hour",
                "vt2_C_vol",
            ]
        ]
        .notna()
        .all(axis=1)
        & selected["projected_t0_hour"].lt(selected["projected_t2_hour"])
        & selected["projected_t2_hour"].lt(selected["projected_t3_hour"])
    )
    selected["has_assignment_speed_profile"] = (
        selected.get(
            "assignment_curve_source",
            pd.Series("", index=selected.index),
        ).eq("taplite_spd_profile")
        & selected.get(
            "assignment_speed_profile_json",
            pd.Series("", index=selected.index),
        ).fillna("").astype(str).ne("")
    )
    selected["has_assignment_curve"] = (
        selected["has_assignment_boundaries"]
        | selected["has_assignment_speed_profile"]
    )
    selected["projection_status"] = np.select(
        [
            ~selected["has_network_mapping"],
            ~selected["has_assignment"],
            ~selected["has_assignment_curve"],
        ],
        ["unmapped", "no_assignment", "no_assignment_curve"],
        default="ready",
    )
    selected["network_order"] = (
        pd.to_numeric(selected["road_order"], errors="coerce").fillna(0.0)
        + pd.to_numeric(
            selected["network_sequence"], errors="coerce"
        ).fillna(0.0)
        / 1000.0
    )
    selected = selected.sort_values(
        ["corridor", "period", "network_order", "net_link_id"],
        na_position="last",
    ).reset_index(drop=True)

    coverage = products.coverage.copy()
    projected = (
        selected.groupby("corridor")
        .agg(
            selected_link_periods=("period", "size"),
            ready_link_periods=(
                "projection_status",
                lambda values: int((values == "ready").sum()),
            ),
            assignment_links=("net_link_id", "nunique"),
        )
        .reset_index()
    )
    coverage = coverage.merge(projected, on="corridor", how="left")
    for column in (
        "selected_link_periods",
        "ready_link_periods",
        "assignment_links",
    ):
        coverage[column] = coverage[column].fillna(0).astype(int)
    coverage["coverage_status"] = np.select(
        [
            coverage["ready_link_periods"].eq(0),
        ],
        ["no_assignment_projection"],
        default="ready",
    )
    return selected, coverage


def build_corridor_period_summary(
    projection: pd.DataFrame,
    coverage: pd.DataFrame,
    settings: DashboardSettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for corridor in coverage["corridor"]:
        for period in settings.periods:
            group = projection[
                projection["corridor"].eq(corridor)
                & projection["period"].eq(period)
            ]
            ready = group[group["projection_status"].eq("ready")]
            rows.append(
                {
                    "corridor": corridor,
                    "period": period,
                    "n_selected_links": int(group["net_link_id"].nunique()),
                    "n_ready_links": int(ready["net_link_id"].nunique()),
                    "P_A_mean": ready["P_A"].mean(),
                    "P_B_mean": ready["P_B"].mean(),
                    "P_C_mean": ready["P_C_vol"].mean(),
                    "DC_obs_mean": ready["DC_obs"].mean(),
                    "DC_assignment_mean": ready["dc_dta_vol"].mean(),
                    "duration_absolute_error_mean": (
                        ready["P_C_vol"] - ready["P_A"]
                    ).abs().mean(),
                    "projection_available": bool(not ready.empty),
                }
            )
    return pd.DataFrame(rows)
