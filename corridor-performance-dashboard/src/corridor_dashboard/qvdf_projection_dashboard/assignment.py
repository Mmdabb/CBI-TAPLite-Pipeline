from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import period_duration_hours
from .settings import DashboardSettings


ASSIGNMENT_COLUMNS = [
    "net_link_id",
    "period",
    "dc_dta_doc",
    "dc_dta_vol",
    "assignment_period_duration_hours",
    "assignment_capacity_volume",
    "assignment_volume",
    "assignment_link_capacity",
    "assignment_iteration",
    "assignment_P_hour",
    "assignment_t0_hour",
    "assignment_t2_hour",
    "assignment_t3_hour",
    "assignment_boundary_source",
    "assignment_curve_source",
    "assignment_speed_profile_json",
    "assignment_vt2_mph",
    "assignment_mu_vphpl",
    "assignment_free_speed_mph",
    "assignment_cutoff_speed_mph",
]


def _speed_profile_json(
    row: pd.Series,
    speed_columns: list[str],
) -> str:
    minutes: list[int] = []
    speeds: list[float] = []
    for column in speed_columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value):
            continue
        clock = column.removeprefix("spd_mph_")
        hour, minute = clock.split(":", maxsplit=1)
        minutes.append(int(hour) * 60 + int(minute))
        speeds.append(float(value))
    if not speeds:
        return ""
    return json.dumps(
        {"time_minutes": minutes, "speed_mph": speeds},
        separators=(",", ":"),
    )


def _read_period(
    path: Path,
    period: str,
    period_duration_hours: float,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing assignment result: {path}")
    required = {
        "iteration_no",
        "link_id",
        "volume",
        "link_capacity",
        "doc",
    }
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required - header)
    if missing:
        raise ValueError(f"{path} is missing assignment columns: {missing}")
    optional = {
        "P",
        "t0",
        "t2",
        "t3",
        "vt2_mph",
        "mu",
        "free_speed_mph",
        "cutoff_speed_mph",
    }
    speed_columns = sorted(
        (column for column in header if column.startswith("spd_mph_")),
        key=lambda column: column.removeprefix("spd_mph_"),
    )
    frame = pd.read_csv(
        path,
        usecols=sorted(required | (optional & header) | set(speed_columns)),
        low_memory=False,
    )
    for column in required | (optional & header) | set(speed_columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in optional - set(frame.columns):
        frame[column] = np.nan
    frame = frame.dropna(subset=["link_id"]).copy()
    frame["link_id"] = frame["link_id"].astype("int64")
    frame = (
        frame.sort_values(["link_id", "iteration_no"], kind="mergesort")
        .drop_duplicates("link_id", keep="last")
        .reset_index(drop=True)
    )
    if not np.isfinite(period_duration_hours) or period_duration_hours <= 0:
        raise ValueError(
            f"Period {period} must have a positive duration in hours"
        )
    capacity = frame["link_capacity"].where(frame["link_capacity"] > 0)
    frame["assignment_period_duration_hours"] = period_duration_hours
    frame["assignment_capacity_volume"] = capacity * period_duration_hours
    frame["dc_dta_vol"] = (
        frame["volume"] / frame["assignment_capacity_volume"]
    )
    frame["dc_dta_doc"] = frame["doc"]
    frame["period"] = period
    frame = frame.rename(
        columns={
            "link_id": "net_link_id",
            "volume": "assignment_volume",
            "link_capacity": "assignment_link_capacity",
            "iteration_no": "assignment_iteration",
            "P": "assignment_P_hour",
            "t0": "assignment_t0_hour",
            "t2": "assignment_t2_hour",
            "t3": "assignment_t3_hour",
            "vt2_mph": "assignment_vt2_mph",
            "mu": "assignment_mu_vphpl",
            "free_speed_mph": "assignment_free_speed_mph",
            "cutoff_speed_mph": "assignment_cutoff_speed_mph",
        }
    )
    boundary_valid = (
        frame[
            [
                "assignment_t0_hour",
                "assignment_t2_hour",
                "assignment_t3_hour",
            ]
        ]
        .notna()
        .all(axis=1)
        & frame["assignment_t0_hour"].lt(frame["assignment_t2_hour"])
        & frame["assignment_t2_hour"].lt(frame["assignment_t3_hour"])
    )
    frame["assignment_speed_profile_json"] = ""
    for index in frame.index[~boundary_valid]:
        frame.at[index, "assignment_speed_profile_json"] = (
            _speed_profile_json(frame.loc[index], speed_columns)
        )
    has_speed_profile = frame["assignment_speed_profile_json"].ne("")
    frame["assignment_boundary_source"] = "taplite_link_performance"
    frame["assignment_curve_source"] = np.select(
        [boundary_valid, has_speed_profile],
        ["qvdf_from_taplite_boundaries", "taplite_spd_profile"],
        default="missing",
    )
    frame["dc_dta_vol"] = frame["dc_dta_vol"].replace(
        [np.inf, -np.inf], np.nan
    )
    return frame[ASSIGNMENT_COLUMNS]


def build_assignment_extract(
    settings: DashboardSettings,
) -> tuple[Path, pd.DataFrame]:
    """Create the authoritative AM/MD/PM dashboard assignment table.

    ``dc_dta_vol`` is calculated as period volume divided by hourly link
    capacity times the period duration. The TAPLite ``doc`` field is retained
    as an independent audit of the same convention.
    """

    parts = []
    for period in settings.periods:
        path = (
            settings.assignment_root
            / period.lower()
            / "link_performance.csv"
        )
        duration_hours = period_duration_hours(period, settings.periods)
        parts.append(_read_period(path, period, duration_hours))
    assignment = pd.concat(parts, ignore_index=True)
    assignment = assignment.sort_values(["period", "net_link_id"])
    if assignment.duplicated(["net_link_id", "period"]).any():
        raise ValueError("Assignment extract contains duplicate link-period keys")

    settings.dashboard_data_root.mkdir(parents=True, exist_ok=True)
    output = settings.dashboard_data_root / "dtalite_assignment_dc.csv"
    assignment.to_csv(output, index=False)
    return output, assignment
