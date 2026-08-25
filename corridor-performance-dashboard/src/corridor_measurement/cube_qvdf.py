"""Reconstruct TAPLite QVDF link profiles with Cube period volumes.

The equations mirror ``Link_QueueVDF`` in TAPLite4MPO's C++ kernel at the
recorded public commit.  Keeping this as a separate module makes the Cube
counterfactual independently testable and usable outside the corridor plots.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


TAPLITE_QVDF_KERNEL_COMMIT = "ab9bd2e369e0e1f00b327603d5bc9f479956b4ee"
TAPLITE_QVDF_KERNEL_BLOB = "fe025de83a3c1ce9eeb4b30b63dd0e595c92ef1e"
TAPLITE_QVDF_KERNEL_URL = (
    "https://github.com/asu-trans-ai-lab/TAPLite4MPO/blob/"
    f"{TAPLITE_QVDF_KERNEL_COMMIT}/kernel/src/TAPLite.cpp#L7224-L7624"
)
CUBE_VOLUME_COLUMNS = {
    "am": "I4AMVOL",
    "md": "I4MDVOL",
    "pm": "I4PMVOL",
}


def _finite_number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_hour(parameters: Mapping[str, object], name: str) -> float | None:
    value = parameters.get(name)
    if value is None or str(value).strip() == "":
        return None
    hour = _finite_number(value, float("nan"))
    if not math.isfinite(hour) or not 0.0 <= hour <= 24.0:
        return None
    return hour


def _optional_positive(
    parameters: Mapping[str, object], name: str
) -> float | None:
    value = _finite_number(parameters.get(name), float("nan"))
    return value if math.isfinite(value) and value > 0.0 else None


def _monotone_hermite_value(
    start_speed: float,
    end_speed: float,
    start_slope: float,
    end_slope: float,
    span_hours: float,
    factor: float,
) -> float:
    position = min(1.0, max(0.0, factor))
    position2 = position * position
    position3 = position2 * position
    value = (
        (2.0 * position3 - 3.0 * position2 + 1.0) * start_speed
        + (position3 - 2.0 * position2 + position)
        * span_hours
        * start_slope
        + (-2.0 * position3 + 3.0 * position2) * end_speed
        + (position3 - position2) * span_hours * end_slope
    )
    return min(
        max(start_speed, end_speed),
        max(min(start_speed, end_speed), value),
    )


def _smoothstep01(value: float) -> float:
    position = min(1.0, max(0.0, value))
    return position * position * (3.0 - 2.0 * position)


def qvdf_link_profile(
    parameters: Mapping[str, object],
    volume: float,
    *,
    period_start_min: int,
    period_end_min: int,
    interval_minutes: int = 5,
) -> Dict[str, object]:
    """Return the C++ ``Link_QueueVDF`` quantities and in-period speed profile."""
    if interval_minutes <= 0 or interval_minutes % 5:
        raise ValueError(
            "interval_minutes must be a positive multiple of 5."
        )
    if period_end_min <= period_start_min:
        raise ValueError("The QVDF period end must be later than its start.")
    volume = float(volume)
    if not math.isfinite(volume) or volume < 0.0:
        raise ValueError("Cube period volume must be finite and nonnegative.")

    start_hour = period_start_min / 60.0
    end_hour = period_end_min / 60.0
    period_hours = end_hour - start_hour
    lanes = _finite_number(parameters.get("lanes"), 1.0)
    lane_capacity = _finite_number(parameters.get("capacity"), 1.0)
    plf = _finite_number(parameters.get("vdf_plf"), 1.0)
    if plf == 0.0:
        plf = 1.0
    alpha = _finite_number(parameters.get("vdf_alpha"), 0.15)
    beta = _finite_number(parameters.get("vdf_beta"), 4.0)

    free_speed = _finite_number(parameters.get("vdf_free_speed_mph"), 0.0)
    if free_speed <= 0.0:
        free_speed = _finite_number(parameters.get("free_speed"), 0.0) / 1.609
    length_mi = _finite_number(parameters.get("vdf_length_mi"), -1.0)
    if length_mi < 0.0:
        length_mi = _finite_number(parameters.get("length"), 0.0) / 1609.0
    if free_speed <= 0.0 or length_mi < 0.0:
        raise ValueError("QVDF reconstruction requires positive speed and valid length.")

    cutoff_speed = _finite_number(parameters.get("cutoff_speed"), 0.0)
    if cutoff_speed <= 0.0:
        cutoff_speed = free_speed * 0.75
    q_cp = _finite_number(parameters.get("vdf_cp"), 0.28125)
    q_cd = _finite_number(parameters.get("vdf_cd"), 1.0)
    q_n = _finite_number(parameters.get("vdf_n"), 1.0)
    q_s = _finite_number(parameters.get("vdf_s"), 4.0)

    incoming_demand = (
        volume
        / max(0.01, lanes)
        / max(0.001, period_hours)
        / max(0.0001, plf)
    )
    doc = incoming_demand / max(0.1, lane_capacity)
    congestion_ref_speed = cutoff_speed
    if doc < 1.0:
        congestion_ref_speed = (
            (1.0 - doc) * free_speed + doc * cutoff_speed
        )
    avg_queue_speed = congestion_ref_speed / (
        1.0 + alpha * math.pow(doc, beta)
    )
    congestion_duration = q_cd * math.pow(doc, q_n)
    if congestion_duration > period_hours:
        avg_period_speed = avg_queue_speed
    else:
        avg_period_speed = (
            congestion_duration / period_hours * avg_queue_speed
            + (1.0 - congestion_duration / period_hours)
            * (congestion_ref_speed + free_speed)
            / 2.0
        )
    avg_period_travel_time = length_mi / max(0.1, avg_period_speed) * 60.0

    base = q_cp * math.pow(congestion_duration, q_s) + 1.0
    vt2 = cutoff_speed / max(0.001, base)
    midpoint = (start_hour + end_hour) / 2.0
    observed_t2 = _optional_hour(parameters, "t2_hour")
    if observed_t2 is None:
        observed_t2 = _optional_hour(parameters, "t2")
    t2_within_period = (
        observed_t2 is not None and start_hour <= observed_t2 <= end_hour
    )
    t2 = float(observed_t2) if t2_within_period else midpoint

    observed_left_fraction = 0.5
    observed_t0 = _optional_hour(parameters, "t0_hour")
    observed_t3 = _optional_hour(parameters, "t3_hour")
    if (
        t2_within_period
        and observed_t0 is not None
        and observed_t3 is not None
        and observed_t0 < t2 < observed_t3
    ):
        observed_duration = observed_t3 - observed_t0
        if observed_duration > 1e-6:
            observed_left_fraction = min(
                0.95,
                max(
                    0.05,
                    (t2 - observed_t0) / max(1e-6, observed_duration),
                ),
            )

    t0 = max(start_hour, t2 - observed_left_fraction * congestion_duration)
    t3 = min(
        end_hour,
        t2 + (1.0 - observed_left_fraction) * congestion_duration,
    )
    discharge_rate = min(
        lane_capacity,
        incoming_demand / max(0.01, congestion_duration),
    )
    running_time_hours = length_mi / max(0.01, congestion_ref_speed)
    peak_wait_hours = length_mi / vt2 - running_time_hours
    gamma = (
        peak_wait_hours
        * 64.0
        * discharge_rate
        / math.pow(congestion_duration, 4.0)
        if congestion_duration > 0.0
        else 0.0
    )
    boundary_speed = max(congestion_ref_speed, avg_queue_speed)

    def speed_at(minute: int) -> float:
        hour = minute / 60.0
        if t0 <= hour <= t3:
            window_span = t3 - t0
            queue_shape = 0.0
            if window_span > 1e-9:
                position = min(1.0, max(0.0, (hour - t0) / window_span))
                peak_fraction = min(1.0, max(0.0, (t2 - t0) / window_span))
                if peak_fraction <= 1e-9:
                    queue_shape = math.pow(1.0 - position, 4.0)
                elif peak_fraction >= 1.0 - 1e-9:
                    queue_shape = math.pow(position, 4.0)
                else:
                    left_exponent = 4.0 * peak_fraction
                    right_exponent = 4.0 * (1.0 - peak_fraction)
                    queue_shape = math.pow(
                        position / peak_fraction,
                        left_exponent,
                    ) * math.pow(
                        (1.0 - position) / (1.0 - peak_fraction),
                        right_exponent,
                    )
            elif abs(hour - t2) <= 1e-9:
                queue_shape = 1.0
            queue_shape = min(1.0, max(0.0, queue_shape))
            peak_queue = peak_wait_hours * discharge_rate
            time_dependent_wait = (
                peak_queue * queue_shape / max(0.001, discharge_rate)
            )
            return length_mi / (time_dependent_wait + running_time_hours)
        if hour < t0:
            factor = (hour - start_hour) / max(0.001, t0 - start_hour)
            smooth_factor = _smoothstep01(factor)
            return (
                (1.0 - smooth_factor) * free_speed
                + smooth_factor * boundary_speed
            )
        factor = (hour - t3) / max(0.001, end_hour - t3)
        smooth_factor = _smoothstep01(factor)
        return (
            (1.0 - smooth_factor) * boundary_speed
            + smooth_factor * free_speed
        )

    # TAPLite always builds its reporting profile at five-minute resolution,
    # applies any observed boundary anchors, and only then emits the requested
    # columns. Reproduce that sequence before selecting the caller's interval.
    model_speed = {
        minute: speed_at(minute)
        for minute in range(period_start_min, period_end_min, 5)
    }
    start_speed = _optional_positive(parameters, "qvdf_start_speed_mph")
    end_speed = _optional_positive(parameters, "qvdf_end_speed_mph")
    # The kernel's reporting dispatcher treats positive volume as a hard guard:
    # a zero-volume profile remains flat and does not use observed anchors.
    if volume > 0.0 and (start_speed is not None or end_speed is not None):
        profile_last_min = max(period_start_min, period_end_min - 5)
        profile_start_hour = period_start_min / 60.0
        profile_last_hour = profile_last_min / 60.0
        anchor_pivot = min(profile_last_hour, max(profile_start_hour, t2))
        anchor_margin = max(2.0, 0.10 * max(0.0, boundary_speed - vt2))

        def use_monotone_hermite(anchor_speed: float) -> bool:
            return (
                t2_within_period
                and anchor_speed > vt2 + anchor_margin
                and anchor_speed < boundary_speed
            )

        def use_low_anchor_connector(anchor_speed: float) -> bool:
            return (
                t2_within_period
                and anchor_speed < boundary_speed
                and anchor_speed <= vt2 + anchor_margin
            )

        if start_speed is not None:
            used_low_anchor_connector = False
            if use_low_anchor_connector(start_speed):
                left_span = anchor_pivot - profile_start_hour
                if left_span > 1e-9:
                    for minute in range(
                        period_start_min, profile_last_min + 1, 5
                    ):
                        hour = minute / 60.0
                        if hour > anchor_pivot:
                            break
                        smooth_factor = _smoothstep01(
                            (hour - profile_start_hour) / left_span
                        )
                        model_speed[minute] = (
                            (1.0 - smooth_factor) * start_speed
                            + smooth_factor * vt2
                        )
                used_low_anchor_connector = True
            used_monotone_hermite = False
            if (
                not used_low_anchor_connector
                and use_monotone_hermite(start_speed)
            ):
                for join_min in range(
                    period_start_min + 5, profile_last_min + 1, 5
                ):
                    if join_min / 60.0 > anchor_pivot:
                        break
                    join_speed = model_speed[join_min]
                    if join_speed >= start_speed:
                        continue
                    span_hours = (join_min - period_start_min) / 60.0
                    secant_slope = (join_speed - start_speed) / span_hours
                    next_min = min(profile_last_min, join_min + 5)
                    slope_span_hours = (next_min - join_min) / 60.0
                    join_slope = (
                        (model_speed[next_min] - join_speed) / slope_span_hours
                        if slope_span_hours > 1e-9
                        else 0.0
                    )
                    if join_slope > 0.0 or join_slope / secant_slope > 3.0:
                        continue
                    for minute in range(period_start_min, join_min + 1, 5):
                        factor = (minute - period_start_min) / (
                            join_min - period_start_min
                        )
                        model_speed[minute] = _monotone_hermite_value(
                            start_speed,
                            join_speed,
                            0.0,
                            join_slope,
                            span_hours,
                            factor,
                        )
                    used_monotone_hermite = True
                    break
            if not used_low_anchor_connector and not used_monotone_hermite:
                left_span = anchor_pivot - profile_start_hour
                for minute in range(period_start_min, profile_last_min + 1, 5):
                    hour = minute / 60.0
                    if hour > anchor_pivot:
                        break
                    observed_weight = (
                        1.0
                        - _smoothstep01(
                            (hour - profile_start_hour) / left_span
                        )
                        if left_span > 1e-9
                        else (1.0 if minute == period_start_min else 0.0)
                    )
                    model_speed[minute] = (
                        observed_weight * start_speed
                        + (1.0 - observed_weight) * model_speed[minute]
                    )
            model_speed[period_start_min] = start_speed

        if end_speed is not None:
            used_low_anchor_connector = False
            if use_low_anchor_connector(end_speed):
                right_span = profile_last_hour - anchor_pivot
                if right_span > 1e-9:
                    for minute in range(
                        period_start_min, profile_last_min + 1, 5
                    ):
                        hour = minute / 60.0
                        if hour < anchor_pivot:
                            continue
                        smooth_factor = _smoothstep01(
                            (hour - anchor_pivot) / right_span
                        )
                        model_speed[minute] = (
                            (1.0 - smooth_factor) * vt2
                            + smooth_factor * end_speed
                        )
                used_low_anchor_connector = True
            used_monotone_hermite = False
            if (
                not used_low_anchor_connector
                and use_monotone_hermite(end_speed)
            ):
                for join_min in range(
                    profile_last_min - 5, period_start_min - 1, -5
                ):
                    if join_min / 60.0 < anchor_pivot:
                        break
                    join_speed = model_speed[join_min]
                    if join_speed >= end_speed:
                        continue
                    span_hours = (profile_last_min - join_min) / 60.0
                    secant_slope = (end_speed - join_speed) / span_hours
                    adjacent_min = (
                        max(period_start_min, join_min - 5)
                        if join_min > anchor_pivot * 60.0 + 1e-9
                        else min(profile_last_min, join_min + 5)
                    )
                    slope_span_hours = abs(adjacent_min - join_min) / 60.0
                    if slope_span_hours > 1e-9:
                        join_slope = (
                            (join_speed - model_speed[adjacent_min])
                            / slope_span_hours
                            if adjacent_min < join_min
                            else (model_speed[adjacent_min] - join_speed)
                            / slope_span_hours
                        )
                    else:
                        join_slope = 0.0
                    if join_slope < 0.0 or join_slope / secant_slope > 3.0:
                        continue
                    for minute in range(join_min, profile_last_min + 1, 5):
                        factor = (minute - join_min) / (
                            profile_last_min - join_min
                        )
                        model_speed[minute] = _monotone_hermite_value(
                            join_speed,
                            end_speed,
                            join_slope,
                            0.0,
                            span_hours,
                            factor,
                        )
                    used_monotone_hermite = True
                    break
            if not used_low_anchor_connector and not used_monotone_hermite:
                right_span = profile_last_hour - anchor_pivot
                for minute in range(period_start_min, profile_last_min + 1, 5):
                    hour = minute / 60.0
                    if hour < anchor_pivot:
                        continue
                    observed_weight = (
                        _smoothstep01((hour - anchor_pivot) / right_span)
                        if right_span > 1e-9
                        else (1.0 if minute == profile_last_min else 0.0)
                    )
                    model_speed[minute] = (
                        (1.0 - observed_weight) * model_speed[minute]
                        + observed_weight * end_speed
                    )
            model_speed[profile_last_min] = end_speed

    speed_by_minute = {
        minute: model_speed[minute]
        for minute in range(period_start_min, period_end_min, interval_minutes)
    }
    severe_duration = sum(
        5.0 / 60.0
        for speed in model_speed.values()
        if speed < free_speed * 0.5
    )
    return {
        "D": incoming_demand,
        "doc": doc,
        "P": congestion_duration,
        "t0": t0,
        "t2": t2,
        "t3": t3,
        "vt2_mph": vt2,
        "mu": discharge_rate,
        "Q_gamma": gamma,
        "free_speed_mph": free_speed,
        "cutoff_speed_mph": cutoff_speed,
        "congestion_ref_speed_mph": congestion_ref_speed,
        "avg_queue_speed_mph": avg_queue_speed,
        "avg_QVDF_period_speed_mph": avg_period_speed,
        "avg_QVDF_period_travel_time": avg_period_travel_time,
        "Severe_Congestion_P": severe_duration,
        "speed_by_minute": speed_by_minute,
    }


def load_cube_qvdf_profiles(
    link_source: Path,
    *,
    cube_volume_column: str,
    period_start_min: int,
    period_end_min: int,
    interval_minutes: int = 5,
    link_ids: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    """Load GMNS links and return a link-performance-shaped Cube QVDF table."""
    if not link_source.is_file():
        raise FileNotFoundError(f"TAPlite GMNS link file not found: {link_source}")
    header = pd.read_csv(link_source, nrows=0)
    required = {
        "link_id",
        cube_volume_column,
        "lanes",
        "capacity",
        "vdf_plf",
        "vdf_alpha",
        "vdf_beta",
        "vdf_free_speed_mph",
        "vdf_length_mi",
        "vdf_cp",
        "vdf_cd",
        "vdf_n",
        "vdf_s",
    }
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(
            f"{link_source} is missing Cube QVDF columns: {', '.join(missing)}"
        )
    optional = {
        "free_speed",
        "length",
        "cutoff_speed",
        "t0_hour",
        "t2_hour",
        "t2",
        "t3_hour",
        "qvdf_start_speed_mph",
        "qvdf_end_speed_mph",
        "link_type",
    }
    columns = sorted(required.union(optional.intersection(header.columns)))
    links = pd.read_csv(
        link_source,
        usecols=columns,
        dtype={"link_id": "string"},
    )
    links["link_id"] = links["link_id"].str.strip()
    if links["link_id"].duplicated().any():
        raise ValueError(f"Duplicate link_id rows found in {link_source}")
    if link_ids is not None:
        selected_ids = {str(value).strip() for value in link_ids}
        links = links[links["link_id"].isin(selected_ids)].copy()
    links = links.sort_values("link_id", kind="stable")

    minutes = list(
        range(period_start_min, period_end_min, interval_minutes)
    )
    speed_columns = {
        minute: f"spd_mph_{minute // 60:02d}:{minute % 60:02d}"
        for minute in minutes
    }
    profile_rows = []
    audit_rows = []
    for row in links.to_dict(orient="records"):
        volume = _finite_number(row.get(cube_volume_column), float("nan"))
        audit = {
            "link_id": str(row["link_id"]),
            "cube_volume_column": cube_volume_column,
            "cube_period_volume": volume,
            "cube_volume_available": math.isfinite(volume) and volume >= 0.0,
        }
        if not audit["cube_volume_available"]:
            profile_rows.append(
                {
                    "link_id": str(row["link_id"]),
                    "volume": np.nan,
                    "doc": np.nan,
                    "P": np.nan,
                    **{column: np.nan for column in speed_columns.values()},
                }
            )
            audit_rows.append(audit)
            continue
        result = qvdf_link_profile(
            row,
            volume,
            period_start_min=period_start_min,
            period_end_min=period_end_min,
            interval_minutes=interval_minutes,
        )
        profile_rows.append(
            {
                "link_id": str(row["link_id"]),
                "volume": volume,
                "doc": result["doc"],
                "P": result["P"],
                **{
                    speed_columns[minute]: result["speed_by_minute"][minute]
                    for minute in minutes
                },
            }
        )
        audit.update(
            {
                "cube_qvdf_D": result["D"],
                "cube_qvdf_doc": result["doc"],
                "cube_qvdf_p_hours": result["P"],
                "cube_qvdf_t0": result["t0"],
                "cube_qvdf_t2": result["t2"],
                "cube_qvdf_t3": result["t3"],
                "cube_qvdf_vt2_mph": result["vt2_mph"],
                "cube_qvdf_mu": result["mu"],
                "cube_qvdf_gamma": result["Q_gamma"],
                "cube_qvdf_free_speed_mph": result["free_speed_mph"],
                "cube_qvdf_cutoff_speed_mph": result["cutoff_speed_mph"],
                "cube_qvdf_avg_queue_speed_mph": result["avg_queue_speed_mph"],
                "cube_qvdf_avg_period_speed_mph": result[
                    "avg_QVDF_period_speed_mph"
                ],
                "cube_qvdf_severe_congestion_p_hours": result[
                    "Severe_Congestion_P"
                ],
                "qvdf_kernel_commit": TAPLITE_QVDF_KERNEL_COMMIT,
                "qvdf_kernel_blob": TAPLITE_QVDF_KERNEL_BLOB,
                "qvdf_kernel_url": TAPLITE_QVDF_KERNEL_URL,
            }
        )
        audit_rows.append(audit)

    profile = pd.DataFrame(profile_rows)
    if profile.empty:
        profile = pd.DataFrame(
            columns=["link_id", "volume", "doc", "P", *speed_columns.values()]
        )
    profile = profile.set_index("link_id", verify_integrity=True)
    minute_lookup = {column: minute for minute, column in speed_columns.items()}
    return profile, minute_lookup, pd.DataFrame(audit_rows)
