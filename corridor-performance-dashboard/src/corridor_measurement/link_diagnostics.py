"""Mapped-link assignment, INRIX/S3, and kernel-formula diagnostics."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


NUMERIC_TOLERANCE = 1e-4
PLF_ROUNDING_TOLERANCE = 5e-3


def kernel_reconciliation(
    frame: pd.DataFrame,
    *,
    period_duration_hours: float,
) -> pd.DataFrame:
    """Recompute Link_QueueVDF D, D/C, and PLF from recorded fields.

    The kernel defines ``D`` (``IncomingDemand``) in vehicles per hour per
    lane. For a fixed assignment period of ``H`` hours:

    ``D = V / (lanes * H * PLF)`` and ``D/C = D / lane_capacity``.
    """

    result = frame.copy()
    volume = pd.to_numeric(result.get("volume"), errors="coerce")
    demand = pd.to_numeric(result.get("D"), errors="coerce")
    doc = pd.to_numeric(result.get("doc"), errors="coerce")
    plf = pd.to_numeric(result.get("vdf_plf"), errors="coerce")
    lane_capacity = pd.to_numeric(
        result.get("lane_capacity"), errors="coerce"
    )
    link_capacity = pd.to_numeric(
        result.get("link_capacity"), errors="coerce"
    )
    lanes = (link_capacity / lane_capacity).where(lane_capacity.gt(0))
    if "lanes" in result:
        fallback = pd.to_numeric(result["lanes"], errors="coerce")
        lanes = lanes.where(lanes.gt(0), fallback)

    denominator = lanes * float(period_duration_hours) * plf
    result["kernel_recomputed_D_vphpl"] = volume.div(
        denominator.where(denominator.gt(0))
    )
    result["kernel_recomputed_doc"] = result[
        "kernel_recomputed_D_vphpl"
    ].div(lane_capacity.where(lane_capacity.gt(0)))
    plf_denominator = demand * lanes * float(period_duration_hours)
    result["kernel_recomputed_plf"] = volume.div(
        plf_denominator.where(plf_denominator.gt(0))
    )
    result["kernel_D_residual"] = demand - result["kernel_recomputed_D_vphpl"]
    result["kernel_doc_residual"] = doc - result["kernel_recomputed_doc"]
    # D is written to four decimals. Inverting PLF is numerically unstable for
    # trace flows such as V=0.01, so retain the audit only where rounding cannot
    # dominate the result. The direct D and D/C checks still cover every row.
    stable_plf = volume.ge(1.0) & demand.ge(0.01)
    result["kernel_plf_residual"] = (
        plf - result["kernel_recomputed_plf"]
    ).where(stable_plf)
    direct_checks = pd.concat(
        [
            result["kernel_D_residual"].abs(),
            result["kernel_doc_residual"].abs(),
        ],
        axis=1,
    )
    plf_check = result["kernel_plf_residual"].abs()
    result["kernel_formula_status"] = np.where(
        direct_checks.max(axis=1, skipna=True).le(NUMERIC_TOLERANCE)
        & (plf_check.isna() | plf_check.le(PLF_ROUNDING_TOLERANCE)),
        "reconciled",
        "review",
    )
    result["gmns_lanes"] = lanes
    return result


def integrate_profile_window(
    profile: pd.DataFrame,
    *,
    start_min: float,
    end_min: float,
    value_column: str,
    interval_minutes: float = 15.0,
) -> float:
    """Integrate a piecewise-constant hourly rate with partial-bin overlap."""

    if end_min <= start_min or profile.empty:
        return float("nan")
    minute = pd.to_numeric(profile["t_min"], errors="coerce")
    rate = pd.to_numeric(profile[value_column], errors="coerce")
    bin_start = minute
    bin_end = minute + float(interval_minutes)
    overlap = np.maximum(
        0.0,
        np.minimum(bin_end, float(end_min))
        - np.maximum(bin_start, float(start_min)),
    )
    valid = rate.notna() & np.isfinite(overlap) & (overlap > 0)
    if not valid.any():
        return float("nan")
    return float((rate[valid] * overlap[valid] / 60.0).sum())


def eligible_episodes(
    episodes: pd.DataFrame,
    periods: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    """Keep accepted episodes whose t2 is inside the labeled period."""

    rows: list[pd.DataFrame] = []
    for period_key, definition in periods.items():
        period = str(period_key).upper()
        start_min = float(definition["start_min"])
        end_min = float(definition["end_min"])
        current = episodes[
            episodes["period"].astype("string").str.upper().eq(period)
        ].copy()
        t0 = pd.to_numeric(current["t0_hour"], errors="coerce") * 60.0
        t2 = pd.to_numeric(current["t2_hour"], errors="coerce") * 60.0
        t3 = pd.to_numeric(current["t3_hour"], errors="coerce") * 60.0
        current = current[t2.ge(start_min) & t2.lt(end_min)].copy()
        if current.empty:
            continue
        t0 = pd.to_numeric(current["t0_hour"], errors="coerce") * 60.0
        t2 = pd.to_numeric(current["t2_hour"], errors="coerce") * 60.0
        t3 = pd.to_numeric(current["t3_hour"], errors="coerce") * 60.0
        current["period_start_min"] = start_min
        current["period_end_min"] = end_min
        current["episode_t0_min"] = t0
        current["episode_t2_min"] = t2
        current["episode_t3_min"] = t3
        current["clipped_t0_min"] = np.maximum(t0, start_min)
        current["clipped_t3_min"] = np.minimum(t3, end_min)
        current["clipped_duration_hours"] = (
            current["clipped_t3_min"] - current["clipped_t0_min"]
        ).clip(lower=0.0) / 60.0
        current = current[current["clipped_duration_hours"].gt(0)].copy()
        rows.append(current)
    if not rows:
        return episodes.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def build_episode_diagnostics(
    episodes: pd.DataFrame,
    mapping: pd.DataFrame,
    observed_profiles: pd.DataFrame,
    performance_by_period: Mapping[str, pd.DataFrame],
    periods: Mapping[str, Mapping[str, int]],
    *,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Compare assignment and inverse-S3 demand over clipped episodes."""

    accepted = eligible_episodes(episodes, periods)
    if accepted.empty:
        return pd.DataFrame()
    if "link_id" in accepted:
        accepted = accepted.rename(columns={"link_id": "cbi_internal_link_id"})
    accepted["period"] = accepted["period"].astype("string").str.upper()
    accepted["tmc_code"] = accepted["tmc_code"].astype("string").str.strip()
    links = mapping.copy()
    links["period"] = links["period"].astype("string").str.upper()
    links["tmc_code"] = links["tmc_code"].astype("string").str.strip()
    links["link_id"] = links["link_id"].astype("string").str.strip()
    joined = accepted.merge(
        links,
        on=["corridor", "period", "tmc_code"],
        how="inner",
        suffixes=("_episode", "_mapping"),
        validate="many_to_many",
    )

    performance_parts: list[pd.DataFrame] = []
    for period, performance in performance_by_period.items():
        current = performance.copy()
        current["period"] = str(period).upper()
        current["link_id"] = current["link_id"].astype("string").str.strip()
        keep = [
            column
            for column in (
                "period",
                "link_id",
                "volume",
                "D",
                "doc",
                "vdf_plf",
                "P",
                "t0",
                "t2",
                "t3",
                "gmns_lanes",
                "lane_capacity",
                "link_capacity",
            )
            if column in current
        ]
        performance_parts.append(current[keep])
    performance_lookup = pd.concat(performance_parts, ignore_index=True)
    joined = joined.merge(
        performance_lookup,
        on=["period", "link_id"],
        how="left",
        validate="many_to_one",
    )

    profile = observed_profiles.copy()
    profile["tmc_code"] = profile["tmc_code"].astype("string").str.strip()
    profile_groups = {
        (str(corridor), str(tmc)): group
        for (corridor, tmc), group in profile.groupby(
            ["corridor", "tmc_code"], sort=False
        )
    }
    synthetic_volume: list[float] = []
    for row in joined.itertuples(index=False):
        current = profile_groups.get((str(row.corridor), str(row.tmc_code)))
        if current is None:
            synthetic_volume.append(float("nan"))
            continue
        per_lane_volume = integrate_profile_window(
            current,
            start_min=float(row.clipped_t0_min),
            end_min=float(row.clipped_t3_min),
            value_column="observed_derived_flow_vphpl",
            interval_minutes=float(interval_minutes),
        )
        lanes = float(row.gmns_lanes) if pd.notna(row.gmns_lanes) else np.nan
        synthetic_volume.append(per_lane_volume * lanes)
    joined["synthetic_episode_volume"] = synthetic_volume
    lanes = pd.to_numeric(joined["gmns_lanes"], errors="coerce")
    duration = pd.to_numeric(
        joined["clipped_duration_hours"], errors="coerce"
    )
    joined["synthetic_episode_D_vphpl"] = joined[
        "synthetic_episode_volume"
    ].div((lanes * duration).where((lanes * duration).gt(0)))
    assigned_d = pd.to_numeric(joined["D"], errors="coerce")
    joined["assignment_episode_volume_at_D"] = (
        assigned_d * lanes * duration
    )
    joined["synthetic_minus_assignment_D_vphpl"] = (
        joined["synthetic_episode_D_vphpl"] - assigned_d
    )
    return joined


def classify_link_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach transparent suspicious-link and likely-cause labels."""

    result = frame.copy()
    volume = pd.to_numeric(result["assignment_volume"], errors="coerce")
    doc = pd.to_numeric(result["assignment_doc"], errors="coerce")
    synthetic = pd.to_numeric(
        result["synthetic_period_volume"], errors="coerce"
    )
    confidence = pd.to_numeric(result.get("map_confidence"), errors="coerce")
    status = result.get("map_status", pd.Series("", index=result.index))
    status = status.astype("string").str.casefold()
    formula = result.get(
        "kernel_formula_status", pd.Series("reconciled", index=result.index)
    ).astype("string")
    map_review = (
        ~status.eq("matched")
        | confidence.fillna(0).lt(50)
        | result.get(
            "qa_status", pd.Series("", index=result.index)
        ).astype("string").str.startswith("review")
    )
    positive_observed = synthetic.fillna(0).gt(0)
    zero = volume.fillna(0).le(0)
    low = doc.fillna(np.inf).le(0.10)
    moderately_low = doc.fillna(np.inf).le(0.25)
    ratio = volume.div(synthetic.where(synthetic.gt(0)))
    result["assignment_to_synthetic_volume_ratio"] = ratio
    result["assignment_minus_synthetic_period_volume"] = volume - synthetic
    result["assignment_zero_flag"] = zero
    result["assignment_doc_le_0_10_flag"] = low
    result["assignment_doc_le_0_25_flag"] = moderately_low
    result["mapmatching_review_flag"] = map_review
    result["suspicious_flag"] = zero | low | formula.ne("reconciled") | map_review
    conditions = [
        condition.fillna(False).to_numpy(dtype=bool)
        for condition in (
            formula.ne("reconciled"),
            map_review,
            zero & positive_observed,
            low & positive_observed & ratio.lt(0.25),
            low,
        )
    ]
    cause = np.select(
        conditions,
        [
            "calculation_mismatch",
            "mapmatching_review",
            "assignment_zero_despite_observed_demand",
            "assignment_underloaded_vs_observed",
            "plausible_low_assignment_loading",
        ],
        default="no_primary_issue",
    )
    result["likely_cause"] = cause
    return result


def deterministic_review_sample(
    frame: pd.DataFrame,
    *,
    per_group: int = 5,
    priority_corridors: Iterable[str] = ("I66_EB", "I66_WB"),
) -> pd.DataFrame:
    """Select repeatable zero/low-volume samples, prioritizing major routes."""

    suspicious = frame[frame["suspicious_flag"]].copy()
    if suspicious.empty:
        return suspicious
    priority = set(priority_corridors)
    suspicious["_priority"] = ~suspicious["corridor"].isin(priority)
    suspicious["_severity"] = np.select(
        [
            suspicious["assignment_zero_flag"],
            suspicious["assignment_doc_le_0_10_flag"],
            suspicious["assignment_doc_le_0_25_flag"],
        ],
        [0, 1, 2],
        default=3,
    )
    suspicious["_confidence_sort"] = -pd.to_numeric(
        suspicious.get("map_confidence"), errors="coerce"
    ).fillna(-1)
    suspicious = suspicious.sort_values(
        [
            "_priority",
            "corridor",
            "period",
            "_severity",
            "_confidence_sort",
            "tmc_code",
            "link_id",
        ],
        kind="stable",
    )
    sample = (
        suspicious.groupby(
            ["corridor", "period", "_severity"], sort=False, group_keys=False
        )
        .head(int(per_group))
        .drop(columns=["_priority", "_severity", "_confidence_sort"])
    )
    return sample.reset_index(drop=True)
