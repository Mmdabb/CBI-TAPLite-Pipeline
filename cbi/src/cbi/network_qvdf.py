from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .calibration import QVDFCalibration, fit_qvdf
from .config import PERIODS, period_duration_hours


PERIOD_SEQUENCE = {"AM": 1, "MD": 2, "PM": 3}
NETWORK_PERIOD_FOLDERS = {"AM": "am", "MD": "md", "PM": "pm"}
UINT32_MAX = np.iinfo(np.uint32).max
OBSERVED_LINK_PLF_DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("plf_am", "<f4"),
        ("plf_md", "<f4"),
        ("plf_pm", "<f4"),
    ]
)
OBSERVED_LINK_SPEED_BOUNDARY_DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("qvdf_start_speed_mph_am", "<f4"),
        ("qvdf_end_speed_mph_am", "<f4"),
        ("qvdf_start_speed_mph_md", "<f4"),
        ("qvdf_end_speed_mph_md", "<f4"),
        ("qvdf_start_speed_mph_pm", "<f4"),
        ("qvdf_end_speed_mph_pm", "<f4"),
    ]
)
OBSERVED_LINK_T2_DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("observed_t0_hour_am", "<f4"),
        ("observed_t2_hour_am", "<f4"),
        ("observed_t3_hour_am", "<f4"),
        ("observed_t0_hour_md", "<f4"),
        ("observed_t2_hour_md", "<f4"),
        ("observed_t3_hour_md", "<f4"),
        ("observed_t0_hour_pm", "<f4"),
        ("observed_t2_hour_pm", "<f4"),
        ("observed_t3_hour_pm", "<f4"),
    ]
)
RESOURCE_ID_COLUMNS = [
    "data_type",
    "link_id",
    "tmc_corridor_name",
    "from_node_id",
    "to_node_id",
    "vdf_code",
]
RESOURCE_PARAMETERS = ("plf", "qdf", "n", "s", "cp", "cd", "alpha", "beta")
RESOURCE_COLUMNS = [
    *RESOURCE_ID_COLUMNS,
    *[
        f"QVDF_{parameter}{sequence}"
        for sequence in PERIOD_SEQUENCE.values()
        for parameter in RESOURCE_PARAMETERS
    ],
]
CALIBRATION_VALUE_COLUMNS = [
    "demand_capacity_ratio",
    "P_hr",
    "min_speed_mph",
    "threshold_used",
    "qdf",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_vdf_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _code_sort_key(code: str) -> tuple[int, float | str]:
    try:
        return (0, float(code))
    except ValueError:
        return (1, code)


def _normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


def _pack_node_pairs(
    from_node_id: np.ndarray,
    to_node_id: np.ndarray,
) -> np.ndarray:
    from_values = np.asarray(from_node_id, dtype=float)
    to_values = np.asarray(to_node_id, dtype=float)
    if from_values.shape != to_values.shape:
        raise ValueError("from_node_id and to_node_id shapes differ")
    for label, values in (
        ("from_node_id", from_values),
        ("to_node_id", to_values),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite values")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label} contains non-integer values")
        if (values < 0).any() or (values > UINT32_MAX).any():
            raise ValueError(f"{label} must fit unsigned 32-bit integers")
    return (
        from_values.astype(np.uint64) << np.uint64(32)
    ) | to_values.astype(np.uint64)


def discover_link_reference_files(cbi_run_dir: Path) -> list[Path]:
    root = Path(cbi_run_dir).resolve()
    files = sorted(root.glob("*/01-input-and-qc/link_reference.csv"))
    if not files:
        files = sorted(root.glob("**/link_reference.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CBI link_reference.csv files were found under {root}"
        )
    return files


def discover_average_weekday_profile_files(cbi_run_dir: Path) -> list[Path]:
    root = Path(cbi_run_dir).resolve()
    files = sorted(root.glob("*/03-profiles/average_weekday_profile.csv"))
    if not files:
        files = sorted(root.glob("**/average_weekday_profile.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CBI average_weekday_profile.csv files were found under {root}"
        )
    return files


def load_observed_primary_links(cbi_run_dir: Path) -> pd.DataFrame:
    """Expand observed TMC identities onto every frozen node-pair win."""

    frames: list[pd.DataFrame] = []
    for path in discover_link_reference_files(cbi_run_dir):
        header = set(pd.read_csv(path, nrows=0).columns)
        required = {
            "tmc_code",
            "network_from_node_id",
            "network_to_node_id",
        }
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"{path} is missing link-reference columns: {missing}")
        optional = [
            column
            for column in (
                "corridor",
                "network_link_id",
                "network_mapping_status",
                "network_match_distance_ft",
                "network_match_score",
                "road_order",
                "network_node_pair_tmc_rank",
                "network_selected_for_node_pair_lookup",
            )
            if column in header
        ]
        frame = pd.read_csv(
            path,
            usecols=[*required, *optional],
            dtype={"tmc_code": "string"},
            low_memory=False,
        )
        if "corridor" not in frame:
            frame["corridor"] = path.parents[1].name
        frame["link_reference_source"] = str(path.resolve())
        frames.append(frame)

    links = pd.concat(frames, ignore_index=True, sort=False)
    links["tmc_code"] = _normalized_text(links["tmc_code"])
    links["corridor"] = _normalized_text(links["corridor"])
    if "road_order" not in links:
        links["road_order"] = np.nan
    identities = links[
        ["corridor", "tmc_code", "road_order", "link_reference_source"]
    ].drop_duplicates(["corridor", "tmc_code"])
    root = Path(cbi_run_dir).resolve()
    canonical_candidates = [
        root / "shared" / "network-mapping" / "canonical_node_pair_tmc.csv",
        root.parent / "shared" / "network-mapping" / "canonical_node_pair_tmc.csv",
    ]
    canonical_source = next(
        (path for path in canonical_candidates if path.is_file()), None
    )
    if canonical_source is None:
        raise FileNotFoundError(
            "Observed-link QVDF resources require the frozen CBI node-pair "
            f"mapping; checked: {canonical_candidates}"
        )
    if canonical_source is not None:
        required = {
            "tmc",
            "link_id",
            "from_node_id",
            "to_node_id",
            "distance_to_tmc_ft",
            "composite_match_score",
            "node_pair_tmc_rank",
            "selected_for_node_pair_lookup",
        }
        header = set(pd.read_csv(canonical_source, nrows=0).columns)
        missing = sorted(required - header)
        if missing:
            raise ValueError(
                f"{canonical_source} is missing canonical columns: {missing}"
            )
        canonical = pd.read_csv(
            canonical_source,
            usecols=sorted(required),
            dtype={"tmc": "string"},
            low_memory=False,
        )
        canonical["tmc"] = _normalized_text(canonical["tmc"])
        canonical["selected_for_node_pair_lookup"] = (
            canonical["selected_for_node_pair_lookup"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        canonical = canonical.loc[
            canonical["selected_for_node_pair_lookup"]
        ].rename(
            columns={
                "tmc": "tmc_code",
                "link_id": "network_link_id",
                "from_node_id": "network_from_node_id",
                "to_node_id": "network_to_node_id",
                "distance_to_tmc_ft": "network_match_distance_ft",
                "composite_match_score": "network_match_score",
                "node_pair_tmc_rank": "network_node_pair_tmc_rank",
                "selected_for_node_pair_lookup": (
                    "network_selected_for_node_pair_lookup"
                ),
            }
        )
        links = identities.merge(
            canonical,
            on="tmc_code",
            how="inner",
            validate="one_to_many",
        )
        links["network_mapping_status"] = "frozen_canonical_node_pair_winner"
    for column in (
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_match_distance_ft",
        "network_match_score",
        "road_order",
        "network_node_pair_tmc_rank",
    ):
        if column not in links:
            links[column] = np.nan
        links[column] = pd.to_numeric(links[column], errors="coerce")
    links = links.dropna(
        subset=["tmc_code", "network_from_node_id", "network_to_node_id"]
    ).copy()
    links["network_from_node_id"] = links["network_from_node_id"].astype(
        np.int64
    )
    links["network_to_node_id"] = links["network_to_node_id"].astype(np.int64)
    if "network_selected_for_node_pair_lookup" in links:
        links["network_selected_for_node_pair_lookup"] = (
            links["network_selected_for_node_pair_lookup"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    links["packed_key"] = _pack_node_pairs(
        links["network_from_node_id"].to_numpy(),
        links["network_to_node_id"].to_numpy(),
    )
    if links.duplicated(
        ["network_from_node_id", "network_to_node_id"]
    ).any():
        duplicates = links.loc[
            links.duplicated(
                ["network_from_node_id", "network_to_node_id"], keep=False
            ),
            ["corridor", "tmc_code", "network_from_node_id", "network_to_node_id"],
        ].head(10)
        raise ValueError(
            "Frozen canonical observations contain duplicate node pairs: "
            + ", ".join(
                f"{row.corridor}/{row.tmc_code}/"
                f"{row.network_from_node_id}->{row.network_to_node_id}"
                for row in duplicates.itertuples(index=False)
            )
        )
    return links


def _rank_node_pair_records(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the frozen mapmatching-derived node-pair winner without reranking."""

    ranked = records.copy()
    rank_source = "network_node_pair_tmc_rank"
    selected_source = "network_selected_for_node_pair_lookup"
    if rank_source in ranked and selected_source in ranked:
        ranked["node_pair_tmc_rank"] = pd.to_numeric(
            ranked[rank_source], errors="coerce"
        ).astype("Int64")
        ranked["selected_for_node_pair_lookup"] = ranked[selected_source].fillna(
            False
        ).astype(bool)
    else:
        raise ValueError(
            "Observed-link records do not carry the frozen node-pair rank "
            "and selection fields"
        )
    selected = ranked[ranked["selected_for_node_pair_lookup"]].copy()
    if not selected["node_pair_tmc_rank"].eq(1).all():
        raise ValueError("A frozen node-pair winner does not have rank 1")
    if selected["packed_key"].duplicated().any():
        raise ValueError("Observed node-pair keys are not unique")
    return ranked, selected


def build_observed_link_plf_overrides(
    cbi_run_dir: Path,
    average_weekday_episodes: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    """Build direct observed-link PLF overrides and a mmap-friendly lookup."""

    links = load_observed_primary_links(cbi_run_dir)
    episodes = average_weekday_episodes.copy()
    if "tmc_code" not in episodes:
        raise ValueError("Average-weekday episodes are missing tmc_code")
    if "corridor" not in episodes:
        episodes["corridor"] = ""
    episodes["tmc_code"] = _normalized_text(episodes["tmc_code"])
    episodes["corridor"] = _normalized_text(episodes["corridor"])
    episodes["period"] = episodes["period"].astype(str).str.upper()
    episodes["qdf"] = pd.to_numeric(episodes["qdf"], errors="coerce")
    invalid = episodes[
        episodes["period"].isin(PERIOD_SEQUENCE)
        & ~(episodes["qdf"].gt(0.0) & episodes["qdf"].le(1.0))
    ]
    if not invalid.empty:
        raise ValueError(
            "Accepted average-weekday episodes contain invalid QDF values"
        )
    observed_qdf = (
        episodes[episodes["period"].isin(PERIOD_SEQUENCE)]
        .groupby(["corridor", "tmc_code", "period"], as_index=False)
        .agg(
            observed_qdf=("qdf", "median"),
            accepted_average_weekday_episode_count=("qdf", "size"),
        )
    )

    records = links.copy()
    for period in PERIOD_SEQUENCE:
        duration = period_duration_hours(period, PERIODS)
        period_qdf = observed_qdf[observed_qdf["period"].eq(period)][
            [
                "corridor",
                "tmc_code",
                "observed_qdf",
                "accepted_average_weekday_episode_count",
            ]
        ].rename(
            columns={
                "observed_qdf": f"qdf_{period.lower()}",
                "accepted_average_weekday_episode_count": (
                    f"accepted_episode_count_{period.lower()}"
                ),
            }
        )
        records = records.merge(
            period_qdf,
            on=["corridor", "tmc_code"],
            how="left",
            validate="many_to_one",
        )
        qdf_column = f"qdf_{period.lower()}"
        count_column = f"accepted_episode_count_{period.lower()}"
        congestion_column = f"observed_congestion_{period.lower()}"
        plf_column = f"plf_{period.lower()}"
        records[congestion_column] = records[qdf_column].notna()
        records[qdf_column] = records[qdf_column].fillna(1.0 / duration)
        records[count_column] = records[count_column].fillna(0).astype(int)
        records[plf_column] = 1.0 / (records[qdf_column] * duration)

    records, selected = _rank_node_pair_records(records)
    plf_columns = [f"plf_{period.lower()}" for period in PERIOD_SEQUENCE]
    if not np.isfinite(selected[plf_columns].to_numpy(dtype=float)).all():
        raise ValueError("Observed PLF overrides contain non-finite values")
    if selected[plf_columns].le(0.0).any().any():
        raise ValueError("Observed PLF overrides must be positive")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audit_path = destination / "observed_link_plf_overrides.csv"
    records.drop(
        columns=["_missing_score", "_missing_distance"], errors="ignore"
    ).to_csv(
        audit_path, index=False
    )
    lookup_path = destination / "observed_link_plf_overrides.npy"
    lookup = np.empty(len(selected), dtype=OBSERVED_LINK_PLF_DTYPE)
    lookup["packed_key"] = selected["packed_key"].to_numpy(dtype=np.uint64)
    lookup["from_node_id"] = selected["network_from_node_id"].to_numpy(
        dtype=np.uint32
    )
    lookup["to_node_id"] = selected["network_to_node_id"].to_numpy(
        dtype=np.uint32
    )
    for period in PERIOD_SEQUENCE:
        field = f"plf_{period.lower()}"
        lookup[field] = selected[field].to_numpy(dtype=np.float32)
    lookup.sort(order="packed_key")
    np.save(lookup_path, lookup, allow_pickle=False)
    restored = np.load(lookup_path, mmap_mode="r", allow_pickle=False)
    if restored.dtype != OBSERVED_LINK_PLF_DTYPE or len(restored) != len(selected):
        raise ValueError("Observed PLF lookup round-trip validation failed")
    metadata = {
        "format": "NumPy .npy structured array sorted by packed_key",
        "unit": "dimensionless",
        "key_definition": "(uint64(from_node_id) << 32) | uint64(to_node_id)",
        "record_dtype": OBSERVED_LINK_PLF_DTYPE.descr,
        "period_fields": {period: f"plf_{period.lower()}" for period in PERIOD_SEQUENCE},
        "no_congestion_rule": "qdf=1/period_duration_hours; plf=1",
        "congestion_rule": "plf=1/(median_accepted_average_weekday_qdf*period_duration_hours)",
        "observed_tmc_rows": int(len(records)),
        "unique_node_pair_rows": int(len(selected)),
        "duplicate_node_pair_tmc_rows": int(len(records) - len(selected)),
        "lookup_path": str(lookup_path),
        "lookup_sha256": _sha256(lookup_path),
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "periods": {
            period: {
                "observed_congestion_rows": int(
                    selected[f"observed_congestion_{period.lower()}"].sum()
                ),
                "no_observed_congestion_rows": int(
                    (~selected[f"observed_congestion_{period.lower()}"]).sum()
                ),
                "plf_min": float(selected[f"plf_{period.lower()}"].min()),
                "plf_max": float(selected[f"plf_{period.lower()}"].max()),
            }
            for period in PERIOD_SEQUENCE
        },
    }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    metadata["metadata_sha256"] = _sha256(metadata_path)
    return metadata


def build_observed_link_t2_lookup(
    cbi_run_dir: Path,
    average_weekday_episodes: pd.DataFrame,
    output_dir: Path,
    *,
    invalid_episode_policy: str = "error",
) -> dict[str, object]:
    """Build accepted weekday-average observed episode times by node pair.

    The historical resource name is retained for compatibility, but each
    period now stores the complete T0/T2/T3 triplet from one representative
    accepted episode. T0 and T3 deliberately retain their observed values
    even when they extend outside the assignment period; TAPLite uses them to
    recover episode asymmetry and clips the emitted profile to the period.
    """

    if invalid_episode_policy not in {"error", "omit"}:
        raise ValueError("invalid_episode_policy must be 'error' or 'omit'")

    links = load_observed_primary_links(cbi_run_dir)
    episodes = average_weekday_episodes.copy()
    required = {
        "tmc_code",
        "period",
        "t0_hour",
        "t2_hour",
        "t3_hour",
        "min_speed_mph",
        "P_hr",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(
            f"Average-weekday episodes are missing observed-T2 columns: {missing}"
        )
    if "corridor" not in episodes:
        episodes["corridor"] = ""
    if "episode_id" not in episodes:
        episodes["episode_id"] = [
            f"accepted-average-weekday-{index}"
            for index in range(len(episodes))
        ]
    episodes["corridor"] = _normalized_text(episodes["corridor"])
    episodes["tmc_code"] = _normalized_text(episodes["tmc_code"])
    episodes["period"] = episodes["period"].astype(str).str.upper()
    episodes["episode_id"] = _normalized_text(episodes["episode_id"])
    for column in ("t0_hour", "t2_hour", "t3_hour", "min_speed_mph", "P_hr"):
        episodes[column] = pd.to_numeric(episodes[column], errors="coerce")

    supported = episodes[episodes["period"].isin(PERIOD_SEQUENCE)].copy()
    valid_t2 = np.zeros(len(supported), dtype=bool)
    for period, (start_minute, end_minute) in PERIODS.items():
        period_rows = supported["period"].eq(period)
        valid_t2 |= (
            period_rows
            & supported["t2_hour"].ge(start_minute / 60.0)
            & supported["t2_hour"].lt(end_minute / 60.0)
        ).to_numpy()
    invalid_t2 = supported.loc[~valid_t2].copy()
    if not invalid_t2.empty and invalid_episode_policy == "error":
        examples = supported.loc[
            ~valid_t2, ["corridor", "tmc_code", "period", "t2_hour"]
        ].head(10)
        raise ValueError(
            "Accepted average-weekday episodes contain missing or out-of-period "
            "observed T2 values: "
            + ", ".join(
                f"{row.corridor}/{row.tmc_code}/{row.period}={row.t2_hour}"
                for row in examples.itertuples(index=False)
            )
        )

    valid_episode = (
        supported["t0_hour"].ge(0.0)
        & supported["t0_hour"].lt(supported["t2_hour"])
        & supported["t2_hour"].lt(supported["t3_hour"])
        & supported["t3_hour"].le(24.0)
    )
    invalid_triplet = supported.loc[valid_t2 & ~valid_episode].copy()
    if not invalid_triplet.empty and invalid_episode_policy == "error":
        examples = supported.loc[
            valid_t2 & ~valid_episode,
            ["corridor", "tmc_code", "period", "t0_hour", "t2_hour", "t3_hour"],
        ].head(10)
        raise ValueError(
            "Accepted average-weekday episodes contain missing or unordered "
            "observed T0/T2/T3 values: "
            + ", ".join(
                f"{row.corridor}/{row.tmc_code}/{row.period}="
                f"{row.t0_hour}/{row.t2_hour}/{row.t3_hour}"
                for row in examples.itertuples(index=False)
            )
        )

    invalid_episodes = pd.concat(
        [
            invalid_t2.assign(invalid_observed_triplet_reason="missing_or_out_of_period_t2"),
            invalid_triplet.assign(invalid_observed_triplet_reason="missing_or_unordered_t0_t2_t3"),
        ],
        ignore_index=True,
        sort=False,
    )
    if invalid_episode_policy == "omit":
        supported = supported.loc[valid_t2 & valid_episode].copy()

    supported["_speed_sort"] = supported["min_speed_mph"].fillna(np.inf)
    supported["_duration_sort"] = -supported["P_hr"].fillna(-np.inf)
    supported["_t2_sort"] = supported["t2_hour"].fillna(np.inf)
    supported = supported.sort_values(
        [
            "corridor",
            "tmc_code",
            "period",
            "_speed_sort",
            "_duration_sort",
            "_t2_sort",
            "episode_id",
        ],
        kind="mergesort",
    )
    supported["observed_t2_representative_rank"] = (
        supported.groupby(
            ["corridor", "tmc_code", "period"], sort=False
        ).cumcount()
        + 1
    )
    group_counts = (
        supported.groupby(
            ["corridor", "tmc_code", "period"], as_index=False
        )
        .size()
        .rename(columns={"size": "accepted_episode_count"})
    )
    representatives = supported[
        supported["observed_t2_representative_rank"].eq(1)
    ].merge(
        group_counts,
        on=["corridor", "tmc_code", "period"],
        how="left",
        validate="one_to_one",
    )

    records = links.copy()
    for period in PERIOD_SEQUENCE:
        suffix = period.lower()
        period_values = representatives[representatives["period"].eq(period)][
            [
                "corridor",
                "tmc_code",
                "t0_hour",
                "t2_hour",
                "t3_hour",
                "episode_id",
                "accepted_episode_count",
            ]
        ].rename(
            columns={
                "t0_hour": f"observed_t0_hour_{suffix}",
                "t2_hour": f"observed_t2_hour_{suffix}",
                "t3_hour": f"observed_t3_hour_{suffix}",
                "episode_id": f"observed_t2_episode_id_{suffix}",
                "accepted_episode_count": f"accepted_episode_count_{suffix}",
            }
        )
        records = records.merge(
            period_values,
            on=["corridor", "tmc_code"],
            how="left",
            validate="many_to_one",
        )
        records[f"accepted_episode_count_{suffix}"] = (
            records[f"accepted_episode_count_{suffix}"].fillna(0).astype(int)
        )

    records, selected = _rank_node_pair_records(records)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    invalid_audit_path = destination / "invalid_observed_episode_triplets.csv"
    invalid_episodes.to_csv(invalid_audit_path, index=False)
    audit_path = destination / "observed_link_t2.csv"
    records.drop(
        columns=["_missing_score", "_missing_distance"], errors="ignore"
    ).to_csv(
        audit_path, index=False
    )
    lookup_path = destination / "observed_link_t2.npy"
    lookup = np.empty(len(selected), dtype=OBSERVED_LINK_T2_DTYPE)
    lookup["packed_key"] = selected["packed_key"].to_numpy(dtype=np.uint64)
    lookup["from_node_id"] = selected["network_from_node_id"].to_numpy(
        dtype=np.uint32
    )
    lookup["to_node_id"] = selected["network_to_node_id"].to_numpy(
        dtype=np.uint32
    )
    for period in PERIOD_SEQUENCE:
        suffix = period.lower()
        for boundary in ("t0", "t2", "t3"):
            field = f"observed_{boundary}_hour_{suffix}"
            lookup[field] = selected[field].to_numpy(dtype=np.float32)
    lookup.sort(order="packed_key")
    np.save(lookup_path, lookup, allow_pickle=False)
    restored = np.load(lookup_path, mmap_mode="r", allow_pickle=False)
    if restored.dtype != OBSERVED_LINK_T2_DTYPE or len(restored) != len(selected):
        raise ValueError("Observed T2 lookup round-trip validation failed")

    metadata = {
        "format": "NumPy .npy structured array sorted by packed_key",
        "key_definition": "(uint64(from_node_id) << 32) | uint64(to_node_id)",
        "record_dtype": OBSERVED_LINK_T2_DTYPE.descr,
        "time_unit": "decimal hour",
        "source": "accepted CBI weekday-average congestion episodes",
        "representative_rule": (
            "lowest minimum speed, then longest duration, then earliest T2, "
            "then episode_id"
        ),
        "endpoint_rule": (
            "preserve observed T0 and T3 even outside the assignment period; "
            "TAPLite uses them only for episode asymmetry and clips the emitted "
            "profile to the period"
        ),
        "no_episode_rule": (
            "NaN triplet; conversion clears t0_hour, t2_hour, and t3_hour on "
            "the matched observed link so spatial or ML completion cannot "
            "create observed congestion"
        ),
        "invalid_episode_policy": invalid_episode_policy,
        "invalid_episode_rows_omitted": int(len(invalid_episodes)),
        "invalid_episode_audit_path": str(invalid_audit_path),
        "invalid_episode_audit_sha256": _sha256(invalid_audit_path),
        "observed_tmc_rows": int(len(records)),
        "unique_node_pair_rows": int(len(selected)),
        "duplicate_node_pair_tmc_rows": int(len(records) - len(selected)),
        "lookup_path": str(lookup_path),
        "lookup_sha256": _sha256(lookup_path),
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "periods": {
            period: {
                "fields": [
                    f"observed_t0_hour_{period.lower()}",
                    f"observed_t2_hour_{period.lower()}",
                    f"observed_t3_hour_{period.lower()}",
                ],
                "node_pairs_with_accepted_episode": int(
                    selected[f"observed_t2_hour_{period.lower()}"].notna().sum()
                ),
                "node_pairs_without_accepted_episode": int(
                    selected[f"observed_t2_hour_{period.lower()}"].isna().sum()
                ),
            }
            for period in PERIOD_SEQUENCE
        },
    }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    metadata["metadata_sha256"] = _sha256(metadata_path)
    return metadata


def build_observed_link_speed_boundaries(
    cbi_run_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build observed weekday-average period-edge speeds by node pair.

    A finite post-QC weekday-average speed is authoritative. Only when that
    exact boundary cell is blank, use the persisted pre-average-QC weekday
    mean. This fallback is deliberately limited to assignment-period edges.
    """

    boundary_minutes = sorted(
        {minute for bounds in PERIODS.values() for minute in bounds}
    )
    frames: list[pd.DataFrame] = []
    source_files = discover_average_weekday_profile_files(cbi_run_dir)
    for path in source_files:
        header = set(pd.read_csv(path, nrows=0).columns)
        required = {"tmc_code", "t_min", "avg_weekday_speed_mph"}
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"{path} is missing profile columns: {missing}")
        usecols = [*required]
        if "avg_weekday_speed_mph_pre_qc" in header:
            usecols.append("avg_weekday_speed_mph_pre_qc")
        if "corridor" in header:
            usecols.append("corridor")
        if "n_days" in header:
            usecols.append("n_days")
        frame = pd.read_csv(
            path,
            usecols=usecols,
            dtype={"tmc_code": "string"},
            low_memory=False,
        )
        if "corridor" not in frame:
            frame["corridor"] = path.parents[1].name
        frame["profile_source_file"] = str(path.resolve())
        frame["t_min"] = pd.to_numeric(frame["t_min"], errors="coerce")
        frame = frame[frame["t_min"].isin(boundary_minutes)].copy()
        frames.append(frame)

    profile = pd.concat(frames, ignore_index=True, sort=False)
    profile["corridor"] = _normalized_text(profile["corridor"])
    profile["tmc_code"] = _normalized_text(profile["tmc_code"])
    profile["avg_weekday_speed_mph"] = pd.to_numeric(
        profile["avg_weekday_speed_mph"], errors="coerce"
    )
    if "avg_weekday_speed_mph_pre_qc" not in profile:
        profile["avg_weekday_speed_mph_pre_qc"] = np.nan
    profile["avg_weekday_speed_mph_pre_qc"] = pd.to_numeric(
        profile["avg_weekday_speed_mph_pre_qc"], errors="coerce"
    )
    profile["boundary_speed_mph"] = profile[
        "avg_weekday_speed_mph"
    ].fillna(profile["avg_weekday_speed_mph_pre_qc"])
    profile["boundary_speed_source"] = np.select(
        [
            profile["avg_weekday_speed_mph"].notna(),
            profile["avg_weekday_speed_mph_pre_qc"].notna(),
        ],
        [
            "post_qc_weekday_average",
            "pre_qc_weekday_average_fallback",
        ],
        default="missing_after_pre_qc_fallback",
    )
    if "n_days" in profile:
        profile["n_days"] = pd.to_numeric(profile["n_days"], errors="coerce")
    profile["source_boundary_row_present"] = True
    key_columns = ["corridor", "tmc_code", "t_min"]
    if profile.duplicated(key_columns).any():
        examples = profile.loc[
            profile.duplicated(key_columns, keep=False), key_columns
        ].head(10)
        raise ValueError(
            "Average-weekday profiles contain duplicate TMC boundary rows: "
            + ", ".join(
                f"{row.corridor}/{row.tmc_code}@{int(row.t_min)}"
                for row in examples.itertuples(index=False)
            )
        )

    wide = (
        profile.set_index(key_columns)["boundary_speed_mph"]
        .unstack("t_min")
        .reindex(columns=boundary_minutes)
        .reset_index()
    )
    post_qc_wide = (
        profile.set_index(key_columns)["avg_weekday_speed_mph"]
        .unstack("t_min")
        .reindex(columns=boundary_minutes)
        .reset_index()
    )
    pre_qc_wide = (
        profile.set_index(key_columns)["avg_weekday_speed_mph_pre_qc"]
        .unstack("t_min")
        .reindex(columns=boundary_minutes)
        .reset_index()
    )
    speed_source_wide = (
        profile.set_index(key_columns)["boundary_speed_source"]
        .unstack("t_min")
        .reindex(columns=boundary_minutes)
        .reset_index()
    )
    presence_wide = (
        profile.set_index(key_columns)["source_boundary_row_present"]
        .unstack("t_min")
        .reindex(columns=boundary_minutes, fill_value=False)
        .fillna(False)
        .astype(bool)
        .reset_index()
    )
    for minute in boundary_minutes:
        wide[f"source_boundary_row_present_{minute}"] = presence_wide[minute]
        wide[f"source_boundary_post_qc_speed_mph_{minute}"] = (
            post_qc_wide[minute]
        )
        wide[f"source_boundary_pre_qc_speed_mph_{minute}"] = (
            pre_qc_wide[minute]
        )
        wide[f"source_boundary_speed_source_{minute}"] = (
            speed_source_wide[minute]
        )
    if "n_days" in profile:
        n_days_wide = (
            profile.set_index(key_columns)["n_days"]
            .unstack("t_min")
            .reindex(columns=boundary_minutes)
            .reset_index()
        )
        for minute in boundary_minutes:
            wide[f"source_boundary_n_days_{minute}"] = n_days_wide[minute]
    speed_fields: list[str] = []
    for period, (start_minute, end_minute) in PERIODS.items():
        start_field = f"qvdf_start_speed_mph_{period.lower()}"
        end_field = f"qvdf_end_speed_mph_{period.lower()}"
        wide[start_field] = wide[start_minute]
        wide[end_field] = wide[end_minute]
        speed_fields.extend([start_field, end_field])
    diagnostic_fields = [
        f"source_boundary_row_present_{minute}"
        for minute in boundary_minutes
    ]
    diagnostic_fields.extend(
        f"source_boundary_post_qc_speed_mph_{minute}"
        for minute in boundary_minutes
    )
    diagnostic_fields.extend(
        f"source_boundary_pre_qc_speed_mph_{minute}"
        for minute in boundary_minutes
    )
    diagnostic_fields.extend(
        f"source_boundary_speed_source_{minute}"
        for minute in boundary_minutes
    )
    diagnostic_fields.extend(
        f"source_boundary_n_days_{minute}"
        for minute in boundary_minutes
        if f"source_boundary_n_days_{minute}" in wide
    )
    wide = wide[
        ["corridor", "tmc_code", *speed_fields, *diagnostic_fields]
    ]

    links = load_observed_primary_links(cbi_run_dir)
    records = links.merge(
        wide,
        on=["corridor", "tmc_code"],
        how="left",
        validate="many_to_one",
    )
    records, selected = _rank_node_pair_records(records)
    finite_values = selected[speed_fields].to_numpy(dtype=float)
    invalid = np.isfinite(finite_values) & (
        (finite_values <= 0.0) | (finite_values > 150.0)
    )
    if invalid.any():
        raise ValueError(
            "Observed weekday-average boundary speeds must be in (0, 150] mph"
        )

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audit_path = destination / "observed_link_speed_boundaries.csv"
    records.drop(
        columns=["_missing_score", "_missing_distance"], errors="ignore"
    ).to_csv(
        audit_path, index=False
    )
    report_frames: list[pd.DataFrame] = []
    report_id_columns = [
        "corridor",
        "tmc_code",
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_match_distance_ft",
        "road_order",
    ]
    for period, (start_minute, end_minute) in PERIODS.items():
        suffix = period.lower()
        start_field = f"qvdf_start_speed_mph_{suffix}"
        end_field = f"qvdf_end_speed_mph_{suffix}"
        report = selected[report_id_columns].copy()
        report["period"] = period
        report["start_minute"] = int(start_minute)
        report["end_minute"] = int(end_minute)
        report["qvdf_start_speed_mph"] = selected[start_field]
        report["qvdf_end_speed_mph"] = selected[end_field]
        start_finite = np.isfinite(report["qvdf_start_speed_mph"])
        end_finite = np.isfinite(report["qvdf_end_speed_mph"])
        report["boundary_status"] = np.select(
            [
                start_finite & end_finite,
                start_finite & ~end_finite,
                ~start_finite & end_finite,
            ],
            ["both", "start_only", "end_only"],
            default="neither",
        )
        for side, minute, finite in (
            ("start", start_minute, start_finite),
            ("end", end_minute, end_finite),
        ):
            present = selected[f"source_boundary_row_present_{minute}"].fillna(
                False
            )
            report[f"{side}_source_row_present"] = present.to_numpy(dtype=bool)
            report[f"{side}_post_qc_speed_mph"] = selected[
                f"source_boundary_post_qc_speed_mph_{minute}"
            ]
            report[f"{side}_pre_qc_speed_mph"] = selected[
                f"source_boundary_pre_qc_speed_mph_{minute}"
            ]
            report[f"{side}_speed_source"] = selected[
                f"source_boundary_speed_source_{minute}"
            ].fillna("source_boundary_row_absent")
            n_days_field = f"source_boundary_n_days_{minute}"
            report[f"{side}_source_n_days"] = (
                selected[n_days_field]
                if n_days_field in selected
                else np.nan
            )
            report[f"{side}_missing_cause"] = np.select(
                [finite, present.to_numpy(dtype=bool)],
                [
                    "available",
                    "source_boundary_speed_missing_after_pre_qc_fallback",
                ],
                default="source_boundary_row_absent",
            )
        report_frames.append(report)
    completeness = pd.concat(report_frames, ignore_index=True)
    completeness_path = destination / "boundary_completeness_report.csv"
    completeness.to_csv(completeness_path, index=False)
    lookup_path = destination / "observed_link_speed_boundaries.npy"
    lookup = np.empty(len(selected), dtype=OBSERVED_LINK_SPEED_BOUNDARY_DTYPE)
    lookup["packed_key"] = selected["packed_key"].to_numpy(dtype=np.uint64)
    lookup["from_node_id"] = selected["network_from_node_id"].to_numpy(
        dtype=np.uint32
    )
    lookup["to_node_id"] = selected["network_to_node_id"].to_numpy(
        dtype=np.uint32
    )
    for field in speed_fields:
        lookup[field] = selected[field].to_numpy(dtype=np.float32)
    lookup.sort(order="packed_key")
    np.save(lookup_path, lookup, allow_pickle=False)
    restored = np.load(lookup_path, mmap_mode="r", allow_pickle=False)
    if (
        restored.dtype != OBSERVED_LINK_SPEED_BOUNDARY_DTYPE
        or len(restored) != len(selected)
    ):
        raise ValueError("Observed speed-boundary lookup round-trip failed")

    metadata = {
        "format": "NumPy .npy structured array sorted by packed_key",
        "key_definition": "(uint64(from_node_id) << 32) | uint64(to_node_id)",
        "record_dtype": OBSERVED_LINK_SPEED_BOUNDARY_DTYPE.descr,
        "speed_unit": "mph",
        "source_profile": "CBI weekday-average 15-minute TMC speed profile",
        "boundary_speed_rule": (
            "Use avg_weekday_speed_mph when finite; otherwise use "
            "avg_weekday_speed_mph_pre_qc only at assignment-period edges"
        ),
        "missing_value_rule": (
            "NaN remains missing only when both post-QC and pre-QC weekday "
            "average boundary speeds are unavailable; conversion then writes "
            "a blank field and the kernel retains its free-speed fallback"
        ),
        "profile_source_files": [str(path.resolve()) for path in source_files],
        "observed_tmc_rows": int(len(records)),
        "unique_node_pair_rows": int(len(selected)),
        "duplicate_node_pair_tmc_rows": int(len(records) - len(selected)),
        "lookup_path": str(lookup_path),
        "lookup_sha256": _sha256(lookup_path),
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "boundary_completeness_report_path": str(completeness_path),
        "boundary_completeness_report_sha256": _sha256(completeness_path),
        "periods": {},
    }
    for period, (start_minute, end_minute) in PERIODS.items():
        start_field = f"qvdf_start_speed_mph_{period.lower()}"
        end_field = f"qvdf_end_speed_mph_{period.lower()}"
        period_values = selected[[start_field, end_field]].to_numpy(dtype=float)
        finite = np.isfinite(period_values)
        metadata["periods"][period] = {
            "start_minute": int(start_minute),
            "end_minute": int(end_minute),
            "start_field": start_field,
            "end_field": end_field,
            "complete_node_pair_rows": int(finite.all(axis=1).sum()),
            "start_only_node_pair_rows": int(
                (finite[:, 0] & ~finite[:, 1]).sum()
            ),
            "end_only_node_pair_rows": int(
                (~finite[:, 0] & finite[:, 1]).sum()
            ),
            "neither_node_pair_rows": int((~finite).all(axis=1).sum()),
            "missing_start_rows": int((~finite[:, 0]).sum()),
            "missing_end_rows": int((~finite[:, 1]).sum()),
            "start_speed_source_counts": (
                completeness.loc[
                    completeness["period"].eq(period),
                    "start_speed_source",
                ]
                .value_counts(dropna=False)
                .astype(int)
                .to_dict()
            ),
            "end_speed_source_counts": (
                completeness.loc[
                    completeness["period"].eq(period),
                    "end_speed_source",
                ]
                .value_counts(dropna=False)
                .astype(int)
                .to_dict()
            ),
            "missing_start_cause_counts": (
                completeness.loc[
                    completeness["period"].eq(period)
                    & completeness["start_missing_cause"].ne("available"),
                    "start_missing_cause",
                ]
                .value_counts()
                .astype(int)
                .to_dict()
            ),
            "missing_end_cause_counts": (
                completeness.loc[
                    completeness["period"].eq(period)
                    & completeness["end_missing_cause"].ne("available"),
                    "end_missing_cause",
                ]
                .value_counts()
                .astype(int)
                .to_dict()
            ),
            "finite_speed_min_mph": float(np.nanmin(period_values)),
            "finite_speed_max_mph": float(np.nanmax(period_values)),
        }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    metadata["metadata_sha256"] = _sha256(metadata_path)
    return metadata


def discover_accepted_episode_files(cbi_run_dir: Path, basis: str) -> list[Path]:
    filename = {
        "daily": "daily_episodes_accepted.csv",
        "average_weekday": "average_weekday_episodes_accepted.csv",
    }[basis]
    root = Path(cbi_run_dir).resolve()
    files = sorted(root.glob(f"**/05-episode-filtering/{filename}"))
    if not files:
        files = sorted(root.glob(f"**/{filename}"))
    if not files:
        raise FileNotFoundError(
            f"No {basis} accepted episode files were found under {root}"
        )
    return files


def load_nvta_network_link_types(
    network_root: Path,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    """Build a strict AM/MD/PM consensus link-type lookup for NVTA."""

    root = Path(network_root).resolve()
    period_frames: list[pd.DataFrame] = []
    sources: dict[str, object] = {}
    type_columns: list[str] = []
    for period, folder in NETWORK_PERIOD_FOLDERS.items():
        path = root / folder / "link.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing NVTA {period} network link file: {path}")
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted({"link_id", "link_type"} - header)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = pd.read_csv(
            path,
            usecols=["link_id", "link_type"],
            low_memory=False,
        )
        frame["link_id"] = pd.to_numeric(
            frame["link_id"], errors="raise"
        ).astype("int64")
        column = f"{period.lower()}_link_type"
        type_columns.append(column)
        frame[column] = frame["link_type"].map(_normalize_vdf_code)
        if frame[column].eq("").any():
            examples = frame.loc[frame[column].eq(""), "link_id"].head(10)
            raise ValueError(
                f"{period} network has blank link_type values for link IDs: "
                + ", ".join(map(str, examples))
            )
        conflicts = frame.groupby("link_id")[column].nunique()
        conflicts = conflicts[conflicts.gt(1)]
        if not conflicts.empty:
            raise ValueError(
                f"{period} network has conflicting link types for link IDs: "
                + ", ".join(map(str, conflicts.index[:10]))
            )
        reduced = frame[["link_id", column]].drop_duplicates("link_id")
        period_frames.append(reduced.set_index("link_id"))
        sources[period] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": int(len(frame)),
            "unique_link_ids": int(frame["link_id"].nunique()),
            "unique_link_types": int(frame[column].nunique()),
        }

    audit = pd.concat(period_frames, axis=1, join="outer").reset_index()
    audit["unique_type_count"] = audit[type_columns].nunique(
        axis=1, dropna=True
    )
    cross_period_conflicts = audit[audit["unique_type_count"].gt(1)]
    if not cross_period_conflicts.empty:
        examples = cross_period_conflicts["link_id"].head(10)
        raise ValueError(
            "NVTA AM/MD/PM networks disagree on link_type for link IDs: "
            + ", ".join(map(str, examples))
        )
    audit["network_link_type"] = audit[type_columns].bfill(axis=1).iloc[:, 0]
    for period in NETWORK_PERIOD_FOLDERS:
        audit[f"present_in_{period.lower()}"] = audit[
            f"{period.lower()}_link_type"
        ].notna()
    present_columns = [f"present_in_{period.lower()}" for period in PERIOD_SEQUENCE]
    audit["period_presence_count"] = audit[present_columns].sum(axis=1).astype(int)
    audit["consensus_status"] = np.where(
        audit["period_presence_count"].eq(len(PERIOD_SEQUENCE)),
        "consistent_all_periods",
        "consistent_partial_period_presence",
    )
    audit = audit.sort_values("link_id").reset_index(drop=True)
    lookup = audit.set_index("link_id")["network_link_type"]
    metadata = {
        "network_root": str(root),
        "sources": sources,
        "consensus_link_ids": int(len(audit)),
        "consensus_link_types": int(audit["network_link_type"].nunique()),
        "all_period_link_ids": int(audit["period_presence_count"].eq(3).sum()),
        "partial_period_link_ids": int(audit["period_presence_count"].lt(3).sum()),
        "cross_period_type_conflicts": 0,
    }
    return lookup, audit, metadata


def load_accepted_episodes(
    paths: Iterable[Path],
    *,
    network_link_types: pd.Series,
) -> pd.DataFrame:
    """Load accepted episodes and attach authoritative NVTA link types."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        if "is_clean_valid_episode" in frame:
            accepted = (
                frame["is_clean_valid_episode"]
                .astype("string")
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
            if not accepted.all():
                raise ValueError(f"Rejected episodes are present in {path}")
        frame["episode_source_file"] = str(Path(path).resolve())
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "period",
        "network_link_id",
        "demand_capacity_ratio",
        "min_speed_mph",
        "threshold_used",
        "qdf",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"Accepted episodes are missing columns: {missing}")
    if "P_hr" not in episodes:
        if "duration_min" not in episodes:
            raise ValueError("Accepted episodes require P_hr or duration_min")
        episodes["P_hr"] = pd.to_numeric(
            episodes["duration_min"], errors="coerce"
        ) / 60.0

    input_types = (
        episodes["network_link_type"].map(_normalize_vdf_code)
        if "network_link_type" in episodes
        else pd.Series("", index=episodes.index, dtype="object")
    )
    network_ids = pd.to_numeric(
        episodes["network_link_id"], errors="coerce"
    ).astype("Int64")
    recovered_types = network_ids.map(network_link_types).fillna("")
    recovered_types = recovered_types.map(_normalize_vdf_code)
    disagreements = input_types.ne("") & recovered_types.ne("") & input_types.ne(
        recovered_types
    )
    if disagreements.any():
        examples = network_ids[disagreements].dropna().astype(str).head(10)
        raise ValueError(
            "Accepted episodes disagree with the authoritative NVTA network "
            "link types for link IDs: " + ", ".join(examples)
        )
    episodes["network_link_type_input"] = input_types
    episodes["network_link_type"] = recovered_types.where(
        recovered_types.ne(""), input_types
    )
    episodes["network_link_type_source"] = np.where(
        recovered_types.ne(""), "nvta_am_md_pm_consensus", "mapping_artifact_fallback"
    )
    episodes["period"] = episodes["period"].astype(str).str.upper()
    episodes["vdf_code"] = episodes["network_link_type"].map(_normalize_vdf_code)
    for column in (*CALIBRATION_VALUE_COLUMNS, "plf"):
        if column in episodes:
            episodes[column] = pd.to_numeric(episodes[column], errors="coerce")

    valid_period = episodes["period"].isin(PERIOD_SEQUENCE)
    mapped_type = episodes["vdf_code"].ne("")
    finite = np.isfinite(
        episodes[CALIBRATION_VALUE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)
    physically_valid = (
        episodes["demand_capacity_ratio"].gt(0.0)
        & episodes["P_hr"].gt(0.0)
        & episodes["min_speed_mph"].gt(0.0)
        & episodes["threshold_used"].gt(0.0)
        & episodes["qdf"].gt(0.0)
        & episodes["qdf"].le(1.0)
    )
    episodes["calibration_eligible"] = (
        valid_period & mapped_type & finite & physically_valid
    )
    reasons = np.select(
        [
            ~valid_period,
            ~mapped_type,
            ~pd.Series(finite, index=episodes.index),
            ~physically_valid,
        ],
        [
            "unsupported_period",
            "missing_network_link_type",
            "nonfinite_calibration_parameter",
            "nonphysical_calibration_parameter",
        ],
        default="eligible",
    )
    episodes["calibration_exclusion_reason"] = reasons
    return episodes


def _fit_period(
    episodes: pd.DataFrame,
    minimum_episodes: int,
) -> tuple[QVDFCalibration, float, float]:
    finite = episodes.dropna(subset=CALIBRATION_VALUE_COLUMNS)
    if finite.empty:
        fit = fit_qvdf([], [], [], np.nan, minimum_episodes)
        return fit, np.nan, np.nan
    vc = float(finite["threshold_used"].median())
    fit = fit_qvdf(
        finite["demand_capacity_ratio"],
        finite["P_hr"],
        finite["min_speed_mph"],
        vc,
        minimum_episodes,
    )
    qdf_values = finite.loc[
        finite["qdf"].gt(0.0) & finite["qdf"].le(1.0), "qdf"
    ]
    qdf = float(qdf_values.median()) if not qdf_values.empty else np.nan
    return fit, qdf, vc


def _period_parameters(
    fit: QVDFCalibration,
    qdf: float,
    period: str,
) -> dict[str, float]:
    if fit.status != "ok" or not np.isfinite(qdf) or qdf <= 0.0:
        raise ValueError(f"No valid QVDF calibration for {period}")
    duration = period_duration_hours(period, PERIODS)
    plf = 1.0 / (qdf * duration)
    return {
        "plf": float(plf),
        "qdf": float(qdf),
        "n": float(fit.n),
        "s": float(fit.s),
        "cp": float(fit.f_p),
        "cd": float(fit.f_d),
        "alpha": float(fit.alpha),
        "beta": float(fit.beta),
    }


def build_resource_from_episodes(
    episodes: pd.DataFrame,
    *,
    minimum_episodes: int = 3,
    vdf_codes: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate every network link type and the final network-wide fallback."""

    work = episodes.copy()
    if "vdf_code" not in work:
        work["vdf_code"] = work["network_link_type"].map(_normalize_vdf_code)
    episode_codes = set(work["vdf_code"].dropna().map(_normalize_vdf_code))
    requested_codes = (
        {_normalize_vdf_code(value) for value in vdf_codes}
        if vdf_codes is not None
        else set()
    )
    codes = sorted(
        (episode_codes | requested_codes) - {"", "all"},
        key=_code_sort_key,
    )
    all_fits: dict[str, tuple[QVDFCalibration, float, float]] = {}
    for period in PERIOD_SEQUENCE:
        all_fits[period] = _fit_period(
            work[work["period"].eq(period)], minimum_episodes
        )
        if all_fits[period][0].status != "ok":
            raise ValueError(
                f"Network-wide {period} calibration has insufficient accepted episodes"
            )

    resource_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for code in [*codes, "all"]:
        row: dict[str, object] = {
            "data_type": "vdf_code",
            "link_id": "",
            "tmc_corridor_name": "",
            "from_node_id": "",
            "to_node_id": "",
            "vdf_code": code,
        }
        for period, sequence in PERIOD_SEQUENCE.items():
            period_all = work[work["period"].eq(period)]
            period_type = (
                period_all[period_all["vdf_code"].eq(code)]
                if code != "all"
                else period_all
            )
            fit, qdf, vc = _fit_period(period_type, minimum_episodes)
            source = "link_type"
            if code == "all":
                source = "network_all"
            elif fit.status != "ok" or not np.isfinite(qdf) or qdf <= 0.0:
                fit, qdf, vc = all_fits[period]
                source = "network_all_fallback"
            parameters = _period_parameters(fit, qdf, period)
            for parameter, value in parameters.items():
                row[f"QVDF_{parameter}{sequence}"] = value
            audit_rows.append(
                {
                    "vdf_code": code,
                    "period": period,
                    "period_sequence": sequence,
                    "calibration_source": source,
                    "candidate_episode_rows": int(len(period_type)),
                    "network_episode_rows": int(len(period_all)),
                    "vc_mph": vc,
                    "qdf_aggregation": "median accepted episode QDF",
                    "plf_formula": "1/(qdf*period_duration_hours)",
                    **asdict(fit),
                    **parameters,
                }
            )
        resource_rows.append(row)
    resource = pd.DataFrame(resource_rows, columns=RESOURCE_COLUMNS)
    if resource.empty or resource.iloc[-1]["vdf_code"] != "all":
        raise AssertionError("QVDF resource must end with vdf_code=all")
    if resource.filter(regex=r"^QVDF_").isna().any().any():
        raise ValueError("QVDF resource contains blank parameter values")
    if not np.isfinite(
        resource.filter(regex=r"^QVDF_").to_numpy(dtype=float)
    ).all():
        raise ValueError("QVDF resource contains non-finite parameter values")
    return resource, pd.DataFrame(audit_rows)


def _write_review_outputs(
    output: Path,
    resources: dict[str, pd.DataFrame],
    audits: dict[str, pd.DataFrame],
    lineages: dict[str, pd.DataFrame],
) -> dict[str, str]:
    review = output / "review"
    review.mkdir(exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for basis, audit in audits.items():
        audit = audit.copy()
        for column in ("duration_bound_active", "speed_bound_active"):
            audit[column] = (
                audit[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
        for column in ("duration_r2", "speed_r2"):
            audit[column] = pd.to_numeric(audit[column], errors="coerce")
        audits[basis] = audit
        for period in PERIOD_SEQUENCE:
            rows = audit[
                audit["period"].eq(period) & audit["vdf_code"].astype(str).ne("all")
            ]
            direct = rows[rows["calibration_source"].eq("link_type")]
            summary_rows.append(
                {
                    "basis": basis,
                    "period": period,
                    "network_vdf_codes": int(len(rows)),
                    "direct_link_type_calibrations": int(
                        rows["calibration_source"].eq("link_type").sum()
                    ),
                    "network_fallback_calibrations": int(
                        rows["calibration_source"].eq("network_all_fallback").sum()
                    ),
                    "eligible_episode_rows": int(
                        lineages[basis].loc[
                            lineages[basis]["calibration_eligible"]
                            & lineages[basis]["period"].eq(period)
                        ].shape[0]
                    ),
                    "direct_median_duration_r2": float(
                        direct["duration_r2"].median()
                    ),
                    "direct_median_speed_r2": float(direct["speed_r2"].median()),
                    "direct_bound_active_calibrations": int(
                        (
                            direct["duration_bound_active"]
                            | direct["speed_bound_active"]
                        ).sum()
                    ),
                    "direct_negative_duration_r2": int(
                        direct["duration_r2"].lt(0.0).sum()
                    ),
                    "direct_negative_speed_r2": int(
                        direct["speed_r2"].lt(0.0).sum()
                    ),
                }
            )
        resource = resources[basis]
        for period, sequence in PERIOD_SEQUENCE.items():
            for parameter in RESOURCE_PARAMETERS:
                column = f"QVDF_{parameter}{sequence}"
                values = pd.to_numeric(resource[column], errors="coerce")
                parameter_rows.append(
                    {
                        "basis": basis,
                        "period": period,
                        "parameter": parameter,
                        "minimum": float(values.min()),
                        "median": float(values.median()),
                        "maximum": float(values.max()),
                    }
                )
    summary_path = review / "calibration_summary_by_period.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    ranges_path = review / "parameter_ranges.csv"
    pd.DataFrame(parameter_rows).to_csv(ranges_path, index=False)
    combined_audit = pd.concat(
        [frame.assign(basis=basis) for basis, frame in audits.items()],
        ignore_index=True,
    )
    fallback_path = review / "fallback_calibrations.csv"
    combined_audit[
        combined_audit["calibration_source"].eq("network_all_fallback")
    ].to_csv(fallback_path, index=False)
    direct_quality = combined_audit[
        combined_audit["calibration_source"].eq("link_type")
    ].copy()
    direct_quality["duration_r2_below_zero"] = direct_quality["duration_r2"].lt(0.0)
    direct_quality["speed_r2_below_zero"] = direct_quality["speed_r2"].lt(0.0)
    direct_quality["bound_active"] = (
        direct_quality["duration_bound_active"]
        | direct_quality["speed_bound_active"]
    )
    direct_quality["requires_review"] = (
        direct_quality["duration_r2_below_zero"]
        | direct_quality["speed_r2_below_zero"]
        | direct_quality["bound_active"]
    )
    direct_quality_path = review / "direct_fit_quality_flags.csv"
    direct_quality.to_csv(direct_quality_path, index=False)
    daily = resources["daily"].set_index("vdf_code").filter(regex=r"^QVDF_")
    average = resources["average_weekday"].set_index("vdf_code").filter(
        regex=r"^QVDF_"
    )
    comparison = daily.join(
        average,
        how="outer",
        lsuffix="_daily",
        rsuffix="_average_weekday",
    ).reset_index()
    comparison_path = review / "daily_vs_average_weekday_parameters.csv"
    comparison.to_csv(comparison_path, index=False)
    markdown_path = review / "REVIEW_SUMMARY.md"
    summary = pd.DataFrame(summary_rows)
    markdown_lines = [
        "# NVTA network QVDF review",
        "",
        "The daily resource is the recommended authoritative TAPLite input. ",
        "The average-weekday resource is retained as a diagnostic comparison.",
        "",
        "| Basis | Period | Episodes | Direct | Fallback | Duration R2 median | Speed R2 median | Bound-active direct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        markdown_lines.append(
            f"| {row.basis} | {row.period} | {row.eligible_episode_rows} | "
            f"{row.direct_link_type_calibrations} | "
            f"{row.network_fallback_calibrations} | "
            f"{row.direct_median_duration_r2:.3f} | "
            f"{row.direct_median_speed_r2:.3f} | "
            f"{row.direct_bound_active_calibrations} |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {
        "calibration_summary": str(summary_path),
        "parameter_ranges": str(ranges_path),
        "fallback_calibrations": str(fallback_path),
        "direct_fit_quality_flags": str(direct_quality_path),
        "daily_vs_average_weekday": str(comparison_path),
        "review_summary": str(markdown_path),
    }


def refresh_review_outputs(output_dir: Path) -> dict[str, str]:
    """Rebuild review tables from an existing network-QVDF product tree."""

    output = Path(output_dir).resolve()
    resources: dict[str, pd.DataFrame] = {}
    audits: dict[str, pd.DataFrame] = {}
    lineages: dict[str, pd.DataFrame] = {}
    for basis in ("daily", "average_weekday"):
        basis_dir = output / basis.replace("_", "-")
        resources[basis] = pd.read_csv(basis_dir / "link_qvdf.csv")
        audits[basis] = pd.read_csv(basis_dir / "calibration_by_link_type.csv")
        lineages[basis] = pd.read_csv(
            basis_dir / "episode_parameter_lineage.csv", low_memory=False
        )
    products = _write_review_outputs(output, resources, audits, lineages)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["review"] = products
        manifest["review_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return products


def build_qvdf_resources(
    cbi_run_dir: Path,
    output_dir: Path,
    *,
    network_root: Path,
    minimum_episodes: int = 3,
    observed_triplet_policy: str = "error",
) -> dict[str, object]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    network_link_types, network_audit, network_metadata = (
        load_nvta_network_link_types(network_root)
    )
    network_audit_path = output / "network_link_type_consensus.csv"
    network_audit.to_csv(network_audit_path, index=False)
    all_vdf_codes = sorted(
        network_audit["network_link_type"].dropna().unique(), key=_code_sort_key
    )
    products: dict[str, object] = {}
    resources: dict[str, pd.DataFrame] = {}
    audits: dict[str, pd.DataFrame] = {}
    lineages: dict[str, pd.DataFrame] = {}
    for basis in ("daily", "average_weekday"):
        files = discover_accepted_episode_files(cbi_run_dir, basis)
        episodes = load_accepted_episodes(
            files, network_link_types=network_link_types
        )
        eligible = episodes[episodes["calibration_eligible"]].copy()
        resource, audit = build_resource_from_episodes(
            eligible,
            minimum_episodes=minimum_episodes,
            vdf_codes=all_vdf_codes,
        )
        basis_dir = output / basis.replace("_", "-")
        basis_dir.mkdir()
        resource_path = basis_dir / "link_qvdf.csv"
        audit_path = basis_dir / "calibration_by_link_type.csv"
        lineage_path = basis_dir / "episode_parameter_lineage.csv"
        exclusions_path = basis_dir / "episode_exclusions.csv"
        resource.to_csv(resource_path, index=False)
        audit.to_csv(audit_path, index=False)
        episodes.to_csv(lineage_path, index=False)
        episodes[~episodes["calibration_eligible"]].to_csv(
            exclusions_path, index=False
        )
        resources[basis] = resource
        audits[basis] = audit
        lineages[basis] = episodes
        products[basis] = {
            "accepted_episode_files": [str(path.resolve()) for path in files],
            "accepted_episode_rows": int(len(episodes)),
            "calibration_eligible_rows": int(len(eligible)),
            "excluded_episode_rows": int((~episodes["calibration_eligible"]).sum()),
            "link_qvdf": str(resource_path),
            "link_qvdf_sha256": _sha256(resource_path),
            "calibration_audit": str(audit_path),
            "episode_lineage": str(lineage_path),
            "episode_exclusions": str(exclusions_path),
            "vdf_codes": resource["vdf_code"].astype(str).tolist(),
        }
    observed_plf_product = build_observed_link_plf_overrides(
        cbi_run_dir,
        lineages["average_weekday"],
        output / "observed-link-plf",
    )
    products["observed_link_plf"] = observed_plf_product
    observed_speed_product = build_observed_link_speed_boundaries(
        cbi_run_dir,
        output / "observed-link-speed-boundaries",
    )
    products["observed_link_speed_boundaries"] = observed_speed_product
    observed_t2_product = build_observed_link_t2_lookup(
        cbi_run_dir,
        lineages["average_weekday"],
        output / "observed-link-t2",
        invalid_episode_policy=observed_triplet_policy,
    )
    products["observed_link_t2"] = observed_t2_product
    review_products = _write_review_outputs(
        output, resources, audits, lineages
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cbi_run_dir": str(Path(cbi_run_dir).resolve()),
        "periods_minutes": PERIODS,
        "period_durations_hours": {
            period: period_duration_hours(period, PERIODS) for period in PERIODS
        },
        "minimum_episodes": minimum_episodes,
        "observed_triplet_policy": observed_triplet_policy,
        "network": network_metadata,
        "network_link_type_consensus": str(network_audit_path),
        "network_link_type_consensus_sha256": _sha256(network_audit_path),
        "qdf_rule": "episode-period overlap synthetic volume / period synthetic volume",
        "plf_rule": "1 / (QDF * period duration hours)",
        "observed_link_plf_rule": (
            "best-match observed TMC link uses average-weekday QDF when an "
            "accepted congestion episode exists; otherwise QDF=1/H and PLF=1"
        ),
        "observed_link_speed_boundary_rule": (
            "best-match observed TMC link uses the post-QC weekday-average "
            "TMC speed at each assignment period's exact start and end "
            "minute, with pre-average-QC weekday speed used only when that "
            "boundary is blank; both missing remains NaN for kernel fallback"
        ),
        "observed_link_t2_rule": (
            "best-match observed TMC link uses the representative accepted "
            "weekday-average episode T0/T2/T3 for that period; no accepted "
            "episode remains a NaN triplet and clears all three boundary "
            "fields to protect the direct observation"
        ),
        "resource_scope": (
            "one row per NVTA network vdf_code plus a final all fallback row"
        ),
        "authoritative_basis": "daily",
        "products": products,
        "review": review_products,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build daily and average-weekday NVTA TAPLite link_qvdf resources."
    )
    parser.add_argument("--cbi-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--network-root",
        type=Path,
        required=True,
        help=(
            "NVTA gmns_network_am_md_pm directory containing am/md/pm/link.csv."
        ),
    )
    parser.add_argument("--minimum-episodes", type=int, default=3)
    parser.add_argument(
        "--observed-triplet-policy",
        choices=["error", "omit"],
        default="error",
        help=(
            "How to handle accepted average-weekday episodes whose observed "
            "T0/T2/T3 cannot form a valid kernel triplet. Actual runs should "
            "keep the strict default; isolated virtual runs may audit and omit."
        ),
    )
    args = parser.parse_args(argv)
    manifest = build_qvdf_resources(
        args.cbi_run_dir,
        args.output_dir,
        network_root=args.network_root,
        minimum_episodes=args.minimum_episodes,
        observed_triplet_policy=args.observed_triplet_policy,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
