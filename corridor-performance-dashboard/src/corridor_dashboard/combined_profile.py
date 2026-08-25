"""TMC-aligned observed and TAPlite dashboard profiles."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .dashboard_filters import is_managed_corridor


OBSERVED_COLOR = "#1f77b4"
TAPLITE_COLOR = "#e87500"
CBI_QVDF_COLOR = "#6f4e9c"
THRESHOLD_COLOR = "#555555"
PERIOD_COLORS = {"AM": "#4c78a8", "MD": "#f2a541", "PM": "#8f63b8"}
PERIOD_BOUNDS = {"AM": (360, 540), "MD": (540, 900), "PM": (900, 1140)}
MINIMUM_FONT_SIZE = 16
PERIOD_TABLE_FONT_SIZE = 15
PRIMARY_LINE_WIDTH = 3.2
SECONDARY_LINE_WIDTH = 2.2
FIGURE_DPI = 180

plt.rcParams.update(
    {
        "font.size": MINIMUM_FONT_SIZE,
        "axes.titlesize": 18,
        "axes.labelsize": MINIMUM_FONT_SIZE,
        "xtick.labelsize": MINIMUM_FONT_SIZE,
        "ytick.labelsize": MINIMUM_FONT_SIZE,
        "legend.fontsize": MINIMUM_FONT_SIZE,
        "legend.title_fontsize": MINIMUM_FONT_SIZE,
    }
)


def _clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _number(value: object, pattern: str, fallback: str = "NA") -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return format(float(number), pattern) if pd.notna(number) else fallback


def _profile_duration_hours(frame: pd.DataFrame) -> float:
    speed = pd.to_numeric(frame["speed_qvdf_model"], errors="coerce")
    threshold = pd.to_numeric(frame["congestion_threshold_mph"], errors="coerce")
    valid = speed.notna() & threshold.notna()
    return float((speed[valid] <= threshold[valid]).sum() * 0.25)


def _tmc_link_label(value: object, link_ids: object = None) -> str:
    count = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(count):
        return "TMC link coverage unavailable"
    rounded = int(round(float(count)))
    noun = "link" if rounded == 1 else "links"
    ordered_ids = [
        link_id.strip()
        for link_id in str(link_ids or "").split(";")
        if link_id.strip()
    ]
    if ordered_ids:
        return f"TMC covers {rounded} {noun}: {' → '.join(ordered_ids)}"
    return f"TMC covers {rounded} {noun}"


def _link_id_text(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return str(int(round(float(number)))) if pd.notna(number) else str(value).strip()


def _load_primary_link_mapping(
    selected: pd.DataFrame, cbi_corridors_root: Path
) -> pd.DataFrame:
    """Load primary-link rows for selected TMCs, including canonical rank."""

    ranked = _load_ranked_primary_link_mapping(cbi_corridors_root)
    if ranked.empty:
        return ranked
    requested = selected[["corridor", "tmc_code"]].drop_duplicates().copy()
    requested["corridor"] = requested["corridor"].astype("string")
    requested["tmc_code"] = requested["tmc_code"].astype("string")
    return requested.merge(
        ranked,
        on=["corridor", "tmc_code"],
        how="left",
        validate="one_to_one",
    )


def _load_general_purpose_tmc_codes(
    mapmatching_product_root: Path,
) -> set[str]:
    """Load the authoritative GP-only visualization membership."""

    source = Path(mapmatching_product_root) / "full_tmc_to_link.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Dashboard facility classification file not found: {source}"
        )
    header = pd.read_csv(source, nrows=0)
    missing = {"tmc", "facility_class"}.difference(header.columns)
    if missing:
        raise ValueError(
            f"{source} is missing dashboard classification columns: "
            + ", ".join(sorted(missing))
        )
    frame = pd.read_csv(
        source,
        usecols=["tmc", "facility_class"],
        dtype={"tmc": "string", "facility_class": "string"},
        low_memory=False,
    )
    frame["tmc"] = frame["tmc"].str.strip()
    frame["facility_class"] = frame["facility_class"].str.strip().str.lower()
    frame = frame.loc[frame["tmc"].notna() & frame["tmc"].ne("")].copy()
    membership = frame.groupby("tmc", sort=False)["facility_class"].agg(
        lambda values: {
            "unclassified" if pd.isna(value) or not str(value).strip() else str(value)
            for value in values
        }
    )
    return {
        str(tmc_code)
        for tmc_code, values in membership.items()
        if values == {"gp"}
    }


def _filter_general_purpose_profiles(
    profiles: pd.DataFrame,
    mapmatching_product_root: Path | None,
) -> pd.DataFrame:
    """Restrict dashboard visualizations to authoritative GP TMC records."""

    if mapmatching_product_root is None:
        return profiles.copy()
    eligible = _load_general_purpose_tmc_codes(mapmatching_product_root)
    return profiles.loc[
        profiles["tmc_code"].astype("string").str.strip().isin(eligible)
    ].copy()


def _load_ranked_primary_link_mapping(
    cbi_corridors_root: Path,
    eligible_tmc_codes: set[str] | None = None,
) -> pd.DataFrame:
    """Load the frozen node-pair winner propagated by CBI without reranking."""

    rows: list[pd.DataFrame] = []
    for source in sorted(
        Path(cbi_corridors_root).glob("*/01-input-and-qc/link_reference.csv")
    ):
        corridor = source.parents[1].name
        requested = {
            "tmc_code", "network_link_id", "network_from_node_id",
            "network_to_node_id", "network_match_distance_ft",
            "network_bearing_diff_deg", "network_mapping_status",
            "network_match_score", "road_order",
            "network_node_pair_tmc_rank",
            "network_selected_for_node_pair_lookup",
        }
        header = pd.read_csv(source, nrows=0)
        frame = pd.read_csv(
            source,
            dtype={"tmc_code": "string"},
            usecols=[column for column in requested if column in header.columns],
            low_memory=False,
        )
        if "tmc_code" not in frame or "network_link_id" not in frame:
            continue
        frame["corridor"] = str(corridor)
        frame["primary_link_id"] = frame["network_link_id"].map(_link_id_text)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["corridor", "tmc_code", "primary_link_id"])
    ranked = pd.concat(rows, ignore_index=True)
    if ranked.duplicated(["corridor", "tmc_code"]).any():
        raise ValueError("CBI link references contain duplicate corridor/TMC rows")
    for column in (
        "network_link_id",
        "network_from_node_id",
        "network_to_node_id",
        "network_match_distance_ft",
        "network_match_score",
        "road_order",
    ):
        if column not in ranked:
            ranked[column] = np.nan
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.dropna(
        subset=["tmc_code", "network_from_node_id", "network_to_node_id"]
    ).copy()
    ranked["corridor"] = ranked["corridor"].astype("string").str.strip()
    ranked["tmc_code"] = ranked["tmc_code"].astype("string").str.strip()
    ranked["primary_link_id"] = ranked["network_link_id"].map(_link_id_text)
    rank_source = "network_node_pair_tmc_rank"
    selected_source = "network_selected_for_node_pair_lookup"
    if rank_source not in ranked or selected_source not in ranked:
        raise ValueError(
            "CBI link references must carry the frozen node-pair rank and "
            "network_selected_for_node_pair_lookup fields"
        )
    ranked["node_pair_tmc_rank"] = pd.to_numeric(
        ranked[rank_source], errors="coerce"
    ).astype("Int64")
    ranked["selected_for_node_pair_lookup"] = (
        ranked[selected_source]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    selected = ranked[ranked["selected_for_node_pair_lookup"]]
    if not selected["node_pair_tmc_rank"].eq(1).all():
        raise ValueError("A dashboard canonical node-pair winner is not rank 1")
    if selected.duplicated(
        ["network_from_node_id", "network_to_node_id"]
    ).any():
        raise ValueError("Dashboard canonical node-pair winners are not unique")
    # Facility-class filtering is downstream of canonical selection. It may
    # hide a managed winner, but it must never promote a losing GP TMC.
    if eligible_tmc_codes is not None:
        ranked = ranked.loc[ranked["tmc_code"].isin(eligible_tmc_codes)].copy()
    return ranked


def _select_canonical_representative_tmcs(
    profiles: pd.DataFrame,
    ranked_mapping: pd.DataFrame,
    *,
    count: int = 5,
) -> pd.DataFrame:
    """Select the most congested canonical TMC in each spatial segment."""

    if (
        ranked_mapping.empty
        or "selected_for_node_pair_lookup" not in ranked_mapping
    ):
        return pd.DataFrame(
            columns=["corridor", "tmc_code", "road_order", "selection_position"]
        )
    winners = ranked_mapping[ranked_mapping["selected_for_node_pair_lookup"]][
        ["corridor", "tmc_code"]
    ].drop_duplicates()
    working = profiles.merge(
        winners,
        on=["corridor", "tmc_code"],
        how="inner",
        validate="many_to_one",
    ).copy()
    for column in (
        "observed_tmc_speed_mph",
        "model_tmc_speed_mph",
        "cube_qvdf_tmc_speed_mph",
        "cbi_tmc_congestion_threshold_mph",
        "speed_at_capacity_mph",
    ):
        if column not in working:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["absolute_speed_error_mph"] = (
        working["model_tmc_speed_mph"] - working["observed_tmc_speed_mph"]
    ).abs()
    working["cube_absolute_speed_error_mph"] = (
        working["cube_qvdf_tmc_speed_mph"]
        - working["observed_tmc_speed_mph"]
    ).abs()
    working["selection_threshold_mph"] = working[
        "cbi_tmc_congestion_threshold_mph"
    ].where(
        working["cbi_tmc_congestion_threshold_mph"].gt(0),
        working["speed_at_capacity_mph"],
    )
    valid_threshold = working["selection_threshold_mph"].gt(0)
    working["observed_congestion_deficit_ratio"] = 0.0
    working.loc[valid_threshold, "observed_congestion_deficit_ratio"] = (
        (
            working.loc[valid_threshold, "selection_threshold_mph"]
            - working.loc[valid_threshold, "observed_tmc_speed_mph"]
        ).clip(lower=0.0)
        / working.loc[valid_threshold, "selection_threshold_mph"]
    )
    working["observed_congested_interval"] = (
        valid_threshold
        & working["observed_tmc_speed_mph"].lt(
            working["selection_threshold_mph"]
        )
    ).astype(int)
    availability = (
        working.groupby(["corridor", "tmc_code"], as_index=False)
        .agg(
            road_order=("road_order", "first"),
            matched_interval_count=(
                "model_tmc_speed_mph",
                lambda values: int(values.notna().sum()),
            ),
            observed_speed_mean_mph=("observed_tmc_speed_mph", "mean"),
            observed_speed_min_mph=("observed_tmc_speed_mph", "min"),
            observed_congestion_severity=(
                "observed_congestion_deficit_ratio", "mean"
            ),
            observed_congested_interval_count=(
                "observed_congested_interval", "sum"
            ),
            observed_speed_range_mph=(
                "observed_tmc_speed_mph",
                lambda values: float(values.max() - values.min()),
            ),
            observed_speed_std_mph=("observed_tmc_speed_mph", "std"),
            model_speed_mean_mph=("model_tmc_speed_mph", "mean"),
            model_speed_range_mph=(
                "model_tmc_speed_mph",
                lambda values: float(values.max() - values.min()),
            ),
            model_speed_std_mph=("model_tmc_speed_mph", "std"),
            cube_speed_mean_mph=("cube_qvdf_tmc_speed_mph", "mean"),
            cube_speed_range_mph=(
                "cube_qvdf_tmc_speed_mph",
                lambda values: float(values.max() - values.min()),
            ),
            cube_speed_std_mph=("cube_qvdf_tmc_speed_mph", "std"),
            mean_absolute_error_mph=("absolute_speed_error_mph", "mean"),
            cube_mean_absolute_error_mph=(
                "cube_absolute_speed_error_mph", "mean"
            ),
        )
        .sort_values(["corridor", "road_order", "tmc_code"], kind="stable")
    )
    availability = availability[availability["matched_interval_count"].gt(0)]
    rows: list[dict[str, object]] = []
    for corridor, group in availability.groupby("corridor", sort=True):
        ordered = group.reset_index(drop=True)
        number_to_select = min(count, len(ordered))
        if number_to_select == 0:
            continue
        if number_to_select == len(ordered):
            indices = list(range(len(ordered)))
        else:
            indices = []
            for segment_positions in np.array_split(
                np.arange(len(ordered)), number_to_select
            ):
                segment = ordered.iloc[segment_positions].copy()
                segment = segment.sort_values(
                    [
                        "observed_congestion_severity",
                        "observed_congested_interval_count",
                        "observed_speed_mean_mph",
                        "observed_speed_min_mph",
                        "road_order",
                        "tmc_code",
                    ],
                    ascending=[False, False, True, True, True, True],
                    kind="stable",
                )
                indices.append(int(segment.index[0]))
            indices = sorted(indices)
        for position, index in enumerate(indices, start=1):
            selected = ordered.iloc[index]
            if len(indices) == 1:
                label = "only"
            elif len(ordered) <= count:
                label = f"all_eligible_{position}_of_{len(indices)}"
            else:
                label = f"segment_{position}_most_congested"
            rows.append(
                {
                    "corridor": corridor,
                    "tmc_code": selected["tmc_code"],
                    "road_order": selected["road_order"],
                    "selection_position": label,
                    "spatial_segment": position,
                    "observed_congestion_severity": selected[
                        "observed_congestion_severity"
                    ],
                    "observed_congested_interval_count": selected[
                        "observed_congested_interval_count"
                    ],
                    "observed_speed_mean_mph": selected[
                        "observed_speed_mean_mph"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _apply_profile_selection_overrides(
    selection: pd.DataFrame,
    profiles: pd.DataFrame,
    ranked_mapping: pd.DataFrame,
    overrides_path: Path | None,
) -> pd.DataFrame:
    """Apply audited display-only replacements to canonical TMC selections.

    An override is accepted only when the requested replacement is a canonical
    node-pair winner in the same corridor, is not already displayed, has
    comparable observed/model samples, and has a lower daily speed MAE than the
    TMC it replaces.  This cannot alter corridor membership or any assignment
    input. If spatial-segment selection already chose the replacement and no
    longer chose the original TMC, the override is an idempotent no-op.
    """

    result = selection.copy()
    result["selection_override_applied"] = False
    result["selection_override_replaced_tmc"] = ""
    if overrides_path is None:
        return result
    source = Path(overrides_path)
    if not source.is_file():
        raise FileNotFoundError(f"Dashboard profile-selection override not found: {source}")
    overrides = pd.read_csv(source, dtype="string").fillna("")
    required = {"corridor", "replace_tmc_code", "replacement_tmc_code"}
    missing = required.difference(overrides.columns)
    if missing:
        raise ValueError(
            "Dashboard profile-selection override is missing columns: "
            + ", ".join(sorted(missing))
        )

    winners = ranked_mapping.loc[
        ranked_mapping["selected_for_node_pair_lookup"].fillna(False),
        ["corridor", "tmc_code", "road_order"],
    ].drop_duplicates(["corridor", "tmc_code"])
    working = profiles[[
        "corridor", "tmc_code", "observed_tmc_speed_mph", "model_tmc_speed_mph"
    ]].copy()
    working["absolute_speed_error_mph"] = (
        pd.to_numeric(working["model_tmc_speed_mph"], errors="coerce")
        - pd.to_numeric(working["observed_tmc_speed_mph"], errors="coerce")
    ).abs()
    metrics = (
        working.groupby(["corridor", "tmc_code"], as_index=False)
        .agg(
            daily_mae_mph=("absolute_speed_error_mph", "mean"),
            comparable_intervals=(
                "absolute_speed_error_mph", lambda values: int(values.notna().sum())
            ),
        )
    )

    for override in overrides.itertuples(index=False):
        corridor = str(override.corridor).strip()
        replaced = str(override.replace_tmc_code).strip()
        replacement = str(override.replacement_tmc_code).strip()
        target = result["corridor"].eq(corridor) & result["tmc_code"].eq(replaced)
        replacement_selected = (
            result["corridor"].eq(corridor)
            & result["tmc_code"].eq(replacement)
        )
        if replacement_selected.any() and not target.any():
            continue
        if int(target.sum()) != 1:
            raise ValueError(
                f"Expected exactly one selected {corridor}/{replaced} row; "
                f"found {int(target.sum())}"
            )
        if replacement_selected.any():
            raise ValueError(
                f"Replacement {corridor}/{replacement} is already selected"
            )
        replacement_row = winners.loc[
            winners["corridor"].eq(corridor)
            & winners["tmc_code"].eq(replacement)
        ]
        if len(replacement_row) != 1:
            raise ValueError(
                f"Replacement {corridor}/{replacement} is not one canonical node-pair winner"
            )
        old_metric = metrics.loc[
            metrics["corridor"].eq(corridor) & metrics["tmc_code"].eq(replaced)
        ]
        new_metric = metrics.loc[
            metrics["corridor"].eq(corridor) & metrics["tmc_code"].eq(replacement)
        ]
        if old_metric.empty or new_metric.empty:
            raise ValueError(
                f"Comparable profile metrics are unavailable for {corridor} override"
            )
        old_mae = float(old_metric.iloc[0]["daily_mae_mph"])
        new_mae = float(new_metric.iloc[0]["daily_mae_mph"])
        new_intervals = int(new_metric.iloc[0]["comparable_intervals"])
        if new_intervals == 0 or not np.isfinite(new_mae) or not new_mae < old_mae:
            raise ValueError(
                f"Replacement {corridor}/{replacement} does not improve daily MAE "
                f"({new_mae:g} versus {old_mae:g})"
            )
        result.loc[target, "tmc_code"] = replacement
        result.loc[target, "road_order"] = float(
            replacement_row.iloc[0]["road_order"]
        )
        result.loc[target, "selection_position"] = (
            str(result.loc[target, "selection_position"].iloc[0])
            .replace("_most_congested", "_display_override")
        )
        result.loc[target, "selection_override_applied"] = True
        result.loc[target, "selection_override_replaced_tmc"] = replaced
    return result.sort_values(
        ["corridor", "road_order", "tmc_code"], kind="stable"
    ).reset_index(drop=True)


def _load_assignment_parameters(
    assignment_root: Path | None,
    requested_link_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if assignment_root is None:
        return frames
    requested_ids = {str(value).strip() for value in (requested_link_ids or set())}
    for period in ("AM", "MD", "PM"):
        source = Path(assignment_root) / period.lower() / "link_performance.csv"
        if not source.is_file():
            continue
        header = pd.read_csv(source, nrows=0)
        columns = [
            column
            for column in (
                "iteration_no", "link_id", "link_capacity", "lane_capacity",
                "vdf_plf", "vdf_alpha", "vdf_beta", "free_speed_mph",
                "volume", "doc", "P",
            )
            if column in header.columns
        ]
        columns.extend(
            column for column in header.columns
            if column.startswith("spd_mph_")
        )
        if "link_id" not in columns:
            continue
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            source,
            usecols=columns,
            dtype={"link_id": "string"},
            chunksize=10_000,
            low_memory=False,
        ):
            chunk["link_id"] = chunk["link_id"].str.strip()
            if requested_ids:
                chunk = chunk[chunk["link_id"].isin(requested_ids)]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            continue
        frame = pd.concat(chunks, ignore_index=True)
        frame["link_id"] = frame["link_id"].str.strip()
        if "iteration_no" in frame:
            frame["iteration_no"] = pd.to_numeric(frame["iteration_no"], errors="coerce")
            frame = frame.sort_values("iteration_no").drop_duplicates("link_id", keep="last")
        else:
            frame = frame.drop_duplicates("link_id", keep="last")
        link_source = Path(assignment_root) / period.lower() / "link.csv"
        if link_source.is_file():
            link_header = pd.read_csv(link_source, nrows=0)
            street_column = next(
                (
                    column for column in ("STREETNAME", "street_name", "name")
                    if column in link_header.columns
                ),
                None,
            )
            cube_volume_column = f"I4{period}VOL"
            cube_vc_column = f"I4{period}VC"
            link_columns = ["link_id"]
            link_columns.extend(
                column
                for column in (
                    street_column,
                    "allowed_use",
                    cube_volume_column,
                    cube_vc_column,
                )
                if column and column in link_header.columns
            )
            link_attributes = pd.read_csv(
                link_source,
                usecols=link_columns,
                dtype={"link_id": "string"},
                low_memory=False,
            )
            link_attributes["link_id"] = link_attributes["link_id"].str.strip()
            rename_columns = {
                cube_volume_column: "cube_volume",
                cube_vc_column: "cube_vc",
            }
            if street_column:
                rename_columns[street_column] = "street_name"
            link_attributes = link_attributes.drop_duplicates(
                "link_id", keep="last"
            ).rename(columns=rename_columns)
            frame = frame.merge(link_attributes, on="link_id", how="left")
        frames[period] = frame.set_index("link_id")
    return frames


def _assignment_summary(
    link_id: object, period: str, assignment: dict[str, pd.DataFrame]
) -> dict[str, object]:
    selected_link = _link_id_text(link_id)
    frame = assignment.get(period)
    if frame is None or not selected_link or selected_link not in frame.index:
        return {}
    available = frame.loc[[selected_link]]
    result: dict[str, object] = {}
    for source, target in (
        ("link_capacity", "capacity"),
        ("vdf_plf", "plf"),
        ("vdf_alpha", "alpha"),
        ("vdf_beta", "beta"),
        ("free_speed_mph", "free_speed"),
        ("volume", "volume"),
        ("doc", "doc"),
        ("P", "p_hours"),
        ("cube_volume", "cube_volume"),
        ("cube_vc", "cube_vc"),
    ):
        if source in available:
            numeric = pd.to_numeric(available[source], errors="coerce").dropna()
            if not numeric.empty:
                result[target] = float(numeric.mean())
    period_key = str(period).upper()
    if "p_hours" in result and period_key in PERIOD_BOUNDS:
        start_minute, end_minute = PERIOD_BOUNDS[period_key]
        period_duration_hours = (end_minute - start_minute) / 60.0
        result["p_hours"] = min(
            period_duration_hours,
            max(0.0, float(result["p_hours"])),
        )
    if "street_name" in available:
        street = available["street_name"].dropna().astype(str).str.strip()
        if not street.empty:
            result["street_name"] = street.iloc[0]
    if "allowed_use" in available:
        allowed_use = available["allowed_use"].fillna("").astype(str).str.strip()
        if not allowed_use.empty:
            result["allowed_use"] = allowed_use.iloc[0]
            result["is_closed"] = allowed_use.iloc[0].casefold() == "closed"
    return result


def _native_link_profile(
    link_id: object, assignment: dict[str, pd.DataFrame]
) -> tuple[list[int], list[float]]:
    selected_link = _link_id_text(link_id)
    points: dict[int, float] = {}
    for period in ("AM", "MD", "PM"):
        frame = assignment.get(period)
        if frame is None or selected_link not in frame.index:
            continue
        values = frame.loc[selected_link]
        for column in frame.columns:
            if not column.startswith("spd_mph_"):
                continue
            clock = column.removeprefix("spd_mph_")
            match = re.fullmatch(r"(\d{2}):(\d{2})", clock)
            if not match:
                continue
            minute = int(match.group(1)) * 60 + int(match.group(2))
            speed = pd.to_numeric(pd.Series([values[column]]), errors="coerce").iloc[0]
            if pd.notna(speed):
                points[minute] = float(speed)
    minutes = sorted(points)
    return minutes, [points[minute] for minute in minutes]


def _plot_corridor(
    *, corridor: str, measurement: pd.DataFrame,
    cbi_profile: pd.DataFrame, destination: Path,
    assignment: dict[str, pd.DataFrame],
    primary_mapping: pd.DataFrame,
) -> int:
    reference = (
        measurement[["tmc_code", "road_order", "selection_position"]]
        .drop_duplicates("tmc_code")
        .sort_values(["road_order", "tmc_code"], kind="stable")
    )
    if reference.empty:
        return 0
    merged = measurement.merge(
        cbi_profile[
            [
                "tmc_code", "t_min", "period", "speed_qvdf_model",
                "congestion_threshold_mph", "count_total_15min", "lanes",
                "capacity_vphpl",
            ]
        ].drop_duplicates(["tmc_code", "t_min"]),
        on=["tmc_code", "t_min"], how="left", validate="many_to_one",
    )
    minutes = list(range(360, 1140, 15))
    # Stack each table beneath its profile. The former 30-inch side-by-side
    # canvas was scaled down to the dashboard column width, making nominal
    # 11-12 pt text unreadable in the browser.
    figure_height = 9.4 * len(reference) + 2.4
    figure = plt.figure(figsize=(22, figure_height))
    grid = figure.add_gridspec(
        len(reference) * 2,
        1,
        height_ratios=[
            value
            for _ in range(len(reference))
            for value in (4.8, 3.4)
        ],
        hspace=0.38,
    )
    axes: list[plt.Axes] = []
    mapping_lookup = (
        primary_mapping.set_index(["corridor", "tmc_code"])
        if not primary_mapping.empty
        else pd.DataFrame()
    )

    for row_index, selection in enumerate(reference.itertuples(index=False)):
        axis = figure.add_subplot(
            grid[row_index * 2, 0],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        )
        axes.append(axis)
        legend_axis = figure.add_subplot(grid[row_index * 2 + 1, 0])
        legend_axis.axis("off")
        frame = (
            merged[merged["tmc_code"].eq(selection.tmc_code)]
            .sort_values("t_min").drop_duplicates("t_min")
            .set_index("t_min").reindex(minutes)
        )
        mapping_key = (str(corridor), str(selection.tmc_code))
        mapping_values = (
            mapping_lookup.loc[mapping_key]
            if not mapping_lookup.empty and mapping_key in mapping_lookup.index
            else pd.Series(dtype="object")
        )
        primary_link_id = _link_id_text(mapping_values.get("primary_link_id", ""))
        period_summaries = {
            period: _assignment_summary(primary_link_id, period, assignment)
            for period in PERIOD_BOUNDS
        }
        for period, (start, end) in PERIOD_BOUNDS.items():
            axis.axvspan(
                start, end, color=PERIOD_COLORS[period],
                alpha=0.035, linewidth=0,
            )
            if period_summaries[period].get("is_closed") is True:
                axis.axvspan(
                    start, end, facecolor="#6b7280", edgecolor="#6b7280",
                    alpha=0.10, hatch="////", linewidth=0.8, zorder=0.25,
                )
                axis.text(
                    (start + end) / 2,
                    0.965,
                    f"{period}: selected link closed\n(allowed_use=closed)",
                    transform=axis.get_xaxis_transform(),
                    ha="center", va="top", fontsize=12, fontweight="bold",
                    color="#7f1d1d", zorder=8,
                    bbox={
                        "boxstyle": "round,pad=0.32",
                        "facecolor": "white",
                        "edgecolor": "#7f1d1d",
                        "alpha": 0.92,
                    },
                )
        axis.plot(
            minutes, frame["observed_tmc_speed_mph"], color=OBSERVED_COLOR,
            linewidth=PRIMARY_LINE_WIDTH, label="Observed weekday mean",
        )
        taplite_minutes, taplite_speeds = _native_link_profile(
            primary_link_id, assignment
        )
        if taplite_minutes:
            axis.plot(
                taplite_minutes, taplite_speeds, color=TAPLITE_COLOR,
                linewidth=PRIMARY_LINE_WIDTH, linestyle="--",
                label="TAPlite best-match link (5 min)",
            )
        else:
            axis.plot(
                minutes, frame["model_tmc_speed_mph"], color=TAPLITE_COLOR,
                linewidth=PRIMARY_LINE_WIDTH, linestyle="--",
                label="TAPlite mapped-path fallback",
            )
        threshold = frame["cbi_tmc_congestion_threshold_mph"].combine_first(
            frame["congestion_threshold_mph"]
        )
        axis.plot(
            minutes, threshold, color=THRESHOLD_COLOR,
            linewidth=SECONDARY_LINE_WIDTH, linestyle=":", label="CBI threshold",
        )
        axis.axvline(540, color="#aaaaaa", linewidth=1.0)
        axis.axvline(900, color="#aaaaaa", linewidth=1.0)
        order = pd.to_numeric(pd.Series([selection.road_order]), errors="coerce").iloc[0]
        order_text = f"{float(order):g}" if pd.notna(order) else "unknown"
        axis.set_title(
            f"{str(selection.selection_position).replace('_', ' ').title()} "
            f"TMC: {selection.tmc_code}  (road order {order_text})",
            loc="left", fontsize=18,
        )
        axis.set_ylabel("Speed (mph)")
        axis.set_ylim(0, 82)
        axis.grid(color="#dddddd", linewidth=0.6, alpha=0.8)

        table_rows: list[list[str]] = []
        for period, (start, end) in PERIOD_BOUNDS.items():
            segment = frame.loc[start : end - 15]
            first = segment.dropna(how="all").head(1)
            if first.empty:
                continue
            values = first.iloc[0]
            observed_speed = pd.to_numeric(
                segment["observed_tmc_speed_mph"], errors="coerce"
            ).mean()
            synthetic_volume = pd.to_numeric(
                segment["count_total_15min"], errors="coerce"
            ).sum(min_count=1)
            assignment_values = period_summaries[period]
            street_name = str(assignment_values.get("street_name", "")).strip()
            street_label = street_name if street_name else "street name unavailable"
            if len(street_label) > 34:
                street_label = street_label[:31].rstrip() + "..."
            match_distance = mapping_values.get("network_match_distance_ft")
            mapping_label = (
                f"Best link {primary_link_id}: {street_label}"
                if primary_link_id
                else "Best link unavailable"
            )
            table_rows.append(
                [
                    period,
                    f"Avg {_number(observed_speed, '.1f')} mph\n"
                    f"Vol {_number(synthetic_volume, ',.0f')}",
                    f"Vol {_number(assignment_values.get('cube_volume'), ',.0f')}\n"
                    f"V/C {_number(assignment_values.get('cube_vc'), '.2f')}",
                    f"Vol {_number(assignment_values.get('volume'), ',.0f')}\n"
                    f"D/C {_number(assignment_values.get('doc'), '.2f')} · "
                    f"P {_number(assignment_values.get('p_hours'), '.2f')} h",
                    f"{mapping_label}\n"
                    f"Map distance {_number(match_distance, ',.1f')} ft | "
                    f"Cap {_number(assignment_values.get('capacity'), ',.0f')} veh/h | "
                    f"FF {_number(assignment_values.get('free_speed'), '.1f')} mph\n"
                    f"PLF {_number(assignment_values.get('plf'), '.3f')} · "
                    f"α {_number(assignment_values.get('alpha'), '.3f')} · "
                    f"β {_number(assignment_values.get('beta'), '.3f')}",
                ]
            )
        table = legend_axis.table(
            cellText=table_rows,
            colLabels=[
                "Period",
                "Observed / synthetic",
                "Cube",
                "TAPlite",
                "Best matched link / QVDF",
            ],
            colWidths=[0.07, 0.18, 0.13, 0.17, 0.45],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(PERIOD_TABLE_FONT_SIZE)
        for (table_row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d7dde4")
            cell.set_linewidth(0.8)
            cell.PAD = 0.055
            cell.set_text_props(va="center", linespacing=1.15)
            if table_row == 0:
                cell.set_height(0.13)
                cell.set_facecolor("#edf3f8")
                cell.set_text_props(
                    weight="bold", color="#243746", va="center"
                )
            else:
                # Header plus three period rows must stay within the table
                # panel. Taller cells extend beyond the axes and collide
                # with the footer on one-TMC and final-TMC figures.
                cell.set_height(0.27)
                period = table_rows[table_row - 1][0]
                cell.set_facecolor(PERIOD_COLORS[period] + "12")

    ticks = list(range(360, 1140, 60))
    for axis in axes:
        axis.set_xticks(ticks)
        axis.set_xticklabels([_clock(value) for value in ticks])
    axes[-1].set_xlabel("Time of day")
    figure.legend(
        handles=[
            Line2D([0], [0], color=OBSERVED_COLOR, linewidth=PRIMARY_LINE_WIDTH,
                   label="Observed weekday mean"),
            Line2D([0], [0], color=TAPLITE_COLOR, linewidth=PRIMARY_LINE_WIDTH,
                   linestyle="--", label="TAPlite best-match link (5 min)"),
            Line2D([0], [0], color=THRESHOLD_COLOR,
                   linewidth=SECONDARY_LINE_WIDTH, linestyle=":",
                   label="CBI threshold"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.966), ncol=3, frameon=False,
    )
    figure.suptitle(
        f"{corridor}: TMC-aligned observed and TAPlite profiles",
        fontsize=22, y=0.995,
    )
    figure.text(
        0.5, 0.18 / figure_height,
        "Observed is the CBI 15-minute average-weekday INRIX TMC profile; "
        "TAPlite is the selected link's native 5-minute output.\n"
        "Each link uses the canonical TMC winner from the direct observed-QVDF "
        "node-pair mapping; the table reports its period diagnostics.\n"
        "Period hatching means allowed_use=closed; zero volume alone is not "
        "treated as a closure.",
        ha="center", va="bottom", fontsize=MINIMUM_FONT_SIZE, color="#555555",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.935,
        bottom=0.90 / figure_height,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the fixed canvas instead of expanding to every table-text extent.
    # A tight bounding box can make the PNG much wider without increasing its
    # height, so the browser scales the profile rows down inside the report.
    figure.savefig(
        destination,
        dpi=FIGURE_DPI,
        pil_kwargs={"compress_level": 3},
    )
    plt.close(figure)
    return 1


def _plot_volume_consistency(
    *, corridor: str, measurement: pd.DataFrame, cbi_profile: pd.DataFrame,
    qvdf_flow: pd.DataFrame, destination: Path,
) -> int:
    reference = (
        measurement[["tmc_code", "road_order", "selection_position"]]
        .drop_duplicates("tmc_code")
        .sort_values(["road_order", "tmc_code"], kind="stable")
    )
    if reference.empty:
        return 0
    flow = qvdf_flow.copy()
    for column in ("qvdf_count_total_15min",):
        flow[column] = pd.to_numeric(flow[column], errors="coerce")
    flow = flow.groupby(["tmc_code", "t_min"], as_index=False)["qvdf_count_total_15min"].mean()
    profile = cbi_profile.merge(flow, on=["tmc_code", "t_min"], how="left")
    minutes = list(range(360, 1140, 15))
    figure, axes = plt.subplots(
        len(reference), 1, figsize=(18, 3.1 * len(reference) + 1.8),
        sharex=True, sharey=False,
    )
    axes = np.atleast_1d(axes).tolist()
    for axis, selection in zip(axes, reference.itertuples(index=False)):
        frame = (
            profile[profile["tmc_code"].eq(selection.tmc_code)]
            .groupby("t_min", as_index=True)
            .agg(
                synthetic_volume=("count_total_15min", "mean"),
                qvdf_volume=("qvdf_count_total_15min", "mean"),
                capacity_vphpl=("capacity_vphpl", "mean"),
                lanes=("lanes", "mean"),
            )
            .reindex(minutes)
        )
        for period, (start, end) in PERIOD_BOUNDS.items():
            axis.axvspan(start, end, color=PERIOD_COLORS[period], alpha=0.035, linewidth=0)
        axis.plot(
            minutes, frame["synthetic_volume"], color=OBSERVED_COLOR,
            linewidth=PRIMARY_LINE_WIDTH, label="Synthetic volume from observed speed",
        )
        axis.plot(
            minutes, frame["qvdf_volume"], color=CBI_QVDF_COLOR,
            linewidth=PRIMARY_LINE_WIDTH, linestyle="-.", label="CBI QVDF conserved volume",
        )
        capacity_15min = frame["capacity_vphpl"] * frame["lanes"] * 0.25
        axis.plot(
            minutes, capacity_15min, color=THRESHOLD_COLOR,
            linewidth=SECONDARY_LINE_WIDTH, linestyle=":", label="15-minute capacity",
        )
        order = pd.to_numeric(pd.Series([selection.road_order]), errors="coerce").iloc[0]
        order_text = f"{float(order):g}" if pd.notna(order) else "unknown"
        axis.set_title(
            f"{str(selection.selection_position).replace('_', ' ').title()} "
            f"TMC: {selection.tmc_code}  (road order {order_text})",
            loc="left", fontsize=14,
        )
        axis.set_ylabel("Volume / 15 min")
        axis.grid(color="#dddddd", linewidth=0.7, alpha=0.8)
    ticks = list(range(360, 1140, 60))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([_clock(value) for value in ticks])
    axes[-1].set_xlabel("Time of day")
    figure.legend(
        handles=[
            Line2D([0], [0], color=OBSERVED_COLOR, linewidth=PRIMARY_LINE_WIDTH,
                   label="Synthetic volume from observed speed"),
            Line2D([0], [0], color=CBI_QVDF_COLOR, linewidth=PRIMARY_LINE_WIDTH,
                   linestyle="-.", label="CBI QVDF conserved volume"),
            Line2D([0], [0], color=THRESHOLD_COLOR, linewidth=SECONDARY_LINE_WIDTH,
                   linestyle=":", label="15-minute link capacity"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False,
    )
    figure.suptitle(
        f"{corridor}: CBI speed-derived and QVDF-conserved volumes for selected TMCs",
        fontsize=17, y=0.995,
    )
    figure.text(
        0.5, 0.012,
        "The TMCs and road order are identical to the combined speed profiles above. "
        "All volumes are displayed per 15-minute interval.",
        ha="center", va="bottom", fontsize=MINIMUM_FONT_SIZE, color="#555555",
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.045, hspace=0.36)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        pil_kwargs={"compress_level": 3},
    )
    plt.close(figure)
    return 1


def generate_combined_tmc_profile_figures(
    *, measurement_root: Path, cbi_corridors_root: Path,
    staged_reports_root: Path,
    assignment_root: Path | None = None,
    mapmatching_product_root: Path | None = None,
    selection_overrides_path: Path | None = None,
    include_volume: bool = True,
    corridor_ids: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Generate combined speed and volume figures for the same TMC selection."""

    selected_path = (
        Path(measurement_root) / "02-tmc-results" / "selected_tmc_profiles.csv"
    )
    if not selected_path.is_file():
        return {}
    all_profiles_path = (
        Path(measurement_root) / "02-tmc-results" / "tmc_daily_profiles.csv"
    )
    profile_source = all_profiles_path if all_profiles_path.is_file() else selected_path
    profiles = pd.read_csv(
        profile_source,
        dtype={"corridor": "string", "tmc_code": "string"},
        low_memory=False,
    )
    profiles = profiles.loc[
        ~profiles["corridor"].map(is_managed_corridor)
    ].copy()
    eligible_tmc_codes = (
        _load_general_purpose_tmc_codes(mapmatching_product_root)
        if mapmatching_product_root is not None
        else None
    )
    if eligible_tmc_codes is not None:
        profiles = profiles.loc[
            profiles["tmc_code"].astype("string").str.strip().isin(
                eligible_tmc_codes
            )
        ].copy()
    generated: dict[str, dict[str, str]] = {}
    ranked_mapping = _load_ranked_primary_link_mapping(
        cbi_corridors_root,
        eligible_tmc_codes=eligible_tmc_codes,
    )
    selection = _select_canonical_representative_tmcs(
        profiles, ranked_mapping, count=5
    )
    selection = _apply_profile_selection_overrides(
        selection,
        profiles,
        ranked_mapping,
        selection_overrides_path,
    )
    if selection.empty:
        # Minimal dashboard fixtures and legacy runs may lack network-node
        # identifiers. Retain their former display-only fallback; complete CBI
        # runs always take the canonical branch below.
        selected = pd.read_csv(
            selected_path,
            dtype={"corridor": "string", "tmc_code": "string"},
            low_memory=False,
        )
        selected = selected.loc[
            ~selected["corridor"].map(is_managed_corridor)
        ].copy()
        selected = _filter_general_purpose_profiles(
            selected, mapmatching_product_root
        )
        primary_mapping = _load_primary_link_mapping(
            selected, cbi_corridors_root
        )
    else:
        selected = profiles.drop(columns="selection_position", errors="ignore").merge(
            selection[["corridor", "tmc_code", "selection_position"]],
            on=["corridor", "tmc_code"],
            how="inner",
            validate="many_to_one",
        )
        primary_mapping = selection[["corridor", "tmc_code"]].merge(
            ranked_mapping,
            on=["corridor", "tmc_code"],
            how="left",
            validate="one_to_one",
        )
        if not primary_mapping["selected_for_node_pair_lookup"].fillna(False).all():
            raise ValueError("Dashboard selection contains a noncanonical TMC")
        dashboard_root = Path(staged_reports_root).parents[1]
        selection_audit = selection.merge(
            primary_mapping.drop(columns="road_order", errors="ignore"),
            on=["corridor", "tmc_code"],
            how="left",
            validate="one_to_one",
        )
        selection_audit["facility_class"] = "gp"
        selection_audit_path = (
            dashboard_root / "data" / "dashboard_canonical_tmc_selection.csv"
        )
        selection_audit_path.parent.mkdir(parents=True, exist_ok=True)
        selection_audit.to_csv(selection_audit_path, index=False)
    if corridor_ids is not None:
        requested_corridors = {str(value).strip() for value in corridor_ids}
        selected = selected.loc[
            selected["corridor"].astype("string").isin(requested_corridors)
        ].copy()
        primary_mapping = primary_mapping.loc[
            primary_mapping["corridor"].astype("string").isin(requested_corridors)
        ].copy()
    requested_link_ids = set(
        primary_mapping["primary_link_id"].dropna().astype(str).str.strip()
    )
    assignment = _load_assignment_parameters(
        assignment_root, requested_link_ids=requested_link_ids
    )
    for corridor, measurement in selected.groupby("corridor", sort=True):
        cbi_path = (
            Path(cbi_corridors_root) / str(corridor)
            / "07-reconstruction-and-handoff"
            / "average_weekday_time_dependent.csv"
        )
        if not cbi_path.is_file():
            continue
        requested_columns = [
            "tmc_code", "t_min", "period", "speed_qvdf_model",
            "congestion_threshold_mph", "count_total_15min", "lanes",
            "capacity_vphpl",
        ]
        cbi_header = pd.read_csv(cbi_path, nrows=0)
        cbi_profile = pd.read_csv(
            cbi_path, dtype={"tmc_code": "string"},
            usecols=[
                column for column in requested_columns
                if column in cbi_header.columns
            ],
            low_memory=False,
        )
        for column in requested_columns:
            if column not in cbi_profile:
                cbi_profile[column] = np.nan
        destination = (
            Path(staged_reports_root) / str(corridor) / "daily_analysis"
            / "tmc_observed_qvdf_taplite.png"
        )
        if _plot_corridor(
            corridor=str(corridor), measurement=measurement,
            cbi_profile=cbi_profile, destination=destination, assignment=assignment,
            primary_mapping=primary_mapping,
        ):
            generated.setdefault(str(corridor), {})["speed"] = (
                "daily_analysis/tmc_observed_qvdf_taplite.png"
            )
        flow_path = cbi_path.with_name("qvdf_conserved_flow.csv")
        if include_volume and flow_path.is_file():
            qvdf_flow = pd.read_csv(
                flow_path,
                dtype={"tmc_code": "string"},
                usecols=["tmc_code", "t_min", "qvdf_count_total_15min"],
                low_memory=False,
            )
            volume_destination = destination.with_name("tmc_cbi_volume_consistency.png")
            if _plot_volume_consistency(
                corridor=str(corridor), measurement=measurement,
                cbi_profile=cbi_profile, qvdf_flow=qvdf_flow,
                destination=volume_destination,
            ):
                generated.setdefault(str(corridor), {})["volume"] = (
                    "daily_analysis/tmc_cbi_volume_consistency.png"
                )
    return generated
