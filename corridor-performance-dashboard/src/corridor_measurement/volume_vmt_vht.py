"""Cube-versus-TAPlite volume, VMT, and VHT comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


CUBE_SPEED_COLUMNS = {
    "am": "I4AMSPD",
    "md": "I4MDSPD",
    "pm": "I4PMSPD",
}

CUBE_VOC_COLUMNS = {
    "am": "I4AMVC",
    "md": "I4MDVC",
    "pm": "I4PMVC",
}

METRICS = {
    "volume": {
        "cube": "cube_volume",
        "taplite": "taplite_volume",
        "title": "Volume",
        "unit": "vehicles",
        "linear_floor": 1.0,
    },
    "vmt": {
        "cube": "cube_vmt",
        "taplite": "taplite_vmt",
        "title": "VMT",
        "unit": "vehicle-miles",
        "linear_floor": 0.1,
    },
    "vht": {
        "cube": "cube_vht",
        "taplite": "taplite_vht",
        "title": "VHT",
        "unit": "vehicle-hours",
        "linear_floor": 0.01,
    },
}

NETWORK_METRIC_GROUPS = {
    "volume-travel-time-speed": {
        "title": "All network links: Cube versus TAPlite loading and performance",
        "metrics": {
            "volume": {
                "cube": "cube_volume",
                "taplite": "taplite_volume",
                "title": "Volume",
                "unit": "vehicles",
            },
            "travel_time": {
                "cube": "cube_travel_time_hours",
                "taplite": "taplite_travel_time_hours",
                "title": "Travel time",
                "unit": "hours",
            },
            "speed": {
                "cube": "cube_speed_mph",
                "taplite": "taplite_speed_mph",
                "title": "Speed",
                "unit": "mph",
            },
        },
    },
    "vmt-vht-doc": {
        "title": "All network links: Cube versus TAPlite derived measures",
        "metrics": {
            "vmt": {
                "cube": "cube_vmt",
                "taplite": "taplite_vmt",
                "title": "VMT",
                "unit": "vehicle-miles",
            },
            "vht": {
                "cube": "cube_vht",
                "taplite": "taplite_vht",
                "title": "VHT",
                "unit": "vehicle-hours",
            },
            "dc": {
                "cube": "cube_voc",
                "taplite": "taplite_doc",
                "title": "Demand/capacity",
                "unit": "ratio",
                "cube_label": "Cube V/C",
                "taplite_label": "TAPlite D/C",
            },
        },
    },
}

PERIOD_COLORS = {"AM": "#4c78a8", "MD": "#f2a541", "PM": "#8f63b8"}
MINIMUM_FONT_SIZE = 11

plt.rcParams.update(
    {
        "font.size": MINIMUM_FONT_SIZE,
        "axes.titlesize": 13,
        "axes.labelsize": MINIMUM_FONT_SIZE,
        "xtick.labelsize": MINIMUM_FONT_SIZE,
        "ytick.labelsize": MINIMUM_FONT_SIZE,
        "legend.fontsize": MINIMUM_FONT_SIZE,
        "legend.title_fontsize": MINIMUM_FONT_SIZE,
    }
)


def _required_columns(
    available: Mapping[str, object] | pd.DataFrame,
    required: List[str],
    source: Path,
) -> None:
    columns = set(available.columns)
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def load_period_link_comparison(
    performance_source: Path,
    link_source: Path,
    *,
    period: str,
    cube_volume_column: str,
    cube_speed_column: str | None = None,
    cube_voc_column: str | None = None,
) -> pd.DataFrame:
    """Build link-level Cube/TAPlite comparison measures for one period."""
    normalized_period = period.strip().lower()
    resolved_speed_column = cube_speed_column or CUBE_SPEED_COLUMNS.get(
        normalized_period
    )
    resolved_voc_column = cube_voc_column or CUBE_VOC_COLUMNS.get(normalized_period)
    if resolved_speed_column is None:
        raise ValueError(f"No Cube speed column configured for period {period!r}.")
    if not performance_source.is_file():
        raise FileNotFoundError(
            f"TAPlite link-performance file not found: {performance_source}"
        )
    if not link_source.is_file():
        raise FileNotFoundError(f"Cube/GMNS link file not found: {link_source}")

    performance_header = pd.read_csv(performance_source, nrows=0)
    _required_columns(
        performance_header,
        ["link_id", "volume"],
        performance_source,
    )
    if not any(
        column in performance_header.columns
        for column in (
            "travel_time",
            "speed_mph",
            "avg_QVDF_period_travel_time",
            "avg_QVDF_period_speed_mph",
        )
    ):
        raise ValueError(
            f"{performance_source} needs TAPlite travel time or speed to calculate VHT."
        )
    performance_columns = ["link_id", "volume"]
    for optional in (
        "iteration_no",
        "VMT",
        "VHT",
        "speed_mph",
        "travel_time",
        "avg_QVDF_period_speed_mph",
        "avg_QVDF_period_travel_time",
        "doc",
    ):
        if optional in performance_header.columns:
            performance_columns.append(optional)
    performance = pd.read_csv(
        performance_source,
        usecols=performance_columns,
        dtype={"link_id": "string"},
    )
    performance["link_id"] = performance["link_id"].str.strip()
    if "iteration_no" in performance:
        performance["iteration_no"] = pd.to_numeric(
            performance["iteration_no"], errors="coerce"
        )
        performance = (
            performance.sort_values("iteration_no")
            .drop_duplicates("link_id", keep="last")
            .drop(columns="iteration_no")
        )
    elif performance["link_id"].duplicated().any():
        raise ValueError(f"Duplicate link_id rows found in {performance_source}")

    link_header = pd.read_csv(link_source, nrows=0)
    _required_columns(
        link_header,
        ["link_id", cube_volume_column, resolved_speed_column],
        link_source,
    )
    length_candidates = [
        column
        for column in ("vdf_length_mi", "length_in_mile")
        if column in link_header.columns
    ]
    if not length_candidates:
        raise ValueError(
            f"{link_source} needs vdf_length_mi or length_in_mile for VMT/VHT."
        )
    link_columns = [
        "link_id",
        cube_volume_column,
        resolved_speed_column,
        *length_candidates,
    ]
    if resolved_voc_column and resolved_voc_column in link_header.columns:
        link_columns.append(resolved_voc_column)
    links = pd.read_csv(
        link_source,
        usecols=link_columns,
        dtype={"link_id": "string"},
    )
    links["link_id"] = links["link_id"].str.strip()
    if links["link_id"].duplicated().any():
        raise ValueError(f"Duplicate link_id rows found in {link_source}")

    links["length_mi"] = np.nan
    for column in length_candidates:
        candidate = pd.to_numeric(links[column], errors="coerce")
        links["length_mi"] = links["length_mi"].where(
            links["length_mi"].gt(0), candidate
        )
    links["cube_volume"] = pd.to_numeric(
        links[cube_volume_column], errors="coerce"
    )
    links["cube_speed_mph"] = pd.to_numeric(
        links[resolved_speed_column], errors="coerce"
    )
    links["cube_voc"] = (
        pd.to_numeric(links[resolved_voc_column], errors="coerce")
        if resolved_voc_column and resolved_voc_column in links
        else np.nan
    )
    links = links[
        ["link_id", "length_mi", "cube_volume", "cube_speed_mph", "cube_voc"]
    ]

    performance = performance.rename(
        columns={
            "volume": "taplite_volume",
            "VMT": "taplite_recorded_vmt",
            "VHT": "taplite_recorded_vht",
            "speed_mph": "taplite_speed_mph",
            "travel_time": "taplite_travel_time_min",
            "avg_QVDF_period_speed_mph": "taplite_qvdf_period_speed_mph",
            "avg_QVDF_period_travel_time": "taplite_qvdf_period_travel_time_min",
            "doc": "taplite_doc",
        }
    )
    for column in performance.columns.difference(["link_id"]):
        performance[column] = pd.to_numeric(performance[column], errors="coerce")

    comparison = links.merge(
        performance,
        on="link_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    comparison.insert(0, "period", normalized_period.upper())
    comparison["link_pair_status"] = comparison.pop("_merge").map(
        {
            "both": "paired",
            "left_only": "cube_only",
            "right_only": "taplite_only",
        }
    )
    valid_cube_volume = comparison["cube_volume"].ge(0)
    valid_taplite_volume = comparison["taplite_volume"].ge(0)
    valid_length = comparison["length_mi"].gt(0)
    valid_cube_speed = comparison["cube_speed_mph"].gt(0)
    comparison["cube_travel_time_hours"] = (
        comparison["length_mi"] / comparison["cube_speed_mph"]
    ).where(valid_length & valid_cube_speed)
    comparison["cube_vmt"] = (
        comparison["cube_volume"] * comparison["length_mi"]
    ).where(valid_cube_volume & valid_length)
    comparison["cube_vht"] = (
        comparison["cube_volume"] * comparison["cube_travel_time_hours"]
    ).where(valid_cube_volume & comparison["cube_travel_time_hours"].notna())
    comparison["taplite_travel_time_hours"] = np.nan
    if "taplite_travel_time_min" in comparison:
        travel_time_min = pd.to_numeric(
            comparison["taplite_travel_time_min"], errors="coerce"
        )
        comparison["taplite_travel_time_hours"] = (
            travel_time_min / 60.0
        ).where(travel_time_min.ge(0))
    if "taplite_qvdf_period_travel_time_min" in comparison:
        qvdf_travel_time_min = pd.to_numeric(
            comparison["taplite_qvdf_period_travel_time_min"], errors="coerce"
        )
        comparison["taplite_travel_time_hours"] = comparison[
            "taplite_travel_time_hours"
        ].where(
            comparison["taplite_travel_time_hours"].notna(),
            (qvdf_travel_time_min / 60.0).where(qvdf_travel_time_min.ge(0)),
        )
    for speed_column in (
        "taplite_speed_mph",
        "taplite_qvdf_period_speed_mph",
    ):
        if speed_column not in comparison:
            continue
        speed = pd.to_numeric(comparison[speed_column], errors="coerce")
        fallback_hours = (comparison["length_mi"] / speed).where(
            valid_length & speed.gt(0)
        )
        comparison["taplite_travel_time_hours"] = comparison[
            "taplite_travel_time_hours"
        ].where(
            comparison["taplite_travel_time_hours"].notna(),
            fallback_hours,
        )
    if "taplite_speed_mph" not in comparison:
        comparison["taplite_speed_mph"] = np.nan
    if "taplite_qvdf_period_speed_mph" in comparison:
        comparison["taplite_speed_mph"] = comparison["taplite_speed_mph"].where(
            comparison["taplite_speed_mph"].gt(0),
            pd.to_numeric(
                comparison["taplite_qvdf_period_speed_mph"], errors="coerce"
            ),
        )
    if "taplite_doc" not in comparison:
        comparison["taplite_doc"] = np.nan
    comparison["taplite_vmt"] = (
        comparison["taplite_volume"] * comparison["length_mi"]
    ).where(valid_taplite_volume & valid_length)
    comparison["taplite_vht"] = (
        comparison["taplite_volume"]
        * comparison["taplite_travel_time_hours"]
    ).where(
        valid_taplite_volume
        & comparison["taplite_travel_time_hours"].notna()
    )
    comparison["cube_volume_column"] = cube_volume_column
    comparison["cube_speed_column"] = resolved_speed_column
    comparison["cube_voc_column"] = resolved_voc_column or ""
    comparison["cube_vmt_method"] = "cube_volume_x_length_mi"
    comparison["cube_vht_method"] = (
        f"cube_volume_x_length_mi_div_{resolved_speed_column}"
    )
    comparison["taplite_vmt_method"] = "taplite_volume_x_length_mi"
    comparison["taplite_vht_method"] = (
        "taplite_volume_x_link_performance_travel_time_min_div_60"
    )
    return comparison.sort_values("link_id", kind="stable").reset_index(drop=True)


def _origin_constrained_slope(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def build_network_scatter_metrics(link_comparison: pd.DataFrame) -> pd.DataFrame:
    """Fit all-network Cube/TAPlite metrics together and by assignment period."""
    rows: List[Dict[str, object]] = []
    paired = link_comparison[
        link_comparison["link_pair_status"].eq("paired")
    ].copy()
    paired["period"] = paired["period"].astype(str).str.upper()
    available_periods = [
        period for period in PERIOD_COLORS if paired["period"].eq(period).any()
    ]
    frames = [("ALL", paired)] + [
        (period, paired[paired["period"].eq(period)])
        for period in available_periods
    ]
    for period, period_frame in frames:
        for group_name, group in NETWORK_METRIC_GROUPS.items():
            for metric, specifications in group["metrics"].items():
                cube = pd.to_numeric(
                    period_frame[str(specifications["cube"])], errors="coerce"
                )
                taplite = pd.to_numeric(
                    period_frame[str(specifications["taplite"])], errors="coerce"
                )
                valid = cube.ge(0) & taplite.ge(0) & cube.notna() & taplite.notna()
                x = cube[valid].to_numpy(dtype=float)
                y = taplite[valid].to_numpy(dtype=float)
                slope = _origin_constrained_slope(x, y)
                rows.append(
                    {
                        "scope": "all_network_links",
                        "period": period,
                        "figure_group": group_name,
                        "metric": metric,
                        "paired_point_count": int(len(x)),
                        "origin_fit_slope": slope,
                        "pearson_r": (
                            float(np.corrcoef(x, y)[0, 1])
                            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
                            else float("nan")
                        ),
                        "mae": (
                            float(np.mean(np.abs(y - x)))
                            if len(x)
                            else float("nan")
                        ),
                        "rmse": (
                            float(np.sqrt(np.mean(np.square(y - x))))
                            if len(x)
                            else float("nan")
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    period_order = {period: index for index, period in enumerate(("ALL", *PERIOD_COLORS))}
    metrics["_period_order"] = metrics["period"].map(period_order)
    return metrics.sort_values(
        ["_period_order", "figure_group", "metric"], kind="stable"
    ).drop(columns="_period_order").reset_index(drop=True)


def build_corridor_period_comparison(
    link_comparison: pd.DataFrame,
    corridor_links: pd.DataFrame,
) -> pd.DataFrame:
    """Sum link measures across each period's de-duplicated corridor links."""
    required_links = ["period", "corridor", "link_id"]
    missing = sorted(set(required_links).difference(corridor_links.columns))
    if missing:
        raise ValueError(
            "Corridor link table is missing columns: " + ", ".join(missing)
        )
    metrics = [
        "cube_volume",
        "taplite_volume",
        "cube_vmt",
        "taplite_vmt",
        "cube_vht",
        "taplite_vht",
    ]
    link_values = link_comparison[["period", "link_id", *metrics]].copy()
    link_values["period"] = link_values["period"].astype(str).str.upper()
    link_values["link_id"] = link_values["link_id"].astype("string").str.strip()
    membership = corridor_links[required_links].copy()
    membership["period"] = membership["period"].astype(str).str.upper()
    membership["link_id"] = membership["link_id"].astype("string").str.strip()
    membership = membership.drop_duplicates(required_links)
    joined = membership.merge(
        link_values,
        on=["period", "link_id"],
        how="left",
        validate="many_to_one",
    )
    rows: List[Dict[str, object]] = []
    for (period, corridor), group in joined.groupby(
        ["period", "corridor"], sort=True
    ):
        row: Dict[str, object] = {
            "period": period,
            "corridor": corridor,
            "gmns_link_count": int(group["link_id"].nunique()),
        }
        for column in metrics:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = values.sum(min_count=1)
            row[f"{column}_link_count"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["period", "corridor"], kind="stable"
    )


def build_scatter_metrics(
    link_comparison: pd.DataFrame,
    corridor_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return audit statistics for each scatter panel."""
    rows: List[Dict[str, object]] = []
    for scope, frame in (
        ("all_links", link_comparison),
        ("corridors", corridor_comparison),
    ):
        for period, period_frame in frame.groupby("period", sort=True):
            for metric, specifications in METRICS.items():
                cube_column = str(specifications["cube"])
                taplite_column = str(specifications["taplite"])
                cube = pd.to_numeric(period_frame[cube_column], errors="coerce")
                taplite = pd.to_numeric(
                    period_frame[taplite_column], errors="coerce"
                )
                valid = cube.ge(0) & taplite.ge(0) & cube.notna() & taplite.notna()
                x = cube[valid].to_numpy(dtype=float)
                y = taplite[valid].to_numpy(dtype=float)
                correlation = float("nan")
                if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                    correlation = float(np.corrcoef(x, y)[0, 1])
                difference = y - x
                cube_total = float(x.sum()) if len(x) else float("nan")
                taplite_total = float(y.sum()) if len(y) else float("nan")
                rows.append(
                    {
                        "scope": scope,
                        "period": period,
                        "metric": metric,
                        "paired_point_count": int(len(x)),
                        "cube_total": cube_total,
                        "taplite_total": taplite_total,
                        "taplite_to_cube_total_ratio": (
                            taplite_total / cube_total
                            if np.isfinite(cube_total) and cube_total > 0
                            else float("nan")
                        ),
                        "pearson_r": correlation,
                        "mae": (
                            float(np.mean(np.abs(difference)))
                            if len(difference)
                            else float("nan")
                        ),
                        "rmse": (
                            float(np.sqrt(np.mean(np.square(difference))))
                            if len(difference)
                            else float("nan")
                        ),
                        "cube_zero_count": int(np.isclose(x, 0.0).sum()),
                        "taplite_zero_count": int(np.isclose(y, 0.0).sum()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["scope", "period", "metric"], kind="stable"
    )


def _compact_number(value: float, _position: int | None = None) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if absolute >= 1_000:
        return f"{value / 1_000:g}k"
    if absolute >= 1:
        return f"{value:g}"
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.2g}"


def _plot_period_scatter(
    frame: pd.DataFrame,
    period_metrics: pd.DataFrame,
    destination: Path,
    *,
    period: str,
    scope: str,
    figure_dpi: int,
) -> None:
    scope_label = (
        "all GMNS links"
        if scope == "all_links"
        else "CBI corridors (unique mapped links summed)"
    )
    figure, axes = plt.subplots(1, 3, figsize=(19.5, 6.5))
    point_color = "#1769aa" if scope == "all_links" else "#6f3fa0"
    for axis, (metric, specifications) in zip(axes, METRICS.items()):
        cube_column = str(specifications["cube"])
        taplite_column = str(specifications["taplite"])
        cube = pd.to_numeric(frame[cube_column], errors="coerce")
        taplite = pd.to_numeric(frame[taplite_column], errors="coerce")
        valid = cube.ge(0) & taplite.ge(0) & cube.notna() & taplite.notna()
        x = cube[valid].to_numpy(dtype=float)
        y = taplite[valid].to_numpy(dtype=float)
        maximum = float(max(np.max(x, initial=0), np.max(y, initial=0)))
        limit = maximum * 1.06 if maximum > 0 else 1.0
        linthresh = max(
            float(specifications["linear_floor"]),
            limit * 0.00001,
        )
        axis.scatter(
            x,
            y,
            s=7 if scope == "all_links" else 28,
            alpha=0.16 if scope == "all_links" else 0.68,
            color=point_color,
            edgecolors="none",
        )
        axis.plot(
            [0, limit],
            [0, limit],
            color="#555555",
            linewidth=1.2,
            linestyle="--",
            label="1:1",
        )
        axis.set_xscale("symlog", linthresh=linthresh)
        axis.set_yscale("symlog", linthresh=linthresh)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        formatter = FuncFormatter(_compact_number)
        axis.xaxis.set_major_formatter(formatter)
        axis.yaxis.set_major_formatter(formatter)
        axis.grid(color="#d9d9d9", linewidth=0.6, alpha=0.75)
        axis.set_title(
            str(specifications["title"]),
            fontsize=12,
            pad=8,
            zorder=100,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )
        axis.set_xlabel(
            f"Cube {specifications['title']} ({specifications['unit']})"
        )
        axis.set_ylabel(
            f"TAPlite {specifications['title']} ({specifications['unit']})"
        )
        stats = period_metrics[period_metrics["metric"].eq(metric)].iloc[0]
        correlation = stats["pearson_r"]
        correlation_text = (
            f"{float(correlation):.3f}" if pd.notna(correlation) else "NA"
        )
        ratio = stats["taplite_to_cube_total_ratio"]
        ratio_text = f"{float(ratio):.3f}" if pd.notna(ratio) else "NA"
        axis.text(
            0.04,
            0.96,
            f"n = {int(stats['paired_point_count']):,}\n"
            f"Pearson r = {correlation_text}\n"
            f"Total TAPlite/Cube = {ratio_text}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=MINIMUM_FONT_SIZE,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.90,
            },
        )
        axis.legend(loc="lower right", frameon=False)

    figure.suptitle(
        f"{period}: Cube versus TAPlite volume, VMT, and VHT — {scope_label}",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        "All points are retained; symmetric-log axes preserve zero values. "
        "Cube VMT = Cube volume × link miles; Cube VHT = Cube volume × "
        "link miles ÷ period Cube speed. TAPlite VMT = TAPlite volume × link "
        "miles; TAPlite VHT = TAPlite volume × link_performance travel time ÷ 60.",
        ha="center",
        va="bottom",
        fontsize=MINIMUM_FONT_SIZE,
        color="#555555",
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.15, wspace=0.26)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=figure_dpi, pil_kwargs={"compress_level": 3})
    plt.close(figure)


def _plot_network_scatter_group(
    frame: pd.DataFrame,
    network_metrics: pd.DataFrame,
    destination: Path,
    *,
    group_name: str,
    period: str = "ALL",
    figure_dpi: int,
) -> None:
    group = NETWORK_METRIC_GROUPS[group_name]
    figure, axes = plt.subplots(1, 3, figsize=(20.5, 7.0))
    paired = frame[frame["link_pair_status"].eq("paired")].copy()
    paired["period"] = paired["period"].astype(str).str.upper()
    normalized_period = period.strip().upper()
    plotted_periods = (
        tuple(PERIOD_COLORS)
        if normalized_period == "ALL"
        else (normalized_period,)
    )
    if normalized_period != "ALL":
        paired = paired[paired["period"].eq(normalized_period)]
    for axis, (metric, specifications) in zip(axes, group["metrics"].items()):
        all_values: list[np.ndarray] = []
        for plotted_period in plotted_periods:
            period_frame = paired[paired["period"].eq(plotted_period)]
            cube = pd.to_numeric(
                period_frame[str(specifications["cube"])], errors="coerce"
            )
            taplite = pd.to_numeric(
                period_frame[str(specifications["taplite"])], errors="coerce"
            )
            valid = cube.ge(0) & taplite.ge(0) & cube.notna() & taplite.notna()
            x = cube[valid].to_numpy(dtype=float)
            y = taplite[valid].to_numpy(dtype=float)
            if len(x):
                all_values.extend((x, y))
            axis.scatter(
                x,
                y,
                s=8,
                alpha=0.16,
                color=PERIOD_COLORS[plotted_period],
                edgecolors="none",
                label=plotted_period,
                rasterized=True,
            )
        maximum = max((float(np.max(values)) for values in all_values), default=1.0)
        limit = maximum * 1.04 if maximum > 0 else 1.0
        axis.plot(
            [0, limit], [0, limit], color="#555555", linewidth=1.6,
            linestyle="--", label="45-degree line",
        )
        statistics = network_metrics[
            network_metrics["figure_group"].eq(group_name)
            & network_metrics["metric"].eq(metric)
            & network_metrics["period"].eq(normalized_period)
        ].iloc[0]
        slope = float(statistics["origin_fit_slope"])
        if np.isfinite(slope):
            axis.plot(
                [0, limit], [0, slope * limit], color="#c33c8c",
                linewidth=2.0, label=f"Origin fit: y = {slope:.3f}x",
            )
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
        axis.yaxis.set_major_formatter(FuncFormatter(_compact_number))
        axis.set_title(str(specifications["title"]), fontsize=14, pad=9)
        cube_label = str(
            specifications.get("cube_label", f"Cube {specifications['title']}")
        )
        taplite_label = str(
            specifications.get(
                "taplite_label", f"TAPlite {specifications['title']}"
            )
        )
        unit = str(specifications["unit"])
        axis.set_xlabel(f"{cube_label} ({unit})", fontsize=MINIMUM_FONT_SIZE)
        axis.set_ylabel(f"{taplite_label} ({unit})", fontsize=MINIMUM_FONT_SIZE)
        r_value = statistics["pearson_r"]
        r_text = f"{float(r_value):.3f}" if pd.notna(r_value) else "NA"
        equation = f"y = {slope:.3f}x" if np.isfinite(slope) else "fit unavailable"
        axis.text(
            0.04, 0.96,
            f"n = {int(statistics['paired_point_count']):,}\n"
            f"Origin fit: {equation}\nPearson r = {r_text}",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=MINIMUM_FONT_SIZE,
            bbox={
                "boxstyle": "round,pad=0.35", "facecolor": "white",
                "edgecolor": "#cccccc", "alpha": 0.92,
            },
        )
        axis.legend(loc="lower right", frameon=False, fontsize=MINIMUM_FONT_SIZE)
    title = str(group["title"])
    if normalized_period != "ALL":
        title = f"{normalized_period}: {title}"
    figure.suptitle(title, fontsize=17, y=0.98)
    if normalized_period == "ALL":
        footer = (
            "Each point is one network link-period; colors identify AM, MD, and PM. "
            "Dashed gray is parity and magenta is the origin-constrained y = ax fit."
        )
    else:
        footer = (
            f"Each point is one network link in {normalized_period}. Dashed gray is "
            "parity and magenta is the origin-constrained y = ax fit."
        )
    if group_name == "vmt-vht-doc":
        footer += (
            " Cube VMT = volume x miles; Cube VHT = volume x miles / Cube speed. "
            "TAPlite VMT/VHT use TAPlite volume and travel time calculated manually."
        )
    figure.text(
        0.5, 0.018, footer, ha="center", va="bottom",
        fontsize=MINIMUM_FONT_SIZE, color="#555555",
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.15, wspace=0.24)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=figure_dpi, pil_kwargs={"compress_level": 3})
    plt.close(figure)


def create_scatter_figures(
    link_comparison: pd.DataFrame,
    corridor_comparison: pd.DataFrame,
    scatter_metrics: pd.DataFrame,
    output_dir: Path,
    *,
    network_scatter_metrics: pd.DataFrame | None = None,
    figure_dpi: int = 120,
    replace: bool = False,
) -> pd.DataFrame:
    """Create one three-panel figure per period and comparison scope."""
    stage_root = output_dir / "08-volume-vmt-vht-comparison"
    figure_root = stage_root / "figures"
    link_root = figure_root / "link-level"
    corridor_root = figure_root / "corridor-level"
    if stage_root.exists() and not replace:
        raise FileExistsError(
            f"Volume/VMT/VHT comparison output already exists: {stage_root}"
        )
    link_root.mkdir(parents=True, exist_ok=replace)
    corridor_root.mkdir(parents=True, exist_ok=replace)
    network_root = figure_root / "all-network"
    network_root.mkdir(parents=True, exist_ok=replace)
    network_period_root = network_root / "by-period"
    rows: List[Dict[str, object]] = []
    for scope, frame, destination_root in (
        ("all_links", link_comparison, link_root),
        ("corridors", corridor_comparison, corridor_root),
    ):
        for period in sorted(frame["period"].dropna().astype(str).unique()):
            destination = destination_root / f"{period}.png"
            _plot_period_scatter(
                frame[frame["period"].eq(period)],
                scatter_metrics[
                    scatter_metrics["scope"].eq(scope)
                    & scatter_metrics["period"].eq(period)
                ],
                destination,
                period=period,
                scope=scope,
                figure_dpi=figure_dpi,
            )
            rows.append(
                {
                    "scope": scope,
                    "period": period,
                    "figure": destination.relative_to(output_dir).as_posix(),
                }
            )
    resolved_network_metrics = (
        network_scatter_metrics
        if network_scatter_metrics is not None
        else build_network_scatter_metrics(link_comparison)
    )
    for group_name in NETWORK_METRIC_GROUPS:
        destination = network_root / f"cube-vs-taplite-{group_name}.png"
        _plot_network_scatter_group(
            link_comparison,
            resolved_network_metrics,
            destination,
            group_name=group_name,
            period="ALL",
            figure_dpi=figure_dpi,
        )
        rows.append(
            {
                "scope": "all_network_links",
                "period": "ALL",
                "figure": destination.relative_to(output_dir).as_posix(),
            }
        )
    available_periods = [
        period
        for period in PERIOD_COLORS
        if link_comparison["period"].astype(str).str.upper().eq(period).any()
    ]
    for period in available_periods:
        for group_name in NETWORK_METRIC_GROUPS:
            destination = (
                network_period_root
                / period
                / f"cube-vs-taplite-{group_name}.png"
            )
            _plot_network_scatter_group(
                link_comparison,
                resolved_network_metrics,
                destination,
                group_name=group_name,
                period=period,
                figure_dpi=figure_dpi,
            )
            rows.append(
                {
                    "scope": "all_network_links",
                    "period": period,
                    "figure": destination.relative_to(output_dir).as_posix(),
                }
            )
    manifest = pd.DataFrame(rows)
    lines = [
        "# Cube versus TAPlite volume, VMT, and VHT scatters",
        "",
        "The period figures contain Volume, VMT, and VHT panels with Cube on the "
        "x-axis, TAPlite on the y-axis, and a 1:1 reference line. All network "
        "links are retained in the link-level plots. Corridor measures sum each "
        "period's de-duplicated mapped GMNS links. Both sides' VMT and VHT are "
        "calculated manually; the erroneous recorded TAPlite VMT/VHT fields are "
        "retained only as audit columns in the link data. The all-network "
        "Volume/Travel Time/Speed and VMT/VHT/D-C groups are produced once with "
        "AM, MD, and PM combined and once separately for every available period. "
        "All include parity and origin-constrained fit lines.",
        "",
        "| Scope | Period | Figure |",
        "|---|---|---|",
    ]
    for row in manifest.itertuples(index=False):
        relative = Path(row.figure).relative_to(
            "08-volume-vmt-vht-comparison"
        ).as_posix()
        lines.append(f"| {row.scope} | {row.period} | [scatter]({relative}) |")
    (stage_root / "SCATTERS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return manifest


def write_scatter_data(
    output_dir: Path,
    link_comparison: pd.DataFrame,
    corridor_comparison: pd.DataFrame,
    scatter_metrics: pd.DataFrame,
    scatter_manifest: pd.DataFrame,
    network_scatter_metrics: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    """Write auditable scatter data products and return their relative paths."""
    outputs = {
        "08-volume-vmt-vht-comparison/data/link_period_comparison.csv": (
            link_comparison
        ),
        "08-volume-vmt-vht-comparison/data/corridor_period_comparison.csv": (
            corridor_comparison
        ),
        "08-volume-vmt-vht-comparison/data/scatter_metrics.csv": scatter_metrics,
        "08-volume-vmt-vht-comparison/scatter_manifest.csv": scatter_manifest,
    }
    if network_scatter_metrics is not None:
        outputs[
            "08-volume-vmt-vht-comparison/data/all_network_scatter_metrics.csv"
        ] = network_scatter_metrics
    for relative_path, frame in outputs.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, float_format="%.6f")
    return outputs
