"""Static corridor profile and heatmap figures."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OBSERVED_COLOR = "#1f77b4"
MODEL_COLOR = "#e87500"
CUBE_COLOR = "#2a9d55"
THRESHOLD_COLOR = "#555555"
MISSING_COLOR = "#d9d9d9"
PERIOD_COLORS = {
    "AM": "#4c78a8",
    "MD": "#f2a541",
    "PM": "#8f63b8",
}
MINIMUM_FONT_SIZE = 11
PRIMARY_LINE_WIDTH = 2.4
SECONDARY_LINE_WIDTH = 1.6

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


def select_representative_tmcs(
    profiles: pd.DataFrame, *, count: int = 5
) -> pd.DataFrame:
    """Select spatially distributed TMCs with diverse speed/error behavior."""
    if count < 1:
        raise ValueError("Selected TMC count must be positive.")

    working = profiles.copy()
    for column in (
        "observed_tmc_speed_mph",
        "model_tmc_speed_mph",
        "cube_qvdf_tmc_speed_mph",
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
    availability = (
        working.groupby(["corridor", "tmc_code"], as_index=False)
        .agg(
            direction=("direction", "first"),
            road_order=("road_order", "first"),
            matched_interval_count=(
                "model_tmc_speed_mph",
                lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum()),
            ),
            observed_speed_mean_mph=("observed_tmc_speed_mph", "mean"),
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
                "cube_absolute_speed_error_mph",
                "mean",
            ),
        )
        .sort_values(["corridor", "road_order", "tmc_code"], kind="stable")
    )
    availability = availability[availability["matched_interval_count"].gt(0)]
    rows: List[Dict[str, object]] = []
    for corridor, group in availability.groupby("corridor", sort=True):
        ordered = group.reset_index(drop=True)
        number_to_select = min(count, len(ordered))
        if number_to_select == 0:
            continue

        if number_to_select == len(ordered):
            indices = list(range(len(ordered)))
            labels = (
                ["only"]
                if number_to_select == 1
                else [
                    f"all_{position + 1}_of_{number_to_select}"
                    for position in range(number_to_select)
                ]
            )
        elif number_to_select == 1:
            indices = [0]
            labels = ["only"]
        else:
            feature_columns = [
                "observed_speed_mean_mph",
                "observed_speed_range_mph",
                "observed_speed_std_mph",
                "model_speed_mean_mph",
                "model_speed_range_mph",
                "model_speed_std_mph",
                "cube_speed_mean_mph",
                "cube_speed_range_mph",
                "cube_speed_std_mph",
                "mean_absolute_error_mph",
                "cube_mean_absolute_error_mph",
            ]
            features = ordered[feature_columns].apply(pd.to_numeric, errors="coerce")
            features.insert(
                0,
                "corridor_position",
                np.linspace(0.0, 1.0, len(ordered)),
            )
            for column in features:
                median = features[column].median()
                features[column] = features[column].fillna(
                    float(median) if pd.notna(median) else 0.0
                )
            matrix = features.to_numpy(dtype=float)
            scale = np.nanstd(matrix, axis=0)
            scale[~np.isfinite(scale) | np.isclose(scale, 0.0)] = 1.0
            matrix = (matrix - np.nanmean(matrix, axis=0)) / scale

            selected_indices = [0, len(ordered) - 1]
            while len(selected_indices) < number_to_select:
                distances = np.linalg.norm(
                    matrix[:, np.newaxis, :]
                    - matrix[np.asarray(selected_indices), :][np.newaxis, :, :],
                    axis=2,
                ).min(axis=1)
                distances[np.asarray(selected_indices)] = -np.inf
                selected_indices.append(int(np.argmax(distances)))
            indices = sorted(selected_indices)
            if len(indices) == 3:
                labels = ["first", "middle", "last"]
            else:
                labels = []
                for position, index in enumerate(indices, start=1):
                    if index == 0:
                        labels.append("corridor_start")
                    elif index == len(ordered) - 1:
                        labels.append("corridor_end")
                    else:
                        labels.append(
                            f"behavior_diverse_{position}_of_{len(indices)}"
                        )
        for label, index in zip(labels, indices):
            selected = ordered.iloc[index]
            rows.append(
                {
                    "corridor": corridor,
                    "tmc_code": selected["tmc_code"],
                    "direction": selected["direction"],
                    "road_order": selected["road_order"],
                    "selection_position": label,
                    "matched_interval_count": selected["matched_interval_count"],
                    "observed_speed_range_mph": selected[
                        "observed_speed_range_mph"
                    ],
                    "model_speed_range_mph": selected["model_speed_range_mph"],
                    "mean_absolute_error_mph": selected[
                        "mean_absolute_error_mph"
                    ],
                    "cube_mean_absolute_error_mph": selected[
                        "cube_mean_absolute_error_mph"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _configured_minutes(settings: Mapping[str, object]) -> List[int]:
    interval = int(settings["comparison_interval_minutes"])
    return [
        minute
        for period in settings["periods"].values()
        for minute in range(int(period["start_min"]), int(period["end_min"]), interval)
    ]


def _clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _plot_selected_profiles(
    corridor: str,
    corridor_profiles: pd.DataFrame,
    selected: pd.DataFrame,
    destination: Path,
    *,
    settings: Mapping[str, object],
) -> None:
    count = len(selected)
    if count == 0:
        return
    minutes = _configured_minutes(settings)
    figure = plt.figure(figsize=(23, 3.7 * count + 2.0))
    grid = figure.add_gridspec(
        count,
        2,
        width_ratios=(4.7, 2.3),
        hspace=0.30,
        wspace=0.03,
    )
    axes = []

    for row_index, selection in enumerate(selected.itertuples(index=False)):
        axis = figure.add_subplot(
            grid[row_index, 0],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        )
        axes.append(axis)
        legend_axis = figure.add_subplot(grid[row_index, 1])
        legend_axis.axis("off")
        frame = corridor_profiles[
            corridor_profiles["tmc_code"].eq(selection.tmc_code)
        ].set_index("t_min").reindex(minutes)
        for period_name, period_settings in settings["periods"].items():
            period_label = period_name.upper()
            axis.axvspan(
                int(period_settings["start_min"]),
                int(period_settings["end_min"]),
                color=PERIOD_COLORS[period_label],
                alpha=0.035,
                linewidth=0,
            )
        axis.plot(
            minutes,
            frame["observed_tmc_speed_mph"],
            color=OBSERVED_COLOR,
            linewidth=PRIMARY_LINE_WIDTH,
            label="CBI observed",
        )
        axis.plot(
            minutes,
            frame["model_tmc_speed_mph"],
            color=MODEL_COLOR,
            linewidth=PRIMARY_LINE_WIDTH,
            linestyle="--",
            label="TAPlite model",
        )
        axis.plot(
            minutes,
            frame["cube_qvdf_tmc_speed_mph"],
            color=CUBE_COLOR,
            linewidth=PRIMARY_LINE_WIDTH,
            linestyle="-.",
            label="Cube-volume QVDF",
        )
        axis.plot(
            minutes,
            frame["cbi_tmc_congestion_threshold_mph"],
            color=THRESHOLD_COLOR,
            linewidth=SECONDARY_LINE_WIDTH,
            linestyle=":",
            label="CBI threshold",
        )
        axis.axvline(540, color="#aaaaaa", linewidth=1.0)
        axis.axvline(900, color="#aaaaaa", linewidth=1.0)
        order_text = (
            f"{float(selection.road_order):g}"
            if pd.notna(selection.road_order)
            else "unknown"
        )
        axis.set_title(
            f"{str(selection.selection_position).replace('_', ' ').title()} TMC: "
            f"{selection.tmc_code}  "
            f"(road order {order_text})",
            loc="left",
            fontsize=12,
        )
        axis.set_ylabel("Speed (mph)")
        axis.set_ylim(0, 82)
        axis.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.8)

        period_rows = (
            corridor_profiles[corridor_profiles["tmc_code"].eq(selection.tmc_code)]
            .sort_values("t_min")
            .groupby("period", as_index=False)
            .first()
            .set_index("period")
        )
        period_handles = []
        for period_name in settings["periods"]:
            period_label = period_name.upper()
            if period_label not in period_rows.index:
                continue
            values = period_rows.loc[period_label]
            volume = values.get("taplite_period_volume")
            doc = values.get("taplite_period_doc")
            p_hours = values.get("taplite_period_p_hours")
            cube_volume = values.get("cube_period_volume")
            cube_doc = values.get("cube_period_doc")
            cube_p_hours = values.get("cube_period_p_hours")
            link_count = values.get("gmns_link_count")
            volume_text = f"{float(volume):,.0f}" if pd.notna(volume) else "NA"
            doc_text = f"{float(doc):.2f}" if pd.notna(doc) else "NA"
            p_text = f"{float(p_hours):.2f}" if pd.notna(p_hours) else "NA"
            cube_volume_text = (
                f"{float(cube_volume):,.0f}" if pd.notna(cube_volume) else "NA"
            )
            cube_doc_text = (
                f"{float(cube_doc):.2f}" if pd.notna(cube_doc) else "NA"
            )
            cube_p_text = (
                f"{float(cube_p_hours):.2f}" if pd.notna(cube_p_hours) else "NA"
            )
            link_text = f"{int(link_count)}" if pd.notna(link_count) else "0"
            metric_label = (
                f"{period_label}   TAPlite: Vol {volume_text}  D/C {doc_text}  "
                f"P {p_text} h\n"
                f"       Cube: Vol {cube_volume_text}  D/C {cube_doc_text}  "
                f"P {cube_p_text} h   Links {link_text}"
            )
            period_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=PERIOD_COLORS[period_label],
                    marker="s",
                    linestyle="none",
                    markersize=7,
                    label=metric_label,
                )
            )
        legend_axis.legend(
            handles=period_handles,
            title="Period QVDF inputs\n(length-weighted TMC path)",
            loc="center left",
            frameon=False,
            fontsize=MINIMUM_FONT_SIZE,
            title_fontsize=MINIMUM_FONT_SIZE,
            handlelength=1.0,
            labelspacing=1.0,
        )

    tick_minutes = list(range(minutes[0], minutes[-1] + 1, 60))
    axes[-1].set_xticks(tick_minutes)
    axes[-1].set_xticklabels([_clock(value) for value in tick_minutes])
    axes[-1].set_xlabel("Time of day")
    legend_handles = [
        Line2D(
            [0], [0], color=OBSERVED_COLOR,
            linewidth=PRIMARY_LINE_WIDTH, label="CBI observed"
        ),
        Line2D(
            [0], [0], color=MODEL_COLOR, linewidth=PRIMARY_LINE_WIDTH,
            linestyle="--", label="TAPlite model"
        ),
        Line2D(
            [0],
            [0],
            color=CUBE_COLOR,
            linewidth=PRIMARY_LINE_WIDTH,
            linestyle="-.",
            label="Cube-volume QVDF",
        ),
        Line2D(
            [0],
            [0],
            color=THRESHOLD_COLOR,
            linewidth=SECONDARY_LINE_WIDTH,
            linestyle=":",
            label="CBI threshold",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"{corridor}: selected TMC daily speed profiles", fontsize=14, y=0.99
    )
    figure.text(
        0.5,
        0.012,
        "TAPlite speed: spd_mph_HH:MM; Cube speed: Link_QueueVDF reconstructed "
        "from I4AMVOL/I4MDVOL/I4PMVOL. Path values are GMNS-link-length weighted.",
        ha="center",
        va="bottom",
        fontsize=MINIMUM_FONT_SIZE,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.11)
    figure.savefig(
        destination,
        dpi=int(settings["figure_dpi"]),
        pil_kwargs={"compress_level": 3},
    )
    plt.close(figure)


def _ordered_heatmap_data(
    corridor_profiles: pd.DataFrame,
    settings: Mapping[str, object],
) -> Tuple[pd.DataFrame, List[int], np.ndarray, np.ndarray, np.ndarray]:
    minutes = _configured_minutes(settings)
    reference = (
        corridor_profiles[["tmc_code", "road_order", "direction"]]
        .drop_duplicates("tmc_code")
        .sort_values(["road_order", "tmc_code"], kind="stable")
    )
    tmcs = reference["tmc_code"].tolist()

    def pivot(column: str) -> np.ndarray:
        return (
            corridor_profiles.pivot_table(
                index="tmc_code",
                columns="t_min",
                values=column,
                aggfunc="first",
            )
            .reindex(index=tmcs, columns=minutes)
            .to_numpy(dtype=float)
        )

    return (
        reference,
        minutes,
        pivot("observed_tmc_speed_mph"),
        pivot("model_tmc_speed_mph"),
        pivot("cube_qvdf_tmc_speed_mph"),
    )


def _heatmap_y_labels(reference: pd.DataFrame) -> List[str]:
    return [
        f"{row.tmc_code}  [{float(row.road_order):g}]"
        if pd.notna(row.road_order)
        else str(row.tmc_code)
        for row in reference.itertuples(index=False)
    ]


def _plot_heatmaps(
    corridor: str,
    corridor_profiles: pd.DataFrame,
    destination: Path,
    *,
    settings: Mapping[str, object],
) -> None:
    reference, minutes, observed, modeled, _cube = _ordered_heatmap_data(
        corridor_profiles, settings
    )
    tmcs = reference["tmc_code"].tolist()
    absolute_error = np.abs(modeled - observed)

    height = min(28.0, max(7.0, 0.30 * len(tmcs) + 3.0))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(28, height),
        sharex=True,
        sharey=True,
    )
    speed_color_map = plt.get_cmap("RdYlGn").copy()
    speed_color_map.set_bad(MISSING_COLOR)
    error_color_map = plt.get_cmap("RdYlGn_r").copy()
    error_color_map.set_bad(MISSING_COLOR)
    speed_vmin = float(settings["heatmap_speed_min_mph"])
    speed_vmax = float(settings["heatmap_speed_max_mph"])
    error_vmax = float(settings["heatmap_error_max_mph"])
    images = []
    for axis, values, title, color_map, vmin, vmax in zip(
        axes,
        (observed, modeled, absolute_error),
        (
            "CBI observed speed",
            "TAPlite speed",
            "TAPlite vs observed absolute error",
        ),
        (speed_color_map, speed_color_map, error_color_map),
        (speed_vmin, speed_vmin, 0.0),
        (speed_vmax, speed_vmax, error_vmax),
    ):
        image = axis.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            interpolation="nearest",
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        axis.set_title(title, fontsize=12)
        tick_positions = list(range(0, len(minutes), 4))
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([_clock(minutes[index]) for index in tick_positions], rotation=45)
        axis.set_xlabel("Time of day")
        for boundary in (540, 900):
            if boundary in minutes:
                axis.axvline(minutes.index(boundary) - 0.5, color="white", linewidth=1.2)

    y_labels = _heatmap_y_labels(reference)
    axes[0].set_yticks(range(len(tmcs)))
    axes[0].set_yticklabels(y_labels, fontsize=MINIMUM_FONT_SIZE)
    axes[0].set_ylabel("TMC in corridor order  [road order]")
    direction = ", ".join(sorted(reference["direction"].dropna().astype(str).unique()))
    figure.suptitle(
        f"{corridor}: observed and TAPlite 15-minute TMC heatmaps ({direction})",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.11, right=0.89, top=0.92, bottom=0.11, wspace=0.08
    )
    speed_color_axis = figure.add_axes((0.91, 0.25, 0.010, 0.50))
    speed_color_bar = figure.colorbar(images[0], cax=speed_color_axis)
    speed_color_bar.set_label("Speed (mph)")
    speed_color_bar.ax.yaxis.set_label_position("left")
    speed_color_bar.ax.yaxis.tick_left()
    error_color_axis = figure.add_axes((0.965, 0.25, 0.010, 0.50))
    error_color_bar = figure.colorbar(
        images[2], cax=error_color_axis, extend="max"
    )
    error_color_bar.set_label("Absolute error (mph)")
    figure.legend(
        handles=[Patch(facecolor=MISSING_COLOR, edgecolor="#999999", label="Missing / unmapped")],
        loc="lower center",
        frameon=False,
    )
    figure.savefig(
        destination,
        dpi=int(settings["figure_dpi"]),
        pil_kwargs={"compress_level": 3},
    )
    plt.close(figure)


def _plot_absolute_error_heatmap(
    corridor: str,
    corridor_profiles: pd.DataFrame,
    destination: Path,
    *,
    settings: Mapping[str, object],
    comparison: str,
) -> None:
    reference, minutes, observed, modeled, cube = _ordered_heatmap_data(
        corridor_profiles, settings
    )
    comparisons = {
        "taplite-vs-observed": (
            np.abs(modeled - observed),
            "TAPlite assigned-volume versus CBI observed",
        ),
        "cube-vs-observed": (
            np.abs(cube - observed),
            "Cube-volume QVDF versus CBI observed",
        ),
        "taplite-vs-cube": (
            np.abs(modeled - cube),
            "TAPlite assigned-volume versus Cube-volume QVDF",
        ),
    }
    if comparison not in comparisons:
        raise ValueError(f"Unknown error-heatmap comparison: {comparison}")
    errors, comparison_label = comparisons[comparison]
    tmcs = reference["tmc_code"].tolist()
    height = min(28.0, max(7.0, 0.30 * len(tmcs) + 3.0))
    figure, axis = plt.subplots(figsize=(14.5, height))
    color_map = plt.get_cmap("RdYlGn_r").copy()
    color_map.set_bad(MISSING_COLOR)
    vmax = float(settings["heatmap_error_max_mph"])
    image = axis.imshow(
        np.ma.masked_invalid(errors),
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=0.0,
        vmax=vmax,
    )
    tick_positions = list(range(0, len(minutes), 4))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [_clock(minutes[index]) for index in tick_positions], rotation=45
    )
    for boundary in (540, 900):
        if boundary in minutes:
            axis.axvline(
                minutes.index(boundary) - 0.5,
                color="white",
                linewidth=1.2,
            )
    axis.set_yticks(range(len(tmcs)))
    axis.set_yticklabels(
        _heatmap_y_labels(reference),
        fontsize=MINIMUM_FONT_SIZE,
    )
    axis.set_xlabel("Time of day")
    axis.set_ylabel("TMC in corridor order  [road order]")
    direction = ", ".join(
        sorted(reference["direction"].dropna().astype(str).unique())
    )
    axis.set_title(
        f"{corridor}: absolute {comparison_label} speed error ({direction})",
        fontsize=13,
    )
    figure.subplots_adjust(left=0.24, right=0.86, top=0.93, bottom=0.12)
    color_axis = figure.add_axes((0.89, 0.25, 0.018, 0.50))
    color_bar = figure.colorbar(image, cax=color_axis, extend="max")
    color_bar.set_label("Absolute speed error (mph)")
    figure.legend(
        handles=[
            Patch(
                facecolor=MISSING_COLOR,
                edgecolor="#999999",
                label="Missing / unmapped",
            )
        ],
        loc="lower center",
        frameon=False,
    )
    figure.savefig(
        destination,
        dpi=int(settings["figure_dpi"]),
        pil_kwargs={"compress_level": 3},
    )
    plt.close(figure)


def _render_corridor_figure_task(
    task: Tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        Path,
        Path,
        Path,
        Path,
        Path,
        Dict[str, object],
    ],
) -> Dict[str, object]:
    (
        corridor,
        corridor_profiles,
        selected,
        profile_path,
        speed_heatmap_path,
        taplite_observed_error_path,
        cube_observed_error_path,
        taplite_cube_error_path,
        settings,
    ) = task
    _plot_selected_profiles(
        corridor,
        corridor_profiles,
        selected,
        profile_path,
        settings=settings,
    )
    _plot_heatmaps(
        corridor,
        corridor_profiles,
        speed_heatmap_path,
        settings=settings,
    )
    _plot_absolute_error_heatmap(
        corridor,
        corridor_profiles,
        taplite_observed_error_path,
        settings=settings,
        comparison="taplite-vs-observed",
    )
    _plot_absolute_error_heatmap(
        corridor,
        corridor_profiles,
        cube_observed_error_path,
        settings=settings,
        comparison="cube-vs-observed",
    )
    _plot_absolute_error_heatmap(
        corridor,
        corridor_profiles,
        taplite_cube_error_path,
        settings=settings,
        comparison="taplite-vs-cube",
    )
    return {
        "corridor": corridor,
        "tmc_count": int(corridor_profiles["tmc_code"].nunique()),
        "selected_tmc_count": int(selected["tmc_code"].nunique()),
        "selected_tmc_codes": ";".join(selected["tmc_code"].astype(str)),
        "selected_profile_figure": profile_path,
        "speed_heatmap_figure": speed_heatmap_path,
        "taplite_vs_observed_error_heatmap_figure": taplite_observed_error_path,
        "cube_vs_observed_error_heatmap_figure": cube_observed_error_path,
        "taplite_vs_cube_error_heatmap_figure": taplite_cube_error_path,
    }


def create_corridor_figures(
    profiles: pd.DataFrame,
    output_dir: Path,
    *,
    settings: Mapping[str, object],
    workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create profiles, observed/model/error heatmaps, and error archives."""
    figure_root = output_dir / "06-figures"
    figure_root.mkdir(parents=True, exist_ok=False)
    profile_root = figure_root / "selected-tmc-profiles"
    speed_heatmap_root = figure_root / "speed-heatmaps"
    error_heatmap_root = figure_root / "absolute-error-heatmaps"
    taplite_observed_error_root = error_heatmap_root / "taplite-vs-observed"
    cube_observed_error_root = error_heatmap_root / "cube-vs-observed"
    taplite_cube_error_root = error_heatmap_root / "taplite-vs-cube"
    for directory in (profile_root, speed_heatmap_root, error_heatmap_root):
        directory.mkdir(parents=False, exist_ok=False)
    for directory in (
        taplite_observed_error_root,
        cube_observed_error_root,
        taplite_cube_error_root,
    ):
        directory.mkdir(parents=False, exist_ok=False)
    selected_reference = select_representative_tmcs(
        profiles, count=int(settings["selected_tmc_count_per_corridor"])
    )
    selected_profiles = profiles.merge(
        selected_reference[
            ["corridor", "tmc_code", "selection_position"]
        ],
        on=["corridor", "tmc_code"],
        how="inner",
        validate="many_to_one",
    )
    selected_period_metrics = (
        selected_profiles.groupby(
            [
                "corridor",
                "tmc_code",
                "selection_position",
                "period",
            ],
            as_index=False,
        )
        .agg(
            road_order=("road_order", "first"),
            gmns_link_count=("gmns_link_count", "first"),
            gmns_link_ids=("gmns_link_ids", "first"),
            taplite_period_volume=("taplite_period_volume", "first"),
            taplite_period_doc=("taplite_period_doc", "first"),
            taplite_period_p_hours=("taplite_period_p_hours", "first"),
            cube_period_volume=("cube_period_volume", "first"),
            cube_period_doc=("cube_period_doc", "first"),
            cube_period_p_hours=("cube_period_p_hours", "first"),
            model_speed_min_mph=("model_tmc_speed_mph", "min"),
            model_speed_max_mph=("model_tmc_speed_mph", "max"),
            model_speed_unique_value_count=("model_tmc_speed_mph", "nunique"),
            cube_speed_min_mph=("cube_qvdf_tmc_speed_mph", "min"),
            cube_speed_max_mph=("cube_qvdf_tmc_speed_mph", "max"),
            cube_speed_unique_value_count=(
                "cube_qvdf_tmc_speed_mph",
                "nunique",
            ),
        )
        .sort_values(["corridor", "road_order", "period"])
    )
    if workers < 1:
        raise ValueError("Figure worker count must be positive.")
    tasks = []
    for corridor, corridor_profiles in profiles.groupby("corridor", sort=True):
        selected = selected_reference[selected_reference["corridor"].eq(corridor)]
        profile_path = profile_root / f"{corridor}.png"
        speed_heatmap_path = speed_heatmap_root / f"{corridor}.png"
        taplite_observed_error_path = (
            taplite_observed_error_root / f"{corridor}.png"
        )
        cube_observed_error_path = cube_observed_error_root / f"{corridor}.png"
        taplite_cube_error_path = taplite_cube_error_root / f"{corridor}.png"
        tasks.append(
            (
                corridor,
                corridor_profiles.copy(),
                selected.copy(),
                profile_path,
                speed_heatmap_path,
                taplite_observed_error_path,
                cube_observed_error_path,
                taplite_cube_error_path,
                dict(settings),
            )
        )
    if workers == 1 or len(tasks) <= 1:
        rendered = [_render_corridor_figure_task(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
                rendered = list(executor.map(_render_corridor_figure_task, tasks))
        except (OSError, PermissionError):
            rendered = [_render_corridor_figure_task(task) for task in tasks]
    manifest_rows: List[Dict[str, object]] = []
    for row in rendered:
        manifest_rows.append(
            {
                **row,
                "selected_profile_figure": Path(row["selected_profile_figure"])
                .relative_to(output_dir)
                .as_posix(),
                "speed_heatmap_figure": Path(row["speed_heatmap_figure"])
                .relative_to(output_dir)
                .as_posix(),
                "taplite_vs_observed_error_heatmap_figure": Path(
                    row["taplite_vs_observed_error_heatmap_figure"]
                )
                .relative_to(output_dir)
                .as_posix(),
                "cube_vs_observed_error_heatmap_figure": Path(
                    row["cube_vs_observed_error_heatmap_figure"]
                )
                .relative_to(output_dir)
                .as_posix(),
                "taplite_vs_cube_error_heatmap_figure": Path(
                    row["taplite_vs_cube_error_heatmap_figure"]
                )
                .relative_to(output_dir)
                .as_posix(),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    lines = [
        "# Corridor TMC profile and heatmap figures",
        "",
        "Each corridor includes five behavior-diverse comparable-TMC profiles "
        "(or every comparable TMC when fewer than five exist), complete "
        "three-panel observed/TAPlite/absolute-error heatmaps, and archived "
        "pairwise absolute-error heatmaps. Heatmap rows include all CBI TMCs "
        "in road order; gray cells identify missing "
        "or unmapped values.",
        "",
        "| Corridor | Selected TMC profiles | Speed heatmaps | TAPlite vs observed | Cube vs observed | TAPlite vs Cube |",
        "|---|---|---|---|---|---|",
    ]
    for row in manifest.itertuples(index=False):
        profile_link = Path(row.selected_profile_figure).relative_to(
            "06-figures"
        ).as_posix()
        speed_link = Path(row.speed_heatmap_figure).relative_to(
            "06-figures"
        ).as_posix()
        taplite_observed_error_link = Path(
            row.taplite_vs_observed_error_heatmap_figure
        ).relative_to(
            "06-figures"
        ).as_posix()
        cube_observed_error_link = Path(
            row.cube_vs_observed_error_heatmap_figure
        ).relative_to("06-figures").as_posix()
        taplite_cube_error_link = Path(
            row.taplite_vs_cube_error_heatmap_figure
        ).relative_to("06-figures").as_posix()
        lines.append(
            f"| {row.corridor} | [profiles]({profile_link}) | "
            f"[speed]({speed_link}) | [error]({taplite_observed_error_link}) | "
            f"[error]({cube_observed_error_link}) | "
            f"[error]({taplite_cube_error_link}) |"
        )
    (figure_root / "FIGURES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return manifest, selected_profiles, selected_period_metrics
