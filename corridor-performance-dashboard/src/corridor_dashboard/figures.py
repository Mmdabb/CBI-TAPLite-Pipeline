from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import calibration_lookup
from .config import CorridorSpec, PipelineSettings
from .fundamental_diagram import s3_flow, s3_speed
from .output_contract import step_dir
from .reconstruction import (
    detection_smoothed_speed,
    predict_duration_hours,
    predict_minimum_speed,
    predicted_bounds_about_t2,
    reconstruct_episode_speed,
)


OBSERVED_COLOR = "#1f77b4"
CBI_RECONSTRUCTION_COLOR = "#6f4e9c"
THRESHOLD_COLOR = "#555555"
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


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _paginate(
    figure_dir: Path,
    period: str,
    links: list[int],
    rows: dict[int, pd.Series],
    draw,
    prefix: str,
    title: str,
) -> int:
    products = 0
    per_page = 20
    for page_start in range(0, len(links), per_page):
        chunk = links[page_start : page_start + per_page]
        columns = min(4, len(chunk))
        rows_count = math.ceil(len(chunk) / columns)
        figure, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(3.7 * columns, 2.5 * rows_count),
            squeeze=False,
        )
        flat = list(axes.flat)
        for index, link_id in enumerate(chunk):
            draw(flat[index], link_id, rows[link_id], index == 0)
        for axis in flat[len(chunk) :]:
            axis.axis("off")
        figure.suptitle(title, fontsize=11)
        suffix = (
            ""
            if len(links) <= per_page
            else f"_p{page_start // per_page + 1}"
        )
        _save(figure, figure_dir / f"{prefix}_{period}{suffix}.png")
        products += 1
    return products


def _validation_handoff(
    handoff: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    start_minute: int = 6 * 60,
    end_minute: int = 21 * 60,
) -> pd.DataFrame:
    """Prepare the observed-versus-model validation figure input."""

    data = handoff.copy()
    for column in (
        "link_id",
        "t_min",
        "speed_raw",
        "speed_qvdf_model",
        "congestion_threshold_mph",
    ):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        data["t_min"].ge(start_minute) & data["t_min"].lt(end_minute)
    ].copy()
    data["cutoff"] = data["congestion_threshold_mph"]
    observed_free_flow = {}
    if "observed_speed_p99_mph" in fd:
        observed_free_flow = dict(
            zip(
                pd.to_numeric(fd["link_id"], errors="coerce"),
                pd.to_numeric(
                    fd["observed_speed_p99_mph"], errors="coerce"
                ),
            )
        )
    data["obs_ff"] = data["link_id"].map(observed_free_flow)
    data["obs_ff"] = data["obs_ff"].fillna(
        data.groupby("link_id")["speed_raw"].transform("max")
    )
    return data


def _pick_validation_links(
    data: pd.DataFrame,
    n_links: int = 4,
) -> tuple[list[int], list[str]]:
    """Select a mix of congested and free-flow-recovery links."""

    grouped = data.groupby("link_id")
    minimum = grouped["speed_raw"].min().dropna()
    maximum = grouped["speed_raw"].max()
    free_flow = grouped["obs_ff"].median()
    cutoff = grouped["cutoff"].median()
    if minimum.empty:
        return [], []
    count = min(max(1, n_links), len(minimum))
    congested_count = min(max(1, count // 2), count)
    congested = [
        int(value)
        for value in minimum.sort_values().index[:congested_count]
    ]
    recovered = [
        int(link_id)
        for link_id in maximum.sort_values(ascending=False).index
        if int(link_id) not in congested
        and minimum.get(link_id, np.inf) < cutoff.get(link_id, -np.inf)
        and maximum.get(link_id, -np.inf)
        >= 0.9 * free_flow.get(link_id, np.inf)
    ]
    if len(recovered) < count - congested_count:
        recovered.extend(
            int(link_id)
            for link_id in minimum.sort_values(ascending=False).index
            if int(link_id) not in congested
            and int(link_id) not in recovered
        )
    recovered = recovered[: count - congested_count]
    links = congested + recovered
    tags = ["congested"] * len(congested) + [
        "free-flow recovery"
    ] * len(recovered)
    return links, tags


def _plot_sensor_vs_model_fullday(
    *,
    spec: CorridorSpec,
    figure_dir: Path,
    data: pd.DataFrame,
) -> int:
    links, tags = _pick_validation_links(data)
    if not links:
        return 0
    columns = min(2, len(links))
    rows = int(np.ceil(len(links) / columns))
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 4.0 * rows),
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    for index, link_id in enumerate(links):
        group = data[data["link_id"].eq(link_id)].sort_values("t_min")
        cutoff = float(group["cutoff"].median())
        free_flow = float(group["obs_ff"].median())
        tmc_code = str(group["tmc_code"].dropna().iloc[0])
        hour = group["t_min"] / 60.0
        axes[index].plot(
            hour,
            group["speed_raw"],
            color=OBSERVED_COLOR,
            lw=PRIMARY_LINE_WIDTH,
            marker=".",
            ms=3,
            label="CBI observed",
        )
        axes[index].plot(
            hour,
            group["speed_qvdf_model"],
            color=CBI_RECONSTRUCTION_COLOR,
            lw=PRIMARY_LINE_WIDTH,
            label="CBI QVDF reconstruction",
        )
        axes[index].axhline(
            free_flow,
            color=THRESHOLD_COLOR,
            ls="-.",
            lw=SECONDARY_LINE_WIDTH,
            label="free-flow (observed)",
        )
        axes[index].axhline(
            cutoff,
            color=THRESHOLD_COLOR,
            ls="--",
            lw=SECONDARY_LINE_WIDTH,
            label="congestion threshold",
        )
        axes[index].axhspan(cutoff, 80, color="tab:green", alpha=0.05)
        axes[index].set_xlim(6, 21)
        axes[index].set_ylim(0, 80)
        axes[index].set_title(
            f"TMC {tmc_code} ({tags[index]})",
            fontsize=12,
        )
        axes[index].set_xlabel("Hour of day", fontsize=MINIMUM_FONT_SIZE)
        axes[index].set_ylabel("Speed (mph)", fontsize=MINIMUM_FONT_SIZE)
        axes[index].tick_params(labelsize=MINIMUM_FONT_SIZE)
        if index == 0:
            axes[index].legend(fontsize=MINIMUM_FONT_SIZE, loc="lower right")
    for axis in axes[len(links) :]:
        axis.axis("off")
    figure.suptitle(
        f"{spec.key} — CBI observed vs CBI QVDF reconstruction, average weekday "
        "(06:00–21:00)",
        fontsize=14,
    )
    _save(figure, figure_dir / "sensor_vs_model_fullday.png")
    return 1


def _plot_validation_speed_heatmap(
    *,
    spec: CorridorSpec,
    figure_dir: Path,
    data: pd.DataFrame,
) -> int:
    if data.empty:
        return 0
    panel = data.copy()
    panel["hour"] = (panel["t_min"] // 60).astype(int)
    tmc_order = (
        panel.sort_values("link_id", kind="stable")["tmc_code"]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    hours = list(range(6, 21))

    def pivot(column: str) -> pd.DataFrame:
        values = panel.pivot_table(
            index="tmc_code",
            columns="hour",
            values=column,
            aggfunc="mean",
        )
        return values.reindex(index=tmc_order, columns=hours)

    observed = pivot("speed_raw")
    modeled = pivot("speed_qvdf_model")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(5.0, 0.28 * len(tmc_order) + 2.0)),
    )
    for axis, (matrix, title) in zip(
        axes,
        (
            (observed, "Observed average-weekday speed"),
            (modeled, "CBI QVDF reconstructed speed"),
        ),
    ):
        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=70,
            extent=[6, 21, len(tmc_order), 0],
        )
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("Hour of day", fontsize=MINIMUM_FONT_SIZE)
        axis.set_ylabel("TMC (corridor order)", fontsize=MINIMUM_FONT_SIZE)
        axis.set_yticks(np.arange(len(tmc_order)) + 0.5)
        axis.set_yticklabels(tmc_order, fontsize=MINIMUM_FONT_SIZE)
        figure.colorbar(
            image,
            ax=axis,
            fraction=0.03,
            pad=0.02,
            label="mph",
        )
    figure.suptitle(
        f"{spec.key} — TMC × hour speed: observed vs CBI reconstruction",
        fontsize=14,
    )
    _save(figure, figure_dir / "speed_heatmap.png")
    return 1


def _plot_validation_speed_volume(
    *,
    spec: CorridorSpec,
    settings: PipelineSettings,
    figure_dir: Path,
    data: pd.DataFrame,
    handoff: pd.DataFrame,
    conserved: pd.DataFrame,
) -> int:
    if data.empty:
        return 0
    minimum = data.groupby("link_id")["speed_raw"].min().dropna()
    if minimum.empty:
        return 0
    link_id = int(minimum.idxmin())
    join_keys = ["link_id", "t_min", "period"]
    if "run_id" in handoff and "run_id" in conserved:
        join_keys.insert(0, "run_id")
    volume = handoff.merge(
        conserved,
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    for column in (
        "t_min",
        "count_per_lane_15min",
        "qvdf_flow_vphpl",
        "qvdf_count_total_15min",
        "capacity_vphpl",
        "lanes",
    ):
        if column in volume:
            volume[column] = pd.to_numeric(
                volume[column], errors="coerce"
            )
    volume = volume[
        volume["t_min"].ge(6 * 60) & volume["t_min"].lt(21 * 60)
    ]
    speed_panel = data[data["link_id"].eq(link_id)].sort_values("t_min")
    volume_panel = volume[volume["link_id"].eq(link_id)].sort_values("t_min")
    if speed_panel.empty or volume_panel.empty:
        return 0
    tmc_code = str(speed_panel["tmc_code"].dropna().iloc[0])

    cutoff = float(speed_panel["cutoff"].median())
    free_flow = float(speed_panel["obs_ff"].median())
    capacity = float(volume_panel["capacity_vphpl"].median())
    lanes = float(volume_panel["lanes"].median())
    lanes = lanes if np.isfinite(lanes) and lanes > 0 else 1.0
    observed_flow = volume_panel["count_per_lane_15min"] * 4.0
    modeled_flow = volume_panel["qvdf_flow_vphpl"]

    figure, (speed_axis, flow_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    speed_axis.plot(
        speed_panel["t_min"] / 60.0,
        speed_panel["speed_raw"],
        color=OBSERVED_COLOR,
        lw=PRIMARY_LINE_WIDTH,
        marker=".",
        ms=3,
        label="CBI observed speed",
    )
    speed_axis.plot(
        speed_panel["t_min"] / 60.0,
        speed_panel["speed_qvdf_model"],
        color=CBI_RECONSTRUCTION_COLOR,
        lw=PRIMARY_LINE_WIDTH,
        label="CBI QVDF reconstruction",
    )
    speed_axis.axhline(
        free_flow,
        color=THRESHOLD_COLOR,
        ls="-.",
        lw=SECONDARY_LINE_WIDTH,
        label="free-flow (observed)",
    )
    speed_axis.axhline(
        cutoff,
        color=THRESHOLD_COLOR,
        ls="--",
        lw=SECONDARY_LINE_WIDTH,
        label="congestion threshold",
    )
    speed_axis.set_ylim(0, 80)
    speed_axis.set_ylabel("Speed (mph)", fontsize=MINIMUM_FONT_SIZE)
    speed_axis.legend(fontsize=MINIMUM_FONT_SIZE, loc="lower right")
    speed_axis.set_title(
        f"{spec.key} TMC {tmc_code} — speed ↔ volume consistency",
        fontsize=13,
    )

    hour = volume_panel["t_min"] / 60.0
    flow_axis.plot(
        hour,
        observed_flow,
        color=OBSERVED_COLOR,
        lw=PRIMARY_LINE_WIDTH,
        marker=".",
        ms=3,
        label="Volume inferred from sensor speed",
    )
    flow_axis.plot(
        hour,
        modeled_flow,
        color=CBI_RECONSTRUCTION_COLOR,
        lw=PRIMARY_LINE_WIDTH,
        label="QVDF conserved flow",
    )
    flow_axis.axhline(
        capacity,
        color="tab:red",
        ls="--",
        lw=SECONDARY_LINE_WIDTH,
        label=f"capacity ({capacity:.0f} vphpl)",
    )
    flow_axis.set_ylim(0, capacity * 1.15)
    flow_axis.set_ylabel("Volume (veh/hr/lane)", fontsize=MINIMUM_FONT_SIZE)
    flow_axis.set_xlabel("Hour of day", fontsize=MINIMUM_FONT_SIZE)
    flow_axis.set_xlim(6, 21)
    flow_axis.legend(
        fontsize=MINIMUM_FONT_SIZE,
        loc="upper center",
        ncol=3,
        framealpha=0.9,
    )
    period_colors = {"AM": "0.85", "MD": "0.92", "PM": "0.85"}
    for period, (start, end) in settings.periods.items():
        segment = volume_panel[
            volume_panel["t_min"].ge(start)
            & volume_panel["t_min"].lt(end)
        ]
        if segment.empty:
            continue
        observed_volume = float(
            segment["count_per_lane_15min"].sum(min_count=1)
        )
        qvdf_demand = float(
            segment["qvdf_count_total_15min"].sum(min_count=1)
        ) / lanes
        for axis in (speed_axis, flow_axis):
            axis.axvspan(
                start / 60.0,
                end / 60.0,
                color=period_colors.get(period, "0.9"),
                alpha=0.35,
                zorder=0,
            )
        flow_axis.text(
            (start + end) / 120.0,
            capacity * 0.14,
            f"{period}\nvolume {observed_volume:,.0f}\n"
            f"QVDF D {qvdf_demand:,.0f}",
            ha="center",
            va="bottom",
            fontsize=MINIMUM_FONT_SIZE,
            color="0.25",
        )
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.93,
        bottom=0.09,
    )
    figure.savefig(
        figure_dir / f"speed_volume_link{link_id}.png",
        dpi=130,
    )
    plt.close(figure)
    return 1


def _generate_standard_validation_figures(
    *,
    spec: CorridorSpec,
    settings: PipelineSettings,
    figure_dir: Path,
    handoff: pd.DataFrame,
    fd: pd.DataFrame,
    conserved: pd.DataFrame,
) -> int:
    """Generate the former testbed validation plots for every corridor."""

    data = _validation_handoff(handoff, fd)
    products = _plot_sensor_vs_model_fullday(
        spec=spec,
        figure_dir=figure_dir,
        data=data,
    )
    products += _plot_validation_speed_heatmap(
        spec=spec,
        figure_dir=figure_dir,
        data=data,
    )
    products += _plot_validation_speed_volume(
        spec=spec,
        settings=settings,
        figure_dir=figure_dir,
        data=data,
        handoff=handoff,
        conserved=conserved,
    )
    return products


def generate_daily_analysis_figures_from_outputs(
    *,
    corridor_key: str,
    corridor_output_dir: Path,
    figure_dir: Path,
    settings: PipelineSettings | None = None,
) -> list[Path]:
    """Rebuild the three standard daily-analysis figures from saved products.

    This public adapter lets downstream reports reuse the integrated CBI
    plotting implementation without rerunning preprocessing, episode detection,
    screening, calibration, or reconstruction.
    """

    corridor_output_dir = Path(corridor_output_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    handoff = pd.read_csv(
        corridor_output_dir
        / "07-reconstruction-and-handoff"
        / "average_weekday_time_dependent.csv",
        low_memory=False,
    )
    fd = pd.read_csv(
        corridor_output_dir
        / "02-fundamental-diagram"
        / "link_fd_context.csv",
        low_memory=False,
    )
    conserved = pd.read_csv(
        corridor_output_dir
        / "07-reconstruction-and-handoff"
        / "qvdf_conserved_flow.csv",
        low_memory=False,
    )
    spec = CorridorSpec(
        key=corridor_key,
        name=corridor_key,
        source="inrix_folder",
        path=corridor_output_dir,
        free_flow_mph=45.0,
        capacity_vphpl=1800.0,
        data_mode="speed_only",
    )
    before = set(figure_dir.glob("*.png"))
    _generate_standard_validation_figures(
        spec=spec,
        settings=settings or PipelineSettings(),
        figure_dir=figure_dir,
        handoff=handoff,
        fd=fd,
        conserved=conserved,
    )
    expected = [
        figure_dir / "sensor_vs_model_fullday.png",
        figure_dir / "speed_heatmap.png",
    ]
    expected.extend(sorted(figure_dir.glob("speed_volume_link*.png")))
    return [
        path
        for path in expected
        if path.is_file() and (path not in before or path.stat().st_size > 0)
    ]


def generate_corridor_figures(
    *,
    spec: CorridorSpec,
    settings: PipelineSettings,
    output_dir: Path,
    observations: pd.DataFrame,
    average: pd.DataFrame,
    fd: pd.DataFrame,
    episodes: pd.DataFrame,
    applied: pd.DataFrame,
    average_episodes: pd.DataFrame,
    average_applied: pd.DataFrame,
) -> int:
    """Write the established 3_CBI figure products from the integrated core."""

    figure_dir = step_dir(output_dir, "figures", create=True)
    products = 0
    metadata = observations.groupby("sensor_uid", as_index=False).agg(
        link_id=("link_id", "first")
    )
    contexts = fd.merge(metadata, on="sensor_uid", how="left").set_index("link_id")
    parameter_map = calibration_lookup(applied)
    average_parameter_map = calibration_lookup(average_applied)
    clean = (
        episodes[episodes["is_clean_valid_episode"].fillna(False)]
        if not episodes.empty
        else episodes
    )
    average_clean = (
        average_episodes[
            average_episodes["is_clean_valid_episode"].fillna(False)
        ]
        if not average_episodes.empty
        else average_episodes
    )
    handoff = pd.read_csv(
        step_dir(output_dir, "handoff")
        / "average_weekday_time_dependent.csv"
    )
    conserved = pd.read_csv(
        step_dir(output_dir, "handoff") / "qvdf_conserved_flow.csv"
    )

    if not applied.empty:
        top = (
            applied.sort_values("n_episodes", ascending=False)
            .drop_duplicates("link_id")
            .head(4)
        )
        if not top.empty:
            figure, axes = plt.subplots(2, 2, figsize=(11, 8))
            for axis, fit in zip(axes.flat, top.itertuples(index=False)):
                link_id = int(fit.link_id)
                group = observations[observations["link_id"].eq(link_id)]
                context = contexts.loc[link_id]
                flow = pd.to_numeric(group["flow_vph"], errors="coerce")
                speed = pd.to_numeric(group["speed_mph"], errors="coerce")
                valid = flow.notna() & speed.notna()
                axis.scatter(flow[valid], speed[valid], s=3, c="k", alpha=0.2)
                density = np.linspace(0.1, 200.0, 300)
                axis.plot(
                    s3_flow(
                        density,
                        context["vf_mph"],
                        context["kc_vpmpl"],
                        context["s3_m"],
                    ),
                    s3_speed(
                        density,
                        context["vf_mph"],
                        context["kc_vpmpl"],
                        context["s3_m"],
                    ),
                    "b--",
                    lw=1.6,
                )
                axis.set_xlim(0, 2400)
                axis.set_ylim(0, 85)
                axis.set_title(
                    f"ID {link_id} vf={context['vf_mph']:.0f} "
                    f"uc={context['vc_mph']:.0f} cap={context['capacity_vphpl']:.0f}",
                    fontsize=MINIMUM_FONT_SIZE,
                )
                axis.set_xlabel("Volume (veh/h/lane)")
                axis.set_ylabel("Speed (mph)")
            figure.suptitle(f"Fig 8 {spec.name} — volume-speed FD")
            _save(figure, figure_dir / "Fig8_FD.png")
            products += 1

    for period in settings.periods:
        period_fits = (
            applied[applied["period"].eq(period)]
            if "period" in applied.columns
            else applied
        )
        period_episodes = (
            clean[clean["period"].eq(period)] if not clean.empty else clean
        )
        if period_fits.empty or period_episodes.empty:
            continue
        top_fits = (
            period_fits.sort_values("n_episodes", ascending=False)
            .drop_duplicates("link_id")
            .head(4)
        )
        links = [int(value) for value in top_fits["link_id"]]

        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        for axis, (column, label) in zip(
            axes.flat,
            [
                ("P_hr", "Congestion duration (h)"),
                ("episode_demand", "Inflow demand (veh/lane)"),
                ("demand_capacity_ratio", "D/C ratio"),
                ("qdf", "QDF"),
            ],
        ):
            values = pd.to_numeric(period_episodes[column], errors="coerce").dropna()
            if len(values):
                axis.hist(
                    values,
                    bins=12,
                    color="white",
                    edgecolor="black",
                    weights=np.ones(len(values)) / len(values),
                )
                axis.set_xlabel(f"{label}\nMean = {values.mean():.2f}")
            axis.set_ylabel("Frequency")
        figure.suptitle(f"Fig 9 {spec.name} {period} — distributions")
        _save(figure, figure_dir / f"Fig9_distributions_{period}.png")
        products += 1

        for number, x_column, y_column, title in (
            (10, "demand_capacity_ratio", "P_hr", "D/C vs P"),
            (11, "P_hr", "magnitude", "P vs magnitude"),
            (
                12,
                "demand_capacity_ratio",
                "mean_speed_mph",
                "D/C vs avg speed",
            ),
        ):
            figure, axes = plt.subplots(2, 2, figsize=(11, 8))
            for axis, link_id in zip(axes.flat, links):
                group = period_episodes[
                    period_episodes["link_id"].eq(link_id)
                ]
                parameters = parameter_map.get((link_id, period))
                if group.empty or parameters is None:
                    continue
                axis.scatter(
                    group[x_column], group[y_column], s=8, c="k"
                )
                x_range = np.linspace(
                    0.01,
                    max(float(group[x_column].max()), 0.1),
                    100,
                )
                if number == 10:
                    model = parameters["f_d"] * x_range ** parameters["n"]
                elif number == 11:
                    model = parameters["f_p"] * x_range ** parameters["s"]
                else:
                    cutoff = float(group["threshold_used"].median())
                    model = cutoff / (
                        1.0
                        + parameters["alpha"] * x_range ** parameters["beta"]
                    )
                axis.plot(x_range, model, "b--", lw=1.6)
                axis.set_title(
                    f"ID {link_id} ({parameters['reliability']})", fontsize=MINIMUM_FONT_SIZE
                )
                axis.set_xlabel(x_column)
                axis.set_ylabel(y_column)
            figure.suptitle(f"Fig {number} {spec.name} {period} — {title}")
            filename = {
                10: f"Fig10_DC_P_{period}.png",
                11: f"Fig11_P_magnitude_{period}.png",
                12: f"Fig12_DC_cd_mean_speed_{period}.png",
            }[number]
            _save(figure, figure_dir / filename)
            products += 1

        average_period_episodes = (
            average_clean[average_clean["period"].eq(period)]
            if not average_clean.empty
            else average_clean
        )
        average_period_fits = (
            average_applied[average_applied["period"].eq(period)]
            if "period" in average_applied.columns
            else average_applied
        )
        average_links = (
            [
                int(value)
                for value in (
                    average_period_fits.sort_values(
                        "n_episodes", ascending=False
                    )
                    .drop_duplicates("link_id")
                    .head(4)["link_id"]
                )
            ]
            if not average_period_fits.empty
            else []
        )
        for figure_index, link_id in enumerate(average_links, start=14):
            if "link_id" not in average_period_episodes.columns:
                # A valid no-congestion corridor can have applied-period
                # metadata but no accepted average-weekday episode rows.  In
                # that case there is no profile reconstruction panel to draw.
                continue
            candidates = average_period_episodes[
                average_period_episodes["link_id"].eq(link_id)
            ].sort_values("P_hr", ascending=False)
            parameters = average_parameter_map.get((link_id, period))
            if candidates.empty or parameters is None:
                continue
            episode = candidates.iloc[0]
            predicted_duration = predict_duration_hours(
                episode["demand_capacity_ratio"], parameters
            )
            t0, t2, t3 = predicted_bounds_about_t2(
                episode["t0_hour"],
                episode["t2_hour"],
                episode["t3_hour"],
                predicted_duration,
            )
            minimum = predict_minimum_speed(
                episode["threshold_used"], predicted_duration, parameters
            )
            start = max(
                settings.wide_window[0], int(episode["t0_hour"] * 60 - 75)
            )
            end = min(
                settings.wide_window[1], int(episode["t3_hour"] * 60 + 75)
            )
            average_panel = average[
                average["link_id"].eq(link_id)
                & average["t_min"].ge(start)
                & average["t_min"].lt(end)
            ].sort_values("t_min")
            if average_panel.empty:
                continue
            context = contexts.loc[link_id]
            model = reconstruct_episode_speed(
                average_panel["t_min"].to_numpy(dtype=float),
                t0_hour=t0,
                t2_hour=t2,
                t3_hour=t3,
                minimum_speed_mph=minimum,
                cutoff_mph=float(episode["threshold_used"]),
                free_flow_mph=float(context["vf_mph"]),
                length_mi=float(episode["length_mi"]),
                discharge_vphpl=float(episode["mu_obs_vphpl"]),
                window_start_minute=start,
                window_end_minute=end,
            )
            figure, axis = plt.subplots(figsize=(9, 4.5))
            axis.plot(
                average_panel["t_min"] / 60.0,
                average_panel["speed_mph"],
                "r--s",
                ms=3,
                label="Observed (avg weekday)",
            )
            axis.plot(
                average_panel["t_min"] / 60.0,
                model,
                "k-o",
                ms=3,
                label="Estimated QVDF",
            )
            axis.axhline(
                episode["threshold_used"],
                color="green",
                ls="--",
                lw=1,
                label=f"cut-off {episode['threshold_used']:.0f}",
            )
            axis.axvline(t2, color="red", ls=":", lw=1)
            axis.set_xlim(start / 60.0, end / 60.0)
            axis.set_ylim(0, 80)
            axis.set_ylabel("Mean speed (mph)")
            axis.set_title(
                f"Fig {figure_index} {spec.name} {period} — obs vs QVDF, ID {link_id}",
                fontsize=MINIMUM_FONT_SIZE,
            )
            axis.legend(fontsize=MINIMUM_FONT_SIZE)
            axis.grid(alpha=0.3)
            _save(
                figure,
                figure_dir
                / f"Fig{figure_index}_td_{period}_{link_id}.png",
            )
            products += 1

        diagnostic_columns = {
            "P_hr",
            "link_id",
            "threshold_used",
            "t0_hour",
            "t2_hour",
            "t3_hour",
        }
        diagnostic_candidates = (
            average_period_episodes.sort_values("P_hr", ascending=False)
            .drop_duplicates("link_id")
            .head(4)
            if diagnostic_columns.issubset(average_period_episodes.columns)
            else pd.DataFrame(columns=sorted(diagnostic_columns))
        )
        if not diagnostic_candidates.empty:
            figure, axes = plt.subplots(2, 2, figsize=(12, 8))
            for axis, episode in zip(
                axes.flat, diagnostic_candidates.itertuples(index=False)
            ):
                panel = average[
                    average["link_id"].eq(episode.link_id)
                    & average["t_min"].ge(settings.wide_window[0])
                    & average["t_min"].lt(settings.wide_window[1])
                ].sort_values("t_min")
                if panel.empty:
                    continue
                speed = panel["speed_mph"].to_numpy(dtype=float)
                smoothed = detection_smoothed_speed(speed, settings)
                axis.plot(
                    panel["t_min"] / 60.0,
                    speed,
                    color="0.7",
                    lw=1,
                    marker=".",
                    ms=3,
                    label="observed 15-min",
                )
                axis.plot(
                    panel["t_min"] / 60.0,
                    smoothed,
                    "b-",
                    lw=2,
                    label="detector-smoothed",
                )
                axis.axhline(
                    episode.threshold_used,
                    color="green",
                    ls="--",
                    lw=1.2,
                    label=f"cut-off {episode.threshold_used:.0f} mph",
                )
                for value, color, label in (
                    (episode.t0_hour, "gray", "t0"),
                    (episode.t2_hour, "red", "t2"),
                    (episode.t3_hour, "gray", "t3"),
                ):
                    axis.axvline(value, color=color, ls=":", lw=1.4)
                    axis.text(value, 3, label, color=color, fontsize=MINIMUM_FONT_SIZE, ha="center")
                axis.axvspan(
                    episode.t0_hour,
                    episode.t3_hour,
                    color="orange",
                    alpha=0.12,
                )
                axis.set_xlim(
                    settings.wide_window[0] / 60.0,
                    settings.wide_window[1] / 60.0,
                )
                axis.set_ylim(0, max(80, float(np.nanmax(speed)) * 1.1))
                axis.set_title(
                    f"link {episode.link_id} [{period}] P={episode.P_hr:.2f} h "
                    f"t2={episode.t2_hour:.2f}",
                    fontsize=MINIMUM_FONT_SIZE,
                )
                axis.set_xlabel("hour")
                axis.set_ylabel("speed (mph)")
                axis.legend(fontsize=MINIMUM_FONT_SIZE, loc="lower right")
            figure.suptitle(
                f"t0/t2/t3 state-transition identification — {spec.name} {period}",
                fontsize=11,
            )
            _save(figure, figure_dir / f"DIAG_t0t3_{period}.png")
            products += 1

    products += _generate_standard_validation_figures(
        spec=spec,
        settings=settings,
        figure_dir=figure_dir,
        handoff=handoff,
        fd=fd,
        conserved=conserved,
    )

    if not average_clean.empty:
        for period in settings.periods:
            candidates = (
                average_clean[average_clean["period"].eq(period)]
                .sort_values("P_hr", ascending=False)
                .drop_duplicates("link_id")
            )
            if candidates.empty:
                continue
            links = [int(value) for value in candidates["link_id"]]
            row_map = {
                int(row["link_id"]): row for _, row in candidates.iterrows()
            }

            def draw_raw(axis, link_id, episode, first):
                start = max(
                    settings.wide_window[0],
                    int(episode["t0_hour"] * 60 - 75),
                )
                end = min(
                    settings.wide_window[1],
                    int(episode["t3_hour"] * 60 + 75),
                )
                daily = observations[
                    observations["link_id"].eq(link_id)
                    & observations["weekday"].lt(5)
                    & observations["t_min"].ge(start)
                    & observations["t_min"].lt(end)
                ]
                for index, (_, day) in enumerate(daily.groupby("date")):
                    day = day.sort_values("t_min")
                    axis.plot(
                        day["t_min"] / 60.0,
                        day["speed_mph"],
                        color="0.7",
                        lw=0.5,
                        alpha=0.5,
                        label="raw daily (weekdays)"
                        if first and index == 0
                        else None,
                    )
                avg = average[
                    average["link_id"].eq(link_id)
                    & average["t_min"].ge(start)
                    & average["t_min"].lt(end)
                ].sort_values("t_min")
                axis.plot(
                    avg["t_min"] / 60.0,
                    avg["speed_mph"],
                    color="tab:blue",
                    lw=2.2,
                    label="average weekday" if first else None,
                )
                axis.axhline(
                    episode["threshold_used"],
                    color="green",
                    ls="--",
                    lw=1,
                    label="cut-off" if first else None,
                )
                axis.set_xlim(start / 60.0, end / 60.0)
                axis.set_ylim(0, 80)
                axis.set_title(
                    f"link {link_id}: P={episode['P_hr']:.1f}h "
                    f"vt2={episode['min_speed_mph']:.0f}",
                    fontsize=MINIMUM_FONT_SIZE,
                )
                axis.tick_params(labelsize=6)
                if first:
                    axis.legend(fontsize=MINIMUM_FONT_SIZE, loc="lower right")

            def draw_model(axis, link_id, episode, first):
                start = max(
                    settings.wide_window[0],
                    int(episode["t0_hour"] * 60 - 75),
                )
                end = min(
                    settings.wide_window[1],
                    int(episode["t3_hour"] * 60 + 75),
                )
                panel = handoff[
                    handoff["link_id"].eq(link_id)
                    & handoff["t_min"].ge(start)
                    & handoff["t_min"].lt(end)
                ].sort_values("t_min")
                axis.plot(
                    panel["t_min"] / 60.0,
                    panel["speed_raw"],
                    color="tab:blue",
                    lw=1.6,
                    marker=".",
                    ms=2,
                    label="average weekday (obs)" if first else None,
                )
                axis.plot(
                    panel["t_min"] / 60.0,
                    panel["speed_qvdf_model"],
                    "k-",
                    lw=1.8,
                    label="QVDF model" if first else None,
                )
                axis.axhline(
                    episode["threshold_used"],
                    color="green",
                    ls="--",
                    lw=1,
                    label="cut-off" if first else None,
                )
                axis.axvline(episode["t2_hour"], color="red", ls=":", lw=1)
                axis.set_xlim(start / 60.0, end / 60.0)
                axis.set_ylim(0, 80)
                axis.set_title(
                    f"link {link_id}: D/C={episode['demand_capacity_ratio']:.1f} "
                    f"P={episode['P_hr']:.1f}h",
                    fontsize=MINIMUM_FONT_SIZE,
                )
                axis.tick_params(labelsize=6)
                if first:
                    axis.legend(fontsize=MINIMUM_FONT_SIZE, loc="lower right")

            products += _paginate(
                figure_dir,
                period,
                links,
                row_map,
                draw_raw,
                "RAW_vs_AVGWEEKDAY",
                f"{spec.name} {period} — raw daily data vs average weekday (per link)",
            )
            products += _paginate(
                figure_dir,
                period,
                links,
                row_map,
                draw_model,
                "AVGWEEKDAY_vs_QVDF",
                f"{spec.name} {period} — average weekday vs QVDF model (per link)",
            )
    return products
