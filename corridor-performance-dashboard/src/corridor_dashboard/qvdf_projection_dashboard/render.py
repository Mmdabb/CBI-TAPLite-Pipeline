from __future__ import annotations

import html
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import reconstruction_curves
from .settings import DashboardSettings


COLORS = {
    "observed": "#1f77b4",
    "cbi": "#6f4e9c",
    "projected": "#e87500",
}
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


def render_summary_figures(
    corridor_summary: pd.DataFrame,
    projection: pd.DataFrame,
    output_root: Path,
) -> list[Path]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    pm = corridor_summary[
        corridor_summary["period"].eq("PM")
        & corridor_summary["projection_available"]
    ].sort_values("P_A_mean")
    height = max(6.0, 0.28 * max(len(pm), 1))
    fig, ax = plt.subplots(figsize=(12, height))
    y = np.arange(len(pm))
    ax.barh(y - 0.22, pm["P_A_mean"], 0.22, label="Observed CBI P", color="#d62728")
    ax.barh(y, pm["P_B_mean"], 0.22, label="QVDF at observed D/C", color="#2ca02c")
    ax.barh(
        y + 0.22,
        pm["P_C_mean"],
        0.22,
        label="QVDF at assignment D/C",
        color="#1f77b4",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pm["corridor"], fontsize=MINIMUM_FONT_SIZE)
    ax.set_xlabel("Congestion duration (hours)")
    ax.set_title("PM duration audit — every corridor with a ready projection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = figure_root / "pm_duration_all_corridors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    ready = projection[projection["projection_status"].eq("ready")]
    fig, ax = plt.subplots(figsize=(8, 7))
    for period, color in (("AM", "#1f77b4"), ("MD", "#ff7f0e"), ("PM", "#d62728")):
        group = ready[ready["period"].eq(period)]
        ax.scatter(
            group["DC_obs"],
            group["dc_dta_vol"],
            s=13,
            alpha=0.45,
            color=color,
            label=period,
        )
    maximum = float(
        np.nanmax(
            [
                ready["DC_obs"].max() if not ready.empty else 1.0,
                ready["dc_dta_vol"].max() if not ready.empty else 1.0,
                1.0,
            ]
        )
    )
    maximum = min(maximum, 8.0)
    ax.plot([0, maximum], [0, maximum], "k--", lw=1)
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_xlabel("Observed CBI D/C")
    ax.set_ylabel("Assignment volume / period capacity")
    ax.set_title("Assignment loading audit")
    ax.legend()
    fig.tight_layout()
    path = figure_root / "dc_assignment_audit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)
    return outputs


def _observed_matrix(
    heat: pd.DataFrame,
    tmc_order: list[str],
    time: np.ndarray,
) -> np.ndarray:
    lookup = {
        str(tmc): group.set_index("time_slot_min")["avg_speed_mph"]
        for tmc, group in heat.groupby("tmc_code")
    }
    return np.array(
        [
            pd.to_numeric(
                lookup.get(tmc, pd.Series(dtype=float)).reindex(time),
                errors="coerce",
            ).to_numpy(dtype=float)
            for tmc in tmc_order
        ]
    )


def _model_matrices(
    corridor_projection: pd.DataFrame,
    tmc_order: list[str],
    period: str,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cbi = np.full((len(tmc_order), len(time)), np.nan)
    projected = np.full_like(cbi, np.nan)
    candidates = corridor_projection[
        corridor_projection["period"].eq(period)
    ].sort_values(
        [
            "tmc_code",
            "network_match_distance_ft",
            "_map_occurrence",
        ],
        na_position="last",
        kind="mergesort",
    )
    candidates = candidates.drop_duplicates("tmc_code", keep="first")
    by_tmc = {str(row.tmc_code): row for row in candidates.itertuples(index=False)}
    for index, tmc in enumerate(tmc_order):
        item = by_tmc.get(tmc)
        if item is None:
            continue
        row = pd.Series(item._asdict())
        cbi[index], projected[index] = reconstruction_curves(row, time)
    return cbi, projected


def _render_corridor_task(
    task: tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        str,
        dict[str, tuple[int, int]],
        int,
        str,
        str,
    ],
) -> tuple[str, str | None]:
    (
        corridor,
        corridor_projection,
        profile,
        corridor_heat,
        status,
        periods,
        interval_minutes,
        report_root,
        cbi_products_root,
    ) = task
    order = (
        profile[["tmc_code", "road_order"]]
        .drop_duplicates("tmc_code")
        .sort_values(["road_order", "tmc_code"], na_position="last")
    )
    tmc_order = order["tmc_code"].astype(str).tolist()
    if not tmc_order:
        return corridor, None
    fig, axes = plt.subplots(
        len(periods),
        3,
        figsize=(18, 12.5),
        squeeze=False,
    )
    last_image = None
    for row_index, (period, (start, end)) in enumerate(periods.items()):
        time = np.arange(start, end, interval_minutes, dtype=float)
        observed = _observed_matrix(corridor_heat, tmc_order, time)
        cbi, projected = _model_matrices(
            corridor_projection, tmc_order, period, time
        )
        for column_index, (matrix, label) in enumerate(
            (
                (observed, "Observed average-weekday speed"),
                (cbi, "Integrated CBI reconstruction"),
                (projected, "Assignment D/C projection"),
            )
        ):
            axis = axes[row_index, column_index]
            if np.isfinite(matrix).any():
                shown = np.ma.masked_invalid(matrix)
                last_image = axis.imshow(
                    shown,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=70,
                    extent=[start / 60.0, end / 60.0, 0, len(tmc_order)],
                )
            else:
                axis.set_facecolor("#f3f4f6")
                message = (
                    "No accepted average-weekday episode"
                    if column_index == 1
                    else "No ready assignment projection"
                    if column_index == 2
                    else "No observed average-weekday speed"
                )
                axis.text(
                    0.5,
                    0.5,
                    message,
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=MINIMUM_FONT_SIZE,
                    color="#4b5563",
                    wrap=True,
                )
            axis.set_title(f"{period} — {label}", fontsize=13)
            axis.set_xlabel("Hour")
            axis.set_xlim(start / 60.0, end / 60.0)
            axis.set_ylim(0, len(tmc_order))
            if column_index == 0:
                axis.set_ylabel("Ordered TMC")
    fig.suptitle(
        f"{corridor} — current-period CBI and assignment projection "
        f"(coverage: {status})",
        fontsize=15,
    )
    fig.subplots_adjust(top=0.93, right=0.90, hspace=0.38, wspace=0.20)
    if last_image is not None:
        colorbar_axis = fig.add_axes([0.92, 0.15, 0.015, 0.70])
        fig.colorbar(last_image, cax=colorbar_axis, label="mph")
    authoritative_figure_dir = (
        Path(cbi_products_root) / corridor / "10-figures"
    )
    authoritative_figure_dir.mkdir(parents=True, exist_ok=True)
    authoritative_path = authoritative_figure_dir / "qvdf_projection.png"
    fig.savefig(authoritative_path, dpi=130)
    plt.close(fig)

    corridor_report_dir = Path(report_root) / corridor
    corridor_report_dir.mkdir(parents=True, exist_ok=True)
    path = corridor_report_dir / "projection.png"
    shutil.copy2(authoritative_path, path)

    from ..figures import generate_daily_analysis_figures_from_outputs

    daily_dir = corridor_report_dir / "daily_analysis"
    daily_figures = generate_daily_analysis_figures_from_outputs(
        corridor_key=corridor,
        corridor_output_dir=Path(cbi_products_root) / corridor,
        figure_dir=daily_dir,
    )
    speed_volume = next(
        (
            figure
            for figure in daily_figures
            if figure.name.startswith("speed_volume_link")
        ),
        None,
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(corridor)} — NVTA CBI dashboard</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:auto;padding:24px;color:#17223b}}
h1,h2{{color:#16295c}} img{{max-width:100%;height:auto;margin:8px 0 24px}}
.note{{background:#f4f6fb;border-left:4px solid #16295c;padding:10px 14px}}
</style></head><body>
<p><a href="../../index.html">← All corridors</a></p>
<h1>{html.escape(corridor)}</h1>
<div class="note">Coverage status: {html.escape(status)}. AM 06:00–09:00,
MD 09:00–15:00, PM 15:00–19:00.</div>
<h2>Assignment projection analysis</h2>
<img src="projection.png" alt="Observed, CBI, and assignment-projected speed fields">
<h2>Daily analysis</h2>
<h3>Sensor versus model, full day</h3>
<img src="daily_analysis/sensor_vs_model_fullday.png" alt="Sensor versus model full day">
<h3>Speed heatmap</h3>
<img src="daily_analysis/speed_heatmap.png" alt="Observed and modeled speed heatmap">
<h3>Speed and volume</h3>
{
    f'<img src="daily_analysis/{html.escape(speed_volume.name)}" '
    'alt="Speed and volume">' if speed_volume is not None
    else '<p>No speed-volume figure was available.</p>'
}
</body></html>"""
    page_path = corridor_report_dir / "index.html"
    page_path.write_text(page, encoding="utf-8")
    return corridor, str(page_path)


def render_corridor_figures(
    projection: pd.DataFrame,
    profiles: pd.DataFrame,
    heatmap: pd.DataFrame,
    coverage: pd.DataFrame,
    settings: DashboardSettings,
    *,
    workers: int = 1,
) -> dict[str, Path]:
    settings.corridor_report_root.mkdir(parents=True, exist_ok=True)
    heatmap = heatmap.copy()
    heatmap["tmc_code"] = heatmap["tmc_code"].astype(str)
    tasks = []
    for corridor in coverage["corridor"]:
        profile = profiles[profiles["corridor"].eq(corridor)].copy()
        tmc = set(profile["tmc_code"].astype(str))
        tasks.append(
            (
                str(corridor),
                projection[projection["corridor"].eq(corridor)].copy(),
                profile,
                heatmap[heatmap["tmc_code"].isin(tmc)].copy(),
                str(
                    coverage.loc[
                        coverage["corridor"].eq(corridor), "coverage_status"
                    ].iloc[0]
                ),
                settings.periods,
                settings.profile_interval_minutes,
                str(settings.corridor_report_root),
                str(settings.cbi_products_root),
            )
        )
    if workers > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_render_corridor_task, tasks))
        except PermissionError:
            # Managed Windows environments can deny multiprocessing pipe
            # creation. Rendering is deterministic, so retain correctness by
            # falling back to the in-process path.
            results = [_render_corridor_task(task) for task in tasks]
    else:
        results = [_render_corridor_task(task) for task in tasks]
    return {
        corridor: Path(path)
        for corridor, path in results
        if path is not None
    }


def _table_html(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    return frame[available].to_html(
        index=False,
        escape=True,
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.3f}",
    )


def render_html(
    settings: DashboardSettings,
    coverage: pd.DataFrame,
    corridor_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    corridor_figures: dict[str, Path],
) -> Path:
    rows = []
    for item in coverage.itertuples(index=False):
        corridor = str(item.corridor)
        figure = corridor_figures.get(corridor)
        label = (
            f"<a href='{html.escape(figure.relative_to(settings.output_root).as_posix())}'>"
            f"{html.escape(corridor)}</a>"
            if figure is not None
            else html.escape(corridor)
        )
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{int(item.profile_tmc)}</td>"
            f"<td>{int(item.mapped_tmc)}</td>"
            f"<td>{int(item.average_accepted_episodes)}</td>"
            f"<td>{int(item.calibrated_link_periods)}</td>"
            f"<td>{int(item.ready_link_periods)}</td>"
            f"<td>{html.escape(str(item.coverage_status))}</td>"
            "</tr>"
        )
    coverage_rows = "\n".join(rows)
    periods = ", ".join(
        f"{name} {start // 60:02d}:{start % 60:02d}–"
        f"{end // 60:02d}:{end % 60:02d}"
        for name, (start, end) in settings.periods.items()
    )
    metrics = _table_html(
        metric_summary,
        [
            "period",
            "n_ready_link_periods",
            "P_A_mean",
            "P_C_mean",
            "MAE_P_hr",
            "MAE_DC",
            "MAE_speed_CBI_mph",
            "MAE_speed_assignment_projection_mph",
        ],
    )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>NVTA integrated CBI QVDF projection dashboard</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#17223b;background:#fff}}
header{{background:#16295c;color:#fff;padding:20px 36px}}
main{{max-width:1180px;margin:auto;padding:20px 34px 60px}}
h1{{font-size:22px;margin:0}} h2{{color:#16295c;margin-top:32px}}
.note{{background:#f4f6fb;border-left:4px solid #16295c;padding:12px 16px}}
img{{max-width:100%;height:auto}}
table.data,table.coverage{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #dce1ec;padding:5px 7px;text-align:right}}
th{{background:#16295c;color:white}} td:first-child,th:first-child{{text-align:left}}
a{{color:#244f9e}}
</style></head><body>
<header><h1>NVTA integrated CBI × assignment QVDF projection</h1>
<div>Average-weekday observed profiles · all 70 corridor inputs</div>
</header><main>
<div class="note"><b>Authoritative analytical engine:</b> cbi state-transition
episode detection, accepted-episode screening, robust bounded calibration, and
t2-anchored reconstruction. Periods: {html.escape(periods)}. No CBI+ self-demo
or transferred/default QVDF calibration is used.</div>
<h2>Coverage</h2>
<table class="coverage"><tr><th>Corridor</th><th>Input TMC</th><th>Mapped TMC</th>
<th>Accepted average episodes</th><th>Calibrated link-periods</th>
<th>Ready assignment projections</th><th>Status</th></tr>
{coverage_rows}</table>
<h2>PM duration audit</h2>
<img src="figures/pm_duration_all_corridors.png" alt="PM duration audit">
<h2>Assignment D/C audit</h2>
<img src="figures/dc_assignment_audit.png" alt="D/C audit">
<h2>Clean-link metrics</h2>
{metrics}
<h2>Daily analysis</h2>
<p>Open any corridor in the coverage table to view the integrated CBI daily
analysis figures: <code>sensor_vs_model_fullday</code>,
<code>speed_heatmap</code>, and the selected
<code>speed_volume_link{{link_id}}</code> figure.</p>
<h2>Data outputs</h2>
<ul>
<li><a href="data/link_episode_projection.csv">Link-period projection table</a></li>
<li><a href="data/corridor_period_summary.csv">Corridor-period summary</a></li>
<li><a href="data/corridor_coverage.csv">Corridor coverage</a></li>
<li><a href="data/dtalite_assignment_dc.csv">Assignment D/C extract</a></li>
</ul>
</main></body></html>"""
    path = settings.output_root / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path
