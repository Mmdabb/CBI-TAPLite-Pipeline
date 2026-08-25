"""Mapped-link volume comparison data and origin-constrained scatter figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from corridor_measurement.figures import (
    MINIMUM_FONT_SIZE,
    PERIOD_COLORS,
    PRIMARY_LINE_WIDTH,
    SECONDARY_LINE_WIDTH,
)


PROBLEM_KEY_COLUMNS = ("period", "link_id")
VOLUME_COLUMNS = (
    "assignment_volume",
    "cube_period_volume",
    "synthetic_period_volume",
)


def extract_problematic_tmc_link_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every TMC-link occurrence with zero volume or D/C at most 0.10."""

    required = {
        "assignment_volume",
        "assignment_doc",
        "period",
        "link_id",
        "tmc_code",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing problematic-link columns: {sorted(missing)}")

    volume = pd.to_numeric(frame["assignment_volume"], errors="coerce")
    doc = pd.to_numeric(frame["assignment_doc"], errors="coerce")
    mask = (volume.notna() & volume.le(0.0)) | (doc.notna() & doc.le(0.10))
    result = frame.loc[mask].copy()
    result["problem_zero_volume_flag"] = volume.loc[mask].le(0.0).to_numpy()
    result["problem_doc_le_0_10_flag"] = doc.loc[mask].le(0.10).to_numpy()
    return result.sort_values(
        ["corridor", "period", "road_order", "tmc_code", "link_id"],
        kind="stable",
    ).reset_index(drop=True)


def _joined_unique(values: Iterable[object]) -> str:
    cleaned = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }
    return "|".join(sorted(cleaned))


def build_unique_matched_link_periods(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one scatter point per physical link-period with match provenance."""

    required = {
        "period",
        "link_id",
        "corridor",
        "tmc_code",
        "assignment_doc",
        *VOLUME_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing scatter-data columns: {sorted(missing)}")

    working = frame.copy()
    for column in (*VOLUME_COLUMNS, "assignment_doc"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.sort_values(
        ["period", "link_id", "corridor", "tmc_code"], kind="stable"
    )
    group = working.groupby(list(PROBLEM_KEY_COLUMNS), sort=True, dropna=False)
    provenance = group.agg(
        tmc_match_count=("tmc_code", "nunique"),
        corridor_match_count=("corridor", "nunique"),
        matched_tmc_codes=("tmc_code", _joined_unique),
        matched_corridors=("corridor", _joined_unique),
    ).reset_index()
    unique = working.drop_duplicates(list(PROBLEM_KEY_COLUMNS), keep="first")
    unique = unique.merge(
        provenance,
        on=list(PROBLEM_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    unique["problem_zero_volume_flag"] = unique["assignment_volume"].le(0.0)
    unique["problem_doc_le_0_10_flag"] = unique["assignment_doc"].le(0.10)
    unique["problematic_flag"] = (
        unique["problem_zero_volume_flag"]
        | unique["problem_doc_le_0_10_flag"]
    )
    return unique.sort_values(list(PROBLEM_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def origin_constrained_fit(
    x_values: Iterable[object], y_values: Iterable[object]
) -> dict[str, float | int]:
    """Fit y = slope*x by ordinary least squares with the intercept fixed at zero."""

    x = pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    x = x[valid]
    y = y[valid]
    denominator = float(np.dot(x, x))
    slope = float(np.dot(x, y) / denominator) if denominator > 0.0 else np.nan
    fitted = slope * x if np.isfinite(slope) else np.full_like(y, np.nan)
    residual = y - fitted
    y_energy = float(np.dot(y, y))
    origin_r_squared = (
        float(1.0 - np.dot(residual, residual) / y_energy)
        if y_energy > 0.0 and np.isfinite(slope)
        else np.nan
    )
    return {
        "n": int(len(x)),
        "slope": slope,
        "slope_minus_parity_percent": (
            float((slope - 1.0) * 100.0) if np.isfinite(slope) else np.nan
        ),
        "origin_r_squared": origin_r_squared,
        "mae": float(np.mean(np.abs(y - x))) if len(x) else np.nan,
    }


def _format_slope(slope: float) -> str:
    if not np.isfinite(slope):
        return "not estimable"
    if abs(slope) >= 100:
        return f"{slope:.1f}"
    if abs(slope) >= 10:
        return f"{slope:.2f}"
    return f"{slope:.3f}"


def create_volume_scatter_figure(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    population_label: str,
) -> pd.DataFrame:
    """Render the three requested comparisons and return panel fit statistics."""

    comparisons = (
        (
            "assignment_volume",
            "synthetic_period_volume",
            "Assigned volume",
            "Synthetic volume",
            "Assigned vs synthetic",
        ),
        (
            "cube_period_volume",
            "synthetic_period_volume",
            "Cube volume",
            "Synthetic volume",
            "Cube vs synthetic",
        ),
        (
            "assignment_volume",
            "cube_period_volume",
            "Assigned volume",
            "Cube volume",
            "Assigned vs Cube",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.6), constrained_layout=False)
    statistics: list[dict[str, object]] = []
    period = frame["period"].astype("string").str.upper()

    for axis, (x_column, y_column, x_label, y_label, title) in zip(
        axes, comparisons
    ):
        x = pd.to_numeric(frame[x_column], errors="coerce")
        y = pd.to_numeric(frame[y_column], errors="coerce")
        valid = x.notna() & y.notna() & x.ge(0.0) & y.ge(0.0)
        panel = pd.DataFrame({"x": x[valid], "y": y[valid], "period": period[valid]})
        fit = origin_constrained_fit(panel["x"], panel["y"])

        for period_name in ("AM", "MD", "PM"):
            subset = panel[panel["period"].eq(period_name)]
            if subset.empty:
                continue
            axis.scatter(
                subset["x"],
                subset["y"],
                s=20,
                alpha=0.34,
                color=PERIOD_COLORS[period_name],
                edgecolors="none",
                rasterized=True,
            )

        maximum = float(panel[["x", "y"]].to_numpy().max()) if not panel.empty else 1.0
        maximum = max(maximum * 1.04, 1.0)
        line_x = np.asarray([0.0, maximum])
        axis.plot(
            line_x,
            line_x,
            color="#333333",
            linestyle="--",
            linewidth=SECONDARY_LINE_WIDTH,
            zorder=3,
        )
        slope = float(fit["slope"])
        if np.isfinite(slope):
            axis.plot(
                line_x,
                slope * line_x,
                color="#d627a2",
                linewidth=PRIMARY_LINE_WIDTH,
                zorder=4,
            )
        axis.set_xlim(0.0, maximum)
        axis.set_ylim(0.0, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#e4e4e4", linewidth=0.8, alpha=0.8)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xlabel(f"{x_label} (vehicles)", fontsize=MINIMUM_FONT_SIZE)
        axis.set_ylabel(f"{y_label} (vehicles)", fontsize=MINIMUM_FONT_SIZE)
        difference = float(fit["slope_minus_parity_percent"])
        difference_text = (
            f"{difference:+.1f}% vs parity" if np.isfinite(difference) else "parity difference n/a"
        )
        axis.text(
            0.04,
            0.96,
            f"Origin fit: y = {_format_slope(slope)}x\n{difference_text}\nN = {int(fit['n']):,}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=MINIMUM_FONT_SIZE,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.92,
            },
        )
        statistics.append(
            {
                "population": population_label,
                "comparison": title,
                "x_column": x_column,
                "y_column": y_column,
                **fit,
            }
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=PERIOD_COLORS[name],
            markeredgecolor="none",
            label=name,
        )
        for name in ("AM", "MD", "PM")
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="--",
                linewidth=SECONDARY_LINE_WIDTH,
                label="45-degree parity",
            ),
            Line2D(
                [0],
                [0],
                color="#d627a2",
                linewidth=PRIMARY_LINE_WIDTH,
                label="Origin-constrained fit",
            ),
        ]
    )
    fig.suptitle(
        f"Mapped-link period volume comparisons — {population_label}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "One point per physical GMNS link-period; repeated TMC/corridor matches are not double-weighted.",
        ha="center",
        fontsize=MINIMUM_FONT_SIZE,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=5,
        frameon=False,
        fontsize=MINIMUM_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.87, bottom=0.19, wspace=0.26)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pd.DataFrame(statistics)
