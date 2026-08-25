from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

from ..dashboard_filters import load_general_purpose_tmc_codes
from ..workers import recommend_workers
from .analysis import (
    build_corridor_period_summary,
    build_projection_table,
    collect_cbi_products,
)
from .assignment import build_assignment_extract
from .metrics import build_speed_metrics
from .render import (
    render_corridor_figures,
    render_html,
    render_summary_figures,
)
from .settings import DashboardSettings


def _prepare_output_root(settings: DashboardSettings) -> None:
    root = settings.output_root.resolve()
    package_root = settings.package_root.resolve()
    if root == package_root or package_root not in root.parents:
        raise ValueError(
            "Dashboard output must be a descendant of the package root"
        )
    if root.exists() and any(root.iterdir()):
        if not settings.force:
            raise FileExistsError(
                f"Dashboard output is not empty: {root}. Use --force to replace it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def run_dashboard(settings: DashboardSettings) -> dict[str, object]:
    """Build the dashboard from the current all-corridor CBI products."""

    started = time.perf_counter()
    _prepare_output_root(settings)
    required = [
        settings.cbi_products_root,
        settings.mapmatching_product_root,
        settings.model_link_map_path,
        settings.assignment_root,
        settings.ritis_15min_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dashboard inputs: {missing}")

    general_purpose_tmcs = load_general_purpose_tmc_codes(
        settings.mapmatching_product_root
    )
    products = collect_cbi_products(
        settings.cbi_products_root,
        eligible_tmc_codes=general_purpose_tmcs,
    )
    corridor_count = int(products.coverage["corridor"].nunique())
    worker_plan = recommend_workers(
        corridor_count,
        target_fraction=settings.worker_fraction,
        explicit_workers=settings.workers,
    )
    assignment_path, assignment = build_assignment_extract(settings)
    projection, coverage = build_projection_table(
        products, assignment, settings
    )
    corridor_summary = build_corridor_period_summary(
        projection, coverage, settings
    )
    heatmap = products.profiles[
        [
            "corridor",
            "tmc_code",
            "road_order",
            "t_min",
            "avg_weekday_speed_mph",
            "n_days",
        ]
    ].copy()
    heatmap = heatmap.rename(
        columns={
            "t_min": "time_slot_min",
            "avg_weekday_speed_mph": "avg_speed_mph",
        }
    )
    heatmap_path = (
        settings.dashboard_data_root
        / "observed_speed_average_weekday.csv"
    )
    heatmap.to_csv(heatmap_path, index=False)
    link_metrics, metric_summary = build_speed_metrics(
        projection, heatmap, settings
    )

    settings.dashboard_data_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "heatmap": heatmap_path,
        "assignment": assignment_path,
        "projection": (
            settings.dashboard_data_root / "link_episode_projection.csv"
        ),
        "coverage": settings.dashboard_data_root / "corridor_coverage.csv",
        "corridor_summary": (
            settings.dashboard_data_root / "corridor_period_summary.csv"
        ),
        "speed_metrics": (
            settings.dashboard_data_root / "link_speed_metrics.csv"
        ),
        "metric_summary": (
            settings.dashboard_data_root / "metric_summary.csv"
        ),
        "accepted_average_episodes": (
            settings.dashboard_data_root
            / "accepted_average_weekday_episodes.csv"
        ),
        "selected_parameters": (
            settings.dashboard_data_root / "qvdf_selected_parameters.csv"
        ),
    }
    projection.to_csv(outputs["projection"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    corridor_summary.to_csv(outputs["corridor_summary"], index=False)
    link_metrics.to_csv(outputs["speed_metrics"], index=False)
    metric_summary.to_csv(outputs["metric_summary"], index=False)
    products.average_accepted.to_csv(
        outputs["accepted_average_episodes"], index=False
    )
    products.parameters.to_csv(outputs["selected_parameters"], index=False)

    summary_figures = render_summary_figures(
        corridor_summary, projection, settings.output_root
    )
    corridor_figures = (
        render_corridor_figures(
            projection,
            products.profiles,
            heatmap,
            coverage,
            settings,
            workers=worker_plan.workers,
        )
        if settings.generate_corridor_figures
        else {}
    )
    html_path = render_html(
        settings,
        coverage,
        corridor_summary,
        metric_summary,
        corridor_figures,
    )

    manifest = {
        "status": "PASS",
        "observed_profile_basis": "average_weekday",
        "profile_interval_minutes": settings.profile_interval_minutes,
        "periods_minutes": {
            label: [start, end]
            for label, (start, end) in settings.periods.items()
        },
        "periods_display": {
            "AM": "06:00-09:00",
            "MD": "09:00-15:00",
            "PM": "15:00-19:00",
        },
        "analytical_engine": "cbi",
        "episode_basis": "accepted_average_weekday_episode",
        "calibration_basis": "accepted_daily_episodes",
        "legacy_qvdf_selfdemo_used": False,
        "cbi_products_root": str(settings.cbi_products_root),
        "mapmatching_product_root": str(settings.mapmatching_product_root),
        "model_link_map": str(settings.model_link_map_path),
        "assignment_root": str(settings.assignment_root),
        "ritis_source": str(settings.ritis_15min_path),
        "worker_plan": worker_plan.to_dict(),
        "corridors": int(len(coverage)),
        "corridors_ready": int(coverage["coverage_status"].eq("ready").sum()),
        "selected_link_periods": int(len(projection)),
        "selected_tmc_periods": int(len(projection)),
        "ready_link_periods": int(
            projection["projection_status"].eq("ready").sum()
        ),
        "ready_tmc_periods": int(
            projection["projection_status"].eq("ready").sum()
        ),
        "summary_figures": [str(path) for path in summary_figures],
        "corridor_figures": len(corridor_figures),
        "html": str(html_path),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = settings.output_root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=str)
    return manifest
