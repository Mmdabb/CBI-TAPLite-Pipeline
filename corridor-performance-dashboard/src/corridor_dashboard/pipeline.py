from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .qvdf_projection_dashboard.pipeline import run_dashboard as run_projection
from .qvdf_projection_dashboard.settings import DashboardSettings
from .builder import DashboardBuildSettings, build_dashboard


def run_integrated_dashboard(
    *,
    package_root: Path,
    corridor_results_root: Path | None = None,
    mapmatching_product_root: Path | None = None,
    model_link_map_path: Path | None = None,
    assignment_root: Path | None = None,
    ritis_15min_path: Path | None = None,
    profile_selection_overrides_path: Path | None = None,
    corridor_measurement_root: Path | None = None,
    output_root: Path | None = None,
    workers: int | None = None,
    worker_fraction: float = 0.50,
    force: bool = False,
) -> dict[str, Any]:
    """Build projection products and the map/report site as one dashboard."""

    required = {
        "corridor_results_root": corridor_results_root,
        "mapmatching_product_root": mapmatching_product_root,
        "model_link_map_path": model_link_map_path,
        "assignment_root": assignment_root,
        "ritis_15min_path": ritis_15min_path,
        "corridor_measurement_root": corridor_measurement_root,
        "output_root": output_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Explicit integrated-dashboard paths are required: " + ", ".join(missing)
        )
    output_root = Path(output_root).resolve()

    build_settings = DashboardBuildSettings(
        package_root=package_root,
        corridor_results_root=corridor_results_root,
        mapmatching_product_root=mapmatching_product_root,
        ritis_15min_path=ritis_15min_path,
        profile_selection_overrides_path=profile_selection_overrides_path,
        corridor_measurement_root=corridor_measurement_root,
        assignment_root=assignment_root,
        qvdf_report_root=output_root.parent / ".projection-input",
        output_root=output_root,
        force=force,
    )
    run_root = build_settings.corridor_results_root.parent
    canonical_map = (
        run_root / "shared" / "network-mapping" / "canonical_node_pair_tmc.csv"
    )
    selected_map = (
        Path(model_link_map_path).resolve()
        if model_link_map_path is not None
        else canonical_map
    )
    if not selected_map.is_file():
        raise FileNotFoundError(
            "The integrated dashboard requires the frozen CBI node-pair "
            f"mapping: {selected_map}"
        )

    # Keep projection products private to this dashboard build. A shared
    # sibling staging directory makes concurrent scenario builds overwrite
    # and delete one another's files during final cleanup.
    staging_root = (
        build_settings.output_root.parent
        / f".projection-staging-{build_settings.output_root.name}"
    )
    projection_settings = DashboardSettings(
        package_root=build_settings.package_root,
        output_root=staging_root,
        cbi_products_root=build_settings.corridor_results_root,
        mapmatching_product_root=build_settings.mapmatching_product_root,
        model_link_map_path=selected_map,
        assignment_root=build_settings.assignment_root,
        ritis_15min_path=build_settings.ritis_15min_path,
        workers=workers,
        worker_fraction=worker_fraction,
        force=True,
    )
    projection_manifest: dict[str, Any] | None = None
    try:
        projection_manifest = run_projection(projection_settings)
        integrated_settings = DashboardBuildSettings(
            package_root=build_settings.package_root,
            corridor_results_root=build_settings.corridor_results_root,
            mapmatching_product_root=build_settings.mapmatching_product_root,
            qvdf_report_root=staging_root,
            corridor_measurement_root=(
                build_settings.corridor_measurement_root
            ),
            assignment_root=build_settings.assignment_root,
            ritis_15min_path=build_settings.ritis_15min_path,
            profile_selection_overrides_path=(
                build_settings.profile_selection_overrides_path
            ),
            output_root=build_settings.output_root,
            force=force,
        )
        manifest = build_dashboard(integrated_settings)
    finally:
        resolved_stage = staging_root.resolve()
        resolved_dashboard_parent = build_settings.output_root.parent.resolve()
        if (
            resolved_stage.is_dir()
            and resolved_stage.parent == resolved_dashboard_parent
            and resolved_stage == staging_root.resolve()
            and resolved_stage.name.startswith(".projection-staging-")
        ):
            shutil.rmtree(resolved_stage)

    manifest["projection"] = {
        "selected_link_periods": projection_manifest["selected_link_periods"],
        "ready_link_periods": projection_manifest["ready_link_periods"],
        "selected_tmc_periods": projection_manifest["selected_tmc_periods"],
        "ready_tmc_periods": projection_manifest["ready_tmc_periods"],
        "assignment_root": projection_manifest["assignment_root"],
        "model_link_map": projection_manifest["model_link_map"],
        "worker_plan": projection_manifest["worker_plan"],
    }
    (build_settings.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return manifest
