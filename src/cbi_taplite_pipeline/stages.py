from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import PipelineConfig


LOGGER = logging.getLogger("cbi_taplite_pipeline")
PERIODS = ("am", "md", "pm")


def _stage(root: Path, number: int, name: str) -> Path:
    return root / f"{number:02d}-{name}"


def stage_paths(config: PipelineConfig) -> dict[str, Path]:
    root = config.output_root
    return {
        "matching": _stage(root, 1, "tmc-matching"),
        "corridors": _stage(root, 2, "corridor-speed-data"),
        "canonical": _stage(root, 3, "canonical-node-pair-map"),
        "coverage": _stage(root, 4, "observation-coverage"),
        "cbi": _stage(root, 5, "cbi"),
        "spatial": _stage(root, 6, "spatial-t2"),
        "boundary_seed": _stage(root, 7, "boundary-candidates"),
        "ridge": _stage(root, 8, "ridge-retraining"),
        "boundaries": _stage(root, 9, "congestion-boundaries"),
        "qvdf": _stage(root, 10, "network-qvdf"),
        "resources": _stage(root, 11, "taplite-resources"),
        "assignment1": _stage(root, 12, "taplite-stage-1"),
        "anchors": _stage(root, 13, "hybrid-anchors"),
        "assignment2": _stage(root, 14, "taplite-stage-2"),
        "measurement": _stage(root, 15, "corridor-measurement"),
        "dashboard": _stage(root, 16, "integrated-dashboard"),
    }


def python_environment(config: PipelineConfig) -> dict[str, str]:
    env = os.environ.copy()
    source_roots = [
        config.repository_root / "src",
        config.repository_root / "tmc-matching" / "src",
        config.repository_root / "cbi" / "src",
        config.repository_root / "corridor-performance-dashboard" / "src",
        config.repository_root / "nvta-taplite-workflow" / "src",
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in source_roots), *([existing] if existing else [])]
    )
    env.update(
        {
            "OMP_NUM_THREADS": str(config.workers),
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def run_command(
    config: PipelineConfig,
    command: list[str],
    output: Path,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    log_dir = config.output_root / "logs" / output.name
    log_dir.mkdir(parents=True, exist_ok=True)
    command_log = log_dir / "command.txt"
    command_log.write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    LOGGER.info("Running %s", output.name)
    with (log_dir / "engine.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=config.repository_root,
            env=environment or python_environment(config),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _module(name: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", name, *(str(value) for value in arguments)]


def _option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _copytree_hardlink(source: Path, destination: Path) -> None:
    def copy(source_file: str, destination_file: str) -> str:
        try:
            os.link(source_file, destination_file)
            return destination_file
        except OSError:
            return shutil.copy2(source_file, destination_file)

    shutil.copytree(source, destination, copy_function=copy)


def run_matching(config: PipelineConfig, output: Path) -> dict[str, str]:
    settings = config.section("matching")
    command = _module(
        "tmc_matching.cli",
        "run",
        "--input-dir",
        config.files["matching_input"],
        "--output-dir",
        output,
        "--source-crs-epsg",
        settings.get("source_crs_epsg", 4326),
        "--working-crs-epsg",
        settings.get("working_crs_epsg", 2248),
        "--lane-class",
        settings.get("lane_class", "auto"),
    )
    if settings.get("write_candidates", False):
        command.append("--write-candidates")
    run_command(config, command, output)
    return {"mapmatching_root": str(output)}


def run_corridor_slices(config: PipelineConfig, output: Path) -> dict[str, str]:
    command = _module(
        "tmc_matching.create_cbi_corridor_slices",
        "--metadata",
        config.files["tmc_metadata"],
        "--readings",
        config.files["tmc_readings"],
        "--output-dir",
        output,
    )
    definitions = config.files.get("corridor_definitions")
    if definitions is None:
        command.append("--all-road-directions")
    else:
        command.extend(["--definitions-csv", str(definitions)])
    run_command(config, command, output)
    return {"corridor_root": str(output / "corridors")}


def run_canonical(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    command = _module(
        "cbi.canonical_cli",
        "--mapping",
        paths["matching"] / "combined" / "full_tmc_to_link.csv",
        "--corridor-root",
        paths["corridors"] / "corridors",
        "--output-dir",
        output,
    )
    run_command(config, command, output)
    return {"canonical_map": str(output / "canonical_node_pair_tmc.csv")}


def run_observation_coverage(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    audit = output / "unmatched-link-audit"
    run_command(
        config,
        _module(
            "tmc_matching.audit_unmatched_dashboard_corridor_links",
            "--all-corridor-inputs",
            "--corridor-inputs",
            paths["corridors"] / "corridors",
            "--mapmatch-product",
            paths["matching"] / "combined",
            "--network-root",
            config.files["base_network"],
            "--metadata",
            config.files["tmc_metadata"],
            "--output-dir",
            audit,
        ),
        audit,
    )
    treatments = output / "treatments"
    run_command(
        config,
        _module(
            "tmc_matching.build_observation_coverage_treatments",
            "--canonical",
            paths["canonical"] / "canonical_node_pair_tmc.csv",
            "--audit",
            audit / "tables" / "unmatched_links_link_by_link_reasons.csv",
            "--metadata",
            config.files["tmc_metadata"],
            "--readings",
            config.files["tmc_readings"],
            "--output-dir",
            treatments,
        ),
        treatments,
    )
    run_command(
        config,
        _module(
            "tmc_matching.build_treatment_direct_mappings",
            "--coverage-root",
            treatments,
            "--canonical",
            paths["canonical"] / "canonical_node_pair_tmc.csv",
        ),
        output / "direct-mapping",
    )
    maps = output / "period-maps"
    run_command(
        config,
        _module(
            "tmc_matching.build_treatment_period_maps",
            "--coverage-root",
            treatments,
            "--mapmatching-run",
            paths["matching"],
            "--network-root",
            config.files["base_network"],
            "--output-dir",
            maps,
        ),
        maps,
    )
    return {
        "treatment_manifest": str(
            treatments / "manifests" / "treatment_manifest.json"
        ),
        "combined_canonical_map": str(maps / "canonical_node_pair_tmc.csv"),
        "virtual_cbi_inputs": str(treatments / "virtual" / "cbi-corridors"),
    }


def _cbi_command(
    config: PipelineConfig,
    input_root: Path,
    mapping: Path,
    output: Path,
) -> list[str]:
    settings = config.section("cbi")
    command = _module(
        "cbi.cli",
        "run",
        "--input-dir",
        input_root,
        "--model-link-map",
        mapping,
        "--output-dir",
        output,
        "--workers",
        config.workers,
    )
    if not settings.get("generate_figures", True):
        command.append("--no-figures")
    _option(command, "--settings", config.repository_path(settings.get("settings")))
    return command


def run_cbi(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    treatments = paths["coverage"] / "treatments"
    actual_map = (
        treatments
        / "actual"
        / "combined-direct-mapping"
        / "actual_tmc_to_link.csv"
    )
    virtual_inputs = treatments / "virtual" / "cbi-corridors"
    virtual_map = treatments / "virtual" / "virtual_tmc_to_link.csv"
    actual_output = output / "actual"
    virtual_output = output / "virtual"
    run_command(
        config,
        _cbi_command(
            config,
            paths["corridors"] / "corridors",
            actual_map,
            actual_output,
        ),
        actual_output,
    )
    run_command(
        config,
        _cbi_command(config, virtual_inputs, virtual_map, virtual_output),
        virtual_output,
    )
    combined_outputs = output / "combined-corridors"
    combined_inputs = output / "combined-inputs"
    combined_outputs.mkdir(parents=True, exist_ok=False)
    combined_inputs.mkdir(parents=True, exist_ok=False)
    for source_root, destination_root in (
        (actual_output / "corridors", combined_outputs),
        (virtual_output / "corridors", combined_outputs),
        (paths["corridors"] / "corridors", combined_inputs),
        (virtual_inputs, combined_inputs),
    ):
        for source in sorted(path for path in source_root.iterdir() if path.is_dir()):
            destination = destination_root / source.name
            if destination.exists():
                raise ValueError(f"Actual/virtual corridor key collision: {source.name}")
            _copytree_hardlink(source, destination)
    return {
        "actual_cbi_corridors": str(actual_output / "corridors"),
        "virtual_cbi_corridors": str(virtual_output / "corridors"),
        "combined_cbi_corridors": str(combined_outputs),
    }


def run_spatial_t2(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    section = config.section("spatial_t2")
    if not section.get("enabled", True):
        raise ValueError("spatial_t2 cannot be disabled while Ridge completion is enabled")
    command = _module(
        "t2_coverage_expansion.cli",
        "all",
        "--config",
        config.repository_path(section.get("config")),
        "--package-root",
        config.repository_root,
        "--artifact-root",
        output,
        "--cbi-corridors",
        paths["cbi"] / "actual" / "corridors",
        "--corridor-inputs",
        paths["corridors"] / "corridors",
        "--mapmatching-root",
        paths["matching"],
        "--network-root",
        config.files["base_network"],
        "--workers",
        config.workers,
    )
    run_command(config, command, output)
    return {"expanded_t2": str(output / "outputs" / "expanded_link_t2.csv")}


def _boundary_command(config: PipelineConfig, output: Path) -> list[str]:
    paths = stage_paths(config)
    return _module(
        "congestion_boundary_mapping.cli",
        "--cbi-output-root",
        paths["cbi"] / "combined-corridors",
        "--canonical-node-pair-map",
        paths["coverage"] / "period-maps" / "canonical_node_pair_tmc.csv",
        "--am-map",
        paths["coverage"] / "period-maps" / "am_full_tmc_to_link.csv",
        "--md-map",
        paths["coverage"] / "period-maps" / "md_full_tmc_to_link.csv",
        "--pm-map",
        paths["coverage"] / "period-maps" / "pm_full_tmc_to_link.csv",
        "--network-root",
        config.files["base_network"],
        "--spatial-output",
        paths["spatial"] / "outputs" / "expanded_link_t2.csv",
        "--output-dir",
        output / "link-t2",
        "--workers",
        config.workers,
    )


def run_boundary_seed(config: PipelineConfig, output: Path) -> dict[str, str]:
    command = _boundary_command(config, output)
    command.extend(["--completion-mode", "vdf_class"])
    run_command(config, command, output)
    return {"boundary_candidates": str(output / "link-t2")}


def run_ridge(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    section = config.section("ridge")
    if not section.get("enabled", True):
        raise ValueError("Ridge retraining is required for a full run")
    output.mkdir(parents=True, exist_ok=True)
    ridge_config = {
        # The Ridge loader owns the internal `corridors/` suffix. Keep this at
        # the CBI producer root to avoid an invalid `corridors/corridors` path.
        "cbi_run_dir": str(paths["cbi"] / "actual"),
        "boundary_mapping_run_dir": str(paths["boundary_seed"]),
        "spatial_run_dir": str(paths["spatial"]),
        "output_root": str(output),
        "random_seed": int(section.get("random_seed", 42)),
        "cv_folds": int(section.get("cv_folds", 5)),
        "worker_fraction": 1.0,
        "max_workers": config.workers,
        "forest_estimators": int(section.get("forest_estimators", 250)),
        "reliable_minimum_days": 3,
        "reliable_maximum_t2_std_hours": 1.5,
        "temporal_holdout_days": 5,
        "model_names": list(section.get("model_names", ["ridge_core"])),
    }
    config_path = output / "ridge_config.json"
    config_path.write_text(json.dumps(ridge_config, indent=2) + "\n", encoding="utf-8")
    run_command(
        config,
        _module("t2_ml_experiment.cli", "--config", config_path, "--output-run-name", "model"),
        output,
    )
    comparison = output / "comparison" / "outputs"
    comparison.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        paths["spatial"] / "outputs" / "validation_predictions.csv",
        comparison / "validation_predictions.csv",
    )
    return {"ridge_model": str(output / "model")}


def run_final_boundaries(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    _copytree_hardlink(paths["boundary_seed"] / "link-t2", output / "link-t2")
    command = _boundary_command(config, output)
    command.extend(
        [
            "--completion-only",
            "--completion-mode",
            "ml",
            "--ml-run-dir",
            str(paths["ridge"] / "model"),
            "--comparison-run-dir",
            str(paths["ridge"] / "comparison"),
        ]
    )
    run_command(config, command, output)
    return {
        "boundary_lookup": str(output / "link-boundaries" / "node_pair_lookup")
    }


def run_network_qvdf(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    section = config.section("cbi")
    for scope in ("actual", "virtual"):
        destination = output / scope
        command = _module(
            "cbi.network_qvdf",
            "--cbi-run-dir",
            paths["cbi"] / scope,
            "--output-dir",
            destination,
            "--network-root",
            config.files["base_network"],
            "--minimum-episodes",
            section.get("minimum_network_qvdf_episodes", 3),
        )
        if scope == "virtual":
            command.extend(["--observed-triplet-policy", "omit"])
        run_command(config, command, destination)
    return {
        "actual_link_qvdf": str(output / "actual" / "daily" / "link_qvdf.csv"),
        "actual_direct_resources": str(output / "actual"),
        "virtual_direct_resources": str(output / "virtual"),
    }


def _replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    _copytree_hardlink(source, destination)


def _copytree_detached(source: Path, destination: Path) -> None:
    """Copy mutable stage seeds without linking them back to package data."""

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _merge_disjoint_lookups(
    actual_path: Path,
    virtual_path: Path,
    destination: Path,
    *,
    scope: str,
    lineage_root: Path,
    metadata_fields: Mapping[str, str],
) -> None:
    actual = np.load(actual_path, allow_pickle=False)
    virtual = np.load(virtual_path, allow_pickle=False)
    if actual.dtype != virtual.dtype:
        raise ValueError(f"Actual/virtual {scope} lookup dtypes differ")
    for label, values in (("actual", actual), ("virtual", virtual)):
        if len(values) and np.any(np.diff(np.sort(values["packed_key"])) <= 0):
            raise ValueError(f"{label} {scope} lookup contains duplicate node pairs")
    overlap = np.intersect1d(actual["packed_key"], virtual["packed_key"])
    if len(overlap):
        raise ValueError(
            f"Actual and virtual {scope} lookups overlap on {len(overlap):,} pairs"
        )
    combined = np.concatenate([actual, virtual])
    combined.sort(order="packed_key")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, combined, allow_pickle=False)
    try:
        actual_source = actual_path.resolve().relative_to(
            lineage_root.resolve()
        ).as_posix()
        virtual_source = virtual_path.resolve().relative_to(
            lineage_root.resolve()
        ).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{scope} sources must be inside the declared lineage root"
        ) from exc
    metadata = {
        "status": "PASS",
        "scope": scope,
        "precedence": ["actual direct", "virtual direct"],
        "actual_rows": int(len(actual)),
        "virtual_rows": int(len(virtual)),
        "combined_rows": int(len(combined)),
        "overlapping_node_pairs": 0,
        "actual_source": actual_source,
        "virtual_source": virtual_source,
        **metadata_fields,
    }
    (destination.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def run_resource_bundle(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    packaged = (
        config.repository_root
        / "nvta-taplite-workflow"
        / "src"
        / "dtalite4cube"
        / "resources"
    )
    _copytree_detached(packaged, output)
    actual = paths["qvdf"] / "actual"
    virtual = paths["qvdf"] / "virtual"
    shutil.copy2(actual / "daily" / "link_qvdf.csv", output / "link_qvdf.csv")
    _merge_disjoint_lookups(
        actual / "observed-link-plf" / "observed_link_plf_overrides.npy",
        virtual / "observed-link-plf" / "observed_link_plf_overrides.npy",
        output / "observed_link_plf_lookup" / "observed_link_plf_overrides.npy",
        scope="observed PLF",
        lineage_root=config.output_root,
        metadata_fields={"unit": "dimensionless"},
    )
    _merge_disjoint_lookups(
        actual
        / "observed-link-speed-boundaries"
        / "observed_link_speed_boundaries.npy",
        virtual
        / "observed-link-speed-boundaries"
        / "observed_link_speed_boundaries.npy",
        output
        / "observed_link_speed_boundary_lookup"
        / "observed_link_speed_boundaries.npy",
        scope="observed speed boundaries",
        lineage_root=config.output_root,
        metadata_fields={"speed_unit": "mph"},
    )
    _merge_disjoint_lookups(
        actual / "observed-link-t2" / "observed_link_t2.npy",
        virtual / "observed-link-t2" / "observed_link_t2.npy",
        output / "observed_link_t2_lookup" / "observed_link_t2.npy",
        scope="observed T0/T2/T3",
        lineage_root=config.output_root,
        metadata_fields={"time_unit": "decimal hour"},
    )
    _replace_directory(
        paths["boundaries"] / "link-boundaries" / "node_pair_lookup",
        output / "congestion_t_node_pair_lookup",
    )
    override = config.files.get("qvdf_override_dictionary")
    override_root = output / "qvdf_parameter_override_lookup"
    if override is None and override_root.exists():
        shutil.rmtree(override_root)
    elif override is not None:
        override_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(override, override_root / "qvdf_node_pair_overrides.npy")
    manifest = {
        "status": "PASS",
        "resource_root": str(output),
        "link_qvdf": str(actual / "daily" / "link_qvdf.csv"),
        "network_qvdf_scope": "actual only",
        "direct_observation_scope": "actual plus virtual, disjoint node pairs",
        "observed_plf": str(output / "observed_link_plf_lookup"),
        "observed_speed_boundaries": str(
            output / "observed_link_speed_boundary_lookup"
        ),
        "observed_t2": str(output / "observed_link_t2_lookup"),
        "completed_boundaries": str(
            paths["boundaries"] / "link-boundaries" / "node_pair_lookup"
        ),
        "qvdf_node_pair_override": str(override) if override else None,
    }
    (output / "resource_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"resource_root": str(output)}


def _assignment_command(config: PipelineConfig, output: Path, anchors: Path) -> list[str]:
    taplite = config.section("taplite")
    override = config.files.get("qvdf_override_dictionary")
    command = [
        sys.executable,
        str(config.repository_root / "nvta-taplite-workflow" / "run_assignment.py"),
        str(config.files["cube_scenario"]),
        "--iterations",
        str(taplite.get("iterations", 10)),
        "--processors",
        str(config.workers),
        "--route-output",
        "0",
        "--vehicle-output",
        "0",
        "--vdf-type",
        "qvdf",
        "--network-conversion",
        "true",
        "--demand-conversion",
        "true",
        "--dtalite-assignment",
        "true",
        "--conversion-workers",
        str(config.workers),
        "--conversion-reserve-cores",
        "0",
        "--conversion-adaptive",
        "false",
        "--conversion-cache",
        str(bool(taplite.get("conversion_cache", True))).lower(),
        "--conversion-cache-dir",
        str(output / "conversion-cache"),
        "--observed-speed-boundary-lookup-directory",
        str(anchors),
        "--qvdf-profile-mode",
        str(taplite.get("qvdf_profile_mode", 2)),
        "--qvdf-parameter-override",
        str(override is not None).lower(),
        "--qvdf-smoothing",
        str(bool(taplite.get("qvdf_smoothing", True))).lower(),
        "--qvdf-smoothing-workers",
        str(config.workers),
        "--output-dir",
        str(output / "assignment"),
        "--time-periods",
        *PERIODS,
        "--period-times",
        *(config.periods[period] for period in PERIODS),
        "--kernel-source",
        str(taplite.get("kernel_source", "pypi")),
    ]
    if override is not None:
        command.extend(["--qvdf-override-dictionary", str(override)])
    return command


def _taplite_environment(config: PipelineConfig) -> dict[str, str]:
    env = python_environment(config)
    env["NVTA_TAPLITE_RESOURCE_ROOT"] = str(stage_paths(config)["resources"])
    return env


def run_assignment_stage1(config: PipelineConfig, output: Path) -> dict[str, str]:
    anchors = stage_paths(config)["resources"] / "observed_link_speed_boundary_lookup"
    run_command(
        config,
        _assignment_command(config, output, anchors),
        output,
        environment=_taplite_environment(config),
    )
    return {"assignment_root": str(output / "assignment")}


def run_hybrid_anchors(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    command = [
        sys.executable,
        str(
            config.repository_root
            / "nvta-taplite-workflow"
            / "generate_hybrid_speed_boundaries.py"
        ),
        "--canonical",
        str(paths["coverage"] / "period-maps" / "canonical_node_pair_tmc.csv"),
        "--regional-readings",
        str(config.files["tmc_readings"]),
        "--stable-assignment",
        str(paths["assignment1"] / "assignment"),
        "--existing-observed-lookup",
        str(
            paths["resources"]
            / "observed_link_speed_boundary_lookup"
            / "observed_link_speed_boundaries.npy"
        ),
        "--virtual-corridor-inputs",
        str(
            paths["coverage"]
            / "treatments"
            / "virtual"
            / "cbi-corridors"
        ),
        "--output-dir",
        str(output),
    ]
    run_command(config, command, output, environment=_taplite_environment(config))
    return {"hybrid_anchor_lookup": str(output / "observed_link_speed_boundaries.npy")}


def run_assignment_stage2(config: PipelineConfig, output: Path) -> dict[str, str]:
    run_command(
        config,
        _assignment_command(config, output, stage_paths(config)["anchors"]),
        output,
        environment=_taplite_environment(config),
    )
    return {"assignment_root": str(output / "assignment")}


def _dashboard_base(config: PipelineConfig, command_name: str, output: Path) -> list[str]:
    paths = stage_paths(config)
    command = _module(
        "corridor_performance_dashboard.cli",
        command_name,
        "--cbi-corridors",
        paths["cbi"] / "actual" / "corridors",
        "--mapmatching-root",
        paths["matching"],
        "--assignment-root",
        paths["assignment2"] / "assignment",
        "--observed-15min",
        config.files["tmc_readings"],
        "--model-link-map",
        paths["canonical"] / "canonical_node_pair_tmc.csv",
        "--workers",
        config.workers,
    )
    dashboard = config.section("dashboard")
    settings = config.repository_path(dashboard.get("settings"))
    _option(command, "--settings", settings)
    override = config.files.get("profile_selection_overrides")
    _option(command, "--profile-selection-overrides", override)
    return command


def run_measurement(config: PipelineConfig, output: Path) -> dict[str, str]:
    command = _dashboard_base(config, "measure", output)
    command.extend(["--measurement-output", str(output)])
    run_command(config, command, output)
    return {"measurement_root": str(output)}


def run_dashboard(config: PipelineConfig, output: Path) -> dict[str, str]:
    paths = stage_paths(config)
    command = _dashboard_base(config, "dashboard", output)
    command.extend(
        [
            "--measurement-root",
            str(paths["measurement"]),
            "--dashboard-output",
            str(output),
            "--worker-fraction",
            str(config.section("dashboard").get("worker_fraction", 0.5)),
            "--force-dashboard",
        ]
    )
    run_command(config, command, output)
    return {"dashboard": str(output / "index.html")}


@dataclass(frozen=True)
class Stage:
    key: str
    output_key: str
    description: str
    runner: Callable[[PipelineConfig, Path], dict[str, str]]
    required_outputs: tuple[str, ...]


STAGES = (
    Stage("matching", "matching", "period-aware TMC/network matching", run_matching, ("combined/full_tmc_to_link.csv", "am/full_tmc_to_link.csv", "md/full_tmc_to_link.csv", "pm/full_tmc_to_link.csv")),
    Stage("corridors", "corridors", "raw observed corridor speed slices", run_corridor_slices, ("corridors",)),
    Stage("canonical", "canonical", "frozen composite-ranked node-pair winners", run_canonical, ("canonical_node_pair_tmc.csv", "canonical_mapping_manifest.json")),
    Stage("coverage", "coverage", "actual/managed/virtual observation coverage treatments", run_observation_coverage, ("treatments/manifests/treatment_manifest.json", "period-maps/canonical_node_pair_tmc.csv", "period-maps/manifest.json")),
    Stage("cbi", "cbi", "isolated actual and virtual corridor CBI calibration", run_cbi, ("actual/corridors", "actual/run_manifest.json", "virtual/corridors", "virtual/run_manifest.json", "combined-corridors")),
    Stage("spatial-t2", "spatial", "direct/spatial T2 coverage and validation", run_spatial_t2, ("outputs/expanded_link_t2.csv", "input-snapshot/route_summary.csv")),
    Stage("boundary-seed", "boundary_seed", "direct/spatial boundary candidates", run_boundary_seed, ("link-t2/run_manifest.json", "link-t2/period_link_files")),
    Stage("ridge", "ridge", "fresh leakage-controlled Ridge training", run_ridge, ("model/experimental_network_boundaries.csv", "model/metrics/out_of_fold_predictions.csv")),
    Stage("boundaries", "boundaries", "direct-spatial-Ridge final boundaries", run_final_boundaries, ("link-boundaries/node_pair_lookup/metadata.json",)),
    Stage("network-qvdf", "qvdf", "actual network QVDF plus isolated actual/virtual direct resources", run_network_qvdf, ("actual/daily/link_qvdf.csv", "actual/observed-link-plf/observed_link_plf_overrides.npy", "virtual/observed-link-plf/observed_link_plf_overrides.npy", "actual/observed-link-speed-boundaries/observed_link_speed_boundaries.npy", "virtual/observed-link-speed-boundaries/observed_link_speed_boundaries.npy", "actual/observed-link-t2/observed_link_t2.npy", "virtual/observed-link-t2/observed_link_t2.npy")),
    Stage("resources", "resources", "isolated treatment-aware TAPlite resource bundle", run_resource_bundle, ("link_qvdf.csv", "resource_bundle_manifest.json", "observed_link_plf_lookup/observed_link_plf_overrides.npy", "observed_link_speed_boundary_lookup/observed_link_speed_boundaries.npy", "observed_link_t2_lookup/observed_link_t2.npy", "congestion_t_node_pair_lookup/metadata.json")),
    Stage("assignment-1", "assignment1", "first TAPlite conversion and assignment", run_assignment_stage1, ("assignment/am/link_performance.csv", "assignment/md/link_performance.csv", "assignment/pm/link_performance.csv")),
    Stage("hybrid-anchors", "anchors", "actual/virtual-plus-stage-1 hybrid anchors", run_hybrid_anchors, ("observed_link_speed_boundaries.npy", "metadata.json")),
    Stage("assignment-2", "assignment2", "final hybrid-anchor TAPlite run", run_assignment_stage2, ("assignment/am/link_performance.csv", "assignment/md/link_performance.csv", "assignment/pm/link_performance.csv")),
    Stage("measurement", "measurement", "corridor profile measurement", run_measurement, ("07-run-metadata/run_manifest.json",)),
    Stage("dashboard", "dashboard", "integrated dashboard", run_dashboard, ("index.html",)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()
