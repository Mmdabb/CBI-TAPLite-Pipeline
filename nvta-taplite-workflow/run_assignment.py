from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.dtalite4cube.reproducible_run import DEFAULT_KERNEL_SOURCE, VALID_KERNEL_SOURCES
from src.dtalite4cube.run_logging import install_root_log_capture

from src.dtalite4cube.settings.dtalite_settings_config import (
    SUPPORTED_MODE_TYPES,
    demand_file_name,
)
from src.dtalite4cube.dtab import demand_binary_path
from src.dtalite4cube.settings.generate_dtalite_settings import normalize_period_key

if TYPE_CHECKING:
    from src.dtalite4cube.runner import AssignmentConfig


logger = logging.getLogger(__name__)

DEFAULT_TIME_PERIODS = ["am", "md", "pm", "nt"]
DEFAULT_PERIOD_TIMES = ["0600_0900", "0900_1500", "1500_1900", "1900_0600"]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_list(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def existing_file(*paths: Path) -> bool:
    return any(path.is_file() for path in paths)


def validate_prepared_period_inputs(config: AssignmentConfig) -> None:
    if not config.dtalite_assignment:
        return

    for raw_period in config.active_time_periods:
        period = normalize_period_key(raw_period)
        period_dir = config.network_path / period

        if not config.network_conversion:
            if not existing_file(period_dir / "node.csv", config.network_path / "node.csv"):
                raise FileNotFoundError(
                    f"Missing node.csv for {period}. Expected {period_dir / 'node.csv'} "
                    f"or {config.network_path / 'node.csv'}."
                )
            if not existing_file(
                period_dir / "link.csv",
                config.network_path / f"link_{period}.csv",
                config.network_path / "link.csv",
            ):
                raise FileNotFoundError(
                    f"Missing link.csv for {period}. Expected {period_dir / 'link.csv'}, "
                    f"{config.network_path / f'link_{period}.csv'}, or {config.network_path / 'link.csv'}."
                )

        if not config.demand_conversion:
            missing = []
            for mode in SUPPORTED_MODE_TYPES:
                demand_name = demand_file_name(mode, period)
                candidates = [
                    period_dir / demand_name,
                    config.network_path / demand_name,
                    period_dir / "demand.csv",
                    config.network_path / "demand.csv",
                ]
                if config.demand_output_format in {"binary", "both"}:
                    candidates = [demand_binary_path(path) for path in candidates]
                if not existing_file(*candidates):
                    missing.append(demand_name)

            if missing:
                raise FileNotFoundError(
                    f"Missing demand file(s) for {period}: {', '.join(missing)}. "
                    f"Expected mode demand CSVs in {period_dir} or {config.network_path}."
                )


def validate_assignment_inputs(config: AssignmentConfig) -> None:
    if not config.network_path.exists():
        raise FileNotFoundError(f"DTALite working directory does not exist: {config.network_path}")
    if not config.network_path.is_dir():
        raise NotADirectoryError(f"DTALite working path is not a directory: {config.network_path}")
    if len(config.active_time_periods) != len(config.period_times):
        raise ValueError("time_periods and period_times must have the same length.")

    validate_prepared_period_inputs(config)


def resolve_cube_paths(raw_path: str | None) -> tuple[Path, Path]:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
    else:
        path = Path.cwd().resolve()

    if path.name == "DTALite" and path.parent.name == "Outputs":
        scenario_root = path.parent.parent
        dtalite_workdir = path
        return scenario_root, dtalite_workdir

    candidate_dtalite = path / "Outputs" / "DTALite"
    if candidate_dtalite.exists():
        scenario_root = path
        dtalite_workdir = candidate_dtalite
        return scenario_root, dtalite_workdir

    scenario_root = path
    dtalite_workdir = path
    return scenario_root, dtalite_workdir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DTALite4Cube assignment for one Cube scenario folder."
    )
    parser.add_argument(
        "scenario_dir",
        nargs="?",
        default=None,
        help="Optional Cube scenario folder. If omitted, infer from current working directory.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--processors", type=int, default=20)
    parser.add_argument("--route-output", type=int, choices=[0], default=0)
    parser.add_argument("--vehicle-output", type=int, choices=[0], default=0)
    parser.add_argument(
        "--unit-system",
        choices=["imperial", "metric"],
        default="metric",
        help="Deprecated compatibility flag; TAPlite's fixed mixed-unit schema is always used.",
    )
    parser.add_argument("--vdf-type", choices=["qvdf"], default="qvdf")
    parser.add_argument("--dtalite-run-mode", choices=["assignment", "simulation"], default="assignment")
    parser.add_argument("--network-conversion", type=str_to_bool, default=True)
    parser.add_argument("--demand-conversion", type=str_to_bool, default=True)
    parser.add_argument("--dtalite-assignment", type=str_to_bool, default=True)
    parser.add_argument(
        "--conversion-workers",
        type=int,
        default=0,
        help="Automatic conversion workers, dynamically bounded by the global 20-core ceiling.",
    )
    parser.add_argument(
        "--conversion-reserve-cores",
        type=int,
        default=1,
        help="Physical CPU cores to leave available for the OS and other work.",
    )
    parser.add_argument(
        "--network-chunks",
        type=int,
        default=0,
        help="Chunks per network period; 0 selects automatically.",
    )
    parser.add_argument(
        "--demand-chunks",
        type=int,
        default=0,
        help="Row chunks per period/mode demand matrix; 0 selects automatically.",
    )
    parser.add_argument(
        "--demand-output-format",
        choices=["csv", "binary", "both"],
        default="csv",
        help="Demand conversion output: compatible CSV, fast DTAB binary, or both.",
    )
    parser.add_argument(
        "--conversion-adaptive",
        type=str_to_bool,
        default=True,
        help="Reduce or disable parallel conversion when the machine is already busy.",
    )
    parser.add_argument(
        "--conversion-cache",
        type=str_to_bool,
        default=True,
        help="Reuse a fingerprinted reprojected network and node template across runs.",
    )
    parser.add_argument(
        "--conversion-cache-dir",
        type=Path,
        default=None,
        help="Optional prepared-network cache directory; defaults under the network source folder.",
    )
    parser.add_argument(
        "--observed-speed-boundary-lookup-directory",
        type=Path,
        default=None,
        help="Optional directory containing a scenario-specific observed_link_speed_boundaries.npy.",
    )
    parser.add_argument(
        "--qvdf-profile-mode",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="QVDF profile mode written to every converted link.",
    )
    parser.add_argument(
        "--qvdf-parameter-override",
        type=str_to_bool,
        default=True,
        help="Apply the packaged calibrated node-pair QVDF dictionary after normal network mapping.",
    )
    parser.add_argument(
        "--qvdf-override-dictionary",
        type=Path,
        default=None,
        help="Optional node-pair QVDF override dictionary; defaults to the packaged stable resource.",
    )
    parser.add_argument(
        "--qvdf-smoothing",
        type=str_to_bool,
        default=True,
        help="Run the validated QVDF speed smoother after all assignments and before downstream aggregation.",
    )
    parser.add_argument(
        "--qvdf-smoothing-workers",
        type=int,
        default=0,
        help="Smoother processes; 0 reuses the assignment processor count (maximum 20).",
    )
    parser.add_argument(
        "--qvdf-smoothing-backup",
        type=str_to_bool,
        default=True,
        help="Retain the original period link_performance.csv files before smoothing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional conversion/output folder. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument("--time-periods", nargs="+", default=DEFAULT_TIME_PERIODS)
    parser.add_argument("--period-times", nargs="+", default=DEFAULT_PERIOD_TIMES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kernel-source", choices=sorted(VALID_KERNEL_SOURCES), default=DEFAULT_KERNEL_SOURCE)
    return parser


def build_config(args: argparse.Namespace) -> AssignmentConfig:
    from src.dtalite4cube.runner import AssignmentConfig

    scenario_root, dtalite_workdir = resolve_cube_paths(args.scenario_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None and not args.output_dir.is_absolute()
        else args.output_dir
    )
    conversion_cache_dir = (
        args.conversion_cache_dir.resolve()
        if args.conversion_cache_dir is not None
        and not args.conversion_cache_dir.is_absolute()
        else args.conversion_cache_dir
    )
    return AssignmentConfig(
        network_path=dtalite_workdir,
        scenario_name=scenario_root.name,
        iterations=args.iterations,
        processors=args.processors,
        route_output=args.route_output,
        vehicle_output=args.vehicle_output,
        unit_system=args.unit_system,
        vdf_type=args.vdf_type,
        dtalite_run_mode=args.dtalite_run_mode,
        network_conversion=args.network_conversion,
        demand_conversion=args.demand_conversion,
        conversion_workers=args.conversion_workers,
        conversion_reserve_cores=args.conversion_reserve_cores,
        network_chunks=args.network_chunks,
        demand_chunks=args.demand_chunks,
        demand_output_format=args.demand_output_format,
        conversion_adaptive=args.conversion_adaptive,
        conversion_cache=args.conversion_cache,
        conversion_cache_dir=conversion_cache_dir,
        observed_speed_boundary_lookup_directory=(
            args.observed_speed_boundary_lookup_directory.resolve()
            if args.observed_speed_boundary_lookup_directory is not None
            else None
        ),
        qvdf_profile_mode=args.qvdf_profile_mode,
        qvdf_parameter_override=args.qvdf_parameter_override,
        qvdf_override_dictionary=(
            args.qvdf_override_dictionary.resolve()
            if args.qvdf_override_dictionary is not None
            else None
        ),
        qvdf_smoothing=args.qvdf_smoothing,
        qvdf_smoothing_workers=args.qvdf_smoothing_workers,
        qvdf_smoothing_backup=args.qvdf_smoothing_backup,
        dtalite_assignment=args.dtalite_assignment,
        output_dir=output_dir,
        time_periods=parse_list(args.time_periods),
        period_times=parse_list(args.period_times),
        dry_run=args.dry_run,
        kernel_source=args.kernel_source,
    )


def log_config(config: AssignmentConfig, scenario_root: Path) -> None:
    logger.info("Scenario root: %s", scenario_root)
    logger.info("DTALite working directory: %s", config.network_path)
    logger.info("Scenario name: %s", config.scenario_name)
    logger.info("Time periods: %s", config.active_time_periods)
    logger.info("Period times: %s", config.period_times)
    logger.info("Iterations: %s", config.iterations)
    logger.info("Processors: %s", config.processors)
    logger.info(
        "Conversion scheduling: workers=%s reserve_cores=%s network_chunks=%s "
        "demand_chunks=%s adaptive=%s demand_output_format=%s",
        config.conversion_workers,
        config.conversion_reserve_cores,
        config.network_chunks,
        config.demand_chunks,
        config.conversion_adaptive,
        config.demand_output_format,
    )
    logger.info(
        "Conversion cache: enabled=%s directory=%s",
        config.conversion_cache,
        config.conversion_cache_dir or "<scenario default>",
    )
    logger.info("Kernel source: %s", config.kernel_source)
    logger.info(
        "Post-mapping QVDF override: enabled=%s dictionary=%s",
        config.qvdf_parameter_override,
        config.resolved_qvdf_override_dictionary,
    )
    logger.info(
        "QVDF smoothing: enabled=%s workers=%s backup=%s",
        config.qvdf_smoothing,
        config.qvdf_smoothing_workers or config.processors,
        config.qvdf_smoothing_backup,
    )
    logger.info(
        "Enabled stages: network_conversion=%s, demand_conversion=%s, dtalite_assignment=%s",
        config.network_conversion,
        config.demand_conversion,
        config.dtalite_assignment,
    )


def main() -> None:
    install_root_log_capture("run_assignment")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args()
    scenario_root, _ = resolve_cube_paths(args.scenario_dir)
    config = build_config(args)
    validate_assignment_inputs(config)
    log_config(config, scenario_root)
    from src.dtalite4cube.runner import run_assignment_pipeline

    run_assignment_pipeline(config)


if __name__ == "__main__":
    main()

