from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from .logging_utils import configure_logging
from .qa import (
    InputQAError,
    Inputs,
    load_column_map,
    normalize_layout,
    validate,
    write_report,
)


LOGGER = logging.getLogger("corridor_performance_dashboard")


def _inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cbi-corridors", type=Path, required=True)
    parser.add_argument("--mapmatching-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--am-product", default="am")
    parser.add_argument("--md-product", default="md")
    parser.add_argument("--pm-product", default="pm")
    parser.add_argument("--dashboard-product", default="combined")
    parser.add_argument("--mapping-file-name", default="full_tmc_to_link.csv")
    parser.add_argument("--route-summary-file-name", default="full_route_match_summary.csv")
    parser.add_argument("--performance-file-name", default="link_performance.csv")
    parser.add_argument("--link-file-name", default="link.csv")
    parser.add_argument("--column-map", type=Path,
                        help="JSON canonical-to-source column aliases by input group.")
    parser.add_argument("--observed-15min", type=Path)
    parser.add_argument("--model-link-map", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corridor-dashboard",
        description="QA, measure corridor profiles, and build the integrated static dashboard.",
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    qa = commands.add_parser("qa", help="Validate a requested process and exit.")
    _inputs(qa)
    qa.add_argument("--process", choices=["measure", "dashboard", "all"], default="all")
    qa.add_argument("--measurement-root", type=Path)
    qa.add_argument("--report-dir", type=Path)
    for name in ("measure", "dashboard", "all"):
        command = commands.add_parser(name)
        _inputs(command)
        command.add_argument("--measurement-root", type=Path)
        command.add_argument("--measurement-output", type=Path)
        command.add_argument("--dashboard-output", type=Path)
        command.add_argument("--settings", type=Path,
                             help="Analysis JSON; defaults to the packaged portable settings.")
        command.add_argument("--workers", type=int)
        command.add_argument("--worker-fraction", type=float, default=0.50)
        command.add_argument("--profile-selection-overrides", type=Path)
        command.add_argument("--force-dashboard", action="store_true")
    return parser


def _raw_inputs(args: argparse.Namespace, measurement_root: Path | None = None) -> Inputs:
    return Inputs(
        cbi_corridors=args.cbi_corridors.resolve(),
        mapmatching_root=args.mapmatching_root.resolve(),
        assignment_root=args.assignment_root.resolve(),
        mapping_products={"am": args.am_product, "md": args.md_product, "pm": args.pm_product},
        dashboard_product=args.dashboard_product,
        mapping_file_name=args.mapping_file_name,
        route_summary_file_name=args.route_summary_file_name,
        performance_file_name=args.performance_file_name,
        link_file_name=args.link_file_name,
        observed_15min=args.observed_15min.resolve() if args.observed_15min else None,
        model_link_map=args.model_link_map.resolve() if args.model_link_map else None,
        measurement_root=measurement_root,
    )


def _measurement_output(args: argparse.Namespace) -> Path:
    return (args.measurement_output or (
        args.assignment_root.resolve() / "outputs" / "corridor-performance-dashboard" / "measurement"
    )).resolve()


def _dashboard_output(args: argparse.Namespace, measurement: Path) -> Path:
    return (args.dashboard_output or (
        measurement / "outputs" / "integrated-dashboard"
    )).resolve()


def _effective_settings(args: argparse.Namespace, output: Path) -> Path:
    source = args.settings
    if source is None:
        payload = json.loads(
            files("corridor_performance_dashboard").joinpath("config/default.json").read_text(encoding="utf-8")
        )
    else:
        if not source.is_file():
            raise InputQAError(f"Settings JSON does not exist: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
    payload["mapping_products"] = {
        "am": args.am_product, "md": args.md_product, "pm": args.pm_product,
    }
    payload["results_root"] = str(output)
    target = output / "qa" / "effective_settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _run_measurement(args: argparse.Namespace, inputs: Inputs, output: Path) -> Path:
    from corridor_measurement.pipeline import run_measurement
    settings = _effective_settings(args, output)
    with (output / "logs" / "measurement-engine.log").open("w", encoding="utf-8") as detail:
        with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
            return run_measurement(
                config_path=settings, output_dir=output,
                cbi_corridors_dir=inputs.cbi_corridors,
                mapmatching_run_dir=inputs.mapmatching_root,
                taplite_assignment_dir=inputs.assignment_root,
                workers=args.workers,
            )


def _run_dashboard(args: argparse.Namespace, inputs: Inputs, measurement: Path, output: Path) -> Path:
    from corridor_dashboard.pipeline import run_integrated_dashboard
    product = inputs.mapmatching_root / inputs.dashboard_product
    with (output / "logs" / "dashboard-engine.log").open("w", encoding="utf-8") as detail:
        with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
            run_integrated_dashboard(
                package_root=output.parent,
                corridor_results_root=inputs.cbi_corridors,
                mapmatching_product_root=product,
                model_link_map_path=inputs.model_link_map,
                assignment_root=inputs.assignment_root,
                ritis_15min_path=inputs.observed_15min,
                profile_selection_overrides_path=args.profile_selection_overrides,
                corridor_measurement_root=measurement,
                output_root=output,
                workers=args.workers,
                worker_fraction=args.worker_fraction,
                force=args.force_dashboard,
            )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "qa":
            report_root = (args.report_dir or args.assignment_root.resolve() / "outputs" / "corridor-performance-dashboard" / "qa").resolve()
            configure_logging(report_root, args.verbose)
            prepared = normalize_layout(_raw_inputs(args, args.measurement_root), report_root, load_column_map(args.column_map))
            report = write_report(validate(prepared, args.process), report_root)
            LOGGER.info("Input QA passed (%s)", report)
            return 0

        measurement = (args.measurement_root or _measurement_output(args)).resolve()
        primary_output = measurement if args.command == "measure" else _dashboard_output(args, measurement)
        primary_output.mkdir(parents=True, exist_ok=True)
        log = configure_logging(primary_output, args.verbose)
        prepared = normalize_layout(_raw_inputs(args, measurement), primary_output, load_column_map(args.column_map))

        if args.command in {"measure", "all"}:
            LOGGER.info("Checking corridor-measurement inputs")
            write_report(validate(prepared, "measure"), measurement)
            LOGGER.info("Running corridor measurement")
            measurement = _run_measurement(args, prepared, measurement)

        if args.command in {"dashboard", "all"}:
            dashboard = _dashboard_output(args, measurement)
            dashboard.mkdir(parents=True, exist_ok=True)
            if primary_output != dashboard:
                configure_logging(dashboard, args.verbose)
            dashboard_inputs = Inputs(
                cbi_corridors=prepared.cbi_corridors,
                mapmatching_root=prepared.mapmatching_root,
                assignment_root=prepared.assignment_root,
                mapping_products=prepared.mapping_products,
                dashboard_product=prepared.dashboard_product,
                observed_15min=prepared.observed_15min,
                model_link_map=prepared.model_link_map, measurement_root=measurement,
            )
            LOGGER.info("Checking dashboard inputs")
            write_report(validate(dashboard_inputs, "dashboard"), dashboard)
            LOGGER.info("Building integrated dashboard")
            _run_dashboard(args, dashboard_inputs, measurement, dashboard)
            LOGGER.info("Dashboard complete: %s", dashboard)
        else:
            LOGGER.info("Measurement complete: %s", measurement)
        LOGGER.debug("Detailed log: %s", log)
        return 0
    except (InputQAError, FileNotFoundError, FileExistsError, ValueError, KeyError) as exc:
        logging.getLogger().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
