from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .logging_utils import configure_logging
from .qa import (
    InputQAError,
    normalized_inputs,
    resolve_match_inputs,
    validate_match_inputs,
    write_qa_report,
)


LOGGER = logging.getLogger("tmc_matching")
PERIODS = ("am", "md", "pm")


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tmc-file-name", default="TMC_Identification.csv")
    parser.add_argument("--network-dir-name", default="network")
    parser.add_argument("--link-file-name", default="link.csv")
    parser.add_argument("--node-file-name", default="node.csv")
    parser.add_argument(
        "--column-map",
        type=Path,
        help="JSON mapping canonical TMC/link/node fields to source field names.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmc-matching",
        description="QA and run period-aware TMC route/path matching.",
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    qa = commands.add_parser("qa", help="Validate inputs without matching.")
    _common_inputs(qa)
    qa.add_argument("--report-dir", type=Path)

    run = commands.add_parser("run", help="Validate inputs and run matching.")
    _common_inputs(run)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--source-crs-epsg", type=int, default=4326)
    run.add_argument("--working-crs-epsg", type=int, default=2248)
    run.add_argument("--network-label", default="period-aware GMNS network")
    run.add_argument("--lane-class", choices=["auto", "all_open", "gp", "managed"], default="auto")
    run.add_argument("--road", action="append")
    run.add_argument("--max-corridors", type=int, default=0)
    run.add_argument("--write-candidates", action="store_true")
    run.add_argument("--no-period-products", action="store_true")
    run.add_argument("--combined-product-name", default="combined")
    run.add_argument("--period-product-template", default="{period}")
    return parser


def _inputs(args: argparse.Namespace):
    return resolve_match_inputs(
        args.input_dir,
        tmc_file_name=args.tmc_file_name,
        network_dir_name=args.network_dir_name,
        periods=PERIODS,
        link_file_name=args.link_file_name,
        node_file_name=args.node_file_name,
        column_map_path=args.column_map,
    )


def _default_output(input_dir: Path) -> Path:
    return input_dir.resolve() / "outputs" / "tmc-matching"


def _qa_command(args: argparse.Namespace) -> int:
    inputs = _inputs(args)
    result = validate_match_inputs(inputs)
    report_root = (args.report_dir or _default_output(inputs.input_dir)).resolve()
    log_path = configure_logging(report_root, args.verbose)
    report = write_qa_report(result, report_root)
    LOGGER.info("Input QA passed (%s)", report)
    LOGGER.debug("Detailed log: %s", log_path)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    inputs = _inputs(args)
    output_root = (args.output_dir or _default_output(inputs.input_dir)).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise InputQAError(
            f"Output directory is not empty: {output_root}. "
            "Choose another --output-dir or archive the existing result."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root, args.verbose)
    LOGGER.info("Checking matching inputs")
    qa_result = validate_match_inputs(inputs)
    qa_report = write_qa_report(qa_result, output_root)
    prepared = normalized_inputs(inputs, output_root)
    LOGGER.info("Input QA passed; starting route/path matching")

    from .run_tmc_mapmatching import main as engine_main

    engine_args = [
        "--output-root", str(output_root),
        "--combined-product-name", args.combined_product_name,
        "--period-product-template", args.period_product_template,
        "--tmc-file", str(prepared.tmc_file),
        "--am-link-file", str(prepared.period_link_files["am"]),
        "--md-link-file", str(prepared.period_link_files["md"]),
        "--pm-link-file", str(prepared.period_link_files["pm"]),
        "--am-node-file", str(prepared.period_node_files["am"]),
        "--md-node-file", str(prepared.period_node_files["md"]),
        "--pm-node-file", str(prepared.period_node_files["pm"]),
        "--source-crs-epsg", str(args.source_crs_epsg),
        "--model-crs-epsg", str(args.working_crs_epsg),
        "--network-label", args.network_label,
        "--lane-class", args.lane_class,
        "--max-corridors", str(args.max_corridors),
    ]
    for road in args.road or ():
        engine_args.extend(["--road", road])
    if args.write_candidates:
        engine_args.append("--write-candidates")
    if args.no_period_products:
        engine_args.append("--no-period-products")

    with (output_root / "logs" / "engine.log").open("w", encoding="utf-8") as detail:
        with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
            engine_main(engine_args)
    manifest = {
        "status": "PASS",
        "producer": "tmc-matching",
        "input_root": str(inputs.input_dir),
        "output_root": str(output_root),
        "qa_report": str(qa_report),
        "detailed_log": str(log_path),
        "products": {
            "combined": args.combined_product_name,
            "period_template": args.period_product_template,
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Matching complete: %s", output_root)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _qa_command(args) if args.command == "qa" else _run_command(args)
    except (InputQAError, FileNotFoundError, ValueError) as exc:
        logging.getLogger().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

