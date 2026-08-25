from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigurationError, load_config
from .qa import InputQAError, validate, write_report
from .runner import run_pipeline, stage_table
from .stages import STAGES


LOGGER = logging.getLogger("cbi_taplite_pipeline")


def _configure_logging(output_root: Path, verbose: bool) -> Path:
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "pipeline.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    terminal = logging.StreamHandler()
    terminal.setLevel(logging.DEBUG if verbose else logging.INFO)
    terminal.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    detail = logging.FileHandler(log_path, encoding="utf-8")
    detail.setLevel(logging.DEBUG)
    detail.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(terminal)
    root.addHandler(detail)
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbi-taplite-pipeline",
        description="Run the complete TMC matching, CBI, TAPlite, and dashboard workflow.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/nvta.json"))
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Show the deterministic stage graph and outputs.")
    commands.add_parser("qa", help="Validate every source needed by a full run.")
    commands.add_parser("status", help="Show completion and lineage state.")
    run = commands.add_parser("run", help="Execute the full workflow or a bounded stage range.")
    choices = [stage.key for stage in STAGES]
    run.add_argument("--from-stage", choices=choices)
    run.add_argument("--through-stage", choices=choices)
    run.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a stage only when its config and upstream manifest hashes match.",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected non-reusable stage folders inside output_root.",
    )
    return parser


def _print_table(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            f"{int(row['number']):02d}  {str(row['stage']):18s} "
            f"{str(row['status']):7s}  {row['description']}\n"
            f"    {row['output']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        _configure_logging(config.output_root, args.verbose)
        if args.command in {"plan", "status"}:
            _print_table(stage_table(config))
            return 0
        report = validate(config)
        report_path = write_report(config, report)
        LOGGER.info("Input QA passed: %s", report_path)
        if args.command == "qa":
            print(json.dumps(report, indent=2))
            return 0
        state = run_pipeline(
            config,
            from_stage=args.from_stage,
            through_stage=args.through_stage,
            resume=args.resume,
            force=args.force,
        )
        LOGGER.info("Pipeline complete: %s", config.output_root)
        print(json.dumps(state, indent=2))
        return 0
    except (
        ConfigurationError,
        InputQAError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
        KeyError,
    ) as exc:
        logging.getLogger().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

