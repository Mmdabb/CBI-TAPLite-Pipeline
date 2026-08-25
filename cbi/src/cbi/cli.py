from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from .batch import run_batch
from .config import PipelineSettings
from .logging_utils import configure_logging
from .qa import (
    CBIInputContract,
    InputQAError,
    load_column_map,
    normalize_inputs,
    validate_inputs,
    write_report,
)


LOGGER = logging.getLogger("cbi")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing one folder per corridor.")
    parser.add_argument("--model-link-map", type=Path, required=True,
                        help="Frozen canonical TMC/link mapping CSV.")
    parser.add_argument("--metadata-file-name", default="TMC_Identification.csv")
    parser.add_argument("--readings-file-name", default="Readings.csv")
    parser.add_argument("--column-map", type=Path,
                        help="JSON mapping canonical metadata/readings/mapping fields to source columns.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbi", description="QA and run corridor bottleneck identification/QVDF calibration."
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    qa = commands.add_parser("qa", help="Validate inputs and exit.")
    _common(qa)
    qa.add_argument("--report-dir", type=Path)
    run = commands.add_parser("run", help="Validate inputs and run CBI.")
    _common(run)
    run.add_argument("--output-dir", type=Path,
                     help="Stable run root (default: <input-dir>/outputs/cbi).")
    run.add_argument("--settings", type=Path,
                     help="Optional JSON values for PipelineSettings.")
    run.add_argument("--corridor", action="append",
                     help="Corridor folder to run; repeat. Default: all.")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--no-figures", action="store_true")
    return parser


def _contract(args: argparse.Namespace) -> CBIInputContract:
    return CBIInputContract(
        args.input_dir.resolve(), args.model_link_map.resolve(),
        args.metadata_file_name, args.readings_file_name,
        load_column_map(args.column_map),
    )


def _settings(path: Path | None) -> PipelineSettings:
    if path is None:
        return PipelineSettings()
    if not path.is_file():
        raise InputQAError(f"Settings JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(PipelineSettings)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise InputQAError("Unknown PipelineSettings keys: " + ", ".join(unknown))
    if "periods" in payload:
        payload["periods"] = {
            str(key): tuple(int(value) for value in values)
            for key, values in payload["periods"].items()
        }
    if "wide_window" in payload:
        payload["wide_window"] = tuple(int(value) for value in payload["wide_window"])
    return PipelineSettings(**payload)


def _default_output(input_dir: Path) -> Path:
    return input_dir.resolve() / "outputs" / "cbi"


def _qa(args: argparse.Namespace) -> int:
    contract = _contract(args)
    output = (args.report_dir or _default_output(contract.corridor_root)).resolve()
    configure_logging(output, args.verbose)
    report = write_report(validate_inputs(contract), output)
    LOGGER.info("Input QA passed (%s)", report)
    return 0


def _run(args: argparse.Namespace) -> int:
    contract = _contract(args)
    output = (args.output_dir or _default_output(contract.corridor_root)).resolve()
    if output.exists() and any(output.iterdir()):
        raise InputQAError(
            f"Output directory is not empty: {output}. Choose another --output-dir "
            "or archive the prior result."
        )
    output.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output, args.verbose)
    LOGGER.info("Checking CBI inputs")
    qa_report = write_report(validate_inputs(contract), output)
    corridor_root, model_map = normalize_inputs(contract, output)
    LOGGER.info("Input QA passed; running CBI")
    with (output / "logs" / "engine.log").open("w", encoding="utf-8") as detail:
        with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
            result = run_batch(
                corridor_root,
                output / "corridors",
                output / "summary",
                keys=args.corridor,
                generate_figures=not args.no_figures,
                workers=max(1, args.workers),
                model_link_map=model_map,
                settings=_settings(args.settings),
            )
    manifest = {
        "status": "PASS", "producer": "cbi", "input_root": str(contract.corridor_root),
        "model_link_map": str(contract.model_link_map), "output_root": str(output),
        "qa_report": str(qa_report), "detailed_log": str(log_path),
        "batch_manifest": result["batch_manifest"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("CBI complete: %s", output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _qa(args) if args.command == "qa" else _run(args)
    except (InputQAError, FileNotFoundError, ValueError) as exc:
        logging.getLogger().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
