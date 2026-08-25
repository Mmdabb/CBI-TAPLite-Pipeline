"""Explicit command-line interface for corridor measurement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .runtime import configure_numerical_threads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure observed versus TAPlite corridor profiles.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Portable JSON analysis settings.")
    parser.add_argument("--cbi-corridors", type=Path, required=True)
    parser.add_argument("--mapmatching-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <assignment-root>/outputs/corridor-measurement.")
    parser.add_argument(
        "--workers", type=int,
        help="50%% of currently free logical-core capacity by default; this overrides it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_numerical_threads(1)
    from .pipeline import run_measurement

    output = (
        args.output_dir
        or args.assignment_root.resolve() / "outputs" / "corridor-measurement"
    )
    result = run_measurement(
        config_path=args.config.resolve(),
        output_dir=output,
        cbi_corridors_dir=args.cbi_corridors.resolve(),
        mapmatching_run_dir=args.mapmatching_root.resolve(),
        taplite_assignment_dir=args.assignment_root.resolve(),
        workers=args.workers,
    )
    print(f"Measurement complete: {result}")
    return 0
