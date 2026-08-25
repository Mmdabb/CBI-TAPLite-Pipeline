from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .expansion import run_expansion
from .snapshot import prepare_snapshot
from .validation import run_validation


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PACKAGE_ROOT / "outputs" / "t2-coverage-expansion"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated NVTA T2 coverage expansion prototype"
    )
    parser.add_argument(
        "command",
        choices=("prepare", "validate", "expand", "all"),
        help="Prototype stage to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "config" / "default.json",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=PACKAGE_ROOT,
        help="Source NVTA package used only by prepare",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_ROOT,
        help="Snapshot and spatial-output root for prepare/validate/expand",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Fixed process count; default uses 70%% of free logical cores",
    )
    parser.add_argument("--cbi-corridors", type=Path)
    parser.add_argument("--corridor-inputs", type=Path)
    parser.add_argument("--mapmatching-root", type=Path)
    parser.add_argument("--network-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command in {"prepare", "all"}:
        missing = [
            name
            for name, value in (
                ("--cbi-corridors", args.cbi_corridors),
                ("--corridor-inputs", args.corridor_inputs),
                ("--mapmatching-root", args.mapmatching_root),
                ("--network-root", args.network_root),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "prepare/all requires explicit inputs: " + ", ".join(missing)
            )
    config = load_config(args.config.resolve())
    artifact_root = args.artifact_root.resolve()
    results = {}
    if args.command in {"prepare", "all"}:
        results["prepare"] = prepare_snapshot(
            args.package_root.resolve(),
            artifact_root,
            config,
            explicit_workers=args.workers,
            cbi_output_root=args.cbi_corridors,
            corridor_input_root=args.corridor_inputs,
            mapmatching_run=args.mapmatching_root,
            network_root=args.network_root,
        )
    if args.command in {"validate", "all"}:
        results["validate"] = run_validation(artifact_root, config)
    if args.command in {"expand", "all"}:
        results["expand"] = run_expansion(artifact_root, config)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
