from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import run_integrated_dashboard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the integrated corridor dashboard.")
    parser.add_argument("--cbi-corridors", type=Path, required=True)
    parser.add_argument("--mapmatching-product", type=Path, required=True)
    parser.add_argument("--model-link-map", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--observed-15min", type=Path, required=True)
    parser.add_argument("--corridor-measurement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <corridor-measurement>/outputs/integrated-dashboard.")
    parser.add_argument("--profile-selection-overrides", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--worker-fraction", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir or (
        args.corridor_measurement.resolve() / "outputs" / "integrated-dashboard"
    )
    result = run_integrated_dashboard(
        package_root=Path.cwd(),
        corridor_results_root=args.cbi_corridors,
        mapmatching_product_root=args.mapmatching_product,
        model_link_map_path=args.model_link_map,
        assignment_root=args.assignment_root,
        ritis_15min_path=args.observed_15min,
        profile_selection_overrides_path=args.profile_selection_overrides,
        corridor_measurement_root=args.corridor_measurement,
        output_root=output,
        workers=args.workers,
        worker_fraction=args.worker_fraction,
        force=args.force,
    )
    print(json.dumps({"status": result.get("status", "PASS"), "output": str(output)}, indent=2))
    return 0
