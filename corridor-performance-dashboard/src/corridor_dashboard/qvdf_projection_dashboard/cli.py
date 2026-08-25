from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_dashboard
from .settings import DashboardSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the all-corridor NVTA QVDF projection dashboard with the "
            "integrated CBI engine."
        )
    )
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cbi-products-root", type=Path)
    parser.add_argument("--model-link-map", type=Path)
    parser.add_argument("--assignment-root", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--worker-fraction", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-corridor-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyword = {
        "output_root": args.output_root,
        "cbi_products_root": args.cbi_products_root,
        "model_link_map_path": args.model_link_map,
        "assignment_root": args.assignment_root,
        "workers": args.workers,
        "worker_fraction": args.worker_fraction,
        "force": args.force,
        "generate_corridor_figures": not args.no_corridor_figures,
    }
    if args.package_root is not None:
        keyword["package_root"] = args.package_root
    settings = DashboardSettings(**keyword)
    result = run_dashboard(settings)
    print(json.dumps(result, indent=2, default=str))
    return 0
