from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .network_mapping import build_observation_quality, write_canonical_mapping_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze one composite-ranked TMC winner per directed node pair."
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--corridor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.mapping.is_file():
        raise FileNotFoundError(args.mapping)
    if not args.corridor_root.is_dir():
        raise FileNotFoundError(args.corridor_root)
    quality = build_observation_quality(args.corridor_root)
    artifacts = write_canonical_mapping_artifacts(
        args.mapping, args.output_dir, observation_quality=quality
    )
    print(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
