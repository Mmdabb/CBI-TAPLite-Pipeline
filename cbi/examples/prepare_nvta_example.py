from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage one real NVTA corridor for CBI.")
    parser.add_argument("--source-package-root", type=Path, required=True)
    parser.add_argument("--model-link-map", type=Path, required=True)
    parser.add_argument("--corridor", default="I66_EB")
    parser.add_argument("--corridors-relative-path", default="input-data/cbi/corridors")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_package_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}")
    corridor = source / args.corridors_relative_path / args.corridor
    mapping = args.model_link_map.resolve()
    for required in (corridor / "TMC_Identification.csv", corridor / "Readings.csv", mapping):
        if not required.is_file():
            raise SystemExit(f"Missing source input: {required}")
    target = output / "corridors" / args.corridor
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corridor / "TMC_Identification.csv", target / "TMC_Identification.csv")
    shutil.copy2(corridor / "Readings.csv", target / "Readings.csv")
    shutil.copy2(mapping, output / "canonical_node_pair_tmc.csv")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
