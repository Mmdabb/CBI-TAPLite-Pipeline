from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage read-only NVTA inputs for tmc-matching.")
    parser.add_argument("--source-package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmc-relative-path", default="input-data/shared/ritis/TMC_Identification.csv")
    parser.add_argument("--network-relative-path", default="input-data/nvta-taplite-workflow/dtalite-run-07162026/gmns_network_am_md_pm")
    args = parser.parse_args()
    source = args.source_package_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tmc = source / args.tmc_relative_path
    network = source / args.network_relative_path
    if not tmc.is_file() or not network.is_dir():
        raise SystemExit(f"Missing source TMC/network input: {tmc}; {network}")
    shutil.copy2(tmc, output / "TMC_Identification.csv")
    for period in ("am", "md", "pm"):
        target = output / "network" / period
        target.mkdir(parents=True)
        for name in ("link.csv", "node.csv"):
            candidate = network / period / name
            if not candidate.is_file():
                raise SystemExit(f"Missing source input: {candidate}")
            shutil.copy2(candidate, target / name)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
