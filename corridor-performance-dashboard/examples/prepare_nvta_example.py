from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage explicit NVTA producer outputs.")
    parser.add_argument("--source-package-root", type=Path, required=True)
    parser.add_argument("--cbi-relative-path", required=True)
    parser.add_argument("--matching-relative-path", required=True)
    parser.add_argument("--assignment-relative-path", required=True)
    parser.add_argument("--model-link-map-relative-path", required=True)
    parser.add_argument("--observed-relative-path", default="input-data/shared/ritis/NOVA-Oct1-31-2025--Avg-at-15min-.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_package_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        "cbi": source / args.cbi_relative_path,
        "matching": source / args.matching_relative_path,
        "assignment": source / args.assignment_relative_path,
    }
    for name, path in directories.items():
        if not path.is_dir():
            raise SystemExit(f"Missing source directory: {path}")
        shutil.copytree(path, output / name)
    model_map = source / args.model_link_map_relative_path
    observed = source / args.observed_relative_path
    for path in (model_map, observed):
        if not path.is_file():
            raise SystemExit(f"Missing source file: {path}")
    shutil.copy2(model_map, output / "model_link_map.csv")
    shutil.copy2(observed, output / "observed_15min.csv")
    print(
        "corridor-dashboard all "
        f"--cbi-corridors \"{output / 'cbi' / 'corridors'}\" "
        f"--mapmatching-root \"{output / 'matching'}\" "
        f"--assignment-root \"{output / 'assignment'}\" "
        f"--model-link-map \"{output / 'model_link_map.csv'}\" "
        f"--observed-15min \"{output / 'observed_15min.csv'}\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
