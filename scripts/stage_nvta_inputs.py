"""Stage the minimum full-run NVTA input set without prior assignment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


PERIODS = ("am", "md", "pm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_bundle_readme(destination: Path) -> None:
    (destination / "README.md").write_text(
        """# Complete NVTA pipeline input bundle

This folder contains the complete operational input set required by the
repository's `config/nvta.json` example. Copy the entire `nvta` folder into
`input-data/` in a clone of the repository; no source-machine paths are stored
in the bundle.

Verify every file before running:

```powershell
python .\\scripts\\verify_input_bundle.py .\\input-data\\nvta
python .\\main.py --config .\\config\\nvta.json qa
```

Then execute the complete workflow:

```powershell
python .\\main.py --config .\\config\\nvta.json run
```

`input_manifest.json` lists each required file by relative path, byte size,
and SHA-256 digest. Assignment outputs, caches, link-performance outputs,
routes, and other derived products are intentionally excluded because the
pipeline regenerates them.
""",
        encoding="utf-8",
    )


def transfer(source: Path, destination: Path, mode: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    shutil.copy2(source, destination)
    return "copy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-package",
        type=Path,
        required=True,
        help="Existing nvta-cbi-package containing input-data.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("input-data/nvta"),
    )
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--readings-name",
        default="NOVA-Oct1-31-2025--Avg-at-15min-.csv",
    )
    parser.add_argument(
        "--scenario-name",
        default="dtalite-run-07162026",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source_package.resolve()
    destination = args.destination.resolve()
    existing = [path for path in destination.iterdir()] if destination.exists() else []
    unexpected = [path for path in existing if path.name != ".gitkeep"]
    if unexpected:
        raise FileExistsError(
            f"Destination is not empty: {destination}. Use a fresh input root."
        )
    ritis = source / "input-data" / "shared" / "ritis"
    scenario = (
        source
        / "input-data"
        / "nvta-taplite-workflow"
        / args.scenario_name
    )
    network = scenario / "gmns_network_am_md_pm"
    transfers: list[dict[str, object]] = []

    def add(source_path: Path, relative: Path) -> None:
        target = destination / relative
        method = transfer(source_path, target, args.mode)
        transfers.append(
            {
                "relative_path": target.relative_to(destination).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
                "transfer": method,
            }
        )

    metadata = ritis / "TMC_Identification.csv"
    readings = ritis / args.readings_name
    add(metadata, Path("ritis/TMC_Identification.csv"))
    add(readings, Path("ritis/Readings.csv"))
    add(metadata, Path("matching/TMC_Identification.csv"))
    for period in PERIODS:
        for filename in ("link.csv", "node.csv"):
            source_file = network / period / filename
            add(source_file, Path("matching/network") / period / filename)
            add(source_file, Path("network/gmns_network_am_md_pm") / period / filename)
    profile = network / "CONVERSION_PROFILE.json"
    if profile.is_file():
        add(profile, Path("network/gmns_network_am_md_pm/CONVERSION_PROFILE.json"))
    cube_files = sorted(scenario.glob("DTALiteNetworkInput.*"))
    matrix_files: list[Path] = []
    for period in ("AM", "MD", "PM"):
        matches = sorted(scenario.glob(f"i4_{period}_Trips.omx"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {period} demand matrix under {scenario}; "
                f"found {len(matches)}"
            )
        matrix_files.extend(matches)
    if not cube_files:
        raise FileNotFoundError(f"No Cube network files under {scenario}")
    for path in [*cube_files, *matrix_files]:
        add(path, Path("taplite-scenario") / path.name)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "bundle": "complete-nvta-pipeline-input",
        "files": transfers,
        "total_files": len(transfers),
        "total_bytes": sum(int(item["size_bytes"]) for item in transfers),
        "excluded": [
            "prior AM/MD/PM/NT assignment folders",
            "route_assignment.csv",
            "od_performance.csv",
            "link_performance.csv",
            "conversion caches",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_bundle_readme(destination)
    print(json.dumps({"status": "PASS", "files": len(transfers), "destination": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
