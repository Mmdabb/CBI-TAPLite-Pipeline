"""Create or verify a portable complete-run input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IGNORED_NAMES = {".gitkeep", "README.md", "input_manifest.json"}

INPUT_README = """# Complete NVTA pipeline input bundle

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
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def input_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    )


def create_manifest(root: Path) -> dict[str, object]:
    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_files(root)
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "bundle": "complete-nvta-pipeline-input",
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "excluded": [
            "prior assignment folders",
            "route_assignment.csv",
            "od_performance.csv",
            "link_performance.csv",
            "conversion caches",
        ],
    }
    (root / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(INPUT_README, encoding="utf-8")
    return manifest


def verify(root: Path) -> dict[str, object]:
    manifest_path = root / "input_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; use --create once to build the manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        str(item["relative_path"]): item for item in manifest.get("files", [])
    }
    actual_paths = {
        path.relative_to(root).as_posix(): path for path in input_files(root)
    }
    errors: list[str] = []
    for relative, item in expected.items():
        path = actual_paths.get(relative)
        if path is None:
            errors.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(item["size_bytes"]):
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256(path) != str(item["sha256"]).upper():
            errors.append(f"hash mismatch: {relative}")
    for relative in sorted(set(actual_paths) - set(expected)):
        errors.append(f"unexpected: {relative}")
    missing_count = sum(error.startswith("missing:") for error in errors)
    return {
        "status": "FAIL" if errors else "PASS",
        "input_root": str(root),
        "verified_files": len(expected) - missing_count,
        "total_files": len(expected),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create or replace the relative-path manifest before verification.",
    )
    args = parser.parse_args()
    root = args.input_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if args.create:
        create_manifest(root)
    report = verify(root)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
