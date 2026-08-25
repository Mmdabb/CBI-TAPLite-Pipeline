from __future__ import annotations

import importlib
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT))

from src.dtalite4cube.taplite_runtime import verify_taplite_runtime  # noqa: E402


DEFAULT_CONFIG = Path("configs/project_assignment.json")

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "openmatrix",
    "tqdm",
    "geopandas",
    "shapely",
]


def selected_config_path() -> Path:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return Path(sys.argv[1])
    return DEFAULT_CONFIG


def read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_dirs_from_config(config: dict) -> tuple[Path, list[Path]]:
    scenario_base_dir = Path(config.get("scenario_base_dir", "scenarios"))

    raw_paths = config.get("scenario_paths")
    if raw_paths:
        scenario_dirs = [
            path if (path := Path(raw_path)).is_absolute() else scenario_base_dir / path
            for raw_path in raw_paths
        ]
        return scenario_base_dir, scenario_dirs

    scenario_names = config.get("scenario_names", [])
    scenario_dirs = [scenario_base_dir / name for name in scenario_names]
    return scenario_base_dir, scenario_dirs


def kernel_source_from_config(config: dict) -> str:
    assignment = config.get("assignment", {})
    return str(assignment.get("kernel_source", "pypi")).strip().lower() or "pypi"


def check_installed_pypi_kernel() -> tuple[bool, str]:
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            identity = verify_taplite_runtime()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, (
        f"{identity['distribution']}=={identity['installed_version']} | "
        f"native SHA256 {identity['native_sha256']}"
    )


def external_exe_required(config: dict) -> tuple[bool, str]:
    assignment = config.get("assignment", {})
    exe_name = assignment.get("dta_exe_name", "")

    external_flags = [
        assignment.get("use_external_dtalite_exe"),
        assignment.get("external_dtalite_exe"),
        assignment.get("dtalite_exe_required"),
    ]
    required = any(value is True for value in external_flags)
    return required, exe_name or "DTALite_0324b.exe"


def main() -> int:
    print("=" * 60)
    print("DTALite Pipeline Setup Check")
    print("=" * 60)

    failed = False
    config_path = selected_config_path()
    explicit_config = len(sys.argv) > 1 and sys.argv[1].strip()

    print(f"\nSelected config: {config_path}")

    print("\nChecking required project files/folders:")
    required_paths = [
        Path("run_assignment.py"),
        Path("run_postprocessing.py"),
        Path("src"),
        Path("setup"),
        Path("setup/environment.yml"),
    ]

    for path in required_paths:
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[MISSING] {path}")
            failed = True

    config = {}
    scenario_base_dir = Path("scenarios")
    scenario_dirs: list[Path] = []
    if config_path.exists():
        try:
            config = read_config(config_path)
            scenario_base_dir, scenario_dirs = scenario_dirs_from_config(config)
        except Exception as exc:
            print(f"[ERROR] Could not read selected config: {exc}")
            failed = True
    elif explicit_config:
        print(f"[MISSING] Selected config does not exist: {config_path}")
        failed = True
    else:
        print(f"[SKIP] Default config not found; scenario checks will be skipped: {config_path}")

    print(f"\nScenario base directory: {scenario_base_dir}")
    print(f"Scenario folders checked: {len(scenario_dirs)}")

    kernel_source = kernel_source_from_config(config)
    packages_to_check = ["taplite4mpo", "pytaplite", *REQUIRED_PACKAGES]

    print("\nChecking Python packages:")
    for pkg in packages_to_check:
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                importlib.import_module(pkg)
            print(f"[OK] {pkg}")
        except ImportError:
            print(f"[MISSING] {pkg}")
            failed = True

    print("\nChecking scenario folders:")
    if not scenario_dirs:
        if config_path.exists() or explicit_config:
            print("[MISSING] No scenario folders listed in selected config.")
            failed = True
        else:
            print("[SKIP] No config was provided.")
    for scenario_dir in scenario_dirs:
        if scenario_dir.exists():
            print(f"[OK] {scenario_dir}")
        else:
            print(f"[MISSING] {scenario_dir}")
            failed = True

    print("\nChecking DTALite kernel source:")
    if kernel_source == "pypi":
        print("Using the pinned TAPLite PyPI prerelease distribution.")
        runtime_ok, runtime_message = check_installed_pypi_kernel()
        if runtime_ok:
            print(f"[OK] {runtime_message}")
        else:
            print(f"[ERROR] Installed PyPI kernel could not load: {runtime_message}")
            failed = True
    else:
        print(f"[ERROR] Unknown kernel_source: {kernel_source}")
        failed = True

    required_exe, exe_name = external_exe_required(config)
    if required_exe:
        print("\nChecking external DTALite executables:")
        for scenario_dir in scenario_dirs:
            exe_path = scenario_dir / exe_name
            if exe_path.exists():
                print(f"[OK] {exe_path}")
            else:
                print(f"[MISSING] {exe_path}")
                failed = True
    else:
        print("\nExternal DTALite executable check skipped.")
        print(f"Current workflow uses kernel_source={kernel_source}.")

    print("\n" + "=" * 60)
    if failed:
        print("Setup check completed with issues.")
        print("Please review the messages above.")
        return 1

    print("Setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

