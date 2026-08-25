from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = WORKFLOW_ROOT / "setup" / "taplite_pypi_lock.json"
CONFLICTING_DISTRIBUTION = "taplite4mpo"
REQUIRED_NATIVE_MARKERS = (
    b"qvdf_profile_status",
    b"flat_missing_observation",
    b"generated_observed",
    b"expected blank, 0, 1, or 2",
)


def load_taplite_lock() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    required = {
        "distribution",
        "version",
        "pypi_project_url",
        "artifact_filename",
        "wheel_sha256",
        "native_module",
        "native_sha256",
    }
    missing = sorted(required.difference(lock))
    if missing:
        raise RuntimeError(f"TAPLite PyPI lock is missing fields: {missing}")
    return lock


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_taplite_runtime() -> dict[str, Any]:
    """Fail closed unless the running interpreter uses the pinned PyPI kernel."""

    lock = load_taplite_lock()
    distribution = str(lock["distribution"])
    expected_version = str(lock["version"])
    installed_version = _distribution_version(distribution)
    if installed_version != expected_version:
        raise RuntimeError(
            f"Expected {distribution}=={expected_version}; installed version is "
            f"{installed_version or '(not installed)'}"
        )

    conflicting_version = _distribution_version(CONFLICTING_DISTRIBUTION)
    if conflicting_version is not None:
        raise RuntimeError(
            f"Conflicting distribution {CONFLICTING_DISTRIBUTION}=="
            f"{conflicting_version} is installed. Remove it before running TAPLite."
        )

    taplite4mpo = importlib.import_module("taplite4mpo")
    pytaplite = importlib.import_module("pytaplite")
    native = importlib.import_module(str(lock["native_module"]))
    if not callable(getattr(pytaplite, "assign", None)):
        raise RuntimeError("The pinned PyPI distribution does not expose pytaplite.assign")
    for module_name, module in (
        ("taplite4mpo", taplite4mpo),
        ("pytaplite", pytaplite),
    ):
        module_version = str(getattr(module, "__version__", ""))
        if module_version != expected_version:
            raise RuntimeError(
                f"{module_name} reports {module_version or '(unknown)'}, expected "
                f"{expected_version}"
            )

    native_path = Path(native.__file__).resolve()
    native_sha256 = _sha256(native_path)
    expected_native_sha256 = str(lock["native_sha256"]).upper()
    if native_sha256 != expected_native_sha256:
        raise RuntimeError(
            "TAPLite native binary mismatch: expected SHA256 "
            f"{expected_native_sha256}, got {native_sha256} from {native_path}"
        )
    native_bytes = native_path.read_bytes()
    missing_markers = [
        marker.decode("ascii")
        for marker in REQUIRED_NATIVE_MARKERS
        if marker not in native_bytes
    ]
    if missing_markers:
        raise RuntimeError(
            "TAPLite native binary is missing strict profile-mode markers: "
            + ", ".join(missing_markers)
        )

    return {
        "kernel_source": "pypi",
        "distribution": distribution,
        "installed_version": installed_version,
        "pypi_project_url": str(lock["pypi_project_url"]),
        "artifact": str(lock["artifact_filename"]),
        "artifact_sha256": str(lock["wheel_sha256"]).upper(),
        "taplite4mpo_path": str(Path(taplite4mpo.__file__).resolve()),
        "pytaplite_path": str(Path(pytaplite.__file__).resolve()),
        "native_path": str(native_path),
        "native_sha256": native_sha256,
        "runtime_contract": "strict_qvdf_profile_mode_2_observed_t2_gate",
    }
