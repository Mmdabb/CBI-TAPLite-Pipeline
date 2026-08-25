from __future__ import annotations

import os
from pathlib import Path


RESOURCE_ROOT_ENV = "NVTA_TAPLITE_RESOURCE_ROOT"


def resource_root() -> Path:
    """Return the run-local resource bundle or the packaged template."""

    configured = os.environ.get(RESOURCE_ROOT_ENV, "").strip()
    if configured:
        root = Path(configured).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"{RESOURCE_ROOT_ENV} points to a missing directory: {root}"
            )
        return root
    return Path(__file__).resolve().parent / "resources"


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)
