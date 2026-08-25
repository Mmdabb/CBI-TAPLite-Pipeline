from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output: Path, verbose: bool = False) -> Path:
    log = output / "logs" / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)
    detail = logging.FileHandler(log, mode="w", encoding="utf-8")
    detail.setLevel(logging.DEBUG)
    detail.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(detail)
    return log
