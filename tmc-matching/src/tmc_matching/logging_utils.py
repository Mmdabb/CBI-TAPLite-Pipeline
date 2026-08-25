from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_root: Path, verbose: bool = False) -> Path:
    """Use concise console logging and retain a detailed file log."""

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    detailed = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    detailed.setLevel(logging.DEBUG)
    detailed.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )
    root.addHandler(detailed)
    return log_path

