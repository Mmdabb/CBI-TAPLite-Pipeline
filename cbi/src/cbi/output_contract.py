from __future__ import annotations

from pathlib import Path


STEP_FOLDERS = {
    "input_qc": "01-input-and-qc",
    "fundamental_diagram": "02-fundamental-diagram",
    "profiles": "03-profiles",
    "episode_detection": "04-episode-detection",
    "episode_filtering": "05-episode-filtering",
    "calibration": "06-qvdf-calibration",
    "handoff": "07-reconstruction-and-handoff",
    "quality": "08-quality-assurance",
    "tables": "09-summary-tables",
    "figures": "10-figures",
    "metadata": "11-run-metadata",
}


def step_dir(
    corridor_root: Path,
    step: str,
    *,
    create: bool = False,
) -> Path:
    """Return the numbered output directory for one processing step."""

    try:
        folder = STEP_FOLDERS[step]
    except KeyError as exc:
        raise ValueError(f"Unknown output step: {step}") from exc
    path = Path(corridor_root) / folder
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def create_step_directories(corridor_root: Path) -> dict[str, Path]:
    """Create and return every numbered corridor-output step directory."""

    return {
        step: step_dir(corridor_root, step, create=True)
        for step in STEP_FOLDERS
    }
