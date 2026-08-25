from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ExperimentConfig:
    prototype_root: Path
    cbi_run_dir: Path
    boundary_mapping_run_dir: Path
    spatial_run_dir: Path
    output_root: Path
    random_seed: int
    cv_folds: int
    worker_fraction: float
    max_workers: int
    forest_estimators: int
    reliable_minimum_days: int
    reliable_maximum_t2_std_hours: float
    temporal_holdout_days: int
    model_names: tuple[str, ...]

    @property
    def n_jobs(self) -> int:
        available = os.cpu_count() or 1
        fraction_workers = int(math.floor(available * self.worker_fraction))
        return max(1, min(available, self.max_workers, fraction_workers))


def _resolve(prototype_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = prototype_root / path
    return path.resolve()


def _explicit_input(prototype_root: Path, value: str, name: str) -> Path:
    if not str(value).strip() or str(value).strip().casefold() == "latest":
        raise ValueError(
            f"{name} must be an explicit path; latest-run discovery is disabled"
        )
    return _resolve(prototype_root, value)


def load_config(config_path: Path) -> ExperimentConfig:
    config_path = config_path.resolve()
    payload: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    prototype_root = config_path.parent.parent
    return ExperimentConfig(
        prototype_root=prototype_root,
        cbi_run_dir=_explicit_input(
            prototype_root, payload["cbi_run_dir"], "cbi_run_dir"
        ),
        boundary_mapping_run_dir=_explicit_input(
            prototype_root,
            payload["boundary_mapping_run_dir"],
            "boundary_mapping_run_dir",
        ),
        spatial_run_dir=_explicit_input(
            prototype_root, payload["spatial_run_dir"], "spatial_run_dir"
        ),
        output_root=_resolve(prototype_root, payload["output_root"]),
        random_seed=int(payload.get("random_seed", 42)),
        cv_folds=int(payload.get("cv_folds", 5)),
        worker_fraction=float(payload.get("worker_fraction", 0.65)),
        max_workers=int(payload.get("max_workers", 15)),
        forest_estimators=int(payload.get("forest_estimators", 250)),
        reliable_minimum_days=int(payload.get("reliable_minimum_days", 3)),
        reliable_maximum_t2_std_hours=float(
            payload.get("reliable_maximum_t2_std_hours", 1.5)
        ),
        temporal_holdout_days=int(payload.get("temporal_holdout_days", 5)),
        model_names=tuple(str(name) for name in payload.get("model_names", [])),
    )
