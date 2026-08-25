from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


@dataclass(frozen=True)
class PipelineConfig:
    repository_root: Path
    config_path: Path
    payload: dict[str, Any]
    input_root: Path
    output_root: Path
    workers: int

    @property
    def files(self) -> dict[str, Path | None]:
        return {
            key: _resolve(self.input_root, value)
            for key, value in self.payload["files"].items()
        }

    @property
    def periods(self) -> dict[str, str]:
        return {str(k).lower(): str(v) for k, v in self.payload["periods"].items()}

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.payload.get(name, {}))

    def repository_path(self, value: str | None) -> Path | None:
        return _resolve(self.repository_root, value)


def load_config(path: Path) -> PipelineConfig:
    config_path = path.resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parent.parent.resolve()
    required = {"input_root", "output_root", "workers", "periods", "files"}
    missing = sorted(required - set(payload))
    if missing:
        raise ConfigurationError(f"Configuration is missing keys: {missing}")
    workers = int(payload["workers"])
    available = os.cpu_count() or 1
    if workers < 1 or workers > available:
        raise ConfigurationError(
            f"workers must be between 1 and {available}; received {workers}"
        )
    periods = {str(k).lower(): str(v) for k, v in payload["periods"].items()}
    if tuple(periods) != ("am", "md", "pm"):
        raise ConfigurationError("periods must contain AM, MD, and PM in that order")
    taplite = payload.get("taplite", {})
    if int(taplite.get("vdf_type", 2)) != 2:
        raise ConfigurationError("TAPlite vdf_type must remain 2")
    if int(taplite.get("qvdf_profile_mode", 2)) not in {0, 1, 2}:
        raise ConfigurationError("qvdf_profile_mode must be 0, 1, or 2")
    return PipelineConfig(
        repository_root=repository_root,
        config_path=config_path,
        payload=payload,
        input_root=_resolve(repository_root, payload["input_root"]),
        output_root=_resolve(repository_root, payload["output_root"]),
        workers=workers,
    )
