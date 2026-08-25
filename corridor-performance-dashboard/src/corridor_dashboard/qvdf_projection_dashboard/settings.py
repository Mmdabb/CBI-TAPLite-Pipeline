from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PERIODS


@dataclass(frozen=True)
class DashboardSettings:
    """Explicit, portable inputs for projection diagnostics."""

    package_root: Path = Path.cwd()
    output_root: Path | None = None
    cbi_products_root: Path | None = None
    mapmatching_product_root: Path | None = None
    model_link_map_path: Path | None = None
    assignment_root: Path | None = None
    ritis_15min_path: Path | None = None
    workers: int | None = None
    worker_fraction: float = 0.50
    force: bool = False
    generate_corridor_figures: bool = True
    profile_interval_minutes: int = 15

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_root", Path(self.package_root).resolve())
        if not 0.0 < float(self.worker_fraction) <= 1.0:
            raise ValueError("worker_fraction must be greater than 0 and at most 1")
        if int(self.profile_interval_minutes) != 15:
            raise ValueError("Projection diagnostics require 15-minute profiles")
        path_fields = (
            "output_root", "cbi_products_root", "mapmatching_product_root",
            "model_link_map_path", "assignment_root", "ritis_15min_path",
        )
        for name in path_fields:
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)).resolve())

    @property
    def periods(self) -> dict[str, tuple[int, int]]:
        return dict(PERIODS)

    @property
    def dashboard_data_root(self) -> Path:
        return self.output_root / "data"

    @property
    def corridor_report_root(self) -> Path:
        return self.output_root / "corridors"
