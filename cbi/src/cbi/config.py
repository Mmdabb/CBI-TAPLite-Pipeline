from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


WORKFLOW_PERIODS = {
    "AM": (6 * 60, 9 * 60),
    "MD": (9 * 60, 15 * 60),
    "PM": (15 * 60, 19 * 60),
    "NT": (19 * 60, 6 * 60),
}
PERIODS = {
    label: WORKFLOW_PERIODS[label]
    for label in ("AM", "MD", "PM")
}
WIDE_WINDOW = (5 * 60, 22 * 60)
WEEKDAY_NAME = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def period_duration_hours(
    period: str,
    periods: dict[str, tuple[int, int]] | None = None,
) -> float:
    """Return the positive duration of a named period in hours."""

    definitions = WORKFLOW_PERIODS if periods is None else periods
    if period not in definitions:
        raise KeyError(f"Unknown time period: {period}")
    start, end = definitions[period]
    minutes = (float(end) - float(start)) % (24.0 * 60.0)
    if minutes <= 0:
        minutes = 24.0 * 60.0
    return minutes / 60.0


@dataclass(frozen=True)
class CorridorSpec:
    """One corridor input and its authoritative physical context."""

    key: str
    name: str
    source: str
    path: Path
    free_flow_mph: float
    capacity_vphpl: float
    model_link_map: Path | None = None
    data_mode: str = "speed_only"
    facility_type: str | None = None
    area_type: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"inrix_folder", "avgweekday_csv"}:
            raise ValueError(f"Unsupported source: {self.source}")
        if self.data_mode not in {"speed_only", "measured"}:
            raise ValueError(f"Unsupported data mode: {self.data_mode}")
        object.__setattr__(self, "path", Path(self.path).resolve())
        if self.model_link_map is not None:
            object.__setattr__(self, "model_link_map", Path(self.model_link_map).resolve())


@dataclass(frozen=True)
class PipelineSettings:
    """Shared settings for raw-corridor and average-weekday workflows."""

    interval_minutes: int = 15
    periods: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(PERIODS))
    wide_window: tuple[int, int] = WIDE_WINDOW
    cutoff_ratio: float = 0.70
    confidence_min: int = 30
    minimum_episode_minutes: float = 30.0
    merge_gap_minutes: float = 15.0
    minimum_discharge_minutes: float = 30.0
    minimum_calibration_episodes: int = 3
    output_numeric_rounding: bool = True

    def period_for_minute(self, minute: float) -> str | None:
        value = float(minute) % (24 * 60)
        for label, (start, end) in self.periods.items():
            if (
                start <= value < end
                if start < end
                else value >= start or value < end
            ):
                return label
        return None
