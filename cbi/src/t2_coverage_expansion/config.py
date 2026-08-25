from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ExpansionConfig:
    episode_contract: str = "pre_filter"
    periods: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "AM": (360, 540),
            "MD": (540, 900),
            "PM": (900, 1140),
        }
    )
    worker_fraction: float = 0.70
    map_anchor_statuses: Tuple[str, ...] = ("matched",)
    minimum_map_confidence: float = 50.0
    require_screened_acceptance_for_anchor: bool = True
    minimum_daily_probe_days: int = 3
    maximum_daily_t2_std_hours: float = 1.0
    profile_threshold_ratio: float = 0.70
    profile_minimum_episode_minutes: int = 30
    profile_merge_gap_minutes: int = 15
    profile_minimum_coverage: float = 0.80
    maximum_interpolation_gap_miles: float = 10.0
    maximum_extrapolation_miles: float = 5.0
    maximum_abs_propagation_minutes_per_mile: float = 20.0
    bracket_assignment_method: str = "linear_t2"
    enable_gap_bridging: bool = True
    maximum_bridge_path_miles: float = 3.0
    bridge_off_corridor_penalty: float = 6.0
    bridge_ramp_penalty: float = 12.0
    validation_folds: int = 5
    enable_linear_t2_fallback: bool = True
    enable_analog_template: bool = False

    def __post_init__(self) -> None:
        if self.episode_contract not in {"pre_filter", "accepted"}:
            raise ValueError("episode_contract must be pre_filter or accepted")
        if not 0.0 < float(self.worker_fraction) <= 1.0:
            raise ValueError("worker_fraction must be greater than 0 and at most 1")
        if not 0.0 < float(self.profile_minimum_coverage) <= 1.0:
            raise ValueError(
                "profile_minimum_coverage must be greater than 0 and at most 1"
            )
        if int(self.validation_folds) < 2:
            raise ValueError("validation_folds must be at least 2")
        if self.bracket_assignment_method not in {
            "linear_t2",
            "normalized_profile",
        }:
            raise ValueError(
                "bracket_assignment_method must be linear_t2 or "
                "normalized_profile"
            )

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["periods"] = {
            key: list(value) for key, value in self.periods.items()
        }
        payload["map_anchor_statuses"] = list(self.map_anchor_statuses)
        return payload


def load_config(path: Path) -> ExpansionConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["periods"] = {
        str(key).upper(): tuple(int(item) for item in value)
        for key, value in raw.get("periods", {}).items()
    }
    raw["map_anchor_statuses"] = tuple(
        str(value) for value in raw.get("map_anchor_statuses", ["matched"])
    )
    return ExpansionConfig(**raw)
