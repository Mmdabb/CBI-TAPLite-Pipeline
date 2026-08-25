"""Workflow-facing adapter for the vendored QVDF speed smoother."""

from __future__ import annotations

import argparse
from pathlib import Path

from .qvdf_profile_batch import run_batch


def smooth_assignment_outputs(
    scenario_dir: Path,
    periods: list[str],
    *,
    workers: int,
    backup: bool = True,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Validate and atomically replace period ``spd_mph_*`` profiles."""

    arguments = argparse.Namespace(
        scenario_dir=Path(scenario_dir),
        periods=list(periods),
        workers=str(max(1, int(workers))),
        chunk_size=0,
        write_back=True,
        backup=bool(backup),
        report=Path(report_path) if report_path is not None else None,
        audit_link=[],
        max_five_minute_change=8.0,
        rolling_window_intervals=3,
        max_rolling_average_change=4.0,
        max_acceleration=576.0,
        speed_decimals=9,
    )
    return run_batch(arguments)
