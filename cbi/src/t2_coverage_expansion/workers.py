from __future__ import annotations

import ctypes
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class WorkerPlan:
    logical_cores: int
    free_core_equivalents: float
    target_fraction: float
    workers: int
    task_count: int
    measurement: str
    sample_seconds: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _windows_idle_fraction(sample_seconds: float) -> float:
    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def value(item: FileTime) -> int:
        return (int(item.high) << 32) | int(item.low)

    def snapshot() -> Tuple[int, int, int]:
        idle = FileTime()
        kernel = FileTime()
        user = FileTime()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            raise OSError("GetSystemTimes failed")
        return value(idle), value(kernel), value(user)

    before = snapshot()
    time.sleep(sample_seconds)
    after = snapshot()
    idle_delta = after[0] - before[0]
    total_delta = (after[1] - before[1]) + (after[2] - before[2])
    if total_delta <= 0:
        raise ValueError("Invalid GetSystemTimes sample")
    return min(1.0, max(0.0, idle_delta / total_delta))


def _proc_stat_idle_fraction(sample_seconds: float) -> float:
    stat_path = Path("/proc/stat")

    def snapshot() -> Tuple[int, int]:
        fields = stat_path.read_text(encoding="utf-8").splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return idle, sum(values)

    before = snapshot()
    time.sleep(sample_seconds)
    after = snapshot()
    idle_delta = after[0] - before[0]
    total_delta = after[1] - before[1]
    if total_delta <= 0:
        raise ValueError("Invalid /proc/stat sample")
    return min(1.0, max(0.0, idle_delta / total_delta))


def _measure_free_core_equivalents(
    logical_cores: int, sample_seconds: float
) -> Tuple[float, str]:
    try:
        if platform.system() == "Windows":
            idle_fraction = _windows_idle_fraction(sample_seconds)
            return logical_cores * idle_fraction, "windows_GetSystemTimes"
        if Path("/proc/stat").is_file():
            idle_fraction = _proc_stat_idle_fraction(sample_seconds)
            return logical_cores * idle_fraction, "proc_stat"
        load_1m = float(os.getloadavg()[0])
        return max(0.0, logical_cores - load_1m), "load_average_1m"
    except (AttributeError, OSError, ValueError):
        return float(logical_cores), "logical_cores_fallback"


def recommend_workers(
    task_count: int,
    target_fraction: float = 0.70,
    explicit_workers: Optional[int] = None,
    sample_seconds: float = 0.25,
) -> WorkerPlan:
    if not 0.0 < float(target_fraction) <= 1.0:
        raise ValueError("target_fraction must be greater than 0 and at most 1")
    logical_cores = max(1, int(os.cpu_count() or 1))
    free_equivalents, measurement = _measure_free_core_equivalents(
        logical_cores, sample_seconds
    )
    task_limit = max(1, int(task_count))
    if explicit_workers is None:
        workers = max(1, int(math.floor(free_equivalents * target_fraction)))
        workers = min(
            workers,
            max(1, int(math.floor(logical_cores * target_fraction))),
        )
    else:
        workers = max(1, int(explicit_workers))
        measurement = "explicit_override_after_" + measurement
    workers = min(workers, logical_cores, task_limit)
    return WorkerPlan(
        logical_cores=logical_cores,
        free_core_equivalents=round(float(free_equivalents), 2),
        target_fraction=float(target_fraction),
        workers=workers,
        task_count=max(0, int(task_count)),
        measurement=measurement,
        sample_seconds=float(sample_seconds),
    )

