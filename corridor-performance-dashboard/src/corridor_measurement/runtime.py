"""Automatic process-worker planning and numerical thread controls."""

from __future__ import annotations

import ctypes
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class WorkerPlan:
    logical_cores: int
    free_core_equivalents: float
    target_fraction: float
    workers: int
    task_count: int
    measurement: str
    sample_seconds: float
    threads_per_worker: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_numerical_threads(threads_per_process: int = 1) -> None:
    """Keep each process single-threaded so workers do not oversubscribe CPUs."""

    if threads_per_process < 1:
        raise ValueError("threads_per_process must be positive.")
    value = str(int(threads_per_process))
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = value
    os.environ["OMP_DYNAMIC"] = "FALSE"


def _windows_idle_fraction(sample_seconds: float) -> float:
    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def value(item: FileTime) -> int:
        return (int(item.high) << 32) | int(item.low)

    def snapshot() -> tuple[int, int, int]:
        idle = FileTime()
        kernel = FileTime()
        user = FileTime()
        ok = ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle),
            ctypes.byref(kernel),
            ctypes.byref(user),
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
        raise ValueError("Invalid GetSystemTimes sample.")
    return min(1.0, max(0.0, idle_delta / total_delta))


def _proc_stat_idle_fraction(sample_seconds: float) -> float:
    stat_path = Path("/proc/stat")

    def snapshot() -> tuple[int, int]:
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
        raise ValueError("Invalid /proc/stat sample.")
    return min(1.0, max(0.0, idle_delta / total_delta))


def _measure_free_core_equivalents(
    logical_cores: int,
    sample_seconds: float,
) -> tuple[float, str]:
    try:
        if platform.system() == "Windows":
            return (
                logical_cores * _windows_idle_fraction(sample_seconds),
                "windows_GetSystemTimes",
            )
        if Path("/proc/stat").is_file():
            return (
                logical_cores * _proc_stat_idle_fraction(sample_seconds),
                "proc_stat",
            )
        load_1m = float(os.getloadavg()[0])
        return max(0.0, logical_cores - load_1m), "load_average_1m"
    except (AttributeError, OSError, ValueError):
        return float(logical_cores), "logical_cores_fallback"


def recommend_workers(
    task_count: int,
    *,
    target_fraction: float = 0.50,
    sample_seconds: float = 0.25,
    explicit_workers: int | None = None,
) -> WorkerPlan:
    """Target a configured share of currently free logical-core capacity."""

    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be greater than 0 and at most 1.")
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive.")
    logical_cores = max(1, int(os.cpu_count() or 1))
    free_cores, measurement = _measure_free_core_equivalents(
        logical_cores, sample_seconds
    )
    task_limit = max(1, int(task_count))
    if explicit_workers is None:
        workers = max(1, int(math.floor(free_cores * target_fraction)))
        workers = min(
            workers,
            max(1, int(math.floor(logical_cores * target_fraction))),
        )
    else:
        workers = max(1, int(explicit_workers))
        measurement = f"explicit_override_after_{measurement}"
    workers = min(workers, logical_cores, task_limit)
    return WorkerPlan(
        logical_cores=logical_cores,
        free_core_equivalents=round(free_cores, 2),
        target_fraction=float(target_fraction),
        workers=workers,
        task_count=max(0, int(task_count)),
        measurement=measurement,
        sample_seconds=float(sample_seconds),
    )
