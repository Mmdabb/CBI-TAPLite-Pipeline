"""Integrated NVTA congestion bottleneck identification and QVDF package."""

from .config import CorridorSpec, PipelineSettings


def run_corridor(*args, **kwargs):
    """Load the full pipeline lazily so lightweight consumers start quickly."""

    from .pipeline import run_corridor as _run_corridor

    return _run_corridor(*args, **kwargs)

__all__ = ["CorridorSpec", "PipelineSettings", "run_corridor"]
__version__ = "0.2.0"
