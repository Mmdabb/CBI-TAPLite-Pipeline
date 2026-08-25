"""All-corridor average-weekday NVTA QVDF projection dashboard."""

from .settings import DashboardSettings


def run_dashboard(*args, **kwargs):
    """Load plotting and orchestration modules only when a run is requested."""

    from .pipeline import run_dashboard as _run_dashboard

    return _run_dashboard(*args, **kwargs)

__all__ = ["DashboardSettings", "run_dashboard"]
