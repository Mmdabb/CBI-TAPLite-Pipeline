from __future__ import annotations

import subprocess
import sys


def test_ridge_cli_runs_as_python_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "t2_ml_experiment.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
