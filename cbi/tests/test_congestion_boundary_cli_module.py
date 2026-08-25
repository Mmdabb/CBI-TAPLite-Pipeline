from __future__ import annotations

import subprocess
import sys


def test_congestion_boundary_cli_runs_as_python_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "congestion_boundary_mapping.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--cbi-output-root" in result.stdout
