from __future__ import annotations

from src.dtalite4cube.reproducible_run import _build_dtalite_command


def test_pypi_launcher_verifies_runtime_before_assignment() -> None:
    command = _build_dtalite_command("pypi")
    assert command[-2] == "-c"
    assert "import os" in command[-1]
    assert "verify_taplite_runtime()" in command[-1]
    assert 'runtime_identity["native_sha256"]' in command[-1]
    assert "pytaplite.assign(os.getcwd(), in_place=True)" in command[-1]
