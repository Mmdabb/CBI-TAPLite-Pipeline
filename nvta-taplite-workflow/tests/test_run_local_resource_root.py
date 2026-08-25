from __future__ import annotations

from pathlib import Path

import pytest

from dtalite4cube.resource_paths import RESOURCE_ROOT_ENV, resource_path, resource_root


def test_run_local_resource_root_overrides_packaged_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_resources = tmp_path / "resources"
    run_resources.mkdir()
    monkeypatch.setenv(RESOURCE_ROOT_ENV, str(run_resources))
    assert resource_root() == run_resources.resolve()
    assert resource_path("link_qvdf.csv") == run_resources / "link_qvdf.csv"


def test_missing_run_local_resource_root_is_a_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv(RESOURCE_ROOT_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match=RESOURCE_ROOT_ENV):
        resource_root()
