from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_treatment_resource_overlay import merge_plf_lookups
from generate_assignment_speed_boundaries import BOUNDARY_DTYPE
from install_final_treatment_resources import _merge_observed_overrides


DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("plf_am", "<f4"),
        ("plf_md", "<f4"),
        ("plf_pm", "<f4"),
    ]
)


def _write(path: Path, pairs: list[tuple[int, int]]) -> None:
    values = np.empty(len(pairs), dtype=DTYPE)
    for index, (from_node, to_node) in enumerate(pairs):
        values[index] = (
            (from_node << 32) | to_node,
            from_node,
            to_node,
            1.0,
            1.0,
            1.0,
        )
    np.save(path, values, allow_pickle=False)


def test_merge_plf_lookups_requires_disjoint_keys(tmp_path: Path) -> None:
    actual = tmp_path / "actual.npy"
    virtual = tmp_path / "virtual.npy"
    _write(actual, [(10, 20), (30, 40)])
    _write(virtual, [(25, 35)])
    combined = merge_plf_lookups(actual, virtual)
    assert combined["from_node_id"].tolist() == [10, 25, 30]

    _write(virtual, [(30, 40)])
    with pytest.raises(ValueError, match="overlap"):
        merge_plf_lookups(actual, virtual)


def test_observed_speed_precedence_replaces_assignment_by_key() -> None:
    def values(pairs: list[tuple[int, int]], speed: float) -> np.ndarray:
        result = np.empty(len(pairs), dtype=BOUNDARY_DTYPE)
        for index, (from_node, to_node) in enumerate(pairs):
            result[index]["packed_key"] = (from_node << 32) | to_node
            result[index]["from_node_id"] = from_node
            result[index]["to_node_id"] = to_node
            for field in BOUNDARY_DTYPE.names or ():
                if "speed_mph" in field:
                    result[index][field] = speed
        result.sort(order="packed_key")
        return result

    baseline = values([(10, 20), (30, 40), (50, 60)], 40.0)
    actual = values([(30, 40)], 30.0)
    virtual = values([(50, 60)], 35.0)
    combined, scope = _merge_observed_overrides(baseline, actual, virtual)
    assert combined["qvdf_start_speed_mph_am"].tolist() == [40.0, 30.0, 35.0]
    assert scope.tolist() == ["assignment", "actual", "virtual"]


def test_partial_observed_speed_keeps_assignment_for_missing_boundaries() -> None:
    def values(speed: float) -> np.ndarray:
        result = np.empty(1, dtype=BOUNDARY_DTYPE)
        result[0]["packed_key"] = (10 << 32) | 20
        result[0]["from_node_id"] = 10
        result[0]["to_node_id"] = 20
        for field in BOUNDARY_DTYPE.names or ():
            if "speed_mph" in field:
                result[0][field] = speed
        return result

    baseline = values(40.0)
    actual = np.empty(0, dtype=BOUNDARY_DTYPE)
    virtual = values(35.0)
    virtual[0]["qvdf_end_speed_mph_pm"] = np.nan

    combined, scope = _merge_observed_overrides(baseline, actual, virtual)

    assert combined[0]["qvdf_start_speed_mph_am"] == pytest.approx(35.0)
    assert combined[0]["qvdf_end_speed_mph_pm"] == pytest.approx(40.0)
    assert scope.tolist() == ["virtual_partial_assignment_fallback"]
