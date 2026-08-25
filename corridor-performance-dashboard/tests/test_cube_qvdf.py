from pathlib import Path

import numpy as np
import pandas as pd

from corridor_measurement.cube_qvdf import (
    TAPLITE_QVDF_KERNEL_COMMIT,
    TAPLITE_QVDF_KERNEL_URL,
    load_cube_qvdf_profiles,
    qvdf_link_profile,
)


def _parameters() -> dict[str, float]:
    return {
        "lanes": 2.0,
        "capacity": 1_800.0,
        "vdf_plf": 1.0,
        "vdf_alpha": 0.15,
        "vdf_beta": 4.0,
        "vdf_free_speed_mph": 60.0,
        "vdf_length_mi": 1.0,
        "cutoff_speed": 45.0,
        "vdf_cp": 0.28125,
        "vdf_cd": 1.0,
        "vdf_n": 1.0,
        "vdf_s": 4.0,
        "t0_hour": 6.2,
        "t2_hour": 7.0,
        "t3_hour": 8.4,
    }


def test_zero_cube_volume_reconstructs_free_flow_profile():
    parameters = {
        **_parameters(),
        "qvdf_start_speed_mph": 20.0,
        "qvdf_end_speed_mph": 25.0,
    }
    result = qvdf_link_profile(
        parameters,
        0.0,
        period_start_min=360,
        period_end_min=540,
    )

    assert result["D"] == 0.0
    assert result["doc"] == 0.0
    assert result["P"] == 0.0
    assert set(result["speed_by_minute"]) == set(range(360, 540, 5))
    assert np.allclose(list(result["speed_by_minute"].values()), 60.0)


def test_low_observed_anchors_connect_directly_to_vt2():
    parameters = {
        **_parameters(),
        "qvdf_start_speed_mph": 34.0,
        "qvdf_end_speed_mph": 35.0,
    }
    result = qvdf_link_profile(
        parameters,
        10_800.0,
        period_start_min=360,
        period_end_min=540,
    )

    left = [result["speed_by_minute"][minute] for minute in range(360, 425, 5)]
    right = [result["speed_by_minute"][minute] for minute in range(420, 540, 5)]
    assert np.isclose(left[0], 34.0)
    assert np.isclose(left[-1], result["vt2_mph"])
    assert np.all(np.diff(left) >= -1e-9)
    assert np.isclose(right[0], result["vt2_mph"])
    assert np.isclose(right[-1], 35.0)
    assert np.all(np.diff(right) <= 1e-9)


def test_cube_volume_controls_doc_duration_and_peak_speed():
    # 10,800 vehicles / 2 lanes / 3 hours / PLF 1 = 1,800 vphpl, D/C = 1.
    result = qvdf_link_profile(
        _parameters(),
        10_800.0,
        period_start_min=360,
        period_end_min=540,
    )

    assert np.isclose(result["D"], 1_800.0)
    assert np.isclose(result["doc"], 1.0)
    assert np.isclose(result["P"], 1.0)
    assert np.isclose(result["t2"], 7.0)
    assert np.isclose(
        result["vt2_mph"],
        45.0 / (1.0 + 0.28125),
    )
    assert np.isclose(result["speed_by_minute"][420], result["vt2_mph"])
    assert result["speed_by_minute"][360] > result["speed_by_minute"][420]


def test_observed_boundary_speeds_anchor_the_emitted_profile():
    parameters = {
        **_parameters(),
        "qvdf_start_speed_mph": 52.0,
        "qvdf_end_speed_mph": 55.0,
    }
    result = qvdf_link_profile(
        parameters,
        10_800.0,
        period_start_min=360,
        period_end_min=540,
    )

    assert np.isclose(result["speed_by_minute"][360], 52.0)
    assert np.isclose(result["speed_by_minute"][535], 55.0)
    assert np.isclose(result["speed_by_minute"][420], result["vt2_mph"])
    assert min(result["speed_by_minute"].values()) >= result["vt2_mph"]


def test_loader_returns_link_performance_shape_and_provenance(tmp_path: Path):
    row = {"link_id": "100", "I4AMVOL": 10_800.0, **_parameters()}
    source = tmp_path / "link.csv"
    pd.DataFrame([row]).to_csv(source, index=False)

    profile, lookup, audit = load_cube_qvdf_profiles(
        source,
        cube_volume_column="I4AMVOL",
        period_start_min=360,
        period_end_min=540,
        link_ids=["100"],
    )

    assert profile.index.tolist() == ["100"]
    assert profile.loc["100", "volume"] == 10_800.0
    assert np.isclose(profile.loc["100", "doc"], 1.0)
    assert lookup["spd_mph_06:00"] == 360
    assert lookup["spd_mph_08:55"] == 535
    assert audit.loc[0, "qvdf_kernel_commit"] == TAPLITE_QVDF_KERNEL_COMMIT
    assert audit.loc[0, "qvdf_kernel_url"] == TAPLITE_QVDF_KERNEL_URL
