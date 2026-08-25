from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_low_boundary_low_congestion_links.py"
SPEC = importlib.util.spec_from_file_location("low_boundary_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_links_requires_low_boundaries_and_low_congestion_indicator() -> None:
    link = pd.DataFrame(
        {
            "link_id": ["1", "2", "3", "4"],
            "street_name": ["qualifies_p", "no_weak_indicator", "one_high_anchor", "qualifies_vt2"],
            "qvdf_start_speed_mph": [20.0, 20.0, 55.0, 20.0],
            "qvdf_end_speed_mph": [22.0, 22.0, 22.0, 21.0],
        }
    )
    performance = pd.DataFrame(
        {
            "iteration_no": [10, 10, 10, 10],
            "link_id": ["1", "2", "3", "4"],
            "cutoff_speed_mph": [30.0, 30.0, 30.0, 30.0],
            "congestion_ref_speed_mph": [40.0, 40.0, 40.0, 40.0],
            "P": [0.25, 1.0, 0.25, 1.0],
            "doc": [0.8, 0.8, 0.8, 0.8],
            "speed_mph": [19.0, 19.0, 60.0, 19.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
            "vt2_mph": [18.0, 18.0, 18.0, 25.0],
        }
    )

    selected, stats = MODULE.select_links(link, performance)

    assert selected["link_id"].tolist() == ["1", "4"]
    assert selected["street_name"].tolist() == ["qualifies_p", "qualifies_vt2"]
    assert "performance_volume" in selected
    assert selected["flag_selected"].all()
    assert stats["joined_rows"] == 4
    assert stats["selected_links"] == 2


def _qvdf_link_row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "lanes": 2.0,
        "capacity": 1800.0,
        "vdf_plf": 1.0,
        "vdf_alpha": 0.15,
        "vdf_beta": 4.0,
        "vdf_free_speed_mph": 60.0,
        "vdf_length_mi": 1.0,
        "vdf_cp": 0.28125,
        "vdf_cd": 1.0,
        "vdf_n": 1.0,
        "vdf_s": 4.0,
        "link_type": 100,
        "qvdf_profile_mode": 2,
        "qvdf_start_speed_mph": 30.0,
        "qvdf_end_speed_mph": 40.0,
    }
    values.update(overrides)
    return pd.Series(values)


def test_kernel_dispatcher_uses_smoothed_boundaries_when_mode_2_has_no_t2() -> None:
    result = MODULE._kernel_reporting_profile(
        _qvdf_link_row(),
        pd.Series({"volume": 1000.0, "speed_mph": 55.0, "cutoff_speed_mph": 45.0}),
        period_start_min=360,
        period_end_min=540,
    )

    assert result["predicted_profile_status"] == "smoothed_boundary_missing_observation"
    assert result["reconstruction_method"] == "kernel_smoothed_observed_boundary_fallback"
    assert result["speed_by_minute"][360] == 30.0
    assert result["speed_by_minute"][535] == 40.0


def test_kernel_dispatcher_generates_unanchored_qvdf_when_mode_1_has_no_boundaries() -> None:
    result = MODULE._kernel_reporting_profile(
        _qvdf_link_row(
            qvdf_profile_mode=1,
            qvdf_start_speed_mph=float("nan"),
            qvdf_end_speed_mph=float("nan"),
        ),
        pd.Series({"volume": 10_800.0, "speed_mph": 45.0, "cutoff_speed_mph": 45.0}),
        period_start_min=360,
        period_end_min=540,
    )

    assert result["predicted_profile_status"] == "generated_model"
    assert result["reconstruction_method"] == "kernel_qvdf_unanchored_no_boundary_speed"
    assert min(result["speed_by_minute"].values()) < max(result["speed_by_minute"].values())


def test_forced_unanchored_reconstruction_ignores_both_observed_speed_boundaries() -> None:
    performance = pd.Series(
        {"volume": 10_800.0, "speed_mph": 45.0, "cutoff_speed_mph": 45.0}
    )
    anchored_input = _qvdf_link_row(t2_hour=7.0)
    no_anchor_input = _qvdf_link_row(
        t2_hour=7.0,
        qvdf_start_speed_mph=float("nan"),
        qvdf_end_speed_mph=float("nan"),
    )

    anchored_result = MODULE._forced_unanchored_qvdf_profile(
        anchored_input,
        performance,
        period_start_min=360,
        period_end_min=540,
    )
    no_anchor_result = MODULE._forced_unanchored_qvdf_profile(
        no_anchor_input,
        performance,
        period_start_min=360,
        period_end_min=540,
    )

    assert anchored_result["reconstruction_method"] == "forced_unanchored_link_queue_vdf"
    assert anchored_result["speed_by_minute"] == no_anchor_result["speed_by_minute"]
    assert anchored_result["speed_by_minute"][360] != anchored_input["qvdf_start_speed_mph"]
    assert anchored_result["speed_by_minute"][535] != anchored_input["qvdf_end_speed_mph"]
