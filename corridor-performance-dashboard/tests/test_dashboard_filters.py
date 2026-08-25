import csv
from pathlib import Path

from corridor_dashboard.dashboard_filters import (
    is_managed_corridor,
    load_general_purpose_tmc_codes,
)


def test_managed_corridor_identification() -> None:
    assert is_managed_corridor("I395HOV_NB")
    assert is_managed_corridor("I-95 Express Lanes SB")
    assert is_managed_corridor("US-1 managed northbound")
    assert not is_managed_corridor("I395_NB")
    assert not is_managed_corridor("I95_SB")


def test_general_purpose_tmc_codes_require_exclusive_gp_class(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full_tmc_to_link.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tmc", "facility_class"])
        writer.writeheader()
        writer.writerows(
            [
                {"tmc": "GP-1", "facility_class": "gp"},
                {"tmc": "MANAGED-1", "facility_class": "managed"},
                {"tmc": "MIXED-1", "facility_class": "gp"},
                {"tmc": "MIXED-1", "facility_class": "managed"},
                {"tmc": "BLANK-1", "facility_class": ""},
            ]
        )

    assert load_general_purpose_tmc_codes(tmp_path) == {"GP-1"}
