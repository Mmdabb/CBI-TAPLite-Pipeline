from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cbi.config import CorridorSpec, PipelineSettings
from cbi.preprocessing import load_inrix_folder


def test_cbi_rejects_five_minute_ritis_input(tmp_path: Path) -> None:
    corridor = tmp_path / "I66_EB"
    corridor.mkdir()
    pd.DataFrame(
        {
            "tmc": ["TMC-1"],
            "road": ["I-66"],
            "direction": ["EASTBOUND"],
            "miles": [1.0],
            "road_order": [1],
        }
    ).to_csv(corridor / "TMC_Identification.csv", index=False)
    pd.DataFrame(
        {
            "tmc_code": ["TMC-1", "TMC-1"],
            "measurement_tstamp": [
                "2025-10-01T00:00:00",
                "2025-10-01T00:05:00",
            ],
            "speed": [55.0, 54.0],
        }
    ).to_csv(corridor / "Readings.csv", index=False)
    spec = CorridorSpec(
        key="I66_EB",
        name="I-66 eastbound",
        source="inrix_folder",
        path=corridor,
        free_flow_mph=60.0,
        capacity_vphpl=1800.0,
    )

    with pytest.raises(ValueError, match="non-15-minute RITIS timestamps"):
        load_inrix_folder(spec, PipelineSettings())

