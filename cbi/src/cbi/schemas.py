"""Column contracts for the refreshed CBI package."""

from __future__ import annotations

from typing import Final


TIMESERIES_COLUMNS: Final = [
    "sensor_uid",
    "datetime",
    "speed_mph",
    "corridor_freeflow_speed_mph",
    "flow_vph",
    "density_vpm",
    "has_volume",
    "flow_synthetic",
    "source_format",
    "lanes",
    "length_mi",
    "road_order",
    "direction",
    "corridor",
]


STAGE1_QC_COLUMNS: Final = [
    "speed_mph_raw",
    "qc_hard_range",
    "qc_hampel",
    "hampel_replacement",
    "qc_jump",
    "qc_spatial_lag",
    "qc_pass",
    "speed_mph_clean",
    "speed_mph_clean_repaired",
    "qc_repaired_flag",
    "qc_repair_method",
    "qc_pass_repaired",
    "stage1_dataset_kind",
]


STAGE1_FLAG_COLUMNS: Final = [
    "qc_hard_range",
    "qc_hampel",
    "qc_jump",
    "qc_spatial_lag",
]
