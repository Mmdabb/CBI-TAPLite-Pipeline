"""Shared panel builders for corridor time-series data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_corridor_panel(df: pd.DataFrame, value_col: str = "speed_mph", time_col: str = "datetime") -> dict:
    """Reshape a long dataframe into a time by ordered-sensor panel."""
    if df.empty:
        return {
            "speed_field": np.zeros((0, 0)),
            "time_axis": [],
            "sensor_ids": [],
            "road_order": [],
            "direction": [],
            "source_format": None,
        }

    ro = df.groupby("sensor_uid")["road_order"].first().sort_values(kind="mergesort")
    sensor_ids = ro.index.tolist()
    pivot = (
        df.pivot_table(index=time_col, columns="sensor_uid", values=value_col, aggfunc="first")
        .reindex(columns=sensor_ids)
    )
    meta = (
        df.groupby("sensor_uid")
        .agg(
            direction=("direction", "first"),
            lanes=("lanes", "first"),
            length_mi=("length_mi", "first"),
            road_order=("road_order", "first"),
            has_volume=("has_volume", "first"),
            corridor=("corridor", "first"),
            source_format=("source_format", "first"),
        )
        .reindex(sensor_ids)
    )
    return {
        "speed_field": pivot.to_numpy(dtype=float),
        "time_axis": pivot.index.to_numpy(),
        "sensor_ids": sensor_ids,
        "road_order": meta["road_order"].to_numpy(dtype=float),
        "direction": meta["direction"].to_list(),
        "lanes": meta["lanes"].to_numpy(dtype=float),
        "length_mi": meta["length_mi"].to_numpy(dtype=float),
        "has_volume": bool(meta["has_volume"].iloc[0]),
        "corridor": str(meta["corridor"].iloc[0]) if len(meta) else None,
        "source_format": str(meta["source_format"].iloc[0]) if len(meta) else None,
    }
