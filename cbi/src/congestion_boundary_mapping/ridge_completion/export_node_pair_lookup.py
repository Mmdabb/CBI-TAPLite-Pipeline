from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


PERIODS = ("AM", "MD", "PM")
UINT32_MAX = np.iinfo(np.uint32).max
LOOKUP_DTYPE = np.dtype(
    [
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("t0_hour", "<f4"),
        ("t2_hour", "<f4"),
        ("t3_hour", "<f4"),
    ]
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_node_pairs(
    from_node_id: np.ndarray,
    to_node_id: np.ndarray,
) -> np.ndarray:
    from_values = np.asarray(from_node_id)
    to_values = np.asarray(to_node_id)
    if from_values.shape != to_values.shape:
        raise ValueError("from_node_id and to_node_id shapes differ.")
    if not np.isfinite(from_values).all() or not np.isfinite(to_values).all():
        raise ValueError("Node-pair keys contain null or non-finite values.")
    if not np.equal(from_values, np.floor(from_values)).all():
        raise ValueError("from_node_id contains non-integer values.")
    if not np.equal(to_values, np.floor(to_values)).all():
        raise ValueError("to_node_id contains non-integer values.")
    if (
        (from_values < 0).any()
        or (to_values < 0).any()
        or (from_values > UINT32_MAX).any()
        or (to_values > UINT32_MAX).any()
    ):
        raise ValueError("Node ids must fit unsigned 32-bit integers.")
    from_uint = from_values.astype(np.uint64)
    to_uint = to_values.astype(np.uint64)
    return (from_uint << np.uint64(32)) | to_uint


def build_period_lookup(
    source_path: Path,
    destination_path: Path,
) -> Dict[str, object]:
    boundary_columns = [
        "t0_hybrid_hour",
        "t2_hybrid_hour",
        "t3_hybrid_hour",
    ]
    columns = [
        "from_node_id",
        "to_node_id",
        *boundary_columns,
    ]
    header = set(pd.read_csv(source_path, nrows=0).columns)
    protection_column = "t2_observed_no_congestion_protected"
    if protection_column in header:
        columns.append(protection_column)
    frame = pd.read_csv(
        source_path,
        usecols=columns,
        low_memory=False,
    )
    nulls = frame[boundary_columns].isna()
    partial_null = nulls.any(axis=1) & ~nulls.all(axis=1)
    if partial_null.any():
        raise ValueError(f"Partial null lookup boundaries in {source_path}.")
    protected = (
        frame.get(protection_column, pd.Series(False, index=frame.index))
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    if nulls.all(axis=1).ne(protected).any():
        raise ValueError(
            f"Null lookup boundaries are not protected no-congestion rows in {source_path}."
        )
    if frame.duplicated(["from_node_id", "to_node_id"]).any():
        duplicate_count = int(
            frame.duplicated(
                ["from_node_id", "to_node_id"], keep=False
            ).sum()
        )
        raise ValueError(
            f"{source_path} has {duplicate_count} duplicate pair rows."
        )
    ordered = (
        frame["t0_hybrid_hour"].le(frame["t2_hybrid_hour"])
        & frame["t2_hybrid_hour"].le(frame["t3_hybrid_hour"])
    )
    if not ordered.loc[~protected].all():
        raise ValueError(
            f"{source_path} has {int((~ordered & ~protected).sum())} unordered rows."
        )
    packed = pack_node_pairs(
        frame["from_node_id"].to_numpy(),
        frame["to_node_id"].to_numpy(),
    )
    lookup = np.empty(len(frame), dtype=LOOKUP_DTYPE)
    lookup["packed_key"] = packed
    lookup["from_node_id"] = frame["from_node_id"].to_numpy(
        dtype=np.uint32
    )
    lookup["to_node_id"] = frame["to_node_id"].to_numpy(dtype=np.uint32)
    lookup["t0_hour"] = frame["t0_hybrid_hour"].to_numpy(dtype=np.float32)
    lookup["t2_hour"] = frame["t2_hybrid_hour"].to_numpy(dtype=np.float32)
    lookup["t3_hour"] = frame["t3_hybrid_hour"].to_numpy(dtype=np.float32)
    lookup.sort(order="packed_key")
    if np.any(np.diff(lookup["packed_key"]) == 0):
        raise ValueError(f"Packed keys are not unique in {source_path}.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination_path, lookup, allow_pickle=False)
    restored = np.load(destination_path, mmap_mode="r", allow_pickle=False)
    if len(restored) != len(frame) or restored.dtype != LOOKUP_DTYPE:
        raise ValueError(f"Lookup round-trip failed for {destination_path}.")
    source_values = (
        frame.set_index(["from_node_id", "to_node_id"])[
            ["t0_hybrid_hour", "t2_hybrid_hour", "t3_hybrid_hour"]
        ]
        .sort_index()
        .to_numpy(dtype=np.float64)
    )
    binary_values = np.column_stack(
        [
            restored["t0_hour"],
            restored["t2_hour"],
            restored["t3_hour"],
        ]
    ).astype(np.float64)
    differences = np.abs(binary_values - source_values)
    max_abs_error_minutes = (
        float(np.nanmax(differences) * 60.0)
        if np.isfinite(differences).any()
        else 0.0
    )
    return {
        "source_path": str(source_path.resolve()),
        "lookup_path": str(destination_path.resolve()),
        "rows": len(restored),
        "dtype": str(LOOKUP_DTYPE.descr),
        "source_size_bytes": source_path.stat().st_size,
        "lookup_size_bytes": destination_path.stat().st_size,
        "compression_ratio_vs_source": (
            destination_path.stat().st_size / source_path.stat().st_size
        ),
        "minimum_from_node_id": int(restored["from_node_id"].min()),
        "maximum_from_node_id": int(restored["from_node_id"].max()),
        "minimum_to_node_id": int(restored["to_node_id"].min()),
        "maximum_to_node_id": int(restored["to_node_id"].max()),
        "max_float32_round_trip_error_minutes": max_abs_error_minutes,
        "protected_no_congestion_rows": int(protected.sum()),
        "rows_with_boundaries": int((~protected).sum()),
        "source_sha256": _hash_file(source_path),
        "lookup_sha256": _hash_file(destination_path),
    }


LOADER_TEXT = '''from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


BOUNDARY_FIELDS = ["t0_hour", "t2_hour", "t3_hour"]


def period_path(
    period: str,
    directory: Optional[Union[str, Path]] = None,
) -> Path:
    root = Path(directory) if directory is not None else Path(__file__).parent
    return root / f"{period.lower()}_node_pair_boundaries.npy"


def _pack(from_node_id, to_node_id) -> np.ndarray:
    from_values = np.asarray(from_node_id, dtype=np.uint64)
    to_values = np.asarray(to_node_id, dtype=np.uint64)
    if from_values.shape != to_values.shape:
        raise ValueError("from_node_id and to_node_id shapes differ.")
    return (from_values << np.uint64(32)) | to_values


def load_period(
    period: str,
    directory: Optional[Union[str, Path]] = None,
):
    """Memory-map one period lookup without reading the full file into RAM."""
    return np.load(period_path(period, directory), mmap_mode="r", allow_pickle=False)


def lookup(
    table,
    from_node_id,
    to_node_id,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return [..., 3] T0/T2/T3 array and a same-shape found mask."""
    packed = _pack(from_node_id, to_node_id)
    original_shape = packed.shape
    query = packed.reshape(-1)
    keys = table["packed_key"]
    values = np.full((len(query), len(BOUNDARY_FIELDS)), np.nan, dtype=np.float32)
    if len(keys) == 0:
        return values.reshape((*original_shape, len(BOUNDARY_FIELDS))), np.zeros(
            original_shape,
            dtype=bool,
        )

    positions = np.searchsorted(keys, query)
    clipped = np.minimum(positions, len(keys) - 1)
    found = (positions < len(keys)) & (keys[clipped] == query)
    if found.any():
        selected = table[clipped[found]]
        for field_index, field in enumerate(BOUNDARY_FIELDS):
            values[found, field_index] = selected[field]
    return values.reshape((*original_shape, len(BOUNDARY_FIELDS))), found.reshape(original_shape)


def lookup_period(
    period: str,
    from_node_id,
    to_node_id,
    directory: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return lookup(load_period(period, directory), from_node_id, to_node_id)
'''


README_TEXT = """# Node-pair boundary lookup

These files are a one-time export from the completed NVTA link files. Each AM, MD, and PM
file is a sorted NumPy structured array keyed by `(from_node_id, to_node_id)`.

## Why `.npy`

- `numpy.load(path, mmap_mode="r")` opens the lookup without loading the full array into RAM.
- Sorted packed 64-bit keys support vectorized `numpy.searchsorted` matching.
- The format preserves typed node ids and float boundary values without CSV parsing.
- No dependency is required beyond NumPy.

## Fields

- `packed_key`: `(uint64(from_node_id) << 32) | uint64(to_node_id)`
- `from_node_id`, `to_node_id`: original pair, stored as unsigned 32-bit integers
- `t0_hour`, `t2_hour`, `t3_hour`: completed boundaries stored as float32 hours

No class estimate, source label, uncertainty field, or other reference column is included.

The float32 export changes the source values by far less than one second; the exact maximum
round-trip difference is recorded in `metadata.json`.

## Load and map

```python
import pandas as pd
from load_node_pair_boundaries import lookup_period

network = pd.read_csv("my_network.csv")
values, found = lookup_period(
    "AM",
    network["from_node_id"].to_numpy(),
    network["to_node_id"].to_numpy(),
    directory="path/to/node_pair_lookup",
)
network[["t0_hour", "t2_hour", "t3_hour"]] = values
```

`found` identifies network pairs that exist in the lookup. The files are sorted and contain
one unique row per directed node pair.
"""


def run(period_link_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata: Dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "format": "NumPy .npy structured array, sorted by packed_key",
        "key_definition": (
            "(uint64(from_node_id) << 32) | uint64(to_node_id)"
        ),
        "record_dtype": LOOKUP_DTYPE.descr,
        "time_unit": "decimal hour",
        "load_pattern": "numpy.load(path, mmap_mode='r', allow_pickle=False)",
        "periods": {},
    }
    for period in PERIODS:
        source_path = period_link_root / period.lower() / "link.csv"
        destination_path = (
            output_dir / f"{period.lower()}_node_pair_boundaries.npy"
        )
        metadata["periods"][period] = build_period_lookup(
            source_path, destination_path
        )
    (output_dir / "load_node_pair_boundaries.py").write_text(
        LOADER_TEXT, encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        README_TEXT, encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export fast node-pair T0/T2/T3 lookup files."
    )
    parser.add_argument("--period-link-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.period_link_root, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
