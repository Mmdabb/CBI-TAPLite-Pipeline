from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


DEFAULT_COLUMN_MAP: dict[str, dict[str, str]] = {
    "metadata": {
        "tmc": "tmc",
        "road": "road",
        "direction": "direction",
        "road_order": "road_order",
        "miles": "miles",
        "start_latitude": "start_latitude",
        "start_longitude": "start_longitude",
        "end_latitude": "end_latitude",
        "end_longitude": "end_longitude",
    },
    "readings": {
        "tmc_code": "tmc_code",
        "measurement_tstamp": "measurement_tstamp",
        "speed": "speed",
    },
    "mapping": {
        "tmc": "tmc",
        "link_id": "link_id",
        "from_node_id": "from_node_id",
        "to_node_id": "to_node_id",
    },
}


class InputQAError(ValueError):
    pass


@dataclass(frozen=True)
class CBIInputContract:
    corridor_root: Path
    model_link_map: Path
    metadata_file_name: str
    readings_file_name: str
    column_map: dict[str, dict[str, str]]


@dataclass(frozen=True)
class QAResult:
    status: str
    corridor_count: int
    metadata_rows: int
    reading_rows: int
    mapping_rows: int
    files: list[str]
    column_map: dict[str, dict[str, str]]


def load_column_map(path: Path | None) -> dict[str, dict[str, str]]:
    result = {name: values.copy() for name, values in DEFAULT_COLUMN_MAP.items()}
    if path is None:
        return result
    if not path.is_file():
        raise InputQAError(f"Column-map JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise InputQAError("Column-map JSON must contain an object.")
    for group in result:
        supplied = payload.get(group, {})
        if not isinstance(supplied, Mapping):
            raise InputQAError(f"column-map.{group} must be an object.")
        result[group].update({str(k): str(v) for k, v in supplied.items()})
    return result


def _header(path: Path, required: set[str]) -> list[str]:
    if not path.is_file():
        raise InputQAError(
            f"Required input is unavailable: {path}. "
            "Check the corridor root and filename arguments."
        )
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        raise InputQAError(f"Cannot read CSV header {path}: {exc}") from exc
    missing = sorted(required.difference(columns))
    if missing:
        raise InputQAError(
            f"{path} is missing configured columns: {', '.join(missing)}"
        )
    return columns


def _row_count(path: Path) -> int:
    count = sum(1 for _ in path.open("r", encoding="utf-8-sig", errors="replace")) - 1
    if count <= 0:
        raise InputQAError(f"Required input contains no data rows: {path}")
    return count


def validate_inputs(contract: CBIInputContract) -> QAResult:
    root = contract.corridor_root.resolve()
    if not root.is_dir():
        raise InputQAError(f"Corridor input directory does not exist: {root}")
    _header(contract.model_link_map, set(contract.column_map["mapping"].values()))
    mapping_rows = _row_count(contract.model_link_map)
    folders = []
    files: list[str] = [str(contract.model_link_map)]
    metadata_rows = 0
    reading_rows = 0
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = folder / contract.metadata_file_name
        readings = folder / contract.readings_file_name
        if not metadata.exists() and not readings.exists():
            continue
        _header(metadata, set(contract.column_map["metadata"].values()))
        _header(readings, set(contract.column_map["readings"].values()))
        metadata_rows += _row_count(metadata)
        reading_rows += _row_count(readings)
        files.extend([str(metadata), str(readings)])
        folders.append(folder)
    if not folders:
        raise InputQAError(
            f"No corridor folders containing {contract.metadata_file_name} and "
            f"{contract.readings_file_name} were found under {root}."
        )
    return QAResult(
        "PASS",
        len(folders),
        metadata_rows,
        reading_rows,
        mapping_rows,
        files,
        contract.column_map,
    )


def write_report(result: QAResult, output_root: Path) -> Path:
    target = output_root / "qa" / "input_qa.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return target


def _normalize_csv(
    source: Path,
    target: Path,
    mapping: Mapping[str, str],
) -> Path:
    if all(canonical == actual for canonical, actual in mapping.items()):
        if source.resolve() == target.resolve():
            return source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
    frame = pd.read_csv(source, low_memory=False)
    frame = frame.rename(columns={actual: canonical for canonical, actual in mapping.items()})
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def normalize_inputs(
    contract: CBIInputContract,
    output_root: Path,
) -> tuple[Path, Path]:
    identity = all(
        canonical == actual
        for group in contract.column_map.values()
        for canonical, actual in group.items()
    ) and contract.metadata_file_name == "TMC_Identification.csv" and contract.readings_file_name == "Readings.csv"
    if identity:
        return contract.corridor_root, contract.model_link_map
    root = output_root / "normalized-inputs" / "corridors"
    for folder in sorted(path for path in contract.corridor_root.iterdir() if path.is_dir()):
        metadata = folder / contract.metadata_file_name
        readings = folder / contract.readings_file_name
        if not metadata.exists() and not readings.exists():
            continue
        _normalize_csv(
            metadata,
            root / folder.name / "TMC_Identification.csv",
            contract.column_map["metadata"],
        )
        _normalize_csv(
            readings,
            root / folder.name / "Readings.csv",
            contract.column_map["readings"],
        )
    mapping = _normalize_csv(
        contract.model_link_map,
        output_root / "normalized-inputs" / "full_tmc_to_link.csv",
        contract.column_map["mapping"],
    )
    return root, mapping
