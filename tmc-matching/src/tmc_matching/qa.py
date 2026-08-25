from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


DEFAULT_COLUMN_MAP: dict[str, dict[str, str]] = {
    "tmc": {
        "tmc": "tmc",
        "road": "road",
        "direction": "direction",
        "road_order": "road_order",
        "start_longitude": "start_longitude",
        "start_latitude": "start_latitude",
        "end_longitude": "end_longitude",
        "end_latitude": "end_latitude",
        "miles": "miles",
    },
    "link": {
        "link_id": "link_id",
        "from_node_id": "from_node_id",
        "to_node_id": "to_node_id",
    },
    "node": {
        "node_id": "node_id",
        "x_coord": "x_coord",
        "y_coord": "y_coord",
    },
}


@dataclass(frozen=True)
class MatchInputs:
    input_dir: Path
    tmc_file: Path
    period_link_files: dict[str, Path]
    period_node_files: dict[str, Path]
    column_map: dict[str, dict[str, str]]


@dataclass(frozen=True)
class QAResult:
    status: str
    files: dict[str, str]
    row_counts: dict[str, int]
    column_map: dict[str, dict[str, str]]


class InputQAError(ValueError):
    """Raised when a requested matching process cannot safely start."""


def load_column_map(path: Path | None) -> dict[str, dict[str, str]]:
    mapping = {group: values.copy() for group, values in DEFAULT_COLUMN_MAP.items()}
    if path is None:
        return mapping
    if not path.is_file():
        raise InputQAError(f"Column-map JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise InputQAError("Column-map JSON must contain an object.")
    for group in mapping:
        supplied = payload.get(group, {})
        if not isinstance(supplied, Mapping):
            raise InputQAError(f"column-map.{group} must be an object.")
        mapping[group].update({str(k): str(v) for k, v in supplied.items()})
    return mapping


def resolve_match_inputs(
    input_dir: Path,
    *,
    tmc_file_name: str,
    network_dir_name: str,
    periods: tuple[str, ...],
    link_file_name: str,
    node_file_name: str,
    column_map_path: Path | None,
) -> MatchInputs:
    root = input_dir.expanduser().resolve()
    mapping = load_column_map(column_map_path)
    return MatchInputs(
        input_dir=root,
        tmc_file=root / tmc_file_name,
        period_link_files={
            period: root / network_dir_name / period / link_file_name
            for period in periods
        },
        period_node_files={
            period: root / network_dir_name / period / node_file_name
            for period in periods
        },
        column_map=mapping,
    )


def _check_csv(path: Path, required_source_columns: set[str]) -> tuple[int, list[str]]:
    if not path.is_file():
        raise InputQAError(
            f"Required input is unavailable: {path}. "
            "Check the input directory and filename arguments."
        )
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        raise InputQAError(f"Cannot read CSV header {path}: {exc}") from exc
    missing = sorted(required_source_columns.difference(header.columns))
    if missing:
        raise InputQAError(
            f"{path} is missing configured columns: {', '.join(missing)}"
        )
    row_count = sum(1 for _ in path.open("r", encoding="utf-8-sig", errors="replace")) - 1
    if row_count <= 0:
        raise InputQAError(f"Required input contains no data rows: {path}")
    return row_count, list(header.columns)


def validate_match_inputs(inputs: MatchInputs) -> QAResult:
    if not inputs.input_dir.is_dir():
        raise InputQAError(f"Input directory does not exist: {inputs.input_dir}")
    counts: dict[str, int] = {}
    files: dict[str, str] = {}
    counts["tmc"], _ = _check_csv(
        inputs.tmc_file, set(inputs.column_map["tmc"].values())
    )
    files["tmc"] = str(inputs.tmc_file)
    for period, path in inputs.period_link_files.items():
        key = f"{period}_link"
        counts[key], _ = _check_csv(path, set(inputs.column_map["link"].values()))
        files[key] = str(path)
    for period, path in inputs.period_node_files.items():
        key = f"{period}_node"
        counts[key], _ = _check_csv(path, set(inputs.column_map["node"].values()))
        files[key] = str(path)
    return QAResult("PASS", files, counts, inputs.column_map)


def write_qa_report(result: QAResult, output_root: Path) -> Path:
    target = output_root / "qa" / "input_qa.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return target


def normalize_csv(
    source: Path,
    target: Path,
    canonical_to_source: Mapping[str, str],
) -> Path:
    """Map user-selected input fields to the package's canonical schema."""

    if all(canonical == actual for canonical, actual in canonical_to_source.items()):
        return source
    frame = pd.read_csv(source, low_memory=False)
    reverse = {actual: canonical for canonical, actual in canonical_to_source.items()}
    frame = frame.rename(columns=reverse)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def normalized_inputs(inputs: MatchInputs, output_root: Path) -> MatchInputs:
    normalized_root = output_root / "normalized-inputs"
    return MatchInputs(
        input_dir=inputs.input_dir,
        tmc_file=normalize_csv(
            inputs.tmc_file,
            normalized_root / "TMC_Identification.csv",
            inputs.column_map["tmc"],
        ),
        period_link_files={
            period: normalize_csv(
                source,
                normalized_root / period / "link.csv",
                inputs.column_map["link"],
            )
            for period, source in inputs.period_link_files.items()
        },
        period_node_files={
            period: normalize_csv(
                source,
                normalized_root / period / "node.csv",
                inputs.column_map["node"],
            )
            for period, source in inputs.period_node_files.items()
        },
        column_map=DEFAULT_COLUMN_MAP,
    )

