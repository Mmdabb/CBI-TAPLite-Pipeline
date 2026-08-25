from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

from .config import PipelineConfig


class InputQAError(ValueError):
    pass


def _csv_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return set(next(csv.reader(stream)))


def validate(config: PipelineConfig) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    files = config.files
    for key in ("tmc_metadata", "tmc_readings"):
        path = files[key]
        if path is None or not path.is_file():
            errors.append(f"files.{key} is missing: {path}")
    for key in ("matching_input", "base_network", "cube_scenario"):
        path = files[key]
        if path is None or not path.is_dir():
            errors.append(f"files.{key} is not a directory: {path}")
    metadata = files["tmc_metadata"]
    readings = files["tmc_readings"]
    if metadata and metadata.is_file():
        columns = _csv_columns(metadata)
        if not ({"tmc", "tmc_code"} & columns):
            errors.append(f"TMC metadata has no tmc/tmc_code field: {metadata}")
    if readings and readings.is_file():
        columns = _csv_columns(readings)
        if not ({"tmc", "tmc_code"} & columns):
            errors.append(f"Readings have no tmc/tmc_code field: {readings}")
        if not ({"speed", "speed_mph", "avg_speed"} & columns):
            errors.append(f"Readings have no recognized speed field: {readings}")
    matching = files["matching_input"]
    if matching and matching.is_dir():
        required = [matching / "TMC_Identification.csv"]
        for period in ("am", "md", "pm"):
            required.extend(
                [matching / "network" / period / "link.csv", matching / "network" / period / "node.csv"]
            )
        errors.extend(f"Matching input is missing: {path}" for path in required if not path.is_file())
    network = files["base_network"]
    if network and network.is_dir():
        for period in ("am", "md", "pm"):
            link = network / period / "link.csv"
            node = network / period / "node.csv"
            if not link.is_file() or not node.is_file():
                errors.append(f"Base {period.upper()} network requires link.csv and node.csv")
    cube = files["cube_scenario"]
    if cube and cube.is_dir():
        if not any(cube.glob("*.DBF")) and not any(cube.glob("*.dbf")):
            errors.append(f"Cube scenario has no network DBF: {cube}")
        for token in ("AM", "MD", "PM"):
            if not any(cube.glob(f"*{token}*.omx")):
                errors.append(f"Cube scenario has no {token} OMX matrix: {cube}")
    if config.output_root == config.input_root or config.input_root in config.output_root.parents:
        errors.append("output_root must not be the input root or a descendant of it")
    if importlib.util.find_spec("taplite4mpo") is None:
        errors.append(
            "The pinned taplite4mpo kernel is not installed. Run the local "
            "nvta-taplite-workflow environment setup first."
        )
    qvdf_override = files.get("qvdf_override_dictionary")
    if qvdf_override is not None and not qvdf_override.is_file():
        errors.append(f"QVDF override dictionary does not exist: {qvdf_override}")
    report = {
        "status": "FAIL" if errors else "PASS",
        "repository_root": str(config.repository_root),
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "workers": config.workers,
        "logical_cores": os.cpu_count() or 1,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        raise InputQAError("Input QA failed:\n- " + "\n- ".join(errors))
    return report


def write_report(config: PipelineConfig, report: dict[str, object]) -> Path:
    root = config.output_root / "qa"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "input_qa.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path

