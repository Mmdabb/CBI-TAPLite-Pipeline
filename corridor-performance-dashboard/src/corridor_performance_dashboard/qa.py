from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


class InputQAError(ValueError):
    pass


@dataclass(frozen=True)
class Inputs:
    cbi_corridors: Path
    mapmatching_root: Path
    assignment_root: Path
    mapping_products: dict[str, str]
    dashboard_product: str = "combined"
    mapping_file_name: str = "full_tmc_to_link.csv"
    route_summary_file_name: str = "full_route_match_summary.csv"
    performance_file_name: str = "link_performance.csv"
    link_file_name: str = "link.csv"
    observed_15min: Path | None = None
    model_link_map: Path | None = None
    measurement_root: Path | None = None


@dataclass(frozen=True)
class QAResult:
    status: str
    process: str
    corridor_count: int
    checked_files: list[str]


def load_column_map(path: Path | None) -> dict[str, dict[str, str]]:
    groups = {name: {} for name in ("mapping", "route_summary", "performance", "link", "observed")}
    if path is None:
        return groups
    if not path.is_file():
        raise InputQAError(f"Column-map JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group, values in payload.items():
        if group not in groups or not isinstance(values, dict):
            raise InputQAError(
                "Column-map groups must be mapping, route_summary, performance, link, or observed."
            )
        groups[group] = {str(key): str(value) for key, value in values.items()}
    return groups


def _stage_csv(source: Path, target: Path, aliases: dict[str, str]) -> None:
    if not source.is_file():
        raise InputQAError(f"Cannot normalize missing input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not aliases:
        shutil.copy2(source, target)
        return
    frame = pd.read_csv(source, low_memory=False)
    frame = frame.rename(columns={actual: canonical for canonical, actual in aliases.items()})
    frame.to_csv(target, index=False)


def normalize_layout(
    inputs: Inputs,
    output: Path,
    column_map: dict[str, dict[str, str]],
) -> Inputs:
    needs_mapping = (
        inputs.mapping_file_name != "full_tmc_to_link.csv"
        or inputs.route_summary_file_name != "full_route_match_summary.csv"
        or column_map["mapping"] or column_map["route_summary"]
    )
    needs_assignment = (
        inputs.performance_file_name != "link_performance.csv"
        or inputs.link_file_name != "link.csv"
        or column_map["performance"] or column_map["link"]
    )
    map_root = inputs.mapmatching_root
    assignment_root = inputs.assignment_root
    observed = inputs.observed_15min
    normalized = output / "normalized-inputs"
    if needs_mapping:
        map_root = normalized / "mapmatching"
        for product in sorted(set(inputs.mapping_products.values()) | {inputs.dashboard_product}):
            source = inputs.mapmatching_root / product
            target = map_root / product
            if (
                product == inputs.dashboard_product
                and product not in inputs.mapping_products.values()
                and not (source / inputs.mapping_file_name).is_file()
            ):
                continue
            _stage_csv(source / inputs.mapping_file_name, target / "full_tmc_to_link.csv", column_map["mapping"])
            _stage_csv(source / inputs.route_summary_file_name, target / "full_route_match_summary.csv", column_map["route_summary"])
    if needs_assignment:
        assignment_root = normalized / "assignment"
        for period in ("am", "md", "pm"):
            source = inputs.assignment_root / period
            target = assignment_root / period
            _stage_csv(source / inputs.performance_file_name, target / "link_performance.csv", column_map["performance"])
            _stage_csv(source / inputs.link_file_name, target / "link.csv", column_map["link"])
    if observed is not None and column_map["observed"]:
        target = normalized / "observed" / "observed_15min.csv"
        _stage_csv(observed, target, column_map["observed"])
        observed = target
    return Inputs(
        cbi_corridors=inputs.cbi_corridors,
        mapmatching_root=map_root,
        assignment_root=assignment_root,
        mapping_products=inputs.mapping_products,
        dashboard_product=inputs.dashboard_product,
        observed_15min=observed,
        model_link_map=inputs.model_link_map,
        measurement_root=inputs.measurement_root,
    )


def _csv(path: Path, required: set[str], label: str) -> list[str]:
    if not path.is_file():
        raise InputQAError(f"Missing {label}: {path}")
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        raise InputQAError(f"Cannot read {label} CSV header {path}: {exc}") from exc
    missing = sorted(required - set(columns))
    if missing:
        raise InputQAError(
            f"{label} {path} is missing required fields: {', '.join(missing)}"
        )
    return columns


def validate(inputs: Inputs, process: str) -> QAResult:
    if process not in {"measure", "dashboard", "all"}:
        raise InputQAError(f"Unsupported QA process: {process}")
    checked: list[str] = []
    if not inputs.cbi_corridors.is_dir():
        raise InputQAError(f"CBI corridor directory does not exist: {inputs.cbi_corridors}")
    corridors = []
    for folder in sorted(path for path in inputs.cbi_corridors.iterdir() if path.is_dir()):
        profile = folder / "03-profiles" / "average_weekday_profile.csv"
        reference = folder / "01-input-and-qc" / "link_reference.csv"
        if not profile.exists() and not reference.exists():
            continue
        _csv(profile, {"tmc_code", "t_min", "avg_weekday_speed_mph"}, "CBI profile")
        _csv(reference, {"tmc_code", "network_link_id"}, "CBI link reference")
        checked.extend([str(profile), str(reference)])
        corridors.append(folder.name)
    if not corridors:
        raise InputQAError(
            f"No complete CBI corridor products were found under {inputs.cbi_corridors}."
        )
    canonical = inputs.cbi_corridors.parent / "shared" / "network-mapping" / "canonical_node_pair_tmc.csv"
    _csv(canonical, {"tmc", "from_node_id", "to_node_id", "selected_for_node_pair_lookup"}, "canonical node-pair map")
    checked.append(str(canonical))
    for period in ("am", "md", "pm"):
        product = inputs.mapping_products[period]
        mapping = inputs.mapmatching_root / product / inputs.mapping_file_name
        summary = inputs.mapmatching_root / product / inputs.route_summary_file_name
        _csv(mapping, {"tmc", "link_id", "from_node_id", "to_node_id", "road_order", "facility_class"}, f"{period.upper()} mapping")
        _csv(summary, {"tmc", "route_link_count", "confidence", "status"}, f"{period.upper()} route summary")
        performance = inputs.assignment_root / period / inputs.performance_file_name
        columns = _csv(performance, {"link_id", "volume", "doc", "P"}, f"{period.upper()} link performance")
        if not any(name.startswith("spd_mph_") for name in columns):
            raise InputQAError(f"{performance} has no time-dependent spd_mph_HH:MM fields.")
        link = inputs.assignment_root / period / inputs.link_file_name
        _csv(link, {"link_id", "from_node_id", "to_node_id"}, f"{period.upper()} network link")
        checked.extend(map(str, (mapping, summary, performance, link)))
    if process in {"dashboard", "all"}:
        dashboard_mapping = inputs.mapmatching_root / inputs.dashboard_product / inputs.mapping_file_name
        dashboard_summary = inputs.mapmatching_root / inputs.dashboard_product / inputs.route_summary_file_name
        _csv(dashboard_mapping, {"tmc", "facility_class"}, "dashboard mapping")
        _csv(dashboard_summary, {"tmc", "status"}, "dashboard route summary")
        checked.extend([str(dashboard_mapping), str(dashboard_summary)])
        if inputs.observed_15min is None:
            raise InputQAError("--observed-15min is required for dashboard generation.")
        _csv(inputs.observed_15min, {"tmc_code", "measurement_tstamp", "speed"}, "15-minute observed speeds")
        if inputs.model_link_map is None:
            raise InputQAError("--model-link-map is required for dashboard generation.")
        _csv(inputs.model_link_map, {"tmc", "from_node_id", "to_node_id"}, "dashboard model-link map")
        if inputs.measurement_root is not None:
            required_measurement = (
                "01-corridor-results/corridor_metrics.csv",
                "01-corridor-results/overall_metrics.csv",
                "06-figures/figure_manifest.csv",
                "07-run-metadata/run_manifest.json",
            )
            missing = [name for name in required_measurement if not (inputs.measurement_root / name).is_file()]
            if missing:
                raise InputQAError(
                    f"Corridor measurement is incomplete under {inputs.measurement_root}; missing: {missing}"
                )
        checked.extend([str(inputs.observed_15min), str(inputs.model_link_map)])
    return QAResult("PASS", process, len(corridors), checked)


def write_report(result: QAResult, output: Path) -> Path:
    path = output / "qa" / "input_qa.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return path
