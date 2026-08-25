from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dashboard_filters import is_managed_corridor
from .corridor_profile import stage_corridor_profile_measurement
from .methods import stage_methods_page


@dataclass(frozen=True)
class DashboardBuildSettings:
    package_root: Path
    corridor_results_root: Path | None = None
    mapmatching_product_root: Path | None = None
    qvdf_report_root: Path | None = None
    corridor_measurement_root: Path | None = None
    assignment_root: Path | None = None
    ritis_15min_path: Path | None = None
    profile_selection_overrides_path: Path | None = None
    output_root: Path | None = None
    force: bool = False

    def __post_init__(self) -> None:
        package_root = Path(self.package_root).resolve()
        object.__setattr__(self, "package_root", package_root)
        required = (
            "corridor_results_root", "mapmatching_product_root",
            "qvdf_report_root", "corridor_measurement_root",
            "assignment_root", "ritis_15min_path", "output_root",
        )
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Explicit dashboard paths are required: " + ", ".join(missing)
            )
        for field_name in required:
            object.__setattr__(
                self, field_name, Path(getattr(self, field_name)).resolve()
            )
        if self.profile_selection_overrides_path is not None:
            object.__setattr__(
                self,
                "profile_selection_overrides_path",
                Path(self.profile_selection_overrides_path).resolve(),
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dashboard input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dashboard input: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _number(value: Any, digits: int = 3) -> float | int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    rounded = round(number, digits)
    return int(rounded) if rounded.is_integer() else rounded


def _parse_linestring_wgs84(value: str | None) -> list[list[float]]:
    if not value:
        return []
    match = re.fullmatch(
        r"\s*LINESTRING\s*\((.*)\)\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    points: list[list[float]] = []
    for pair in match.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            return []
        try:
            longitude, latitude = float(parts[0]), float(parts[1])
        except ValueError:
            return []
        points.append([round(latitude, 6), round(longitude, 6)])
    return points if len(points) >= 2 else []


def _workspace_label(path: Path, package_root: Path) -> str:
    try:
        return path.relative_to(package_root).as_posix()
    except ValueError:
        return path.name


def _prepare_output(settings: DashboardBuildSettings) -> None:
    output_root = settings.output_root.resolve()
    package_root = settings.package_root.resolve()
    if output_root == package_root or package_root not in output_root.parents:
        raise ValueError("Dashboard output must be below the explicit package root")
    source_roots = {
        settings.corridor_results_root.resolve(),
        settings.mapmatching_product_root.resolve(),
        settings.qvdf_report_root.resolve(),
        settings.corridor_measurement_root.resolve(),
    }
    if output_root in source_roots:
        raise ValueError("Dashboard output cannot overwrite an input directory")
    existing_products = (
        [item for item in output_root.iterdir() if item.name not in {"logs", "qa", "normalized-inputs"}]
        if output_root.exists() else []
    )
    if existing_products:
        if not settings.force:
            raise FileExistsError(
                f"Dashboard output is not empty: {output_root}. Use --force."
            )
        for item in existing_products:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _load_mapmatching(
    product_root: Path,
) -> tuple[
    dict[str, list[list[list[float]]]],
    dict[str, dict[str, str]],
    set[str],
]:
    mapping_rows = _read_csv(product_root / "full_tmc_to_link.csv")
    facility_classes: dict[str, set[str]] = defaultdict(set)
    for row in mapping_rows:
        tmc = str(row.get("tmc", "")).strip().upper()
        facility_class = str(row.get("facility_class", "")).strip().lower()
        if tmc:
            facility_classes[tmc].add(facility_class or "unclassified")
    general_purpose_tmcs = {
        tmc for tmc, values in facility_classes.items() if values == {"gp"}
    }

    geometry_by_tmc: dict[str, list[list[list[float]]]] = defaultdict(list)
    geometry_seen: dict[str, set[str]] = defaultdict(set)
    for row in mapping_rows:
        tmc = str(row.get("tmc", "")).strip().upper()
        geometry_text = str(row.get("geometry_wgs84", "")).strip()
        if (
            tmc not in general_purpose_tmcs
            or not geometry_text
            or geometry_text in geometry_seen[tmc]
        ):
            continue
        points = _parse_linestring_wgs84(geometry_text)
        if points:
            geometry_by_tmc[tmc].append(points)
            geometry_seen[tmc].add(geometry_text)

    match_by_tmc: dict[str, dict[str, str]] = {}
    for row in _read_csv(product_root / "full_route_match_summary.csv"):
        tmc = str(row.get("tmc", "")).strip().upper()
        if tmc in general_purpose_tmcs:
            match_by_tmc[tmc] = row
    return geometry_by_tmc, match_by_tmc, general_purpose_tmcs


def _load_quality(path: Path) -> dict[str, list[dict[str, Any]]]:
    quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        corridor = str(row.get("key", "")).strip()
        if not corridor:
            continue
        quality[corridor].append(
            {
                "period": row.get("period"),
                "n_links": _number(row.get("n_links"), 0),
                "duration_r2": _number(row.get("step1_DC_P_R2")),
                "speed_r2": _number(row.get("step2_P_mag_R2")),
                "duration_mape_pct": _number(row.get("P_MAPE_pct"), 1),
                "minimum_speed_mape_pct": _number(
                    row.get("vt2_MAPE_pct"), 1
                ),
                "t0_mae_min": _number(row.get("t0_MAE_min"), 1),
                "gates_pass": row.get("gates_pass"),
            }
        )
    period_order = {"AM": 0, "MD": 1, "PM": 2}
    for rows in quality.values():
        rows.sort(key=lambda item: period_order.get(str(item["period"]), 9))
    return quality


def _load_qvdf_coverage(path: Path) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        corridor = str(row.get("corridor", "")).strip()
        if not corridor:
            continue
        coverage[corridor] = {
            "coverage_status": row.get("coverage_status") or "unknown",
            "average_accepted_episodes": _number(
                row.get("average_accepted_episodes"), 0
            ),
            "daily_accepted_episodes": _number(
                row.get("daily_accepted_episodes"), 0
            ),
            "calibrated_link_periods": _number(
                row.get("calibrated_link_periods"), 0
            ),
            "selected_link_periods": _number(
                row.get("selected_link_periods"), 0
            ),
            "ready_link_periods": _number(
                row.get("ready_link_periods"), 0
            ),
            "assignment_links": _number(row.get("assignment_links"), 0),
        }
    return coverage


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _load_tmc_map_metrics(
    measurement_root: Path,
) -> dict[tuple[str, str], dict[str, float | int | None]]:
    """Summarize daily observed/model performance for map tooltips."""

    source = (
        Path(measurement_root)
        / "02-tmc-results"
        / "tmc_daily_profiles.csv"
    )
    if not source.is_file():
        return {}
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            corridor = str(row.get("corridor", "")).strip()
            tmc = str(row.get("tmc_code", "")).strip().upper()
            if not corridor or not tmc:
                continue
            observed = _finite_float(row.get("observed_tmc_speed_mph"))
            modeled = _finite_float(row.get("model_tmc_speed_mph"))
            values = totals[(corridor, tmc)]
            if observed is not None:
                values["observed_sum"] += observed
                values["observed_count"] += 1
            if modeled is not None:
                values["model_sum"] += modeled
                values["model_count"] += 1
            if observed is None or modeled is None:
                continue
            error = abs(modeled - observed)
            values["absolute_error_sum"] += error
            values["matched_count"] += 1
            if observed > 0:
                values["ape_sum"] += error / observed * 100.0
                values["ape_count"] += 1

    result: dict[tuple[str, str], dict[str, float | int | None]] = {}
    for key, values in totals.items():
        observed_count = int(values.get("observed_count", 0))
        model_count = int(values.get("model_count", 0))
        matched_count = int(values.get("matched_count", 0))
        ape_count = int(values.get("ape_count", 0))
        result[key] = {
            "observed_average_speed_mph": (
                values["observed_sum"] / observed_count
                if observed_count
                else None
            ),
            "model_average_speed_mph": (
                values["model_sum"] / model_count if model_count else None
            ),
            "speed_mae_mph": (
                values["absolute_error_sum"] / matched_count
                if matched_count
                else None
            ),
            "speed_mape_pct": (
                values["ape_sum"] / ape_count if ape_count else None
            ),
            "matched_interval_count": matched_count,
        }
    return result


def _bounds_for_segments(
    segments: list[list[list[float]]],
) -> list[list[float]] | None:
    points = [point for segment in segments for point in segment]
    if not points:
        return None
    latitudes = [point[0] for point in points]
    longitudes = [point[1] for point in points]
    return [
        [min(latitudes), min(longitudes)],
        [max(latitudes), max(longitudes)],
    ]


def _qvdf_corridor_report_root(qvdf_root: Path) -> Path:
    """Resolve projection staging or an already integrated report layout."""

    direct = Path(qvdf_root) / "corridors"
    integrated = Path(qvdf_root) / "reports" / "corridors"
    return direct if direct.is_dir() else integrated


def _corridor_record(
    corridor_dir: Path,
    geometry_by_tmc: dict[str, list[list[list[float]]]],
    match_by_tmc: dict[str, dict[str, str]],
    quality: dict[str, list[dict[str, Any]]],
    qvdf_coverage: dict[str, dict[str, Any]],
    qvdf_root: Path,
    general_purpose_tmcs: set[str],
    tmc_map_metrics: dict[
        tuple[str, str], dict[str, float | int | None]
    ],
) -> dict[str, Any]:
    corridor = corridor_dir.name
    manifest = _read_json(
        corridor_dir / "11-run-metadata" / "run_manifest.json"
    )
    link_rows = _read_csv(
        corridor_dir / "01-input-and-qc" / "link_reference.csv"
    )
    link_rows = [
        row
        for row in link_rows
        if str(row.get("tmc_code", "")).strip().upper()
        in general_purpose_tmcs
    ]
    tmc_ids = sorted(
        {
            str(row.get("tmc_code", "")).strip().upper()
            for row in link_rows
            if str(row.get("tmc_code", "")).strip()
        }
    )
    direction = next(
        (
            str(row.get("direction", "")).strip()
            for row in link_rows
            if str(row.get("direction", "")).strip()
        ),
        "",
    )
    total_miles = sum(
        float(row["length_mi"])
        for row in link_rows
        if row.get("length_mi") not in (None, "")
    )
    segments: list[list[list[float]]] = []
    link_by_tmc: dict[str, dict[str, str]] = {}
    for row in link_rows:
        tmc = str(row.get("tmc_code", "")).strip().upper()
        if tmc and tmc not in link_by_tmc:
            link_by_tmc[tmc] = row
    tmc_layers: list[dict[str, Any]] = []
    seen_segments: set[tuple[tuple[float, float], ...]] = set()
    mapped_tmc = 0
    for tmc in tmc_ids:
        tmc_segments = geometry_by_tmc.get(tmc, [])
        if tmc_segments:
            mapped_tmc += 1
        for segment in tmc_segments:
            key = tuple((point[0], point[1]) for point in segment)
            if key not in seen_segments:
                segments.append(segment)
                seen_segments.add(key)
        link = link_by_tmc.get(tmc, {})
        metrics = tmc_map_metrics.get((corridor, tmc), {})
        tmc_layers.append(
            {
                "tmc_code": tmc,
                "direction": str(link.get("direction", direction)).strip(),
                "road_order": _number(link.get("road_order")),
                "link_id": _number(link.get("network_link_id")),
                "from_node_id": _number(
                    link.get("network_from_node_id")
                ),
                "to_node_id": _number(link.get("network_to_node_id")),
                "segments": tmc_segments,
                **metrics,
            }
        )

    status_counts = Counter(
        (match_by_tmc.get(tmc) or {}).get("status", "missing")
        for tmc in tmc_ids
    )
    missing_tmc = [
        {
            "tmc": tmc,
            "status": (match_by_tmc.get(tmc) or {}).get("status", "missing"),
        }
        for tmc in tmc_ids
        if not geometry_by_tmc.get(tmc)
    ]
    report_page = (
        _qvdf_corridor_report_root(qvdf_root)
        / corridor
        / "index.html"
    )
    return {
        "id": corridor,
        "label": corridor.replace("_", " "),
        "direction": direction,
        "run_status": manifest.get("status"),
        "tmc_count": len(tmc_ids),
        "mapped_tmc_count": mapped_tmc,
        "mapping_share_pct": (
            round(mapped_tmc / len(tmc_ids) * 100.0, 1) if tmc_ids else None
        ),
        "corridor_miles": round(total_miles, 2),
        "segments": segments,
        "tmcs": tmc_layers,
        "bounds": _bounds_for_segments(segments),
        "mapping_status_counts": dict(sorted(status_counts.items())),
        "missing_tmc": missing_tmc,
        "result_stats": {
            "raw_qc_pass_pct": (
                round(float(manifest["raw_qc_pass_rate"]) * 100.0, 1)
                if manifest.get("raw_qc_pass_rate") is not None
                else None
            ),
            "episodes_detected": manifest.get("episodes_detected"),
            "episodes_clean": manifest.get("episodes_clean"),
            "calibration_rows": manifest.get("calibration_rows"),
            "average_weekday_calibration_rows": manifest.get(
                "average_weekday_calibration_rows"
            ),
            "figures": manifest.get("figures"),
        },
        "quality": quality.get(corridor, []),
        "qvdf": qvdf_coverage.get(
            corridor,
            {
                "coverage_status": "missing",
                "average_accepted_episodes": 0,
                "daily_accepted_episodes": 0,
                "calibrated_link_periods": 0,
                "selected_link_periods": 0,
                "ready_link_periods": 0,
                "assignment_links": 0,
            },
        ),
        "report_url": (
            f"reports/corridors/{corridor}/index.html"
            if report_page.is_file()
            else None
        ),
    }


def _build_data(settings: DashboardBuildSettings) -> dict[str, Any]:
    geometry_by_tmc, match_by_tmc, general_purpose_tmcs = _load_mapmatching(
        settings.mapmatching_product_root
    )
    quality_path = (
        settings.corridor_results_root.parent
        / "summary"
        / "_QUALITY_SUMMARY.csv"
    )
    if not quality_path.is_file():
        quality_path = (
            settings.corridor_results_root
            / "_run-summary"
            / "_QUALITY_SUMMARY.csv"
        )
    quality = _load_quality(quality_path)
    qvdf_coverage = _load_qvdf_coverage(
        settings.qvdf_report_root / "data" / "corridor_coverage.csv"
    )
    tmc_map_metrics = _load_tmc_map_metrics(
        settings.corridor_measurement_root
    )
    corridor_dirs = sorted(
        (
            path
            for path in settings.corridor_results_root.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith("_")
                and not is_managed_corridor(path.name)
            )
        ),
        key=lambda path: path.name,
    )
    corridors = [
        _corridor_record(
            path,
            geometry_by_tmc,
            match_by_tmc,
            quality,
            qvdf_coverage,
            settings.qvdf_report_root,
            general_purpose_tmcs,
            tmc_map_metrics,
        )
        for path in corridor_dirs
    ]
    corridors = [corridor for corridor in corridors if corridor["tmc_count"] > 0]
    corridors.sort(
        key=lambda corridor: (
            -int(corridor["tmc_count"]),
            str(corridor["id"]),
        )
    )
    report_ids = {
        path.name
        for path in _qvdf_corridor_report_root(
            settings.qvdf_report_root
        ).iterdir()
        if path.is_dir() and (path / "index.html").is_file()
    }
    corridor_ids = {corridor["id"] for corridor in corridors}
    all_bounds = _bounds_for_segments(
        [
            segment
            for corridor in corridors
            for segment in corridor["segments"]
        ]
    )
    total_tmc = sum(int(corridor["tmc_count"]) for corridor in corridors)
    mapped_tmc = sum(
        int(corridor["mapped_tmc_count"]) for corridor in corridors
    )
    return {
        "summary": {
            "corridors": len(corridors),
            "corridors_with_reports": sum(
                corridor["report_url"] is not None for corridor in corridors
            ),
            "corridors_fully_mapped": sum(
                not corridor["missing_tmc"] for corridor in corridors
            ),
            "corridors_partially_mapped": sum(
                bool(corridor["missing_tmc"]) for corridor in corridors
            ),
            "total_tmc": total_tmc,
            "mapped_tmc": mapped_tmc,
            "mapping_share_pct": (
                round(mapped_tmc / total_tmc * 100.0, 1)
                if total_tmc
                else None
            ),
            "missing_tmc": total_tmc - mapped_tmc,
            "qvdf_ready_corridors": sum(
                corridor["qvdf"]["coverage_status"] == "ready"
                for corridor in corridors
            ),
        },
        "bounds": all_bounds,
        "corridors": corridors,
        "audit": {
            "missing_report_corridors": sorted(corridor_ids - report_ids),
            "orphan_report_corridors": sorted(report_ids - corridor_ids),
            "partial_geometry_corridors": [
                {
                    "corridor": corridor["id"],
                    "missing_tmc": len(corridor["missing_tmc"]),
                    "tmc": corridor["missing_tmc"],
                }
                for corridor in corridors
                if corridor["missing_tmc"]
            ],
        },
        "sources": {
            "corridor_results": _workspace_label(
                settings.corridor_results_root, settings.package_root
            ),
            "tmc_mapmatching": _workspace_label(
                settings.mapmatching_product_root, settings.package_root
            ),
            "qvdf_reports": _workspace_label(
                settings.output_root / "data", settings.package_root
            ),
            "ritis_15min": _workspace_label(
                settings.ritis_15min_path, settings.package_root
            ),
        },
    }


def _stage_reports(
    settings: DashboardBuildSettings,
    eligible_corridor_ids: set[str],
) -> int:
    source = settings.qvdf_report_root / "corridors"
    target = settings.output_root / "reports" / "corridors"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for corridor_dir in sorted(source.iterdir()):
        page = corridor_dir / "index.html"
        if (
            not corridor_dir.is_dir()
            or not page.is_file()
            or is_managed_corridor(corridor_dir.name)
            or corridor_dir.name not in eligible_corridor_ids
        ):
            continue
        destination = target / corridor_dir.name
        shutil.copytree(corridor_dir, destination)
        staged_page = destination / "index.html"
        html = staged_page.read_text(encoding="utf-8")
        html = html.replace(
            'href="../../index.html"',
            'href="../../../index.html"',
        ).replace("All corridors", "Interactive corridor map")
        staged_page.write_text(html, encoding="utf-8")
        copied += 1
    return copied


def _stage_projection_assets(settings: DashboardBuildSettings) -> dict[str, int]:
    """Carry projection downloads and summary figures into the one dashboard."""

    counts = {"data_files": 0, "summary_figures": 0}
    for source_name, target_name, pattern in (
        ("data", "data", "*"),
        ("figures", "figures", "*.png"),
    ):
        source = settings.qvdf_report_root / source_name
        if not source.is_dir():
            continue
        target = settings.output_root / target_name
        target.mkdir(parents=True, exist_ok=True)
        for path in source.glob(pattern):
            if not path.is_file():
                continue
            shutil.copy2(path, target / path.name)
            key = "data_files" if source_name == "data" else "summary_figures"
            counts[key] += 1
    return counts


def _script_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NVTA Corridor Performance Explorer</title>
  <link rel="preconnect" href="https://unpkg.com">
  <link rel="preconnect" href="https://tile.openstreetmap.org">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {
      --ink: #12212f;
      --muted: #5c6b78;
      --line: #dbe3e9;
      --paper: #fff;
      --soft: #f4f7f8;
      --navy: #163a5f;
      --blue: #1976a8;
      --teal: #178578;
      --orange: #e3762b;
      --amber-soft: #fff3e8;
      --green-soft: #e8f6f2;
      --shadow: 0 14px 34px rgba(18, 33, 47, .09);
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      padding-left: 182px;
      color: var(--ink);
      background: var(--soft);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .utility-rail {
      position: fixed; inset: 0 auto 0 0; z-index: 1000; width: 182px;
      padding: 18px 12px; color: #fff; background: #102c48;
      box-shadow: 5px 0 18px rgba(12,35,55,.18);
    }
    .utility-brand { margin: 0 7px 18px; color: #a9e4df; font-size: 12px; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }
    .utility-link, .utility-rail summary {
      display: block; width: 100%; margin: 7px 0; padding: 11px 12px;
      color: #fff; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
      border-radius: 8px; font-size: 12px; font-weight: 800; text-decoration: none; cursor: pointer;
    }
    .utility-link:hover, .utility-rail summary:hover { background: rgba(255,255,255,.16); }
    .utility-link.active { color: #12395d; background: #fff; }
    .utility-rail details[open] summary { margin-bottom: 8px; }
    .utility-downloads { max-height: 55vh; overflow: auto; }
    .utility-downloads a { display: block; padding: 7px 8px; color: #d8e9f2; font-size: 11px; line-height: 1.25; text-decoration: none; }
    .utility-downloads a:hover { color: #fff; text-decoration: underline; }
    .utility-downloads .download-all { margin-top: 7px; color: #a9e4df; font-weight: 850; }
    button, input { font: inherit; }
    .page-head {
      color: #fff;
      background:
        radial-gradient(circle at 82% -20%, rgba(62, 177, 173, .55), transparent 36rem),
        linear-gradient(122deg, #102c48, #15577d 72%, #147b80);
      padding: 28px clamp(18px, 4vw, 58px) 36px;
    }
    .head-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 24px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .profile-page-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 10px 14px;
      color: #12395d;
      background: #fff;
      border: 1px solid rgba(255,255,255,.58);
      border-radius: 8px;
      box-shadow: 0 8px 20px rgba(9, 32, 50, .16);
      font-size: 12px;
      font-weight: 850;
      text-decoration: none;
      white-space: nowrap;
    }
    .profile-page-link:hover { transform: translateY(-1px); }
    .eyebrow {
      margin: 0 0 7px;
      color: #a9e4df;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 46px); line-height: 1.05; }
    .lede { max-width: 790px; margin: 12px 0 0; color: #d8e9f2; line-height: 1.55; }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      max-width: 1500px;
      margin: -18px auto 18px;
      padding: 0 18px;
      position: relative;
      z-index: 2;
    }
    .summary-card {
      min-height: 88px;
      padding: 15px 16px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .summary-value { font-size: 27px; font-weight: 850; letter-spacing: -.03em; }
    .summary-label { margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.35; }
    .workspace {
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      gap: 16px;
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 18px 28px;
    }
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .sidebar { min-height: 700px; }
    .sidebar-head { padding: 16px; border-bottom: 1px solid var(--line); }
    .sidebar-head h2, .panel-head h2 { margin: 0; font-size: 16px; }
    .title-with-help { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
    .learn-more { color: var(--blue); font-size: 11px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    .learn-more:hover { text-decoration: underline; }
    .search {
      width: 100%;
      margin-top: 12px;
      padding: 10px 12px;
      color: var(--ink);
      background: #f8fafb;
      border: 1px solid var(--line);
      border-radius: 8px;
      outline: none;
    }
    .search:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(25, 118, 168, .12); }
    .corridor-list { max-height: 720px; overflow-y: auto; padding: 8px; }
    .corridor-button {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 5px 10px;
      width: 100%;
      padding: 11px 10px;
      color: var(--ink);
      text-align: left;
      background: transparent;
      border: 0;
      border-radius: 8px;
      cursor: pointer;
    }
    .corridor-button:hover { background: #f0f6f8; }
    .corridor-button.active { color: #fff; background: var(--navy); }
    .corridor-name { font-size: 13px; font-weight: 820; }
    .corridor-meta { color: var(--muted); font-size: 11px; }
    .corridor-button.active .corridor-meta { color: #cae2ef; }
    .dot { width: 9px; height: 9px; margin-top: 4px; border-radius: 50%; background: var(--teal); }
    .dot.partial { background: var(--orange); }
    .main-stack { display: grid; gap: 16px; min-width: 0; }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .panel-subtitle { margin-top: 3px; color: var(--muted); font-size: 12px; }
    #map { height: 570px; background: #dfe9ed; }
    .map-key {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .speed-key { display: inline-flex; align-items: center; gap: 5px; }
    .speed-swatch { width: 13px; height: 7px; border-radius: 3px; }
    .tmc-map-tooltip { min-width: 340px; padding: 0; }
    .tmc-map-tooltip .leaflet-tooltip-content { margin: 0; }
    .tmc-tooltip-grid {
      display: grid; grid-template-columns: minmax(145px, 1fr) minmax(130px, .9fr);
      gap: 0; color: #243746; font-size: 12px; line-height: 1.45;
    }
    .tmc-tooltip-col { padding: 9px 11px; }
    .tmc-tooltip-col + .tmc-tooltip-col { border-left: 1px solid #d7dde4; background: #f7fafb; }
    .tmc-tooltip-title { margin-bottom: 5px; color: #12395d; font-size: 13px; font-weight: 850; }
    .tmc-tooltip-label { color: #637381; }
    .detail-body { padding: 16px; }
    .detail-top {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    .detail-title { margin: 0; font-size: 25px; letter-spacing: -.02em; }
    .detail-caption { margin: 5px 0 0; color: var(--muted); font-size: 13px; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 27px;
      padding: 5px 10px;
      border-radius: 999px;
      color: #176558;
      background: var(--green-soft);
      font-size: 11px;
      font-weight: 850;
      text-transform: capitalize;
      white-space: nowrap;
    }
    .status-pill.limited { color: #935018; background: var(--amber-soft); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      min-height: 76px;
      padding: 11px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #fbfcfd;
    }
    .metric-value { font-size: 19px; font-weight: 850; overflow-wrap: anywhere; }
    .metric-label { margin-top: 4px; color: var(--muted); font-size: 11px; line-height: 1.3; }
    .quality-wrap { margin-top: 14px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 9px 10px; text-align: right; border-bottom: 1px solid var(--line); }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); background: #f7f9fa; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
    .mapping-note {
      margin-top: 12px;
      padding: 10px 12px;
      color: #7d4318;
      background: var(--amber-soft);
      border: 1px solid #f4d6bd;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.5;
    }
    .report-panel { grid-column: 1 / -1; max-width: 1500px; margin: 0 auto 34px; width: calc(100% - 36px); }
    .report-head-copy { min-width: 0; }
    .report-actions { display: flex; gap: 10px; align-items: center; }
    .report-link {
      color: var(--blue);
      font-size: 12px;
      font-weight: 800;
      text-decoration: none;
      white-space: nowrap;
    }
    .report-link:hover { text-decoration: underline; }
    .report-frame {
      display: block;
      width: 100%;
      min-height: 1200px;
      border: 0;
      background: #fff;
    }
    .report-empty { padding: 60px 20px; color: var(--muted); text-align: center; }
    .downloads {
      max-width: 1500px;
      margin: 0 auto 22px;
      padding: 0 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
    }
    .download-link {
      color: var(--blue);
      background: #fff;
      border: 1px solid #cfd8e5;
      border-radius: 7px;
      padding: 8px 11px;
      font-size: 12px;
      font-weight: 750;
      text-decoration: none;
    }
    .download-link:hover { background: #f2f6fb; }
    .footer {
      max-width: 1500px;
      margin: 0 auto 28px;
      padding: 0 18px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }
    @media (max-width: 1100px) {
      .summary { grid-template-columns: repeat(3, 1fr); }
      .metric-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 800px) {
      body { padding-left: 0; }
      .utility-rail { position: static; width: auto; padding: 10px 12px; }
      .utility-brand { display: none; }
      .utility-link, .utility-rail details { display: inline-block; width: auto; margin: 3px; vertical-align: top; }
      .utility-rail summary { margin: 0; }
      .utility-downloads { position: absolute; z-index: 1001; width: min(320px, 90vw); padding: 8px; background: #102c48; border-radius: 8px; box-shadow: var(--shadow); }
      .summary { grid-template-columns: repeat(2, 1fr); }
      .workspace { grid-template-columns: 1fr; }
      .sidebar { min-height: auto; }
      .corridor-list { max-height: 260px; }
      #map { height: 480px; }
    }
    @media (max-width: 540px) {
      .head-row { flex-direction: column; }
      .summary { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, 1fr); }
      .map-key { display: none; }
      .detail-top { flex-direction: column; }
      .report-actions { align-items: flex-end; flex-direction: column; }
    }
  </style>
</head>
<body>
  <aside class="utility-rail" aria-label="Dashboard navigation">
    <div class="utility-brand">NVTA CBI</div>
    <a class="utility-link active" href="index.html" aria-current="page">Overview</a>
    <a class="utility-link" href="corridor-profile-measurement/index.html">Corridor profiles</a>
    <a class="utility-link" href="learn-more/index.html">Methods</a>
    <details>
      <summary>Download data</summary>
      <div class="utility-downloads">
        <a href="data/link_episode_projection.csv" download>TMC-period projection</a>
        <a href="data/corridor_period_summary.csv" download>Corridor-period summary</a>
        <a href="data/corridor_coverage.csv" download>Corridor coverage</a>
        <a href="data/dtalite_assignment_dc.csv" download>TAPlite assignment</a>
        <a href="data/accepted_average_weekday_episodes.csv" download>Accepted episodes</a>
        <a href="data/qvdf_selected_parameters.csv" download>Selected QVDF parameters</a>
        <a class="download-all" href="corridor-profile-measurement/downloads/corridor-profile-measurement-data.zip" download>Download all corridor data</a>
      </div>
    </details>
  </aside>
  <header class="page-head">
    <div class="head-row">
      <div>
        <p class="eyebrow">NVTA CBI · 2026 Corridor Results</p>
        <h1>Corridor Performance Explorer</h1>
        <p class="lede">Select a corridor on the map or from the list. Its current CBI statistics appear with the observed-versus-TAPlite diagnostic report directly below the map.</p>
      </div>
    </div>
  </header>

  <section class="summary" aria-label="Dashboard summary">
    <article class="summary-card"><div class="summary-value" id="sumCorridors">—</div><div class="summary-label">Corridors</div></article>
    <article class="summary-card"><div class="summary-value" id="sumReports">—</div><div class="summary-label">Complete corridor report pages</div></article>
    <article class="summary-card"><div class="summary-value" id="sumReady">—</div><div class="summary-label">QVDF-ready corridors</div></article>
    <article class="summary-card"><div class="summary-value" id="sumMapping">—</div><div class="summary-label">TMCs with current route geometry</div></article>
    <article class="summary-card"><div class="summary-value" id="sumPartial">—</div><div class="summary-label">Corridors with partial map geometry</div></article>
  </section>

  <main class="workspace">
    <aside class="panel sidebar">
      <div class="sidebar-head">
        <div class="title-with-help"><h2>Choose a corridor</h2><a class="learn-more" href="learn-more/index.html#corridor-selection" target="_blank" rel="noopener">Learn more</a></div>
        <input class="search" id="search" type="search" placeholder="Search I-66, VA-7, direction…" aria-label="Search corridors">
      </div>
      <div class="corridor-list" id="corridorList"></div>
    </aside>

    <div class="main-stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="title-with-help"><h2>Current TMC route map</h2><a class="learn-more" href="learn-more/index.html#route-map" target="_blank" rel="noopener">Learn more</a></div>
            <div class="panel-subtitle">Built from the explicitly selected map-matching product</div>
          </div>
          <div class="map-key">
            <span>Observed avg speed:</span>
            <span class="speed-key"><i class="speed-swatch" style="background:#b2182b"></i>&lt;30</span>
            <span class="speed-key"><i class="speed-swatch" style="background:#ef8a62"></i>30–45</span>
            <span class="speed-key"><i class="speed-swatch" style="background:#f2c94c"></i>45–55</span>
            <span class="speed-key"><i class="speed-swatch" style="background:#66bd63"></i>55–65</span>
            <span class="speed-key"><i class="speed-swatch" style="background:#1a9850"></i>65+</span>
          </div>
        </div>
        <div id="map" role="application" aria-label="Interactive corridor map"></div>
      </section>

      <section class="panel" aria-live="polite">
        <div class="detail-body">
          <div class="detail-top">
            <div>
              <div class="title-with-help"><h2 class="detail-title" id="detailTitle">Select a corridor</h2><a class="learn-more" href="learn-more/index.html#corridor-details" target="_blank" rel="noopener">Learn more</a></div>
              <p class="detail-caption" id="detailCaption">Choose a route to view current corridor and projection statistics.</p>
            </div>
            <span class="status-pill" id="statusPill">Waiting</span>
          </div>
          <div class="metric-grid" id="metricGrid"></div>
          <div class="quality-wrap" id="qualityWrap"></div>
          <div class="mapping-note" id="mappingNote" hidden></div>
        </div>
      </section>
    </div>
  </main>

  <section class="panel report-panel" id="reportPanel">
    <div class="panel-head">
      <div class="report-head-copy">
        <div class="title-with-help"><h2 id="reportTitle">Corridor diagnostic report</h2><a class="learn-more" href="learn-more/index.html#projection-report" target="_blank" rel="noopener">Learn more</a></div>
        <div class="panel-subtitle" id="reportSubtitle">Select a corridor above.</div>
      </div>
      <div class="report-actions">
        <span class="status-pill" id="reportStatus">Waiting</span>
        <a class="report-link" id="reportLink" href="#" target="_blank" rel="noopener" hidden>Open full page</a>
      </div>
    </div>
    <div class="report-empty" id="reportEmpty">Select a corridor from the map or list to load its full page here.</div>
    <iframe class="report-frame" id="reportFrame" title="Selected corridor observed and TAPlite diagnostic report" loading="lazy" hidden></iframe>
  </section>

  <footer class="footer" id="provenance"></footer>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const DATA = __DATA__;
    const corridorById = new Map(DATA.corridors.map(item => [item.id, item]));
    const layerById = new Map();
    let selectedId = "";

    const fmt = (value, digits = 0) => {
      if (value === null || value === undefined || value === "") return "—";
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    };
    const statusLabel = value => String(value || "unknown").replaceAll("_", " ");
    const isLimited = corridor => corridor.qvdf.coverage_status !== "ready" || corridor.missing_tmc.length > 0;
    const escapeHtml = value => String(value ?? "—")
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    document.getElementById("sumCorridors").textContent = fmt(DATA.summary.corridors);
    document.getElementById("sumReports").textContent = fmt(DATA.summary.corridors_with_reports);
    document.getElementById("sumReady").textContent = fmt(DATA.summary.qvdf_ready_corridors);
    document.getElementById("sumMapping").textContent = `${fmt(DATA.summary.mapping_share_pct, 1)}%`;
    document.getElementById("sumPartial").textContent = fmt(DATA.summary.corridors_partially_mapped);
    document.getElementById("provenance").textContent =
      `Sources: ${DATA.sources.corridor_results} · ${DATA.sources.tmc_mapmatching} · ${DATA.sources.qvdf_reports} · ${DATA.sources.ritis_15min}`;

    const map = L.map("map", { preferCanvas: true, zoomControl: true });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      opacity: .20,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    function speedColor(speed) {
      if (speed === null || speed === undefined || Number.isNaN(Number(speed))) return "#7b8794";
      if (Number(speed) < 30) return "#b2182b";
      if (Number(speed) < 45) return "#ef8a62";
      if (Number(speed) < 55) return "#f2c94c";
      if (Number(speed) < 65) return "#66bd63";
      return "#1a9850";
    }

    function tmcStyle(tmc, selected = false) {
      return {
        color: speedColor(tmc.observed_average_speed_mph),
        weight: selected ? 6 : 3.5,
        opacity: selected ? 1 : .35
      };
    }

    function tmcTooltip(corridor, tmc) {
      const nodePair = tmc.from_node_id !== null && tmc.to_node_id !== null
        ? `${fmt(tmc.from_node_id)} → ${fmt(tmc.to_node_id)}`
        : "Unavailable";
      return `<div class="tmc-tooltip-grid">
        <div class="tmc-tooltip-col">
          <div class="tmc-tooltip-title">${escapeHtml(corridor.label)} · ${escapeHtml(tmc.direction || corridor.direction)}</div>
          <div><span class="tmc-tooltip-label">TMC:</span> ${escapeHtml(tmc.tmc_code)}</div>
          <div><span class="tmc-tooltip-label">Link:</span> ${escapeHtml(tmc.link_id ?? "Unavailable")}</div>
          <div><span class="tmc-tooltip-label">Node pair:</span> ${escapeHtml(nodePair)}</div>
        </div>
        <div class="tmc-tooltip-col">
          <div><span class="tmc-tooltip-label">Observed avg:</span> ${fmt(tmc.observed_average_speed_mph, 1)} mph</div>
          <div><span class="tmc-tooltip-label">TAPlite avg:</span> ${fmt(tmc.model_average_speed_mph, 1)} mph</div>
          <div><span class="tmc-tooltip-label">Speed MAE:</span> ${fmt(tmc.speed_mae_mph, 1)} mph</div>
          <div><span class="tmc-tooltip-label">Speed MAPE:</span> ${fmt(tmc.speed_mape_pct, 1)}%</div>
        </div>
      </div>`;
    }

    DATA.corridors.forEach(corridor => {
      const layers = [];
      (corridor.tmcs || []).forEach(tmc => {
        if (!tmc.segments?.length) return;
        const layer = L.polyline(tmc.segments, tmcStyle(tmc)).addTo(map);
        layer.bindTooltip(tmcTooltip(corridor, tmc), {
          sticky: true, direction: "top", className: "tmc-map-tooltip", opacity: .98
        });
        layer.on("click", () => selectCorridor(corridor.id, true));
        layers.push({ layer, tmc });
      });
      if (layers.length) layerById.set(corridor.id, layers);
    });
    if (DATA.bounds) map.fitBounds(DATA.bounds, { padding: [18, 18] });
    else map.setView([38.87, -77.25], 9);

    function renderList() {
      const query = document.getElementById("search").value.trim().toLowerCase();
      const rows = DATA.corridors.filter(corridor =>
        `${corridor.id} ${corridor.label} ${corridor.direction}`.toLowerCase().includes(query)
      );
      document.getElementById("corridorList").innerHTML = rows.map(corridor => `
        <button class="corridor-button ${corridor.id === selectedId ? "active" : ""}" data-id="${corridor.id}">
          <span>
            <span class="corridor-name">${corridor.label}</span><br>
            <span class="corridor-meta">${fmt(corridor.mapped_tmc_count)}/${fmt(corridor.tmc_count)} mapped TMCs · ${statusLabel(corridor.qvdf.coverage_status)}</span>
          </span>
          <i class="dot ${corridor.missing_tmc.length ? "partial" : ""}" aria-hidden="true"></i>
        </button>
      `).join("");
      document.querySelectorAll(".corridor-button").forEach(button => {
        button.addEventListener("click", () => selectCorridor(button.dataset.id, true));
      });
    }

    function metric(value, label) {
      return `<article class="metric"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></article>`;
    }

    function renderDetails(corridor) {
      document.getElementById("detailTitle").textContent = corridor.label;
      document.getElementById("detailCaption").textContent =
        `${corridor.direction || "Direction unavailable"} · ${fmt(corridor.corridor_miles, 2)} corridor miles · run ${corridor.run_status || "status unavailable"}`;
      const pill = document.getElementById("statusPill");
      pill.textContent = statusLabel(corridor.qvdf.coverage_status);
      pill.className = `status-pill ${isLimited(corridor) ? "limited" : ""}`;
      document.getElementById("metricGrid").innerHTML = [
        metric(`${fmt(corridor.mapped_tmc_count)}/${fmt(corridor.tmc_count)}`, "TMCs with current map geometry"),
        metric(`${fmt(corridor.result_stats.raw_qc_pass_pct, 1)}%`, "Raw observation QC pass"),
        metric(fmt(corridor.result_stats.episodes_clean), "Accepted daily episodes"),
        metric(fmt(corridor.result_stats.calibration_rows), "Calibrated link-period rows"),
        metric(`${fmt(corridor.qvdf.ready_link_periods)}/${fmt(corridor.qvdf.selected_link_periods)}`, "QVDF-ready observed TMC-periods"),
        metric(fmt(corridor.result_stats.figures), "CBI result figures")
      ].join("");

      const quality = corridor.quality;
      document.getElementById("qualityWrap").innerHTML = quality.length ? `
        <table>
          <thead><tr><th>Period</th><th>Links</th><th>Duration MAPE</th><th>Min speed MAPE</th><th>t0 MAE</th><th>Gates</th></tr></thead>
          <tbody>${quality.map(row => `
            <tr>
              <td>${row.period || "—"}</td>
              <td>${fmt(row.n_links)}</td>
              <td>${fmt(row.duration_mape_pct, 1)}%</td>
              <td>${fmt(row.minimum_speed_mape_pct, 1)}%</td>
              <td>${fmt(row.t0_mae_min, 1)} min</td>
              <td>${row.gates_pass || "—"}</td>
            </tr>`).join("")}
          </tbody>
        </table>` : "";

      const note = document.getElementById("mappingNote");
      if (corridor.missing_tmc.length) {
        const statuses = Object.entries(corridor.mapping_status_counts)
          .filter(([status]) => status === "no_path" || status === "no_endpoint_candidates" || status === "missing")
          .map(([status, count]) => `${statusLabel(status)}: ${count}`)
          .join(" · ");
        note.textContent =
          `Partial current map geometry: ${corridor.missing_tmc.length} of ${corridor.tmc_count} corridor TMCs have no route geometry in the selected current map-matching product (${statuses}). No geometry was substituted from another project.`;
        note.hidden = false;
      } else {
        note.hidden = true;
      }
    }

    function resizeReport() {
      const frame = document.getElementById("reportFrame");
      if (frame.hidden) return;
      try {
        const doc = frame.contentDocument;
        const height = Math.max(doc?.documentElement?.scrollHeight || 0, doc?.body?.scrollHeight || 0);
        if (height) frame.style.height = `${Math.max(1200, height + 10)}px`;
      } catch {
        frame.style.height = "1800px";
      }
    }

    function renderReport(corridor) {
      document.getElementById("reportTitle").textContent = `${corridor.label} · Corridor diagnostic report`;
      document.getElementById("reportSubtitle").textContent =
        "TMC-aligned observed and TAPlite profiles with speed-and-flow diagnostics.";
      const status = document.getElementById("reportStatus");
      status.textContent = statusLabel(corridor.qvdf.coverage_status);
      status.className = `status-pill ${corridor.qvdf.coverage_status === "ready" ? "" : "limited"}`;
      const frame = document.getElementById("reportFrame");
      const empty = document.getElementById("reportEmpty");
      const link = document.getElementById("reportLink");
      if (!corridor.report_url) {
        empty.textContent = "No generated report page is available for this corridor.";
        empty.hidden = false;
        frame.hidden = true;
        frame.removeAttribute("src");
        link.hidden = true;
        return;
      }
      empty.hidden = true;
      frame.hidden = false;
      link.href = corridor.report_url;
      link.hidden = false;
      if (frame.dataset.corridor !== corridor.id) {
        frame.style.height = "1200px";
        frame.src = corridor.report_url;
        frame.dataset.corridor = corridor.id;
      } else {
        resizeReport();
      }
    }

    function selectCorridor(id, zoom = false) {
      const corridor = corridorById.get(id);
      if (!corridor) return;
      selectedId = id;
      layerById.forEach((layers, layerId) => {
        layers.forEach(({ layer, tmc }) => {
          layer.setStyle(tmcStyle(tmc, layerId === id));
          if (layerId === id) layer.bringToFront();
        });
      });
      if (zoom && corridor.bounds) map.fitBounds(corridor.bounds, { padding: [35, 35], maxZoom: 13 });
      renderList();
      renderDetails(corridor);
      renderReport(corridor);
      const url = new URL(window.location.href);
      url.searchParams.set("corridor", corridor.id);
      history.replaceState(null, "", url);
    }

    document.getElementById("search").addEventListener("input", renderList);
    document.getElementById("reportFrame").addEventListener("load", () => {
      requestAnimationFrame(() => requestAnimationFrame(resizeReport));
    });
    window.addEventListener("resize", resizeReport);
    renderList();
    const requested = new URLSearchParams(window.location.search).get("corridor");
    const initial = corridorById.has(requested) ? requested : (corridorById.has("I66_EB") ? "I66_EB" : DATA.corridors[0]?.id);
    if (initial) selectCorridor(initial, false);
  </script>
</body>
</html>
"""


def build_dashboard(settings: DashboardBuildSettings) -> dict[str, Any]:
    required = [
        settings.corridor_results_root,
        settings.mapmatching_product_root,
        settings.qvdf_report_root,
        settings.corridor_measurement_root,
        settings.ritis_15min_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing integrated dashboard inputs: {missing}")
    _prepare_output(settings)
    data = _build_data(settings)
    eligible_corridor_ids = {
        str(corridor["id"]) for corridor in data["corridors"]
    }
    staged_reports = _stage_reports(
        settings,
        eligible_corridor_ids,
    )
    staged_assets = _stage_projection_assets(settings)
    corridor_profile = stage_corridor_profile_measurement(
        settings,
        eligible_corridor_ids,
    )
    methods_page = stage_methods_page(settings)
    display_overrides_file: str | None = None
    if (
        settings.profile_selection_overrides_path is not None
        and settings.profile_selection_overrides_path.is_file()
    ):
        overrides_target = (
            settings.output_root
            / "data"
            / "dashboard_profile_selection_overrides.csv"
        )
        overrides_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings.profile_selection_overrides_path, overrides_target)
        display_overrides_file = overrides_target.relative_to(
            settings.output_root
        ).as_posix()
    html = HTML_TEMPLATE.replace("__DATA__", _script_json(data))
    (settings.output_root / "index.html").write_text(html, encoding="utf-8")
    (settings.output_root / ".nojekyll").write_text("", encoding="utf-8")
    (settings.output_root / "dashboard_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "status": "PASS",
        "corridors": data["summary"]["corridors"],
        "resolved_inputs": {
            "corridor_results_root": str(settings.corridor_results_root),
            "mapmatching_product_root": str(
                settings.mapmatching_product_root
            ),
            "qvdf_report_root": str(settings.qvdf_report_root),
            "corridor_measurement_root": str(
                settings.corridor_measurement_root
            ),
            "assignment_root": str(settings.assignment_root),
            "ritis_15min_path": str(settings.ritis_15min_path),
        },
        "combined_profile_selection": {
            "scope": "canonical directed-node-pair TMC winners with modeled speed",
            "maximum_profiles_per_corridor": 5,
            "policy_when_more_than_five": (
                "split the road-ordered eligible TMC-link sequence into five "
                "contiguous near-equal segments and select the most congested "
                "observed TMC from each segment"
            ),
            "policy_when_five_or_fewer": "select every eligible TMC-link",
            "primary_score": (
                "daily mean max(congestion_threshold_mph - "
                "observed_speed_mph, 0) / congestion_threshold_mph"
            ),
            "tie_breakers": [
                "more observed congested 15-minute intervals",
                "lower observed daily mean speed",
                "lower observed daily minimum speed",
                "lower road order",
                "TMC code",
            ],
            "analytical_effect": (
                "display only; does not change corridor membership, heatmap "
                "rows, or performance metrics"
            ),
            "display_overrides_file": display_overrides_file,
        },
        "presentation_adjustments": {
            "first_page_removed_quality_columns": [
                "Duration R²",
                "Speed R²",
            ],
        },
        "staged_reports": staged_reports,
        "projection_data_files": staged_assets["data_files"],
        "projection_summary_figures": staged_assets["summary_figures"],
        "corridor_profile_measurement": corridor_profile,
        "methods_page": methods_page.relative_to(settings.output_root).as_posix(),
        "corridors_fully_mapped": data["summary"]["corridors_fully_mapped"],
        "corridors_partially_mapped": data["summary"][
            "corridors_partially_mapped"
        ],
        "missing_tmc_geometry": data["summary"]["missing_tmc"],
        "missing_report_corridors": data["audit"]["missing_report_corridors"],
        "sources": data["sources"],
        "entrypoint": "index.html",
    }
    (settings.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
