from __future__ import annotations

import csv
import gzip
import html
import json
import math
import re
import shutil
import zipfile
from io import StringIO
from pathlib import Path
from typing import Any

from .dashboard_filters import is_managed_corridor
from .combined_profile import generate_combined_tmc_profile_figures


FIGURE_COLUMNS = (
    "speed_heatmap_figure",
)

CORE_DOWNLOADS = (
    (
        "Corridor summary metrics",
        "01-corridor-results/corridor_metrics.csv",
    ),
    (
        "Corridor-period metrics",
        "01-corridor-results/corridor_period_metrics.csv",
    ),
    (
        "Aligned daily corridor profiles",
        "01-corridor-results/daily_corridor_profiles.csv",
    ),
    (
        "Overall performance summary",
        "01-corridor-results/overall_metrics.csv",
    ),
    (
        "Selected TMC-period metrics",
        "02-tmc-results/selected_tmc_period_metrics.csv",
    ),
    (
        "Congestion episodes",
        "03-congestion-results/congestion_episodes.csv",
    ),
    (
        "Figure manifest",
        "06-figures/figure_manifest.csv",
    ),
)

OPTIONAL_DOWNLOADS = (
    (
        "Observed-speed-derived volume summary",
        "09-observed-speed-derived-volume/observed_volume_summary.csv",
    ),
    (
        "Mapped-link assignment diagnostic summary",
        "10-link-volume-diagnostics/corridor_period_summary.csv",
    ),
    (
        "Mapped-link manual review sample",
        "10-link-volume-diagnostics/manual_review_sample.csv",
    ),
    (
        "TAPlite kernel formula reconciliation",
        "10-link-volume-diagnostics/kernel_formula_reconciliation.csv",
    ),
    (
        "CBI-QVDF versus TAPlite duration and D/C audit",
        "11-cbi-taplite-duration-audit/tmc_period_duration_dc_audit.csv",
    ),
    (
        "Duration and D/C audit summary",
        "11-cbi-taplite-duration-audit/corridor_period_summary.csv",
    ),
    (
        "CBI and TAPlite formula reference",
        "11-cbi-taplite-duration-audit/formula_reference.csv",
    ),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corridor measurement input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _json_value(value: str | None) -> str | float | int | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else round(number, 6)


def _typed_row(row: dict[str, str]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in row.items()}


def _safe_source(root: Path, relative_text: str) -> tuple[Path, Path]:
    relative = Path(relative_text.replace("\\", "/"))
    source = (root / relative).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Corridor measurement asset escapes its run root: {relative_text}"
        ) from exc
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing corridor measurement asset: {source}"
        )
    return source, relative


def _script_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")


def _stage_file(root: Path, page_root: Path, relative_text: str) -> str:
    source, relative = _safe_source(root, relative_text)
    target = page_root / "assets" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return (Path("assets") / relative).as_posix()


def _mean(rows: list[dict[str, Any]], column: str) -> float | None:
    values = [float(row[column]) for row in rows if row.get(column) is not None]
    return sum(values) / len(values) if values else None


def _dashboard_overall(corridors: list[dict[str, Any]]) -> dict[str, Any]:
    """Recalculate overview cards after managed corridors are excluded."""

    valid_speed = [row for row in corridors if float(row.get("matched_interval_count") or 0) > 0]
    valid_cube = [row for row in corridors if float(row.get("cube_vs_observed_matched_interval_count") or 0) > 0]
    valid_tap_cube = [row for row in corridors if float(row.get("taplite_vs_cube_matched_interval_count") or 0) > 0]
    valid_duration = [row for row in corridors if float(row.get("observed_congestion_duration_min") or 0) > 0]
    valid_iou = [row for row in corridors if float(row.get("congestion_union_min") or 0) > 0]
    return {
        "corridor_count": len(corridors),
        "corridors_with_speed_results": len(valid_speed),
        "corridors_with_all_periods": sum(float(row.get("periods_with_matched_intervals") or 0) == 3 for row in corridors),
        "corridors_with_complete_interval_coverage": sum(
            float(row.get("matched_interval_count") or 0)
            == float(row.get("expected_interval_count") or -1)
            for row in corridors
        ),
        "corridor_mean_speed_mae_mph": _mean(valid_speed, "mae_mph"),
        "corridor_mean_speed_mape_pct": _mean(valid_speed, "mape_pct"),
        "corridor_mean_speed_rmse_mph": _mean(valid_speed, "rmse_mph"),
        "corridor_mean_cube_vs_observed_speed_mae_mph": _mean(valid_cube, "cube_vs_observed_mae_mph"),
        "corridor_mean_cube_vs_observed_speed_mape_pct": _mean(valid_cube, "cube_vs_observed_mape_pct"),
        "corridor_mean_taplite_vs_cube_speed_mae_mph": _mean(valid_tap_cube, "taplite_vs_cube_mae_mph"),
        "corridor_mean_taplite_vs_cube_speed_mape_pct": _mean(valid_tap_cube, "taplite_vs_cube_mape_pct"),
        "congestion_duration_mae_min": _mean(corridors, "congestion_duration_absolute_error_min"),
        "congestion_duration_mape_pct": _mean(valid_duration, "congestion_duration_ape_pct"),
        "cube_congestion_duration_mae_min": _mean(valid_cube, "cube_vs_observed_congestion_duration_absolute_error_min"),
        "corridors_with_observed_congestion": len(valid_duration),
        "mean_congestion_iou_pct": _mean(valid_iou, "congestion_iou_pct"),
        "mean_tmc_comparison_coverage_pct": _mean(corridors, "minimum_tmc_comparison_coverage_pct"),
    }


def _copy_filtered_dashboard_csv(
    source: Path,
    target: Path,
    eligible_corridor_ids: set[str],
) -> None:
    """Stage a dashboard CSV with managed corridor rows removed when keyed."""

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        corridor_column = next(
            (column for column in ("corridor", "key") if column in fieldnames),
            None,
        )
        if corridor_column is None:
            shutil.copy2(source, target)
            return
        rows = [
            row for row in reader
            if not is_managed_corridor(row.get(corridor_column))
            and str(row.get(corridor_column, "")).strip()
            in eligible_corridor_ids
        ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_page_data(
    measurement_root: Path,
    page_root: Path,
    package_root: Path,
    eligible_corridor_ids: set[str],
) -> tuple[dict[str, Any], int]:
    corridor_rows = _read_csv(
        measurement_root / "01-corridor-results" / "corridor_metrics.csv"
    )
    overall_rows = _read_csv(
        measurement_root / "01-corridor-results" / "overall_metrics.csv"
    )
    figure_rows = _read_csv(
        measurement_root / "06-figures" / "figure_manifest.csv"
    )
    figures_by_corridor: dict[str, dict[str, Any]] = {}
    copied_assets = 0
    for row in figure_rows:
        corridor = str(row.get("corridor", "")).strip()
        if (
            not corridor
            or is_managed_corridor(corridor)
            or corridor not in eligible_corridor_ids
        ):
            continue
        item: dict[str, Any] = {
            "tmc_count": _json_value(row.get("tmc_count")),
            "selected_tmc_count": _json_value(row.get("selected_tmc_count")),
            "selected_tmc_codes": [
                value
                for value in str(row.get("selected_tmc_codes", "")).split(";")
                if value
            ],
        }
        for column in FIGURE_COLUMNS:
            relative = str(row.get(column, "")).strip()
            if relative:
                item[column] = _stage_file(
                    measurement_root, page_root, relative
                )
                copied_assets += 1
            else:
                item[column] = None
        figures_by_corridor[corridor] = item

    corridors: list[dict[str, Any]] = []
    diagnostic_source = (
        measurement_root
        / "10-link-volume-diagnostics"
        / "corridor_period_summary.csv"
    )
    diagnostic_by_corridor: dict[str, list[dict[str, Any]]] = {}
    if diagnostic_source.is_file():
        for row in _read_csv(diagnostic_source):
            corridor = str(row.get("corridor", "")).strip()
            if (
                is_managed_corridor(corridor)
                or corridor not in eligible_corridor_ids
            ):
                continue
            diagnostic_by_corridor.setdefault(corridor, []).append(
                _typed_row(row)
            )
    for row in corridor_rows:
        item = _typed_row(row)
        corridor = str(row.get("corridor", "")).strip()
        if (
            is_managed_corridor(corridor)
            or corridor not in eligible_corridor_ids
        ):
            continue
        item["figures"] = figures_by_corridor.get(corridor, {})
        item["assignment_diagnostics"] = diagnostic_by_corridor.get(
            corridor, []
        )
        corridors.append(item)
    corridors.sort(
        key=lambda item: (
            -int((item.get("figures") or {}).get("tmc_count") or 0),
            str(item.get("corridor", "")),
        )
    )
    dashboard_overall = _dashboard_overall(corridors)

    downloads: list[dict[str, str]] = []
    for label, relative_text in CORE_DOWNLOADS:
        source, relative = _safe_source(measurement_root, relative_text)
        target = page_root / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "01-corridor-results/overall_metrics.csv":
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dashboard_overall))
                writer.writeheader()
                writer.writerow(dashboard_overall)
        else:
            _copy_filtered_dashboard_csv(
                source,
                target,
                eligible_corridor_ids,
            )
        downloads.append(
            {
                "label": label,
                "url": (Path("data") / relative).as_posix(),
            }
        )
    for label, relative_text in OPTIONAL_DOWNLOADS:
        source = measurement_root / relative_text
        if not source.is_file():
            continue
        _, relative = _safe_source(measurement_root, relative_text)
        target = page_root / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_filtered_dashboard_csv(source, target, eligible_corridor_ids)
        downloads.append(
            {
                "label": label,
                "url": (Path("data") / relative).as_posix(),
            }
        )

    try:
        source_label = measurement_root.relative_to(package_root).as_posix()
    except ValueError:
        source_label = measurement_root.name
    return (
        {
            "overall": dashboard_overall,
            "corridors": corridors,
            "downloads": downloads,
            "source": source_label,
        },
        copied_assets,
    )


def _build_download_bundle(
    measurement_root: Path,
    page_root: Path,
    eligible_corridor_ids: set[str],
) -> Path:
    bundle = (
        page_root
        / "downloads"
        / "corridor-profile-measurement-data.zip"
    )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(
        set(measurement_root.rglob("*.csv"))
        | set(measurement_root.rglob("*.csv.gz"))
    )
    with zipfile.ZipFile(
        bundle,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in csv_paths:
            archive_name = path.relative_to(measurement_root).as_posix()
            if archive_name == "01-corridor-results/overall_metrics.csv":
                dashboard_overall = page_root / "data" / archive_name
                archive.write(dashboard_overall, archive_name)
                continue
            is_gzip_csv = path.name.lower().endswith(".csv.gz")
            open_stream = gzip.open if is_gzip_csv else Path.open
            open_kwargs = {
                "mode": "rt" if is_gzip_csv else "r",
                "encoding": "utf-8-sig",
                "newline": "",
            }
            with open_stream(path, **open_kwargs) as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames or []
                corridor_column = next(
                    (
                        column
                        for column in ("corridor", "corridors", "key")
                        if column in fieldnames
                    ),
                    None,
                )
                if corridor_column is None:
                    archive.write(path, archive_name)
                    continue
                buffer = StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    corridor = str(row.get(corridor_column, "")).strip()
                    if (
                        not is_managed_corridor(corridor)
                        and corridor in eligible_corridor_ids
                    ):
                        writer.writerow(row)
                payload = buffer.getvalue().encode("utf-8")
                if is_gzip_csv:
                    payload = gzip.compress(payload, compresslevel=6, mtime=0)
                archive.writestr(archive_name, payload)
    return bundle


def _harmonize_staged_corridor_reports(
    output_root: Path,
    data: dict[str, Any],
    measurement_root: Path,
    cbi_corridors_root: Path,
    assignment_root: Path | None,
    mapmatching_product_root: Path | None,
    selection_overrides_path: Path | None,
) -> int:
    """Replace link-ID profiles with a single TMC-aligned comparison."""

    updated = 0
    combined_figures = generate_combined_tmc_profile_figures(
        measurement_root=measurement_root,
        cbi_corridors_root=cbi_corridors_root,
        staged_reports_root=output_root / "reports" / "corridors",
        assignment_root=assignment_root,
        mapmatching_product_root=mapmatching_product_root,
        selection_overrides_path=selection_overrides_path,
    )
    for item in data["corridors"]:
        corridor = str(item.get("corridor", ""))
        report = output_root / "reports" / "corridors" / corridor / "index.html"
        combined = combined_figures.get(corridor, {})
        if not report.is_file():
            continue
        methods = "../../../learn-more/index.html"
        combined_image = (
            f'<img class="tmc-profile-figure" src="{html.escape(combined["speed"])}" alt="TMC-aligned observed and TAPlite profiles">'
            if combined.get("speed")
            else '<p>No TMC-aligned combined profile was available.</p>'
        )
        volume_image = (
            f'<img class="tmc-profile-figure" src="{html.escape(combined["volume"])}" alt="Selected-TMC synthetic and CBI QVDF volume profiles">'
            if combined.get("volume")
            else '<p>No selected-TMC volume consistency figure was available.</p>'
        )
        daily_header = f"""
<h2>TMC-aligned profile diagnostics <a class="learn-more" href="{methods}#selected-tmc-profiles" target="_blank" rel="noopener">Learn more</a></h2>
<p class="note">Every displayed record is classified <code>facility_class = gp</code> in the authoritative map-matching product. Each displayed link is paired with the canonical winning GP TMC from the same node-pair ranking used for direct PLF, observed speed-boundary, and observed congestion-boundary inputs. Observed points are shown at 15-minute resolution and TAPlite uses that link's native 5-minute output. The table below each plot reports the selected link and its period diagnostics.</p>
<h3>Observed and TAPlite</h3>
{combined_image}
<p><a href="../../../corridor-profile-measurement/index.html?corridor={html.escape(corridor)}">Open the complete corridor profile measurement page →</a></p>
<h2>CBI speed and flow consistency <a class="learn-more" href="{methods}#cbi-speed-volume" target="_blank" rel="noopener">Learn more</a></h2>
<p class="note">The volume profiles below use the same selected TMCs and road order as the combined speed profiles.</p>
{volume_image}"""
        page = report.read_text(encoding="utf-8")
        if ".tmc-profile-figure{" not in page:
            page = page.replace(
                "</style>",
                ".learn-more{margin-left:7px;color:#1976a8;font-size:12px;font-weight:800;text-decoration:none;white-space:nowrap}.learn-more:hover{text-decoration:underline}.tmc-profile-figure{display:block;width:100%;height:auto;margin:8px 0 16px}\n</style>",
            )
        for image_source in (
            "daily_analysis/tmc_observed_qvdf_taplite.png",
            "daily_analysis/tmc_cbi_volume_consistency.png",
        ):
            page = page.replace(
                f'<img src="{image_source}"',
                f'<img class="tmc-profile-figure" src="{image_source}"',
            )
        page = page.replace("<h2>Assignment projection analysis</h2>", "")
        page = re.sub(r'<img src="projection\.png"[^>]*>', "", page)
        projection_asset = report.parent / "projection.png"
        if projection_asset.is_file():
            projection_asset.unlink()
        page = page.replace("<h2>Daily analysis</h2>", daily_header)
        page = page.replace("<h3>Sensor versus model, full day</h3>", "")
        page = page.replace(
            '<img src="daily_analysis/sensor_vs_model_fullday.png" alt="Sensor versus model full day">',
            "",
        )
        page = page.replace("<h3>Speed heatmap</h3>", "")
        page = re.sub(
            r'<img src="daily_analysis/speed_heatmap\.png"[^>]*>', "", page
        )
        page = page.replace("<h3>Speed and volume</h3>", "")
        page = re.sub(
            r'<img src="daily_analysis/speed_volume_link[^\"]+\.png"[^>]*>',
            "", page,
        )
        daily_analysis = report.parent / "daily_analysis"
        for obsolete_name in (
            "sensor_vs_model_fullday.png",
            "speed_heatmap.png",
        ):
            obsolete_asset = daily_analysis / obsolete_name
            if obsolete_asset.is_file():
                obsolete_asset.unlink()
        for obsolete_asset in daily_analysis.glob("speed_volume_link*.png"):
            if obsolete_asset.is_file():
                obsolete_asset.unlink()
        report.write_text(page, encoding="utf-8")
        updated += 1
    return updated


def stage_corridor_profile_measurement(
    settings: Any,
    eligible_corridor_ids: set[str],
) -> dict[str, Any]:
    """Build and stage the second-page corridor profile explorer."""

    measurement_root = Path(settings.corridor_measurement_root).resolve()
    page_root = Path(settings.output_root) / "corridor-profile-measurement"
    if page_root.is_dir():
        shutil.rmtree(page_root)
    page_root.mkdir(parents=True, exist_ok=True)
    data, asset_count = _build_page_data(
        measurement_root,
        page_root,
        Path(settings.package_root).resolve(),
        eligible_corridor_ids,
    )
    bundle = _build_download_bundle(
        measurement_root,
        page_root,
        eligible_corridor_ids,
    )
    html = CORRIDOR_PROFILE_TEMPLATE.replace("__DATA__", _script_json(data))
    (page_root / "index.html").write_text(html, encoding="utf-8")
    (page_root / "corridor_profile_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    harmonized_reports = _harmonize_staged_corridor_reports(
        Path(settings.output_root),
        data,
        measurement_root,
        Path(settings.corridor_results_root),
        Path(settings.assignment_root) if settings.assignment_root else None,
        (
            Path(settings.mapmatching_product_root)
            if settings.mapmatching_product_root
            else None
        ),
        (
            Path(settings.profile_selection_overrides_path)
            if settings.profile_selection_overrides_path
            else None
        ),
    )
    return {
        "page": "corridor-profile-measurement/index.html",
        "corridors": len(data["corridors"]),
        "figure_assets": asset_count,
        "data_downloads": len(data["downloads"]),
        "download_bundle": bundle.relative_to(settings.output_root).as_posix(),
        "harmonized_reports": harmonized_reports,
        "source": data["source"],
    }


CORRIDOR_PROFILE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NVTA Corridor Profile Measurement</title>
  <style>
    :root {
      --ink: #12212f;
      --muted: #60707e;
      --line: #d9e2e8;
      --paper: #fff;
      --soft: #f3f7f8;
      --navy: #12395d;
      --blue: #1976a8;
      --teal: #16867a;
      --orange: #db7029;
      --green-soft: #e8f6f2;
      --amber-soft: #fff3e8;
      --shadow: 0 14px 34px rgba(18, 33, 47, .09);
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      padding-left: 182px;
      color: var(--ink);
      background: var(--soft);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
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
    button, input, select { font: inherit; }
    a { color: inherit; }
    .hero {
      color: #fff;
      background:
        radial-gradient(circle at 84% -35%, rgba(59, 187, 176, .6), transparent 34rem),
        linear-gradient(122deg, #102c48, #15577d 72%, #147b80);
      padding: 26px clamp(18px, 4vw, 58px) 42px;
    }
    .hero-top, .hero-copy, .section, .summary, .control-bar {
      max-width: 1480px;
      margin-left: auto;
      margin-right: auto;
    }
    .hero-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      margin-bottom: 34px;
    }
    .brand {
      color: #a9e4df;
      font-size: 12px;
      font-weight: 850;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    .hero-actions { display: flex; flex-wrap: wrap; gap: 9px; }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 9px 14px;
      border: 1px solid rgba(255,255,255,.42);
      border-radius: 8px;
      color: #fff;
      background: rgba(255,255,255,.09);
      font-size: 12px;
      font-weight: 800;
      text-decoration: none;
    }
    .button.primary { color: var(--navy); background: #fff; }
    .button:hover { transform: translateY(-1px); }
    h1 { margin: 0; font-size: clamp(30px, 4.3vw, 52px); line-height: 1.03; }
    .lede { max-width: 850px; margin: 13px 0 0; color: #d8e9f2; line-height: 1.58; }
    .summary {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: -20px;
      padding: 0 18px;
      position: relative;
      z-index: 2;
    }
    .summary-card, .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .summary-card { min-height: 98px; padding: 16px; }
    .summary-value { font-size: 27px; font-weight: 850; letter-spacing: -.03em; }
    .summary-label { margin-top: 4px; color: var(--muted); font-size: 11px; line-height: 1.35; }
    .control-bar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 12px;
      align-items: end;
      margin-top: 18px;
      padding: 0 18px;
    }
    label { display: block; color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; }
    select, input[type="search"] {
      width: 100%;
      margin-top: 7px;
      padding: 11px 12px;
      color: var(--ink);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      outline: none;
    }
    select:focus, input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(25,118,168,.12); }
    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      padding: 8px 12px;
      color: #176558;
      background: var(--green-soft);
      border-radius: 999px;
      font-size: 11px;
      font-weight: 850;
      text-transform: capitalize;
    }
    .status-pill.limited { color: #935018; background: var(--amber-soft); }
    .section { padding: 20px 18px 0; }
    .section-title { display: flex; justify-content: space-between; align-items: end; gap: 14px; margin: 8px 0 12px; }
    .section-title h2 { margin: 0; color: var(--navy); font-size: 21px; }
    .section-title p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
    .title-with-help { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
    .learn-more { color: var(--blue); font-size: 11px; font-weight: 850; text-decoration: none; white-space: nowrap; }
    .learn-more:hover { text-decoration: underline; }
    .metrics { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 8px; padding: 14px; }
    .metric { min-height: 86px; padding: 12px; background: #f9fbfc; border: 1px solid var(--line); border-radius: 9px; }
    .metric-value { font-size: 20px; font-weight: 850; overflow-wrap: anywhere; }
    .metric-label { margin-top: 5px; color: var(--muted); font-size: 11px; line-height: 1.35; }
    .figure-panel { overflow: hidden; }
    .figure-head { padding: 13px 16px; border-bottom: 1px solid var(--line); }
    .figure-head h3 { margin: 0; font-size: 15px; }
    .figure-head p { margin: 4px 0 0; color: var(--muted); font-size: 11px; }
    .figure-panel img { display: block; width: 100%; height: auto; background: #fff; }
    .figure-stack { display: grid; gap: 14px; }
    .table-wrap { overflow: auto; max-height: 620px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th, td { padding: 9px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { position: sticky; top: 0; z-index: 1; color: #fff; background: var(--navy); font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
    th:first-child, td:first-child { text-align: left; }
    tbody tr { cursor: pointer; }
    tbody tr:hover, tbody tr.active { background: #eaf5f4; }
    .downloads { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 9px; padding: 14px; }
    .data-link { padding: 11px 12px; color: var(--blue); background: #f8fafb; border: 1px solid var(--line); border-radius: 8px; font-size: 11px; font-weight: 800; text-decoration: none; }
    .data-link:hover { background: #edf5f8; }
    footer { max-width: 1480px; margin: 24px auto 0; padding: 0 18px 28px; color: var(--muted); font-size: 11px; }
    @media (max-width: 1180px) {
      .summary { grid-template-columns: repeat(3, 1fr); }
      .metrics { grid-template-columns: repeat(4, 1fr); }
    }
    @media (max-width: 720px) {
      body { padding-left: 0; }
      .utility-rail { position: static; width: auto; padding: 10px 12px; }
      .utility-brand { display: none; }
      .utility-link, .utility-rail details { display: inline-block; width: auto; margin: 3px; vertical-align: top; }
      .utility-rail summary { margin: 0; }
      .utility-downloads { position: absolute; z-index: 1001; width: min(320px, 90vw); padding: 8px; background: #102c48; border-radius: 8px; box-shadow: var(--shadow); }
      .hero-top, .section-title { align-items: flex-start; flex-direction: column; }
      .summary { grid-template-columns: repeat(2, 1fr); }
      .control-bar { grid-template-columns: 1fr; align-items: start; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .downloads { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <aside class="utility-rail" aria-label="Dashboard navigation">
    <div class="utility-brand">NVTA CBI</div>
    <a class="utility-link" href="../index.html">Overview</a>
    <a class="utility-link active" href="index.html" aria-current="page">Corridor profiles</a>
    <a class="utility-link" href="../learn-more/index.html">Methods</a>
    <details>
      <summary>Download data</summary>
      <div class="utility-downloads" id="railDownloadLinks">
        <a class="download-all" href="downloads/corridor-profile-measurement-data.zip" download>Download all corridor data</a>
      </div>
    </details>
  </aside>
  <header class="hero">
    <div class="hero-top">
      <div class="brand">NVTA CBI · Model Validation</div>
    </div>
    <div class="hero-copy">
      <h1>Corridor Profile Measurement</h1>
      <p class="lede">Compare CBI-observed speeds with TAPlite link-performance speeds on the same ordered GMNS links. Select any corridor to inspect observed and modeled speed patterns, their spatial-temporal absolute error, and congestion-duration fit.</p>
    </div>
  </header>

  <section class="summary" aria-label="Overall model performance">
    <article class="summary-card"><div class="summary-value" id="sumCorridors">—</div><div class="summary-label">Corridors evaluated</div></article>
    <article class="summary-card"><div class="summary-value" id="sumMae">—</div><div class="summary-label">Mean TAPlite vs CBI speed MAE</div></article>
    <article class="summary-card"><div class="summary-value" id="sumMape">—</div><div class="summary-label">Mean TAPlite vs CBI speed MAPE</div></article>
    <article class="summary-card"><div class="summary-value" id="sumCubeMae">—</div><div class="summary-label">Mean Cube-QVDF vs CBI speed MAE</div></article>
    <article class="summary-card"><div class="summary-value" id="sumTapCube">—</div><div class="summary-label">Mean TAPlite vs Cube-QVDF speed MAE</div></article>
    <article class="summary-card"><div class="summary-value" id="sumDuration">—</div><div class="summary-label">Congestion-duration MAE</div></article>
  </section>

  <div class="control-bar">
    <label>Corridor
      <select id="corridorSelect" aria-label="Select corridor"></select>
    </label>
    <span class="status-pill" id="statusPill">Waiting</span>
  </div>

  <section class="section">
    <div class="section-title"><div><div class="title-with-help"><h2 id="corridorTitle">Corridor details</h2><a class="learn-more" href="../learn-more/index.html#corridor-overall" target="_blank" rel="noopener">Learn more</a></div><p id="corridorSubtitle"></p></div></div>
    <div class="panel metrics" id="metricGrid"></div>
  </section>

  <section class="section">
    <div class="section-title"><div><div class="title-with-help"><h2>Observed and TAPlite heatmaps</h2><a class="learn-more" href="../learn-more/index.html#speed-heatmaps" target="_blank" rel="noopener">Learn more</a></div><p>Observed speed, TAPlite speed, and their absolute error use the same ordered corridor TMCs and time bins.</p></div></div>
    <div class="figure-stack">
      <article class="panel figure-panel"><div class="figure-head"><div class="title-with-help"><h3>Complete ordered-TMC comparison</h3><a class="learn-more" href="../learn-more/index.html#speed-heatmaps" target="_blank" rel="noopener">Learn more</a></div><p>Observed speed, TAPlite speed, and TAPlite-versus-observed absolute error side by side. Green means low error and red means high error in the third panel.</p></div><img id="speedHeatmap" alt="Observed speed, TAPlite speed, and TAPlite versus observed absolute speed error heatmaps" loading="lazy"></article>
    </div>
  </section>

  <section class="section">
    <div class="section-title"><div><div class="title-with-help"><h2>Assignment loading and mapped-link diagnostic</h2><a class="learn-more" href="../learn-more/index.html#assignment-loading-diagnostic" target="_blank" rel="noopener">Learn more</a></div><p>Exact GMNS-link assignment values compared with complete-period INRIX inverse-S3 volume; calculation checks use the TAPlite kernel equations.</p></div></div>
    <div class="panel table-wrap"><table><thead><tr><th>Period</th><th>Mapped links</th><th>Zero volume</th><th>D/C ≤ 0.10</th><th>D/C ≤ 0.25</th><th>Map review</th><th>Formula review</th><th>Median assigned volume</th><th>Median INRIX/S3 volume</th><th>Zero with positive INRIX/S3</th></tr></thead><tbody id="diagnosticBody"></tbody></table></div>
  </section>

  <section class="section">
    <div class="section-title"><div><div class="title-with-help"><h2>All corridor results</h2><a class="learn-more" href="../learn-more/index.html#corridor-results" target="_blank" rel="noopener">Learn more</a></div><p>Corridors are sorted from the most to the fewest mapped TMC-links. Choose a row to update the figures above.</p></div><label>Filter corridors<input id="corridorSearch" type="search" placeholder="I66, VA7, eastbound…"></label></div>
    <div class="panel table-wrap"><table><thead><tr><th>Corridor</th><th>TMC-links</th><th>Status</th><th>Intervals</th><th>TAPlite-CBI MAE</th><th>TAPlite-CBI MAPE</th><th>Observed congestion</th><th>TAPlite congestion</th><th>Duration abs. error</th><th>IoU</th></tr></thead><tbody id="resultsBody"></tbody></table></div>
  </section>

  <footer id="provenance"></footer>

  <script>
    const DATA = __DATA__;
    const byCorridor = new Map(DATA.corridors.map(item => [item.corridor, item]));
    let selectedCorridor = "";
    const fmt = (value, digits = 1) => value === null || value === undefined || value === ""
      ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    const mph = value => value === null || value === undefined ? "—" : `${fmt(value, 2)} mph`;
    const pct = value => value === null || value === undefined ? "—" : `${fmt(value, 1)}%`;
    const minutes = value => value === null || value === undefined ? "—" : `${fmt(value, 0)} min`;
    const statusText = value => String(value || "unknown").replaceAll("_", " ");

    const overall = DATA.overall;
    document.getElementById("sumCorridors").textContent = fmt(overall.corridor_count, 0);
    document.getElementById("sumMae").textContent = mph(overall.corridor_mean_speed_mae_mph);
    document.getElementById("sumMape").textContent = pct(overall.corridor_mean_speed_mape_pct);
    document.getElementById("sumCubeMae").textContent = mph(overall.corridor_mean_cube_vs_observed_speed_mae_mph);
    document.getElementById("sumTapCube").textContent = mph(overall.corridor_mean_taplite_vs_cube_speed_mae_mph);
    document.getElementById("sumDuration").textContent = minutes(overall.congestion_duration_mae_min);
    document.getElementById("provenance").textContent = `Source: ${DATA.source}`;

    const select = document.getElementById("corridorSelect");
    select.innerHTML = DATA.corridors.map(item => `<option value="${item.corridor}">${item.corridor.replaceAll("_", " ")} · ${fmt(item.figures?.tmc_count, 0)} TMC-links</option>`).join("");

    function metric(value, label) {
      return `<article class="metric"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></article>`;
    }

    function setImage(id, source) {
      const image = document.getElementById(id);
      if (source) { image.src = source; image.hidden = false; }
      else { image.removeAttribute("src"); image.hidden = true; }
    }

    function renderCorridor(corridorId, scroll = false) {
      const item = byCorridor.get(corridorId);
      if (!item) return;
      selectedCorridor = corridorId;
      select.value = corridorId;
      document.getElementById("corridorTitle").textContent = corridorId.replaceAll("_", " ");
      document.getElementById("corridorSubtitle").textContent = `${fmt(item.figures.tmc_count, 0)} ordered TMCs · ${fmt(item.matched_interval_count, 0)} matched 15-minute intervals`;
      const pill = document.getElementById("statusPill");
      pill.textContent = statusText(item.result_status);
      pill.className = `status-pill ${item.result_status === "complete" ? "" : "limited"}`;
      document.getElementById("metricGrid").innerHTML = [
        metric(mph(item.mae_mph), "TAPlite vs CBI speed MAE"),
        metric(pct(item.mape_pct), "TAPlite vs CBI speed MAPE"),
        metric(mph(item.cube_vs_observed_mae_mph), "Cube-QVDF vs CBI speed MAE"),
        metric(pct(item.cube_vs_observed_mape_pct), "Cube-QVDF vs CBI speed MAPE"),
        metric(mph(item.taplite_vs_cube_mae_mph), "TAPlite vs Cube-QVDF speed MAE"),
        metric(minutes(item.observed_congestion_duration_min), "Observed congestion duration"),
        metric(minutes(item.model_congestion_duration_min), "TAPlite congestion duration"),
        metric(minutes(item.congestion_duration_absolute_error_min), "TAPlite duration absolute error")
      ].join("");
      const figures = item.figures || {};
      setImage("speedHeatmap", figures.speed_heatmap_figure);
      const diagnostics = item.assignment_diagnostics || [];
      document.getElementById("diagnosticBody").innerHTML = diagnostics.length
        ? diagnostics.map(row => `<tr>
          <td>${row.period}</td><td>${fmt(row.mapped_physical_link_count, 0)}</td><td>${fmt(row.zero_assignment_link_count, 0)}</td>
          <td>${fmt(row.doc_le_0_10_link_count, 0)}</td><td>${fmt(row.doc_le_0_25_link_count, 0)}</td><td>${fmt(row.mapmatching_review_link_count, 0)}</td>
          <td>${fmt(row.formula_review_link_count, 0)}</td><td>${fmt(row.median_assignment_volume, 0)}</td><td>${fmt(row.median_synthetic_period_volume, 0)}</td>
          <td>${fmt(row.zero_assignment_positive_synthetic_count, 0)}</td>
        </tr>`).join("")
        : '<tr><td colspan="10" style="text-align:left">No mapped-link diagnostic is available for this run.</td></tr>';
      renderTable();
      const url = new URL(window.location.href);
      url.searchParams.set("corridor", corridorId);
      history.replaceState(null, "", url);
      if (scroll) document.querySelector(".control-bar").scrollIntoView({ behavior: "smooth" });
    }

    function renderTable() {
      const query = document.getElementById("corridorSearch").value.trim().toLowerCase();
      const rows = DATA.corridors.filter(item => `${item.corridor} ${item.result_status}`.toLowerCase().includes(query));
      document.getElementById("resultsBody").innerHTML = rows.map(item => `<tr data-corridor="${item.corridor}" class="${item.corridor === selectedCorridor ? "active" : ""}">
        <td>${item.corridor.replaceAll("_", " ")}</td><td>${fmt(item.figures?.tmc_count, 0)}</td><td>${statusText(item.result_status)}</td><td>${fmt(item.matched_interval_count, 0)}</td>
        <td>${fmt(item.mae_mph, 2)}</td><td>${fmt(item.mape_pct, 1)}%</td><td>${fmt(item.observed_congestion_duration_min, 0)}</td><td>${fmt(item.model_congestion_duration_min, 0)}</td>
        <td>${fmt(item.congestion_duration_absolute_error_min, 0)}</td><td>${pct(item.congestion_iou_pct)}</td>
      </tr>`).join("");
      document.querySelectorAll("#resultsBody tr").forEach(row => row.addEventListener("click", () => renderCorridor(row.dataset.corridor, true)));
    }

    document.getElementById("railDownloadLinks").insertAdjacentHTML(
      "afterbegin",
      DATA.downloads.map(item => `<a href="${item.url}" download>${item.label}</a>`).join("")
    );
    select.addEventListener("change", () => renderCorridor(select.value));
    document.getElementById("corridorSearch").addEventListener("input", renderTable);
    const requested = new URLSearchParams(window.location.search).get("corridor");
    renderCorridor(byCorridor.has(requested) ? requested : (byCorridor.has("I66_EB") ? "I66_EB" : DATA.corridors[0]?.corridor));
  </script>
</body>
</html>
"""
