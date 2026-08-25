from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import CorridorSpec, PipelineSettings
from .output_contract import step_dir
from .pipeline import run_corridor
from .network_mapping import (
    build_observation_quality,
    write_canonical_mapping_artifacts,
)


def _run_one_corridor(
    work: tuple[CorridorSpec, Path, PipelineSettings, bool],
) -> dict[str, object]:
    spec, output_root, settings, generate_figures = work
    return run_corridor(
        spec,
        output_root,
        settings,
        generate_figures=generate_figures,
    )


def discover_inrix_corridors(
    input_root: Path,
    model_link_map: Path | None = None,
) -> dict[str, CorridorSpec]:
    input_root = Path(input_root).resolve()
    if model_link_map is None:
        raise ValueError(
            "A model-link mapping must be supplied explicitly. "
            "Use --model-link-map; automatic latest-run selection is disabled."
        )
    shared_map = Path(model_link_map).resolve()
    if not shared_map.is_file():
        raise FileNotFoundError(
            "CBI requires the current TMC map-matching product: "
            f"{shared_map}"
        )
    corridors: dict[str, CorridorSpec] = {}
    for folder in sorted(input_root.iterdir()):
        if not folder.is_dir() or folder.name == "model_link":
            continue
        if not (folder / "Readings.csv").is_file() or not (
            folder / "TMC_Identification.csv"
        ).is_file():
            continue
        corridors[folder.name] = CorridorSpec(
            key=folder.name,
            name=f"{folder.name} (NoVA INRIX)",
            source="inrix_folder",
            path=folder,
            free_flow_mph=45.0,
            capacity_vphpl=1800.0,
            model_link_map=shared_map,
            data_mode="speed_only",
        )
    return corridors


def select_corridors_with_frozen_tmcs(
    specs: dict[str, CorridorSpec],
    frozen_mapping: Path,
) -> tuple[list[str], pd.DataFrame]:
    """Select only corridor folders containing a frozen winning TMC.

    Expanded observed corridor catalogs can include TMCs that lost the
    immutable node-pair composite ranking to a better TMC on the same link.
    Running those folders would either calibrate a different target identity
    or fail late after other corridors have completed. This preflight makes
    that exclusion explicit and auditable before any calibration starts.
    """

    mapping_path = Path(frozen_mapping).resolve()
    mapping_header = set(pd.read_csv(mapping_path, nrows=0).columns)
    mapping_tmc_column = "tmc" if "tmc" in mapping_header else "tmc_code"
    if mapping_tmc_column not in mapping_header:
        raise ValueError(
            f"Frozen mapping requires tmc or tmc_code: {mapping_path}"
        )
    mapped_tmcs = set(
        pd.read_csv(
            mapping_path,
            usecols=[mapping_tmc_column],
            dtype={mapping_tmc_column: "string"},
        )[mapping_tmc_column]
        .dropna()
        .astype("string")
        .str.strip()
        .str.upper()
    )
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for key, spec in specs.items():
        metadata_path = spec.path / "TMC_Identification.csv"
        metadata_header = set(pd.read_csv(metadata_path, nrows=0).columns)
        metadata_tmc_column = "tmc" if "tmc" in metadata_header else "tmc_code"
        if metadata_tmc_column not in metadata_header:
            raise ValueError(
                f"Corridor metadata requires tmc or tmc_code: {metadata_path}"
            )
        corridor_tmcs = set(
            pd.read_csv(
                metadata_path,
                usecols=[metadata_tmc_column],
                dtype={metadata_tmc_column: "string"},
            )[metadata_tmc_column]
            .dropna()
            .astype("string")
            .str.strip()
            .str.upper()
        )
        winners = sorted(corridor_tmcs & mapped_tmcs)
        included = bool(winners)
        if included:
            selected.append(key)
        rows.append(
            {
                "corridor_key": key,
                "included": included,
                "reason": (
                    "contains_frozen_node_pair_winner"
                    if included
                    else "no_frozen_node_pair_winner_in_corridor"
                ),
                "corridor_tmc_count": len(corridor_tmcs),
                "frozen_winner_tmc_count": len(winners),
                "frozen_winner_tmcs": "|".join(winners),
            }
        )
    return selected, pd.DataFrame(rows)


def cross_corridor_quality(
    output_root: Path,
    specs: dict[str, CorridorSpec],
    keys: list[str],
    summary_root: Path,
) -> Path:
    rows = []
    for key in keys:
        corridor_output = output_root / key
        quality_dir = step_dir(corridor_output, "quality")
        calibration_path = quality_dir / "qvdf_validation_by_period.csv"
        timeseries_path = (
            quality_dir / "profile_smoothing_quality_by_link.csv"
        )
        gates_path = quality_dir / "quality_gates.csv"
        if not calibration_path.is_file():
            continue
        calibration = pd.read_csv(calibration_path)
        timeseries = pd.read_csv(timeseries_path)
        gates = pd.read_csv(gates_path)
        for result in calibration.itertuples(index=False):
            period_gates = gates[gates["period"].eq(result.period)]
            rows.append(
                {
                    "corridor": specs[key].name,
                    "key": key,
                    "period": result.period,
                    "n_links": result.n_links,
                    "step1_DC_P_R2": result.step1_DC_P_R2,
                    "step2_P_mag_R2": result.step2_P_mag_R2,
                    "P_MAPE_pct": result.P_MAPE_pct,
                    "vt2_MAPE_pct": result.vt2_MAPE_pct,
                    "t0_MAE_min": result.t0_MAE_min,
                    "smooth_R2_med": round(
                        float(timeseries["smooth_vs_raw_R2"].median()), 3
                    ),
                    "gates_pass": (
                        f"{int(period_gates['status'].eq('PASS').sum())}/"
                        f"{len(period_gates)}"
                    ),
                }
            )
    summary_root.mkdir(parents=True, exist_ok=True)
    path = summary_root / "_QUALITY_SUMMARY.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_batch(
    input_root: Path,
    output_root: Path,
    summary_root: Path,
    keys: list[str] | None = None,
    *,
    generate_figures: bool = True,
    workers: int = 1,
    model_link_map: Path | None = None,
    settings: PipelineSettings | None = None,
) -> dict[str, object]:
    batch_started = datetime.now().astimezone()
    specs = discover_inrix_corridors(input_root, model_link_map=model_link_map)
    source_map = next(iter(specs.values())).model_link_map if specs else None
    if source_map is None:
        raise ValueError("No canonical TMC/network-link mapping source was discovered")
    observation_quality = build_observation_quality(input_root)
    mapping_artifacts = write_canonical_mapping_artifacts(
        source_map,
        Path(output_root).parent / "shared" / "network-mapping",
        observation_quality=observation_quality,
    )
    specs = {
        key: replace(spec, model_link_map=mapping_artifacts["node_pair_primary"])
        for key, spec in specs.items()
    }
    mapped_keys, selection_audit = select_corridors_with_frozen_tmcs(
        specs,
        mapping_artifacts["node_pair_primary"],
    )
    Path(summary_root).mkdir(parents=True, exist_ok=True)
    selection_audit_path = Path(summary_root) / "corridor_selection_audit.csv"
    selection_audit.to_csv(selection_audit_path, index=False)
    selected = list(keys) if keys is not None else mapped_keys
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise ValueError(f"Unknown corridor keys: {unknown}")
    unmapped = sorted(set(selected) - set(mapped_keys))
    if unmapped:
        raise ValueError(
            "Requested corridors do not contain a frozen node-pair-winning TMC: "
            + ", ".join(unmapped)
        )
    if not selected:
        raise ValueError("No corridor contains a frozen node-pair-winning TMC")
    pipeline_settings = settings or PipelineSettings()
    work = [
        (
            specs[key],
            Path(output_root),
            pipeline_settings,
            generate_figures,
        )
        for key in selected
    ]
    if workers > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            manifests = list(executor.map(_run_one_corridor, work))
    else:
        manifests = [_run_one_corridor(item) for item in work]
    quality_path = cross_corridor_quality(
        Path(output_root), specs, selected, Path(summary_root)
    )
    batch_completed = datetime.now().astimezone()
    result = {
        "status": "PASS",
        "batch_started_at": batch_started.isoformat(timespec="minutes"),
        "batch_completed_at": batch_completed.isoformat(timespec="minutes"),
        "output_root": str(Path(output_root).resolve()),
        "corridors": len(manifests),
        "keys": selected,
        "quality_summary": str(quality_path),
        "corridor_selection_audit": str(selection_audit_path.resolve()),
        "excluded_unmapped_corridors": sorted(set(specs) - set(mapped_keys)),
        "canonical_network_mapping": {
            name: str(path.resolve())
            for name, path in mapping_artifacts.items()
        },
        "requested_model_link_map": (
            str(Path(model_link_map).resolve())
            if model_link_map is not None
            else None
        ),
        "runs": manifests,
    }
    batch_manifest = Path(summary_root) / "batch_manifest.json"
    batch_manifest.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )
    result["batch_manifest"] = str(batch_manifest.resolve())
    return result


def finalize_existing_batch(
    input_root: Path,
    output_root: Path,
    summary_root: Path,
    model_link_map: Path,
    excluded_keys: set[str] | None = None,
) -> dict[str, object]:
    """Rebuild the cross-corridor summary after a safe resumed batch."""

    specs = discover_inrix_corridors(input_root, model_link_map=model_link_map)
    excluded = set(excluded_keys or ())
    keys = [key for key in specs if key not in excluded]
    manifests: list[dict[str, object]] = []
    missing: list[str] = []
    for key in keys:
        manifest_path = (
            Path(output_root)
            / key
            / "11-run-metadata"
            / "run_manifest.json"
        )
        if not manifest_path.is_file():
            missing.append(key)
            continue
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError(
            "Cannot finalize; missing completed corridor manifests: "
            + ", ".join(missing)
        )
    quality_path = cross_corridor_quality(
        Path(output_root), specs, keys, Path(summary_root)
    )
    result = {
        "status": "PASS",
        "mode": "finalize_existing_complete_batch",
        "output_root": str(Path(output_root).resolve()),
        "corridors": len(manifests),
        "keys": keys,
        "excluded_corridors": sorted(excluded),
        "quality_summary": str(quality_path.resolve()),
        "runs": manifests,
    }
    batch_manifest = Path(summary_root) / "batch_manifest.json"
    batch_manifest.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )
    result["batch_manifest"] = str(batch_manifest.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the integrated NVTA CBI/QVDF corridor workflow."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Corridor output root. Default: "
            "<input-root>/outputs/cbi/corridors."
        ),
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        help="Batch summary root (default: <run-root>/summary).",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Rebuild the 70-corridor summary from completed corridor manifests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent corridor workers (default: 1).",
    )
    parser.add_argument(
        "--model-link-map",
        type=Path,
        required=True,
        help=(
            "Explicit frozen TMC/link mapping. Use this for treatment-specific "
            "runs so CBI cannot silently resolve a different producer run."
        ),
    )
    parser.add_argument(
        "--exclude-corridors-file",
        type=Path,
        help=(
            "CSV containing corridor or corridor_key values that have an "
            "audited reason for exclusion from link-based CBI."
        ),
    )
    parser.add_argument("corridors", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excluded_keys: set[str] = set()
    if args.exclude_corridors_file is not None:
        exclusions = pd.read_csv(args.exclude_corridors_file)
        column = next(
            (name for name in ("corridor", "corridor_key") if name in exclusions),
            None,
        )
        if column is None:
            raise ValueError(
                "Excluded-corridor CSV requires corridor or corridor_key"
            )
        excluded_keys = set(exclusions[column].dropna().astype(str).str.strip())
    output_root = (
        args.output_root
        if args.output_root is not None
        else args.input_root.resolve() / "outputs" / "cbi" / "corridors"
    )
    summary_root = (
        args.summary_root
        if args.summary_root is not None
        else output_root.parent / "summary"
    )
    result = (
        finalize_existing_batch(
            args.input_root,
            output_root,
            summary_root,
            args.model_link_map,
            excluded_keys=excluded_keys,
        )
        if args.finalize_only
        else run_batch(
            args.input_root,
            output_root,
            summary_root,
            [key for key in args.corridors if key not in excluded_keys]
            if args.corridors
            else [
                key
                for key in discover_inrix_corridors(
                    args.input_root,
                    model_link_map=args.model_link_map,
                )
                if key not in excluded_keys
            ],
            generate_figures=not args.no_figures,
            workers=max(1, args.workers),
            model_link_map=args.model_link_map,
            settings=PipelineSettings(),
        )
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
