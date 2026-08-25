"""Complete network congestion boundaries using ML or the VDF-class hierarchy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

COMPLETION_MODES = ("ml", "vdf_class")
RESOURCE_ROOT = Path(__file__).resolve().parent / "resources" / "ridge_completion"
ML_REQUIRED_PATHS = (
    "experimental_network_boundaries.csv",
    "metrics/out_of_fold_predictions.csv",
)
COMPARISON_REQUIRED_PATHS = ()


def _load_ml_functions(workspace_root: Path):
    """Load the mapper-owned, tested Ridge application implementation."""

    del workspace_root
    from congestion_boundary_mapping.ridge_completion.apply_network_fill import (
        run as apply_ml_completion,
    )
    from congestion_boundary_mapping.ridge_completion.export_node_pair_lookup import (
        run as export_node_pair_lookup,
    )

    return apply_ml_completion, export_node_pair_lookup


def _required_artifact_root(
    root: Path,
    label: str,
    required_paths: tuple[str, ...],
) -> Path:
    root = Path(root).resolve()
    missing = [relative for relative in required_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"The {label} artifact set is incomplete under {root}; "
            f"missing: {missing}"
        )
    return root


def resolve_ml_inputs(
    workspace_root: Path,
    ml_run_dir: Optional[Path] = None,
    comparison_run_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Resolve explicit overrides or the mapper-owned Ridge artifacts."""

    del workspace_root
    ml_run = (
        Path(ml_run_dir).resolve()
        if ml_run_dir is not None
        else RESOURCE_ROOT / "ml_run"
    )
    comparison_run = (
        Path(comparison_run_dir).resolve()
        if comparison_run_dir is not None
        else RESOURCE_ROOT / "comparison_run"
    )
    comparison_candidates = (
        comparison_run / "outputs" / "validation_benchmark_detail.csv",
        comparison_run / "outputs" / "validation_predictions.csv",
    )
    if not any(path.is_file() for path in comparison_candidates):
        raise FileNotFoundError(
            "The spatial validation artifact set is incomplete under "
            f"{comparison_run}; expected one of: {comparison_candidates}"
        )
    return (
        _required_artifact_root(ml_run, "Ridge completion", ML_REQUIRED_PATHS),
        _required_artifact_root(
            comparison_run,
            "Ridge validation comparison",
            COMPARISON_REQUIRED_PATHS,
        ),
    )


def complete_boundaries(
    workspace_root: Path,
    mapping_run_dir: Path,
    mode: str = "ml",
    ml_run_dir: Optional[Path] = None,
    comparison_run_dir: Optional[Path] = None,
) -> dict:
    """Create the selected final boundary product for one mapping run."""

    if mode not in COMPLETION_MODES:
        raise ValueError(
            f"Unsupported completion mode {mode!r}; choose from {COMPLETION_MODES}."
        )
    workspace_root = Path(workspace_root).resolve()
    mapping_run_dir = Path(mapping_run_dir).resolve()
    link_t2_dir = mapping_run_dir / "link-t2"
    if not link_t2_dir.is_dir():
        raise FileNotFoundError(f"Missing mapped link-T2 input: {link_t2_dir}")

    if mode == "vdf_class":
        return {
            "status": "PASS",
            "mode": "vdf_class",
            "precedence": ["direct", "spatial", "vdf_class"],
            "final_output_dir": str(link_t2_dir),
            "final_period_link_root": str(link_t2_dir / "period_link_files"),
            "final_fields": [
                "t0_hybrid_hour",
                "t2_hybrid_hour",
                "t3_hybrid_hour",
            ],
            "network_wide_t0_t2_t3": False,
            "note": (
                "VDF-class completion estimates T2 only. T0/T3 remain populated "
                "only where direct episode boundaries exist."
            ),
        }

    ml_run, comparison_run = resolve_ml_inputs(
        workspace_root,
        ml_run_dir=ml_run_dir,
        comparison_run_dir=comparison_run_dir,
    )
    apply_ml_completion, export_node_pair_lookup = _load_ml_functions(
        workspace_root
    )
    final_output_dir = mapping_run_dir / "link-boundaries"
    apply_ml_completion(
        mapping_run_dir,
        ml_run,
        comparison_run,
        final_output_dir,
    )
    lookup_dir = final_output_dir / "node_pair_lookup"
    export_node_pair_lookup(
        final_output_dir / "period_link_files",
        lookup_dir,
    )
    lookup_metadata = json.loads(
        (lookup_dir / "metadata.json").read_text(encoding="utf-8")
    )
    return {
        "status": "PASS",
        "mode": "ml",
        "precedence": ["direct", "spatial", "ml"],
        "selected_model": "ridge_core",
        "implementation": "congestion_boundary_mapping.ridge_completion",
        "ml_run_dir": str(ml_run),
        "comparison_run_dir": str(comparison_run),
        "final_output_dir": str(final_output_dir),
        "final_period_link_root": str(final_output_dir / "period_link_files"),
        "node_pair_lookup_dir": str(lookup_dir),
        "node_pair_lookup_periods": lookup_metadata["periods"],
        "final_fields": [
            "t0_hybrid_hour",
            "t2_hybrid_hour",
            "t3_hybrid_hour",
        ],
        "network_wide_t0_t2_t3": False,
        "completion_scope": (
            "all network link-period rows except best-match observed TMC links "
            "with no accepted average-weekday congestion episode"
        ),
    }
