import json
from pathlib import Path

from congestion_boundary_mapping import completion, hybrid_t2


def test_vdf_class_mode_retains_link_t2_product(tmp_path):
    mapping_run = tmp_path / "congestion-boundary-mapping-2026-07-31-10-00"
    (mapping_run / "link-t2").mkdir(parents=True)

    result = completion.complete_boundaries(
        tmp_path,
        mapping_run,
        mode="vdf_class",
    )

    assert result["mode"] == "vdf_class"
    assert result["precedence"] == ["direct", "spatial", "vdf_class"]
    assert result["network_wide_t0_t2_t3"] is False
    assert result["final_output_dir"] == str(mapping_run / "link-t2")


def test_ml_mode_uses_mapper_owned_implementation(monkeypatch, tmp_path):
    mapping_run = tmp_path / "congestion-boundary-mapping-2026-07-31-10-00"
    (mapping_run / "link-t2").mkdir(parents=True)
    ml_run = tmp_path / "ml-run"
    comparison_run = tmp_path / "comparison-run"
    (ml_run / "metrics").mkdir(parents=True)
    (comparison_run / "outputs").mkdir(parents=True)
    (ml_run / "experimental_network_boundaries.csv").write_text(
        "network_link_id,period\n", encoding="utf-8"
    )
    (ml_run / "metrics" / "out_of_fold_predictions.csv").write_text(
        "tmc_code,period\n", encoding="utf-8"
    )
    (
        comparison_run / "outputs" / "validation_benchmark_detail.csv"
    ).write_text("tmc,period\n", encoding="utf-8")
    calls = []

    def fake_apply(mapping_input, ml_input, comparison_input, output_dir):
        calls.append((mapping_input, ml_input, comparison_input, output_dir))
        (output_dir / "period_link_files").mkdir(parents=True)
        return output_dir

    def fake_export(period_link_root, output_dir):
        output_dir.mkdir(parents=True)
        (output_dir / "metadata.json").write_text(
            json.dumps({"periods": {"AM": {}, "MD": {}, "PM": {}}}),
            encoding="utf-8",
        )
        return output_dir

    monkeypatch.setattr(
        completion,
        "_load_ml_functions",
        lambda workspace_root: (fake_apply, fake_export),
    )

    result = completion.complete_boundaries(
        tmp_path,
        mapping_run,
        mode="ml",
        ml_run_dir=ml_run,
        comparison_run_dir=comparison_run,
    )

    assert len(calls) == 1
    assert result["mode"] == "ml"
    assert result["selected_model"] == "ridge_core"
    assert result["implementation"] == (
        "congestion_boundary_mapping.ridge_completion"
    )
    assert result["precedence"] == ["direct", "spatial", "ml"]
    assert result["network_wide_t0_t2_t3"] is False
    assert Path(result["node_pair_lookup_dir"]).name == "node_pair_lookup"


def test_default_completion_resources_are_self_contained(tmp_path):
    ml_run, comparison_run = completion.resolve_ml_inputs(tmp_path)
    assert ml_run.is_relative_to(completion.RESOURCE_ROOT)
    assert comparison_run.is_relative_to(completion.RESOURCE_ROOT)
    for relative in completion.ML_REQUIRED_PATHS:
        assert (ml_run / relative).is_file()
    for relative in completion.COMPARISON_REQUIRED_PATHS:
        assert (comparison_run / relative).is_file()

    apply, export = completion._load_ml_functions(tmp_path)
    assert apply.__module__.startswith(
        "congestion_boundary_mapping.ridge_completion"
    )
    assert export.__module__.startswith(
        "congestion_boundary_mapping.ridge_completion"
    )


def test_default_spatial_resource_is_mapper_owned(tmp_path):
    path = hybrid_t2.latest_spatial_output(tmp_path)
    assert path.is_relative_to(completion.RESOURCE_ROOT)
    assert path.name == "expanded_link_t2.csv"
