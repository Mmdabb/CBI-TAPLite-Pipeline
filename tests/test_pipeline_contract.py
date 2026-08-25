from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from cbi_taplite_pipeline.config import ConfigurationError, load_config
import cbi_taplite_pipeline.stages as stages
from cbi_taplite_pipeline.stages import STAGES, stage_paths


def _config(tmp_path: Path, **taplite) -> Path:
    payload = {
        "input_root": "input-data",
        "output_root": "outputs/full-run",
        "workers": 1,
        "periods": {"am": "0600_0900", "md": "0900_1500", "pm": "1500_1900"},
        "files": {
            "tmc_metadata": "ritis/TMC.csv",
            "tmc_readings": "ritis/Readings.csv",
            "matching_input": "matching",
            "base_network": "network",
            "cube_scenario": "cube",
            "corridor_definitions": None,
            "profile_selection_overrides": None,
            "qvdf_override_dictionary": None,
        },
        "taplite": {"vdf_type": 2, "qvdf_profile_mode": 2, **taplite},
    }
    path = tmp_path / "config" / "test.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stage_order_is_complete_and_deterministic(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    assert [stage.key for stage in STAGES] == [
        "matching", "corridors", "canonical", "coverage", "cbi", "spatial-t2",
        "boundary-seed", "ridge", "boundaries", "network-qvdf", "resources",
        "assignment-1", "hybrid-anchors", "assignment-2", "measurement", "dashboard",
    ]
    assert stage_paths(config)["dashboard"].name == "16-integrated-dashboard"
    measurement = next(stage for stage in STAGES if stage.key == "measurement")
    assert measurement.required_outputs == (
        "07-run-metadata/run_manifest.json",
    )


@pytest.mark.parametrize("value", [0, 1, 3])
def test_vdf_type_two_is_invariant(tmp_path: Path, value: int) -> None:
    with pytest.raises(ConfigurationError, match="vdf_type"):
        load_config(_config(tmp_path, vdf_type=value))


def test_relative_paths_resolve_from_repository_root(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    assert config.input_root == (tmp_path / "input-data").resolve()
    assert config.output_root == (tmp_path / "outputs/full-run").resolve()


def test_ridge_receives_cbi_run_root_not_corridors_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_config(tmp_path))
    paths = stage_paths(config)
    validation = paths["spatial"] / "outputs" / "validation_predictions.csv"
    validation.parent.mkdir(parents=True)
    validation.write_text("row\n", encoding="utf-8")
    monkeypatch.setattr(stages, "run_command", lambda *args, **kwargs: None)
    output = paths["ridge"]

    stages.run_ridge(config, output)

    ridge_config = json.loads((output / "ridge_config.json").read_text())
    assert Path(ridge_config["cbi_run_dir"]) == paths["cbi"] / "actual"


def test_network_qvdf_keeps_actual_strict_and_omits_invalid_virtual_triplets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_config(tmp_path))
    commands: list[list[str]] = []
    monkeypatch.setattr(
        stages,
        "run_command",
        lambda _config, command, _output, **_kwargs: commands.append(command),
    )

    stages.run_network_qvdf(config, stage_paths(config)["qvdf"])

    assert "--observed-triplet-policy" not in commands[0]
    policy_index = commands[1].index("--observed-triplet-policy")
    assert commands[1][policy_index + 1] == "omit"


def test_assignment_conversion_cache_is_kept_out_of_input_bundle(
    tmp_path: Path,
) -> None:
    config = load_config(_config(tmp_path, conversion_cache=True))
    output = stage_paths(config)["assignment1"]
    anchors = stage_paths(config)["resources"] / "anchors"

    command = stages._assignment_command(config, output, anchors)

    cache_index = command.index("--conversion-cache-dir")
    assert Path(command[cache_index + 1]) == output / "conversion-cache"
    assert not Path(command[cache_index + 1]).is_relative_to(config.input_root)


def test_resource_lookup_metadata_is_unit_explicit_and_portable(
    tmp_path: Path,
) -> None:
    dtype = np.dtype([
        ("packed_key", "<u8"),
        ("from_node_id", "<u4"),
        ("to_node_id", "<u4"),
        ("plf_am", "<f4"),
    ])
    lineage = tmp_path / "outputs"
    actual = lineage / "10-network-qvdf" / "actual" / "plf.npy"
    virtual = lineage / "10-network-qvdf" / "virtual" / "plf.npy"
    actual.parent.mkdir(parents=True)
    virtual.parent.mkdir(parents=True)
    np.save(actual, np.array([(1, 1, 2, 1.0)], dtype=dtype))
    np.save(virtual, np.array([(2, 2, 3, 1.0)], dtype=dtype))
    destination = lineage / "11-resources" / "plf" / "lookup.npy"

    stages._merge_disjoint_lookups(
        actual,
        virtual,
        destination,
        scope="observed PLF",
        lineage_root=lineage,
        metadata_fields={"unit": "dimensionless"},
    )

    metadata = json.loads(
        (destination.parent / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["unit"] == "dimensionless"
    assert metadata["actual_source"] == "10-network-qvdf/actual/plf.npy"
    assert metadata["virtual_source"] == "10-network-qvdf/virtual/plf.npy"
    assert not Path(metadata["actual_source"]).is_absolute()


def test_resource_stage_seed_is_detached_from_packaged_resources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "packaged"
    destination = tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text("source", encoding="utf-8")

    stages._copytree_detached(source, destination)
    (destination / "metadata.json").write_text("generated", encoding="utf-8")

    assert (source / "metadata.json").read_text(encoding="utf-8") == "source"
