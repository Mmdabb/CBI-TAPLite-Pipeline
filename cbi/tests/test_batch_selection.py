from __future__ import annotations

from pathlib import Path

import pandas as pd

from cbi.batch import select_corridors_with_frozen_tmcs
from cbi.config import CorridorSpec


def _spec(root: Path, key: str, tmcs: list[str], mapping: Path) -> CorridorSpec:
    folder = root / key
    folder.mkdir()
    pd.DataFrame({"tmc": tmcs}).to_csv(folder / "TMC_Identification.csv", index=False)
    return CorridorSpec(
        key=key,
        name=key,
        source="inrix_folder",
        path=folder,
        free_flow_mph=45.0,
        capacity_vphpl=1800.0,
        model_link_map=mapping,
    )


def test_select_corridors_with_frozen_tmcs_is_explicit_and_auditable(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "canonical.csv"
    pd.DataFrame({"tmc": ["winner-a", "winner-c"]}).to_csv(mapping, index=False)
    specs = {
        "A": _spec(tmp_path, "A", ["WINNER-A", "loser"], mapping),
        "B": _spec(tmp_path, "B", ["loser-only"], mapping),
    }

    selected, audit = select_corridors_with_frozen_tmcs(specs, mapping)

    assert selected == ["A"]
    assert audit.set_index("corridor_key").loc["A", "included"]
    assert not audit.set_index("corridor_key").loc["B", "included"]
    assert (
        audit.set_index("corridor_key").loc["B", "reason"]
        == "no_frozen_node_pair_winner_in_corridor"
    )
