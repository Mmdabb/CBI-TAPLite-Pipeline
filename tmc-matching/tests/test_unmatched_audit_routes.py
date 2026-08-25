from __future__ import annotations

import pytest

from tmc_matching.audit_unmatched_dashboard_corridor_links import route_pattern


@pytest.mark.parametrize(
    ("road", "street_name"),
    [
        ("I-66", "I 66"),
        ("US-15-BR", "US 15-BR"),
        ("I-395 (HOV)", "I-395 HOV"),
        ("I-95/I-495", "I 95 / I-495"),
        ("BROADLANDS BLVD", "BROADLANDS BLVD"),
        ("SHELLHORN RD", "SHELLHORN-RD"),
    ],
)
def test_route_pattern_accepts_tmc_and_network_separator_variants(
    road: str,
    street_name: str,
) -> None:
    assert route_pattern(road).search(street_name.upper())


def test_route_pattern_does_not_match_longer_route_number() -> None:
    assert route_pattern("I-66").search("I-66")
    assert not route_pattern("I-66").search("I-664")


def test_route_pattern_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="searchable token"):
        route_pattern(" -- ")
