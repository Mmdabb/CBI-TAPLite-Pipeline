from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


MANAGED_CORRIDOR_PATTERN = re.compile(r"(?:HOV|EXPRESS|HOT|MANAGED)", re.IGNORECASE)


def is_managed_corridor(value: Any) -> bool:
    """Return whether a dashboard corridor identifier denotes managed lanes."""

    return bool(MANAGED_CORRIDOR_PATTERN.search(str(value or "").strip()))


def load_general_purpose_tmc_codes(product_root: Path) -> set[str]:
    """Return TMCs classified exclusively as general-purpose facilities."""

    source = Path(product_root) / "full_tmc_to_link.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing map-matching facility source: {source}")
    facility_classes: dict[str, set[str]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"tmc", "facility_class"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{source} is missing required fields: {missing}")
        for row in reader:
            tmc = str(row.get("tmc", "")).strip().upper()
            facility_class = (
                str(row.get("facility_class", "")).strip().lower()
                or "unclassified"
            )
            if tmc:
                facility_classes.setdefault(tmc, set()).add(facility_class)
    return {
        tmc for tmc, values in facility_classes.items() if values == {"gp"}
    }
