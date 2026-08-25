"""Create CBI-ready INRIX corridor slices from the shared regional export.

The creator reads the large regional readings file once, routes every row by
TMC, and writes the two-file CBI input contract for each corridor:

* ``TMC_Identification.csv``
* ``Readings.csv``

Corridors can be supplied explicitly in a CSV or generated from every exact
``(road, direction)`` pair in the master TMC inventory.  An optional semantic
comparison ignores harmless CSV formatting differences such as ``T`` versus a
space in timestamps and ``33.80`` versus ``33.8`` in numeric fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


TMC_COLUMNS = ("tmc", "tmc_code")
TIME_COLUMNS = ("measurement_tstamp", "datetime", "timestamp")
METADATA_NUMERIC_COLUMNS = {
    "start_latitude",
    "start_longitude",
    "end_latitude",
    "end_longitude",
    "miles",
    "road_order",
}


def normalize_text(value: object) -> str:
    return str(value or "").strip().upper()


def normalize_number(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text.lower()
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", ""} else normalized


def normalize_timestamp(value: object) -> str:
    return str(value or "").strip().replace("T", " ")


def first_present(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Required column is missing; expected one of " + ", ".join(candidates)
    )


def direction_abbreviation(direction: str) -> str:
    known = {
        "NORTHBOUND": "NB",
        "SOUTHBOUND": "SB",
        "EASTBOUND": "EB",
        "WESTBOUND": "WB",
        "NORTHEASTBOUND": "NEB",
        "NORTHWESTBOUND": "NWB",
        "SOUTHEASTBOUND": "SEB",
        "SOUTHWESTBOUND": "SWB",
        "CLOCKWISE": "CW",
        "COUNTERCLOCKWISE": "CCW",
    }
    normalized = normalize_text(direction)
    if normalized in known:
        return known[normalized]
    words = re.findall(r"[A-Z0-9]+", normalized)
    abbreviation = "".join(word[0] for word in words if word)
    return abbreviation or "UNKNOWN"


def corridor_key(road: str, direction: str) -> str:
    road_key = re.sub(r"[^A-Z0-9]+", "", normalize_text(road))
    if not road_key:
        road_key = "UNNAMED"
    return f"{road_key}_{direction_abbreviation(direction)}"


@dataclass(frozen=True)
class CorridorDefinition:
    key: str
    road: str
    direction: str

    @property
    def pair(self) -> Tuple[str, str]:
        return normalize_text(self.road), normalize_text(self.direction)


class MultisetFingerprint:
    """Order-independent semantic fingerprint that retains duplicate counts."""

    MODULUS = 1 << 256

    def __init__(self) -> None:
        self.rows = 0
        self.digest_xor = 0
        self.digest_sum = 0

    def add(self, token: str) -> None:
        digest = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big")
        self.rows += 1
        self.digest_xor ^= digest
        self.digest_sum = (self.digest_sum + digest) % self.MODULUS

    def as_dict(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "sha256_xor": f"{self.digest_xor:064x}",
            "sha256_sum_mod_2_256": f"{self.digest_sum:064x}",
        }

    def matches(self, other: "MultisetFingerprint") -> bool:
        return self.as_dict() == other.as_dict()


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def load_master_metadata(path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    fieldnames, rows = read_csv_rows(path)
    tmc_column = first_present(fieldnames, TMC_COLUMNS)
    required = {"road", "direction"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing metadata columns: {missing}")
    return fieldnames, rows, tmc_column


def definitions_from_all_pairs(metadata_rows: Iterable[Mapping[str, str]]) -> List[CorridorDefinition]:
    original_values: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for row in metadata_rows:
        pair = (normalize_text(row.get("road")), normalize_text(row.get("direction")))
        if not pair[0] or not pair[1]:
            continue
        original_values.setdefault(pair, (str(row.get("road", "")).strip(), str(row.get("direction", "")).strip()))
    definitions = [
        CorridorDefinition(corridor_key(road, direction), road, direction)
        for road, direction in original_values.values()
    ]
    return validate_definitions(definitions)


def definitions_from_csv(path: Path) -> List[CorridorDefinition]:
    fieldnames, rows = read_csv_rows(path)
    key_column = first_present(fieldnames, ("corridor_key", "corridor", "key"))
    missing = sorted({"road", "direction"} - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing definition columns: {missing}")
    definitions = [
        CorridorDefinition(
            str(row.get(key_column, "")).strip(),
            str(row.get("road", "")).strip(),
            str(row.get("direction", "")).strip(),
        )
        for row in rows
    ]
    return validate_definitions(definitions)


def definitions_from_corridor_root(root: Path) -> List[CorridorDefinition]:
    definitions: List[CorridorDefinition] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = folder / "TMC_Identification.csv"
        if not metadata_path.is_file():
            continue
        _, rows, _ = load_master_metadata(metadata_path)
        pairs = {
            (normalize_text(row.get("road")), normalize_text(row.get("direction")))
            for row in rows
            if normalize_text(row.get("road")) and normalize_text(row.get("direction"))
        }
        if len(pairs) != 1:
            raise ValueError(
                f"{metadata_path} must contain exactly one road/direction pair; found {sorted(pairs)}"
            )
        road, direction = next(iter(pairs))
        definitions.append(CorridorDefinition(folder.name, road, direction))
    return validate_definitions(definitions)


def validate_definitions(definitions: Iterable[CorridorDefinition]) -> List[CorridorDefinition]:
    result = sorted(definitions, key=lambda item: item.key)
    if not result:
        raise ValueError("No corridor definitions were produced")
    seen_keys: Dict[str, Tuple[str, str]] = {}
    seen_pairs: Dict[Tuple[str, str], str] = {}
    for definition in result:
        if not definition.key:
            raise ValueError("Corridor keys cannot be blank")
        pair = definition.pair
        if not pair[0] or not pair[1]:
            raise ValueError(f"{definition.key}: road and direction are required")
        if definition.key in seen_keys and seen_keys[definition.key] != pair:
            raise ValueError(f"Corridor key {definition.key!r} refers to multiple road/direction pairs")
        if pair in seen_pairs and seen_pairs[pair] != definition.key:
            raise ValueError(
                f"Road/direction pair {pair} is assigned to both {seen_pairs[pair]!r} and {definition.key!r}"
            )
        seen_keys[definition.key] = pair
        seen_pairs[pair] = definition.key
    return result


def metadata_token(row: Mapping[str, str], columns: Sequence[str]) -> str:
    parts: List[str] = []
    for column in columns:
        value = row.get(column, "")
        if column in METADATA_NUMERIC_COLUMNS:
            parts.append(normalize_number(value))
        elif column in TMC_COLUMNS or column in {"road", "direction", "state", "country"}:
            parts.append(normalize_text(value))
        else:
            parts.append(str(value or "").strip())
    return "\x1f".join(parts)


def reading_token(
    row: Mapping[str, str],
    columns: Sequence[str],
    tmc_column: str,
    time_column: str,
) -> str:
    parts: List[str] = []
    for column in columns:
        value = row.get(column, "")
        if column == tmc_column:
            parts.append(normalize_text(value))
        elif column == time_column:
            parts.append(normalize_timestamp(value))
        else:
            parts.append(normalize_number(value))
    return "\x1f".join(parts)


def create_slices(
    metadata_path: Path,
    readings_path: Path,
    output_dir: Path,
    definitions: Sequence[CorridorDefinition],
    progress_every: int = 1_000_000,
) -> Dict[str, object]:
    fieldnames, metadata_rows, metadata_tmc_column = load_master_metadata(metadata_path)
    pair_to_definition = {definition.pair: definition for definition in definitions}
    rows_by_key: Dict[str, List[Dict[str, str]]] = {definition.key: [] for definition in definitions}
    tmc_to_key: Dict[str, str] = {}
    metadata_fingerprints = {definition.key: MultisetFingerprint() for definition in definitions}

    for row in metadata_rows:
        pair = (normalize_text(row.get("road")), normalize_text(row.get("direction")))
        definition = pair_to_definition.get(pair)
        if definition is None:
            continue
        tmc = normalize_text(row.get(metadata_tmc_column))
        if not tmc:
            continue
        previous = tmc_to_key.setdefault(tmc, definition.key)
        if previous != definition.key:
            raise ValueError(f"TMC {tmc} is assigned to both {previous} and {definition.key}")
        rows_by_key[definition.key].append(dict(row))
        metadata_fingerprints[definition.key].add(metadata_token(row, fieldnames))

    corridors_root = output_dir / "corridors"
    corridors_root.mkdir(parents=True, exist_ok=False)
    for definition in definitions:
        folder = corridors_root / definition.key
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "TMC_Identification.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows_by_key[definition.key])

    definition_path = output_dir / "corridor_definitions.csv"
    with definition_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["corridor_key", "road", "direction", "tmc_count"], lineterminator="\n"
        )
        writer.writeheader()
        for definition in definitions:
            writer.writerow(
                {
                    "corridor_key": definition.key,
                    "road": definition.road,
                    "direction": definition.direction,
                    "tmc_count": len(rows_by_key[definition.key]),
                }
            )

    with readings_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{readings_path} has no CSV header")
        reading_columns = list(reader.fieldnames)
        reading_tmc_column = first_present(reading_columns, TMC_COLUMNS)
        time_column = first_present(reading_columns, TIME_COLUMNS)
        handles: Dict[str, object] = {}
        writers: Dict[str, csv.DictWriter] = {}
        reading_fingerprints = {definition.key: MultisetFingerprint() for definition in definitions}
        source_rows = 0
        written_rows = 0
        try:
            for definition in definitions:
                handle = (corridors_root / definition.key / "Readings.csv").open(
                    "w", encoding="utf-8", newline=""
                )
                handles[definition.key] = handle
                writer = csv.DictWriter(handle, fieldnames=reading_columns, lineterminator="\n")
                writer.writeheader()
                writers[definition.key] = writer
            for row in reader:
                source_rows += 1
                key = tmc_to_key.get(normalize_text(row.get(reading_tmc_column)))
                if key is not None:
                    writers[key].writerow(row)
                    reading_fingerprints[key].add(
                        reading_token(row, reading_columns, reading_tmc_column, time_column)
                    )
                    written_rows += 1
                if progress_every and source_rows % progress_every == 0:
                    print(
                        f"Processed {source_rows:,} regional rows; wrote {written_rows:,} corridor rows",
                        flush=True,
                    )
        finally:
            for handle in handles.values():
                handle.close()

    return {
        "metadata_columns": fieldnames,
        "readings_columns": reading_columns,
        "metadata_tmc_column": metadata_tmc_column,
        "readings_tmc_column": reading_tmc_column,
        "readings_time_column": time_column,
        "source_metadata_rows": len(metadata_rows),
        "source_reading_rows": source_rows,
        "selected_tmc_count": len(tmc_to_key),
        "written_reading_rows": written_rows,
        "metadata_fingerprints": metadata_fingerprints,
        "reading_fingerprints": reading_fingerprints,
        "rows_by_key": rows_by_key,
    }


def fingerprint_csv(
    path: Path,
    expected_columns: Sequence[str],
    kind: str,
) -> MultisetFingerprint:
    fingerprint = MultisetFingerprint()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        if kind == "metadata":
            missing = sorted(set(expected_columns) - set(reader.fieldnames))
            if missing:
                raise ValueError(f"{path} is missing comparison columns: {missing}")
            for row in reader:
                fingerprint.add(metadata_token(row, expected_columns))
        elif kind == "readings":
            tmc_column = first_present(reader.fieldnames, TMC_COLUMNS)
            time_column = first_present(reader.fieldnames, TIME_COLUMNS)
            missing = sorted(set(expected_columns) - set(reader.fieldnames))
            if missing:
                raise ValueError(f"{path} is missing comparison columns: {missing}")
            for row in reader:
                fingerprint.add(reading_token(row, expected_columns, tmc_column, time_column))
        else:
            raise ValueError(f"Unsupported fingerprint kind: {kind}")
    return fingerprint


def compare_existing(
    existing_root: Path,
    output_dir: Path,
    definitions: Sequence[CorridorDefinition],
    generation: Mapping[str, object],
) -> Dict[str, object]:
    pair_to_key = {definition.pair: definition.key for definition in definitions}
    metadata_columns = list(generation["metadata_columns"])
    reading_columns = list(generation["readings_columns"])
    metadata_fingerprints = generation["metadata_fingerprints"]
    reading_fingerprints = generation["reading_fingerprints"]
    rows: List[Dict[str, object]] = []

    for folder in sorted(path for path in existing_root.iterdir() if path.is_dir()):
        metadata_path = folder / "TMC_Identification.csv"
        readings_path = folder / "Readings.csv"
        if not metadata_path.is_file() or not readings_path.is_file():
            continue
        _, existing_metadata_rows, _ = load_master_metadata(metadata_path)
        pairs = {
            (normalize_text(row.get("road")), normalize_text(row.get("direction")))
            for row in existing_metadata_rows
            if normalize_text(row.get("road")) and normalize_text(row.get("direction"))
        }
        if len(pairs) != 1:
            raise ValueError(f"{metadata_path} contains {len(pairs)} road/direction pairs")
        pair = next(iter(pairs))
        generated_key = pair_to_key.get(pair)
        if generated_key is None:
            rows.append(
                {
                    "existing_corridor": folder.name,
                    "generated_corridor": "",
                    "road": pair[0],
                    "direction": pair[1],
                    "metadata_match": False,
                    "readings_match": False,
                    "status": "missing_generated_pair",
                }
            )
            continue
        existing_metadata_fingerprint = fingerprint_csv(
            metadata_path, metadata_columns, "metadata"
        )
        existing_reading_fingerprint = fingerprint_csv(
            readings_path, reading_columns, "readings"
        )
        generated_metadata_fingerprint = metadata_fingerprints[generated_key]
        generated_reading_fingerprint = reading_fingerprints[generated_key]
        metadata_match = existing_metadata_fingerprint.matches(generated_metadata_fingerprint)
        readings_match = existing_reading_fingerprint.matches(generated_reading_fingerprint)
        rows.append(
            {
                "existing_corridor": folder.name,
                "generated_corridor": generated_key,
                "road": pair[0],
                "direction": pair[1],
                "existing_tmc_rows": existing_metadata_fingerprint.rows,
                "generated_tmc_rows": generated_metadata_fingerprint.rows,
                "existing_reading_rows": existing_reading_fingerprint.rows,
                "generated_reading_rows": generated_reading_fingerprint.rows,
                "metadata_match": metadata_match,
                "readings_match": readings_match,
                "status": "match" if metadata_match and readings_match else "different",
            }
        )
        print(
            f"Compared {folder.name}: metadata={metadata_match}, readings={readings_match}",
            flush=True,
        )

    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "existing_corridor",
        "generated_corridor",
        "road",
        "direction",
        "existing_tmc_rows",
        "generated_tmc_rows",
        "existing_reading_rows",
        "generated_reading_rows",
        "metadata_match",
        "readings_match",
        "status",
    ]
    with (report_dir / "comparison_to_existing.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    matching = sum(row.get("status") == "match" for row in rows)
    result = {
        "existing_corridors_compared": len(rows),
        "exact_semantic_matches": matching,
        "differences": len(rows) - matching,
        "all_existing_corridors_match": bool(rows) and matching == len(rows),
        "comparison_csv": str(report_dir / "comparison_to_existing.csv"),
    }
    (report_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--definitions-csv",
        type=Path,
        help="CSV with corridor_key (or corridor/key), road, and direction columns.",
    )
    selection.add_argument(
        "--definitions-from",
        type=Path,
        help="Existing corridor root used only to derive folder key/road/direction definitions.",
    )
    selection.add_argument(
        "--all-road-directions",
        action="store_true",
        help="Create one corridor for every exact road/direction pair in the master inventory.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Existing corridor root to compare semantically with generated overlapping pairs.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = (
        args.output_dir
        or args.readings.resolve().parent / "outputs" / "cbi-corridor-slices"
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    _, metadata_rows, _ = load_master_metadata(args.metadata.resolve())
    if args.definitions_csv is not None:
        definitions = definitions_from_csv(args.definitions_csv.resolve())
        definition_mode = "definitions_csv"
    elif args.definitions_from is not None:
        definitions = definitions_from_corridor_root(args.definitions_from.resolve())
        definition_mode = "existing_corridor_root"
    else:
        definitions = definitions_from_all_pairs(metadata_rows)
        definition_mode = "all_road_direction_pairs"

    generation = create_slices(
        args.metadata.resolve(),
        args.readings.resolve(),
        output_dir,
        definitions,
        progress_every=max(args.progress_every, 0),
    )
    comparison = None
    if args.compare_to is not None:
        comparison = compare_existing(
            args.compare_to.resolve(), output_dir, definitions, generation
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition_mode": definition_mode,
        "metadata_source": str(args.metadata.resolve()),
        "readings_source": str(args.readings.resolve()),
        "output_dir": str(output_dir),
        "corridor_count": len(definitions),
        "selected_tmc_count": generation["selected_tmc_count"],
        "source_metadata_rows": generation["source_metadata_rows"],
        "source_reading_rows": generation["source_reading_rows"],
        "written_reading_rows": generation["written_reading_rows"],
        "comparison": comparison,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
