import csv
import sys
import tempfile
import unittest
from pathlib import Path

MATCHER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MATCHER_ROOT / "src"))

from tmc_matching.create_cbi_corridor_slices import (
    compare_existing,
    create_slices,
    definitions_from_all_pairs,
)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class CorridorSliceCreatorTests(unittest.TestCase):
    def test_all_pairs_and_semantic_existing_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_columns = ["tmc", "road", "direction", "miles", "road_order"]
            metadata_rows = [
                {"tmc": "A", "road": "I-1", "direction": "EASTBOUND", "miles": "1.20", "road_order": "1.0"},
                {"tmc": "B", "road": "I-1", "direction": "EASTBOUND", "miles": "2", "road_order": "2"},
                {"tmc": "C", "road": "VA-2", "direction": "NORTHBOUND", "miles": "3", "road_order": "1"},
            ]
            reading_columns = ["tmc_code", "measurement_tstamp", "speed", "reference_speed"]
            reading_rows = [
                {"tmc_code": "A", "measurement_tstamp": "2025-10-01 06:00:00", "speed": "33.80", "reference_speed": "55.00"},
                {"tmc_code": "B", "measurement_tstamp": "2025-10-01 06:00:00", "speed": "40.00", "reference_speed": "55.00"},
                {"tmc_code": "C", "measurement_tstamp": "2025-10-01 06:00:00", "speed": "45.00", "reference_speed": "50.00"},
            ]
            metadata = root / "metadata.csv"
            readings = root / "readings.csv"
            write_csv(metadata, metadata_columns, metadata_rows)
            write_csv(readings, reading_columns, reading_rows)

            definitions = definitions_from_all_pairs(metadata_rows)
            self.assertEqual([item.key for item in definitions], ["I1_EB", "VA2_NB"])
            output = root / "generated"
            output.mkdir()
            generation = create_slices(metadata, readings, output, definitions, progress_every=0)
            self.assertEqual(generation["selected_tmc_count"], 3)
            self.assertEqual(generation["written_reading_rows"], 3)

            existing = root / "existing" / "I1_EB"
            write_csv(existing / "TMC_Identification.csv", ["column"] + metadata_columns, [
                {"column": "", "tmc": "A", "road": "I-1", "direction": "EASTBOUND", "miles": "1.2", "road_order": "1"},
                {"column": "", "tmc": "B", "road": "I-1", "direction": "EASTBOUND", "miles": "2.0", "road_order": "2.0"},
            ])
            write_csv(existing / "Readings.csv", reading_columns, [
                {"tmc_code": "A", "measurement_tstamp": "2025-10-01T06:00:00", "speed": "33.8", "reference_speed": "55"},
                {"tmc_code": "B", "measurement_tstamp": "2025-10-01T06:00:00", "speed": "40", "reference_speed": "55"},
            ])
            comparison = compare_existing(existing.parent, output, definitions, generation)
            self.assertTrue(comparison["all_existing_corridors_match"])
            self.assertEqual(comparison["exact_semantic_matches"], 1)


if __name__ == "__main__":
    unittest.main()
