"""Average original T2 by period VDF code and fill a separate t2_est column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd


from cbi.workers import recommend_workers


PERIOD_CONFIG = {
    "AM": {
        "directory": "am",
        "lookup_column": "t2_am",
        "start_hour": 6.0,
        "end_hour": 9.0,
    },
    "MD": {
        "directory": "md",
        "lookup_column": "t2_md",
        "start_hour": 9.0,
        "end_hour": 15.0,
    },
    "PM": {
        "directory": "pm",
        "lookup_column": "t2_pm",
        "start_hour": 15.0,
        "end_hour": 19.0,
    },
}
VDF_SOURCE_COLUMN = "link_type"
LOOKUP_COLUMNS = ["vdf_code", "t2_am", "t2_md", "t2_pm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Explicit existing congestion-boundary output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Explicit process-worker count. By default the system uses 70%% "
            "of currently free logical-core capacity, capped at three periods."
        ),
    )
    parser.add_argument(
        "--worker-fraction",
        type=float,
        default=0.70,
        help="Share of currently free logical-core capacity to use (default: 0.70).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_csv_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="").writerow([value])
    return buffer.getvalue()


def parse_optional_float(value: str) -> float | None:
    text = value.strip()
    if not text or text.lower() in {"na", "nan", "none", "null"}:
        return None
    number = float(text)
    return number if np.isfinite(number) else None


def normalize_vdf_code(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "<na>"}:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return ""
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def resolve_t2_est(
    original_t2: float | None,
    vdf_code: str,
    means: dict[str, float],
) -> tuple[float | None, str]:
    if original_t2 is not None:
        return original_t2, "carried_original"
    propagated = means.get(vdf_code)
    if propagated is None:
        return None, "unmatched_blank"
    return propagated, "propagated_vdf_mean"


def period_vdf_averages(
    source: Path,
    vdf_column: str,
) -> tuple[dict[str, float], dict[str, int], int]:
    frame = pd.read_csv(
        source,
        usecols=[vdf_column, "t2_hour"],
        dtype={vdf_column: "string"},
        low_memory=False,
    )
    frame["vdf_code"] = frame[vdf_column].map(normalize_vdf_code)
    frame["t2_hour"] = pd.to_numeric(frame["t2_hour"], errors="coerce")
    valid = frame[
        frame["vdf_code"].notna()
        & frame["vdf_code"].ne("")
        & frame["t2_hour"].notna()
    ]
    grouped = valid.groupby("vdf_code", sort=True)["t2_hour"].agg(
        ["mean", "count"]
    )
    means = {
        str(code): float(value)
        for code, value in grouped["mean"].items()
    }
    counts = {
        str(code): int(value)
        for code, value in grouped["count"].items()
    }
    all_vdf_codes = int(
        frame.loc[
            frame["vdf_code"].notna() & frame["vdf_code"].ne(""),
            "vdf_code",
        ].nunique()
    )
    return means, counts, all_vdf_codes


def process_period(
    task: tuple[str, Path, Path],
) -> dict[str, object]:
    period, source, target = task
    config = PERIOD_CONFIG[period]
    means, group_counts, all_vdf_codes = period_vdf_averages(
        source,
        VDF_SOURCE_COLUMN,
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    original_t2 = 0
    carried_t2_est = 0
    propagated_t2_est = 0
    unmatched_blank_t2_est = 0
    with source.open("r", encoding="utf-8", newline="") as input_stream:
        header_line = input_stream.readline()
        if not header_line:
            raise ValueError(f"{source} is empty.")
        newline = "\r\n" if header_line.endswith("\r\n") else "\n"
        header = next(csv.reader([header_line]))
        required = {VDF_SOURCE_COLUMN, "t2_hour"}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"{source} is missing columns: {missing}")
        if header.count("t2_est") > 1:
            raise ValueError(f"{source} contains duplicate t2_est columns.")
        existing_t2_est = "t2_est" in header
        if existing_t2_est and header[-1] != "t2_est":
            raise ValueError(
                f"{source} must keep t2_est as its final column for refresh."
            )
        vdf_index = header.index(VDF_SOURCE_COLUMN)
        t2_index = header.index("t2_hour")

        with target.open("w", encoding="utf-8", newline="") as output_stream:
            if existing_t2_est:
                output_stream.write(header_line)
            else:
                output_stream.write(
                    header_line.rstrip("\r\n") + ",t2_est" + newline
                )
            for raw_line in input_stream:
                fields = next(csv.reader([raw_line]))
                if len(fields) != len(header):
                    raise ValueError(
                        f"{source} contains a multiline or malformed CSV row."
                )
                t2 = parse_optional_float(fields[t2_index])
                vdf_code = normalize_vdf_code(fields[vdf_index])
                t2_est, source_method = resolve_t2_est(
                    t2,
                    vdf_code,
                    means,
                )
                if source_method == "carried_original":
                    original_t2 += 1
                    carried_t2_est += 1
                elif source_method == "propagated_vdf_mean":
                    propagated_t2_est += 1
                else:
                    unmatched_blank_t2_est += 1

                prefix = raw_line.rstrip("\r\n")
                if existing_t2_est:
                    prefix = prefix.rsplit(",", 1)[0]
                output_stream.write(
                    prefix
                    + ","
                    + format_csv_value(t2_est)
                    + newline
                )
                rows += 1

    expected_t2_est = carried_t2_est + propagated_t2_est
    check = pd.read_csv(
        target,
        usecols=[VDF_SOURCE_COLUMN, "t2_hour", "t2_est"],
        dtype={VDF_SOURCE_COLUMN: "string"},
        low_memory=False,
    )
    check_t2 = pd.to_numeric(check["t2_hour"], errors="coerce")
    check_est = pd.to_numeric(check["t2_est"], errors="coerce")
    if len(check) != rows:
        raise ValueError(f"{period} output row count changed.")
    if int(check_t2.notna().sum()) != original_t2:
        raise ValueError(f"{period} original T2 count changed.")
    if int(check_est.notna().sum()) != expected_t2_est:
        raise ValueError(f"{period} t2_est count does not match propagation.")
    if not np.allclose(
        check_est[check_t2.notna()].to_numpy(dtype=float),
        check_t2[check_t2.notna()].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{period} did not carry every original T2 to t2_est.")
    check_vdf = check[VDF_SOURCE_COLUMN].map(normalize_vdf_code)
    expected_est = check_t2.copy()
    missing_original = check_t2.isna()
    expected_est.loc[missing_original] = check_vdf.loc[
        missing_original
    ].map(means)
    expected_serialized = expected_est.map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else float(format_csv_value(value))
        )
    )
    if not check_est.isna().equals(
        expected_serialized.isna()
    ) or not np.array_equal(
        check_est.dropna().to_numpy(dtype=float),
        expected_serialized.dropna().to_numpy(dtype=float),
    ):
        raise ValueError(
            f"{period} t2_est does not exactly follow the link_type mean."
        )
    populated_est = check_est.dropna()
    start_hour = float(config["start_hour"])
    end_hour = float(config["end_hour"])
    if (
        (populated_est < start_hour - 1e-12)
        | (populated_est >= end_hour + 1e-12)
    ).any():
        raise ValueError(f"{period} t2_est contains an out-of-period value.")

    return {
        "period": period,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(target.resolve()),
        "rows": rows,
        "vdf_column": VDF_SOURCE_COLUMN,
        "all_vdf_codes": all_vdf_codes,
        "vdf_codes_with_original_t2": len(means),
        "vdf_codes_averaging_multiple_original_t2": int(
            sum(count > 1 for count in group_counts.values())
        ),
        "original_t2": original_t2,
        "carried_original_t2_to_t2_est": carried_t2_est,
        "propagated_t2_est": propagated_t2_est,
        "populated_t2_est": expected_t2_est,
        "unmatched_blank_t2_est": unmatched_blank_t2_est,
        "lookup_means": means,
        "lookup_source_counts": group_counts,
        "output_sha256": sha256(target),
    }


def vdf_sort_key(code: str) -> tuple[int, float | str, str]:
    try:
        return (0, float(code), code)
    except ValueError:
        return (1, code, code)


def build_lookup_table(
    results: list[dict[str, object]],
) -> pd.DataFrame:
    by_period = {str(result["period"]): result for result in results}
    codes: set[str] = set()
    for result in results:
        codes.update(str(code) for code in result["lookup_means"])
    rows: list[dict[str, object]] = []
    for code in sorted(codes, key=vdf_sort_key):
        row: dict[str, object] = {"vdf_code": code}
        for period, config in PERIOD_CONFIG.items():
            means = by_period[period]["lookup_means"]
            row[str(config["lookup_column"])] = means.get(code, np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=LOOKUP_COLUMNS)


def update_run_manifest(
    output_dir: Path,
    propagation_manifest: dict[str, object],
) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    period_link_files = manifest.setdefault("period_link_files", {})
    for period, result in propagation_manifest["periods"].items():
        item = period_link_files.setdefault(period, {})
        item.update(
            {
                "output": str(Path(result["output"]).resolve()),
                "sha256": result["output_sha256"],
                "rows": result["rows"],
                "assigned_t2": result["original_t2"],
                "blank_t2": result["rows"] - result["original_t2"],
                "vdf_grouping_column": result["vdf_column"],
                "t2_est_carried_original": result[
                    "carried_original_t2_to_t2_est"
                ],
                "t2_est_propagated_by_vdf": result["propagated_t2_est"],
                "t2_est_populated": result["populated_t2_est"],
                "t2_est_blank": result["unmatched_blank_t2_est"],
            }
        )
    manifest["vdf_t2_postprocessing"] = {
        "status": "PASS",
        "manifest": str((output_dir / "vdf_t2_propagation_manifest.json").resolve()),
        "lookup_table": str((output_dir / "vdf_code_t2_lookup.csv").resolve()),
        "worker_plan": propagation_manifest["worker_plan"],
        "aggregation_rule": propagation_manifest["aggregation_rule"],
        "fill_rule": propagation_manifest["fill_rule"],
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def propagate_t2_by_vdf(
    output_dir: Path,
    *,
    workers: int | None = None,
    worker_fraction: float = 0.70,
    update_parent_manifest: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    sources = {
        period: output_dir
        / str(config["directory"])
        / "link.csv"
        if output_dir.name == "period_link_files"
        else output_dir
        / "period_link_files"
        / str(config["directory"])
        / "link.csv"
        for period, config in PERIOD_CONFIG.items()
    }
    root = (
        output_dir.parent
        if output_dir.name == "period_link_files"
        else output_dir
    )
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing period link files: {missing}")

    worker_plan = recommend_workers(
        len(PERIOD_CONFIG),
        target_fraction=worker_fraction,
        explicit_workers=workers,
    )
    stage_root = root / f".vdf_t2_propagation_{uuid.uuid4().hex}"
    stage_root.mkdir(parents=False, exist_ok=False)
    committed = False
    try:
        tasks = [
            (
                period,
                sources[period],
                stage_root
                / "period_link_files"
                / str(config["directory"])
                / "link.csv",
            )
            for period, config in PERIOD_CONFIG.items()
        ]
        if worker_plan.workers <= 1:
            results = [process_period(task) for task in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=worker_plan.workers
            ) as executor:
                results = list(executor.map(process_period, tasks))
        results.sort(key=lambda item: list(PERIOD_CONFIG).index(item["period"]))

        lookup = build_lookup_table(results)
        lookup_path = stage_root / "vdf_code_t2_lookup.csv"
        lookup.to_csv(
            lookup_path,
            index=False,
            columns=LOOKUP_COLUMNS,
            float_format="%.12g",
        )
        if list(pd.read_csv(lookup_path, nrows=0).columns) != LOOKUP_COLUMNS:
            raise ValueError("VDF lookup table column order changed.")

        public_results: dict[str, dict[str, object]] = {}
        for result in results:
            period = str(result["period"])
            public_results[period] = {
                key: value
                for key, value in result.items()
                if key not in {"lookup_means", "lookup_source_counts"}
            }
            public_results[period]["output"] = str(sources[period])
        propagation_manifest: dict[str, object] = {
            "status": "PASS",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "aggregation_rule": (
                "For each period, average nonblank original t2_hour values "
                "within the link_type category; label that category vdf_code "
                "in the four-column lookup table."
            ),
            "fill_rule": (
                "t2_est carries original t2_hour when present; otherwise it "
                "uses that period's VDF-code mean; unmatched rows remain blank."
            ),
            "lookup_columns": LOOKUP_COLUMNS,
            "lookup_rows": int(len(lookup)),
            "lookup_sha256": sha256(lookup_path),
            "worker_plan": worker_plan.to_dict(),
            "parallel_stages": [
                "AM VDF aggregation and link-file propagation",
                "MD VDF aggregation and link-file propagation",
                "PM VDF aggregation and link-file propagation",
            ],
            "periods": public_results,
        }
        staged_manifest = stage_root / "vdf_t2_propagation_manifest.json"
        staged_manifest.write_text(
            json.dumps(propagation_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        for period, config in PERIOD_CONFIG.items():
            staged = (
                stage_root
                / "period_link_files"
                / str(config["directory"])
                / "link.csv"
            )
            os.replace(staged, sources[period])
        os.replace(lookup_path, root / "vdf_code_t2_lookup.csv")
        os.replace(
            staged_manifest,
            root / "vdf_t2_propagation_manifest.json",
        )
        committed = True
        if update_parent_manifest:
            update_run_manifest(root, propagation_manifest)
        return propagation_manifest
    finally:
        if stage_root.is_dir():
            if committed:
                shutil.rmtree(stage_root)
            else:
                print(
                    "VDF propagation failed before commit; staged diagnostics "
                    f"remain at {stage_root}",
                    flush=True,
                )


def main() -> int:
    args = parse_args()
    manifest = propagate_t2_by_vdf(
        args.output_dir,
        workers=args.workers,
        worker_fraction=args.worker_fraction,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
