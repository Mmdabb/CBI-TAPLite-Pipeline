from __future__ import annotations

"""Build treatment-aware period maps and one immutable direct canonical map."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PERIODS = ("am", "md", "pm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalized_tmc(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper()


def _open_links(network_root: Path, period: str) -> set[int]:
    path = network_root / period / "link.csv"
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in {"link_id", "lanes"},
        low_memory=False,
    )
    frame["link_id"] = pd.to_numeric(frame["link_id"], errors="coerce")
    frame["lanes"] = pd.to_numeric(frame.get("lanes"), errors="coerce")
    return set(
        frame.loc[frame["link_id"].notna() & frame["lanes"].gt(0), "link_id"]
        .astype("int64")
        .tolist()
    )


def build(
    coverage_root: Path,
    mapmatching_run: Path,
    network_root: Path,
    output_dir: Path,
    *,
    period_product_template: str = "{period}",
    mapping_file_name: str = "full_tmc_to_link.csv",
    actual_mapping_relative_path: str = "actual/combined-direct-mapping/actual_tmc_to_link.csv",
    virtual_mapping_relative_path: str = "virtual/virtual_tmc_to_link.csv",
) -> dict[str, object]:
    coverage_root = Path(coverage_root).resolve()
    mapmatching_run = Path(mapmatching_run).resolve()
    network_root = Path(network_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    actual_path = coverage_root / actual_mapping_relative_path
    virtual_path = coverage_root / virtual_mapping_relative_path
    actual = pd.read_csv(actual_path, dtype={"tmc": "string"}, low_memory=False)
    virtual = pd.read_csv(virtual_path, dtype={"tmc": "string"}, low_memory=False)
    actual["tmc"] = _normalized_tmc(actual["tmc"])
    virtual["tmc"] = _normalized_tmc(virtual["tmc"])
    for frame in (actual, virtual):
        frame["link_id"] = pd.to_numeric(frame["link_id"], errors="coerce").astype("int64")

    canonical_columns = list(actual.columns)
    virtual_canonical = pd.DataFrame(index=virtual.index, columns=canonical_columns)
    shared = [column for column in canonical_columns if column in virtual.columns]
    virtual_canonical[shared] = virtual[shared]
    virtual_canonical["distance_to_tmc_ft"] = 0.0
    virtual_canonical["node_pair_tmc_rank"] = 1
    virtual_canonical["node_pair_tmc_ranking_basis"] = virtual["treatment"]
    virtual_canonical["selected_for_node_pair_lookup"] = True
    virtual_canonical["link_tmc_rank"] = 1
    virtual_canonical["tmc_link_rank"] = 1
    virtual_canonical["match_status"] = virtual["treatment"]
    canonical = pd.concat([actual, virtual_canonical], ignore_index=True, sort=False)
    canonical["selected_for_node_pair_lookup"] = True
    if canonical.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError("Combined actual/virtual canonical map has duplicate node pairs")
    if canonical.duplicated("link_id").any():
        raise ValueError("Combined actual/virtual canonical map has duplicate link IDs")
    canonical_path = output_dir / "canonical_node_pair_tmc.csv"
    canonical.to_csv(canonical_path, index=False)

    actual_keys = actual[["tmc", "link_id"]].drop_duplicates()
    supplemental_actual = actual.merge(
        actual_keys,
        on=["tmc", "link_id"],
        how="inner",
    )
    period_products: dict[str, object] = {}
    for period in PERIODS:
        source = mapmatching_run / period_product_template.format(period=period) / mapping_file_name
        header = set(pd.read_csv(source, nrows=0).columns)
        usecols = [
            column
            for column in ("tmc", "link_id", "distance_to_tmc_ft", f"{period}_is_open")
            if column in header
        ]
        base = pd.read_csv(source, usecols=usecols, dtype={"tmc": "string"}, low_memory=False)
        base["tmc"] = _normalized_tmc(base["tmc"])
        base["link_id"] = pd.to_numeric(base["link_id"], errors="coerce")
        base = base.dropna(subset=["tmc", "link_id"]).copy()
        base["link_id"] = base["link_id"].astype("int64")
        if "distance_to_tmc_ft" not in base:
            base["distance_to_tmc_ft"] = 0.0
        if f"{period}_is_open" not in base:
            base[f"{period}_is_open"] = True

        existing = set(zip(base["tmc"].astype(str), base["link_id"]))
        additions = pd.concat(
            [
                actual[["tmc", "link_id", "distance_to_tmc_ft"]],
                virtual.assign(distance_to_tmc_ft=0.0)[
                    ["tmc", "link_id", "distance_to_tmc_ft"]
                ],
            ],
            ignore_index=True,
        )
        additions = additions[
            ~pd.Series(
                list(zip(additions["tmc"].astype(str), additions["link_id"])),
                index=additions.index,
            ).isin(existing)
        ].copy()
        open_links = _open_links(network_root, period)
        additions[f"{period}_is_open"] = additions["link_id"].isin(open_links)
        combined = pd.concat([base, additions], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(["tmc", "link_id"], keep="first")
        target = output_dir / f"{period}_full_tmc_to_link.csv"
        combined.to_csv(target, index=False)
        period_products[period.upper()] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "base_rows": int(len(base)),
            "added_rows": int(len(additions)),
            "added_open_rows": int(additions[f"{period}_is_open"].sum()),
            "output": str(target),
            "output_sha256": sha256(target),
        }

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "canonical_precedence": ["actual direct", "virtual direct"],
            "period_availability_for_added_rows": "period network lanes > 0",
            "period_winner": "same canonical TMC across all periods",
        },
        "actual_canonical_rows": int(len(actual)),
        "virtual_canonical_rows": int(len(virtual)),
        "combined_canonical_rows": int(len(canonical)),
        "canonical_output": str(canonical_path),
        "canonical_sha256": sha256(canonical_path),
        "periods": period_products,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--mapmatching-run", type=Path, required=True)
    parser.add_argument("--network-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--period-product-template", default="{period}")
    parser.add_argument("--mapping-file-name", default="full_tmc_to_link.csv")
    parser.add_argument(
        "--actual-mapping-relative-path",
        default="actual/combined-direct-mapping/actual_tmc_to_link.csv",
    )
    parser.add_argument(
        "--virtual-mapping-relative-path",
        default="virtual/virtual_tmc_to_link.csv",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.coverage_root,
                args.mapmatching_run,
                args.network_root,
                args.output_dir,
                period_product_template=args.period_product_template,
                mapping_file_name=args.mapping_file_name,
                actual_mapping_relative_path=args.actual_mapping_relative_path,
                virtual_mapping_relative_path=args.virtual_mapping_relative_path,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
