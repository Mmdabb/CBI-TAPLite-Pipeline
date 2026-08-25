# TMC Matching

Period-aware INRIX TMC-to-GMNS route matching for AM, MD, and PM networks. This repository is independent: it does not discover a workspace, select a “latest” run, or depend on the former `tmc_mapmatching_updated` directory name.

## Repository layout

```text
src/tmc_matching/   installable library and CLI
tests/              unit and opt-in NVTA regression tests
examples/           input staging and column-map examples
docs/               input/output contracts
```

## Install and inspect

```console
python -m pip install -e .
tmc-matching --help
tmc-matching qa --help
tmc-matching run --help
```

The CLI prints only milestones and actionable errors. Detailed application logs are written to `logs/run.log`; captured engine output is written to `logs/engine.log`.

## Input contract and QA

`--input-dir` contains the TMC metadata and the three network folders:

```text
INPUT/
  TMC_Identification.csv
  network/
    am/link.csv  am/node.csv
    md/link.csv  md/node.csv
    pm/link.csv  pm/node.csv
```

Run QA without matching:

```console
tmc-matching qa --input-dir INPUT
```

QA verifies every required file, source field, and nonempty dataset. A failure returns exit code 2 with the missing path or field. The machine-readable report is `INPUT/outputs/tmc-matching/qa/input_qa.json` unless `--report-dir` is supplied.

File names are configurable with `--tmc-file-name`, `--network-dir-name`, `--link-file-name`, and `--node-file-name`. Source columns are configurable with `--column-map`; the adapter writes canonical CSVs under `normalized-inputs/` only when aliases are needed. See `examples/column-map.json`.

## Run

```console
tmc-matching run --input-dir INPUT
```

The matcher itself is deterministic and does not require a worker argument; network-period products are written to a stable output root. Use `--output-dir` to override the default `INPUT/outputs/tmc-matching`. The command refuses a nonempty output directory rather than silently replacing a result.

Default products are:

```text
outputs/tmc-matching/
  combined/
  am/
  md/
  pm/
  qa/input_qa.json
  logs/run.log
  logs/engine.log
  run_manifest.json
```

Product names are controlled by `--combined-product-name` and `--period-product-template`. CRS, lane class, road selection, candidates, and period-product behavior are CLI options; inspect `tmc-matching run --help` for defaults.

## Post-processing tools

The installed package also retains the tested corridor-slice, observation-coverage, and unmatched-link audit modules. They take explicit inputs and write fixed producer-named outputs adjacent to the named input; none performs “latest run” discovery:

```console
python -m tmc_matching.create_cbi_corridor_slices --help
python -m tmc_matching.build_observation_coverage_treatments --help
python -m tmc_matching.audit_unmatched_dashboard_corridor_links --help
```

## NVTA working example

Stage the current NVTA inputs without embedding a machine path in this repository:

```console
python examples/prepare_nvta_example.py --source-package-root PATH_TO_NVTA_CBI_PACKAGE --output-dir example-input
tmc-matching qa --input-dir example-input
tmc-matching run --input-dir example-input --max-corridors 2
```

The preparation script copies inputs; it never changes the source package.

## Development

```console
python -m pip install -e ".[dev]"
pytest -q
```

Large NVTA regression tests are opt-in. Set `TMC_MATCHING_NVTA_INPUT_DIR` to a staged input directory to enable them.
