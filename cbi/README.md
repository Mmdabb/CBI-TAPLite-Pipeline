# CBI QVDF

Congestion bottleneck identification (CBI), episode screening, QVDF calibration, and portable link-boundary resource creation. The repository is standalone and never searches another workspace for a “latest” run.

## Repository layout

```text
src/cbi/                           CBI library and primary CLI
src/t2_coverage_expansion/         direct/spatial T2 coverage and validation
src/t2_ml_experiment/              fresh leakage-controlled Ridge training
src/congestion_boundary_mapping/  replaceable link T0/T2/T3 resource producer
  ridge_completion/               tested Ridge application/export code
  resources/ridge_completion/     optional versioned fallback artifacts
tests/                             unit and contract tests
examples/                          NVTA input staging and configuration examples
docs/                              process contracts
```

The former legacy `t2` package is intentionally absent.  The spatial coverage
and Ridge-training code required by a full run is now owned by this package
under the two explicit modules above.  Congestion-boundary completion owns the
application/export layer and may also use its versioned model only for a
standalone inference run.

## Install

The large Ridge network artifact is tracked with Git LFS. Clone with LFS enabled, then install:

```console
git lfs pull
python -m pip install -e .
cbi --help
```

## Input QA

The corridor root contains one directory per corridor, each with metadata and 15-minute readings. A frozen mapping is always explicit:

```console
cbi qa --input-dir INPUT/corridors --model-link-map INPUT/canonical_node_pair_tmc.csv
```

QA checks file presence, readable headers, configured fields, nonempty rows, and at least one complete corridor. It exits with code 2 and an actionable message on failure. The JSON report is written to `INPUT/corridors/outputs/cbi/qa/input_qa.json` unless `--report-dir` is given.

Input filenames are controlled by `--metadata-file-name` and `--readings-file-name`. `--column-map` accepts canonical-to-source aliases for `metadata`, `readings`, and `mapping`; noncanonical inputs are adapted under `normalized-inputs/` without changing the originals.

## Run CBI

```console
cbi run \
  --input-dir INPUT/corridors \
  --model-link-map INPUT/canonical_node_pair_tmc.csv \
  --workers 4
```

The stable default output is `INPUT/corridors/outputs/cbi`. Supply `--output-dir` to place it elsewhere. CBI refuses a nonempty output. Console logging is concise; detailed logs are `logs/run.log` and `logs/engine.log`.

```text
outputs/cbi/
  corridors/       numbered per-corridor process products
  shared/          frozen canonical mapping artifacts
  summary/         cross-corridor QA summary and batch manifest
  qa/
  logs/
  run_manifest.json
```

Use `--corridor` repeatedly to select corridors and `--no-figures` for analysis-only execution. Algorithm settings are supplied in a JSON file through `--settings`; every accepted key is a `PipelineSettings` field. See `examples/settings.json`.

## Congestion-boundary resource producer

This replaceable helper maps observed CBI episode boundaries to period networks, applies spatial/VDF treatment, and optionally completes the network with the bundled Ridge model:

```console
cbi-congestion-boundaries \
  --cbi-output-root OUTPUT/cbi \
  --canonical-node-pair-map OUTPUT/cbi/shared/network-mapping/canonical_node_pair_tmc.csv \
  --am-map MATCHING/am/full_tmc_to_link.csv \
  --md-map MATCHING/md/full_tmc_to_link.csv \
  --pm-map MATCHING/pm/full_tmc_to_link.csv \
  --network-root NETWORK \
  --output-dir OUTPUT/congestion-boundaries/link-t2
```

All producer inputs are explicit. If `--output-dir` is omitted, the stable default is `<cbi-output-root>/outputs/congestion-boundaries/link-t2`. `--spatial-output`, `--ml-run-dir`, and `--comparison-run-dir` replace bundled resources when required. `--completion-mode ml` uses the self-contained Ridge resources; `vdf_class` is the non-ML alternative. Assignment-only and completion-only modes require an explicit existing output directory.

For a full recalibration, create spatial coverage and retrain Ridge before the
final completion pass:

```console
cbi-spatial-t2 all --help
cbi-ridge-train --help
```

The root monorepo runner supplies deterministic input/output paths to both
commands and passes the resulting model and validation products explicitly to
`cbi-congestion-boundaries`.  No helper searches for a dated run.

## NVTA working example

```console
python examples/prepare_nvta_example.py \
  --source-package-root PATH_TO_NVTA_CBI_PACKAGE \
  --model-link-map PATH_TO_CANONICAL_MAP \
  --corridor I66_EB \
  --output-dir example-input
cbi qa --input-dir example-input/corridors --model-link-map example-input/canonical_node_pair_tmc.csv
cbi run --input-dir example-input/corridors --model-link-map example-input/canonical_node_pair_tmc.csv --corridor I66_EB
```

The staging script copies a real corridor and mapping; it does not change the source package.

## Development

```console
python -m pip install -e ".[dev]"
pytest -q
```
