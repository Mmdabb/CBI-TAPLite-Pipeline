# Corridor Performance Dashboard

Corridor-level observed-versus-TAPlite measurement and integrated static dashboard generation in one independent repository. The package consumes explicit outputs from CBI, TMC matching, and `nvta-taplite-workflow`; it does not select a “latest” folder or require those packages to share a workspace.

## Repository layout

```text
src/corridor_measurement/             profile construction, metrics, figures
src/corridor_dashboard/               projection diagnostics and static site
src/corridor_performance_dashboard/   unified CLI, QA, logging, adapters
tests/                                unit and integration-contract tests
examples/                             NVTA staging and source-column aliases
scripts/                              optional explicit-input diagnostics
docs/                                 producer/consumer contracts
```

## Install

```console
python -m pip install -e .
corridor-dashboard --help
```

The unified command is recommended. `corridor-measurement` and `integrated-corridor-dashboard` remain available for direct component use.

## QA before work

```console
corridor-dashboard qa --process all \
  --cbi-corridors CBI/corridors \
  --mapmatching-root MATCHING \
  --assignment-root ASSIGNMENT \
  --model-link-map CBI/shared/network-mapping/canonical_node_pair_tmc.csv \
  --observed-15min OBSERVED.csv
```

QA verifies:

- complete CBI average-weekday profile and link-reference products;
- the frozen canonical node-pair winner map;
- AM/MD/PM matching products and route summaries;
- AM/MD/PM `link.csv` and `link_performance.csv` files, including time-dependent speeds;
- dashboard measurement products, model-link map, and 15-minute observed speeds when requested.

Failures return exit code 2 with the precise missing path or field. `qa/input_qa.json` records every checked file.

The product and file names are CLI options: `--am-product`, `--md-product`, `--pm-product`, `--dashboard-product`, `--mapping-file-name`, `--route-summary-file-name`, `--performance-file-name`, and `--link-file-name`. `--column-map` supports canonical-to-source aliases for mapping, route-summary, performance, link, and observed inputs. When names differ, normalized copies are written below the run’s `normalized-inputs/`; original producers are never changed.

CBI numbered output fields are a versioned producer contract, not guessed source columns. Their contract is documented below and QA fails if an incompatible CBI producer is supplied.

## Measure profiles

```console
corridor-dashboard measure \
  --cbi-corridors CBI/corridors \
  --mapmatching-root MATCHING \
  --assignment-root ASSIGNMENT \
  --workers 4
```

Default output is `ASSIGNMENT/outputs/corridor-performance-dashboard/measurement`. Override it with `--measurement-output`. No datetime is added and a nonempty product root is rejected. Analysis options live in portable JSON; pass `--settings`, or use the packaged defaults. Product names supplied on the CLI override the JSON.

The measurement output retains the numbered data/QA/figure contract used by the existing dashboard, plus `qa/`, `logs/`, and the effective settings file.

## Build the dashboard

```console
corridor-dashboard dashboard \
  --cbi-corridors CBI/corridors \
  --mapmatching-root MATCHING \
  --assignment-root ASSIGNMENT \
  --measurement-root MEASUREMENT \
  --model-link-map CBI/shared/network-mapping/canonical_node_pair_tmc.csv \
  --observed-15min OBSERVED.csv
```

Default output is `MEASUREMENT/outputs/integrated-dashboard`. The result is a self-contained static site with `index.html`, corridor pages, methods, downloadable data, figures, and `build_manifest.json`. Use `--force-dashboard` only when intentional replacement is desired.

## Run both stages

```console
corridor-dashboard all \
  --cbi-corridors CBI/corridors \
  --mapmatching-root MATCHING \
  --assignment-root ASSIGNMENT \
  --model-link-map CBI/shared/network-mapping/canonical_node_pair_tmc.csv \
  --observed-15min OBSERVED.csv \
  --workers 4
```

Console output is limited to QA/stage milestones. Full engine output is saved in `logs/measurement-engine.log` and `logs/dashboard-engine.log`; structured application logging is in `logs/run.log`.

## NVTA working example

Use real producer outputs without embedding their dated names in this repository:

```console
python examples/prepare_nvta_example.py \
  --source-package-root PATH_TO_NVTA_CBI_PACKAGE \
  --cbi-relative-path PATH_RELATIVE_TO_SOURCE \
  --matching-relative-path PATH_RELATIVE_TO_SOURCE \
  --assignment-relative-path PATH_RELATIVE_TO_SOURCE \
  --model-link-map-relative-path PATH_RELATIVE_TO_SOURCE \
  --output-dir example-input
```

Then run the command printed by the preparation script. The script copies source products and does not alter the NVTA package.

## Development

```console
python -m pip install -e ".[dev]"
pytest -q
```
