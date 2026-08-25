# CBI–TAPlite Pipeline

This monorepo contains the reproducible NVTA workflow from period-aware TMC
matching through the published integrated dashboard.  It packages the four
working components together while preserving their independent libraries and
CLIs:

```text
tmc-matching/                       TMC-to-GMNS matching and corridor inputs
cbi/                                CBI, QVDF calibration, T2/Ridge resources
nvta-taplite-workflow/              conversion, assignment, smoothing, aggregation
corridor-performance-dashboard/     corridor measures and integrated dashboard
src/cbi_taplite_pipeline/           root orchestration, QA, lineage, run control
```

The source `NVTA_internal/nvta-cbi-package` is never modified by this
repository.  The staging utility can hard-link or copy only the source inputs
needed for a full run; prior assignments, caches, route files, and other large
derived products are deliberately excluded.

## Reproducible stage sequence

`main.py` runs the following stable, producer-named folders.  No stage searches
for a dated or “latest” directory.

| # | stage | result |
|---:|---|---|
| 1 | `matching` | AM/MD/PM matches and the combined full TMC-to-link product |
| 2 | `corridors` | observed average-weekday corridor metadata and speed slices |
| 3 | `canonical` | one immutable composite-scored winning TMC per directed node pair |
| 4 | `coverage` | canonical actual, managed actual, virtual, and excluded treatments |
| 5 | `cbi` | separate actual/virtual episode and corridor QVDF calibration |
| 6 | `spatial-t2` | direct and spatial T2 coverage plus held-out validation |
| 7 | `boundary-seed` | direct/spatial link T0/T2/T3 candidates |
| 8 | `ridge` | a newly trained, leakage-controlled Ridge model |
| 9 | `boundaries` | direct-first, spatial-second, Ridge-last boundary completion |
| 10 | `network-qvdf` | actual-only network QVDF and isolated actual/virtual resources |
| 11 | `resources` | treatment-aware, run-local TAPlite resource bundle |
| 12 | `assignment-1` | first conversion, assignment, smoothing, and aggregation |
| 13 | `hybrid-anchors` | actual/virtual anchors retained; other links filled from stage 1 |
| 14 | `assignment-2` | final conversion, assignment, smoothing, and aggregation |
| 15 | `measurement` | corridor profile/error measurements and figures |
| 16 | `dashboard` | integrated static dashboard, tables, figures, and downloads |

The stages that are easy to miss in a manual run are source-data QA, freezing
the canonical mapping before calibration, auditable observation-coverage
treatments, keeping virtual CBI separate from actual/network-wide calibration,
spatial T2 expansion, fresh Ridge training, rebuilding and validating the
converted network in both assignment passes, QVDF smoothing before
aggregation, and final cross-stage lineage QA. They are explicit here rather
than implicit side effects.

## Resource isolation

The local `nvta-taplite-workflow` is part of this repository because the public
workflow distribution contains a baseline resource payload.  Stage 11 starts
from that payload and replaces the active products with resources created by
this run:

- `link_qvdf.csv`;
- observed-link PLF overrides;
- observed speed-boundary anchors;
- observed T2 overrides;
- completed T0/T2/T3 node-pair lookups; and
- an optional explicit node-pair QVDF parameter override.

Both network conversions receive `NVTA_TAPLITE_RESOURCE_ROOT` pointing at this
isolated folder.  The baseline package resources are not edited, so experiments
cannot corrupt the repository or another run.  The native TAPlite engine is
still installed and contract-checked using the pinned setup under
`nvta-taplite-workflow/setup/`.

## Environment

Use 64-bit Windows and Python 3.11.  The bootstrap installs all four local
packages in editable mode, installs the root orchestrator, installs the pinned
TAPlite prerelease, and executes its native contract check:

```powershell
git lfs install
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

For development tools, use `-Development`.

## Stage the NVTA input example

The operational input payload is intentionally ignored by Git.  To construct a
working local example from the established package without changing it:

```powershell
python .\scripts\stage_nvta_inputs.py `
  --source-package C:\path\to\NVTA_internal\nvta-cbi-package `
  --destination .\input-data\nvta `
  --mode hardlink
```

Hard links avoid duplicating hundreds of megabytes when the source and target
are on the same volume; the script falls back to copying when necessary.  It
hashes every staged file and writes `input-data/nvta/input_manifest.json`.
Only RITIS metadata/readings, the AM/MD/PM GMNS base networks, the Cube network,
and the AM/MD/PM demand matrices are staged.

The manifest contains only relative paths, file sizes, and SHA-256 hashes, so
the complete `input-data/nvta` folder can be copied to another machine without
carrying source-machine paths. Verify a received bundle before its QA/run:

```powershell
python .\scripts\verify_input_bundle.py .\input-data\nvta
```

If a source uses different filenames, pass `--readings-name` or
`--scenario-name`.  Runtime locations are configured in
`config/nvta.json`; `config/nvta.example.json` is the documented template.

## Inspect and run

These commands work directly from a fresh clone; installing the root package is
not required for `python main.py`.

```powershell
python .\main.py --config .\config\nvta.json plan
python .\main.py --config .\config\nvta.json qa
python .\main.py --config .\config\nvta.json run
python .\main.py --config .\config\nvta.json status
```

The default worker ceiling is 10 logical cores and can be changed in the JSON
configuration.  The same ceiling is propagated to matching helpers, CBI,
coverage expansion, Ridge fitting, conversion, assignment, smoothing, corridor
measurement, and dashboard generation.  BLAS thread pools are held at one per
worker to avoid nested oversubscription.

For a bounded rerun:

```powershell
python .\main.py --config .\config\nvta.json run `
  --from-stage network-qvdf --through-stage dashboard --resume
```

`--resume` reuses a stage only when its configuration hash and every upstream
stage-manifest hash match.  Changed upstream data therefore cannot silently
reuse a stale result.  `--force` removes only the selected producer folder
inside the configured output root and rebuilds it.

## Outputs and logs

The default output root is `outputs/full-run`.  Every stage contains a
`pipeline_stage_manifest.json`; the root contains `pipeline_manifest.json` and
the input QA report.  Concise milestones are printed to the terminal.  Detailed
engine output and exact commands are retained under
`outputs/full-run/logs/<stage>/`.

The final products are:

```text
outputs/full-run/14-taplite-stage-2/assignment/     final link performance
outputs/full-run/15-corridor-measurement/           measures, data, and figures
outputs/full-run/16-integrated-dashboard/index.html dashboard entry page
```

Generated outputs and operational inputs are excluded from Git.  Code,
configuration templates, QA contracts, setup locks, tests, and empty directory
markers are versioned.  Required NumPy lookup resources and the larger bundled
Ridge fallback artifact use Git LFS; the fresh full run still regenerates and
replaces those baseline products in its isolated stage-11 resource bundle.

## Package CLIs

Each component remains independently usable:

```powershell
tmc-matching --help
cbi --help
cbi-congestion-boundaries --help
nvta-taplite --help
corridor-dashboard --help
```

Their READMEs document individual input/field adapters and producer-specific
QA.  The root CLI adds full-pipeline wiring and cross-package lineage; it does
not replace those APIs.

## Tests

```powershell
python -m pytest -q
python -m pytest -q tmc-matching\tests
python -m pytest -q cbi\tests
python -m pytest -q nvta-taplite-workflow\tests
python -m pytest -q corridor-performance-dashboard\tests
```

Large NVTA regression runs are opt-in and are not performed by the unit-test
suite.
