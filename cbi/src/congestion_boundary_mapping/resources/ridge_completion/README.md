# Bundled Ridge completion resources

These resources make `congestion_boundary_mapping` runnable without the
excluded legacy `t2/` code or `outputs/t2/` runs.

Runtime inputs:

- `ml_run/experimental_network_boundaries.csv`: frozen network-wide
  `ridge_core` boundary predictions used by the mapper;
- `ml_run/metrics/out_of_fold_predictions.csv`: only the
  `aggregate_all_days` + `corridor_held_out` + `ridge_core` validation rows
  required by the completion report;
- `comparison_run/outputs/validation_benchmark_detail.csv`: spatial and
  class validation comparison used by that report;
- `spatial_run/outputs/expanded_link_t2.csv`: spatial T2 assignments used
  before Ridge completion;
- `spatial_run/input-snapshot/tmc_period_representatives.csv`: direct
  episode boundaries used to restore T0/T3 for spatial records when needed.

The three small Joblib files under `ml_run/models/` are the fitted Ridge
estimators retained for model lineage. The current mapping stage applies the
already generated network-wide prediction table; it does not retrain or run
the Joblib estimators. Explicit `--ml-run-dir`, `--comparison-run-dir`, and
`--spatial-output` overrides remain available when regenerated artifacts are
provided.

Source runs and SHA-256 hashes are recorded in `resource_manifest.json`.
