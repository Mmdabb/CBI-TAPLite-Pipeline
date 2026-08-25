# Producer and consumer contract

The measurement stage consumes three explicit producer roots:

1. CBI `corridors/` plus its sibling `shared/network-mapping/canonical_node_pair_tmc.csv`.
2. TMC matching products for AM, MD, PM, and the dashboard geometry product.
3. TAPlite assignment `am/`, `md/`, and `pm/` link/performance files.

The dashboard additionally consumes the measurement root, the explicit model-link map, and observed 15-minute speeds.

File and source-field aliases are resolved only at the boundary. Internally, canonical schemas remain stable so equations, metrics, and figure logic cannot silently bind to the wrong column. QA reports provide the trace from each canonical contract to its actual source.

Outputs are fixed process names rather than timestamps. A run can therefore be addressed by its input location plus producer name. Manifests, effective settings, and logs contain the reproducibility record.
