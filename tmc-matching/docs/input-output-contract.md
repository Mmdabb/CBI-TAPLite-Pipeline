# Input and output contract

The public contract is the CLI, not a repository-relative data path. Canonical semantic fields are mapped at the input boundary and are then stable inside the algorithm.

- TMC: `tmc`, `road`, `direction`, `road_order`, endpoint coordinates and names.
- Link: `link_id`, `from_node_id`, `to_node_id`, geometry, length, lanes, facility/use/status fields.
- Node: `node_id`, `x_coord`, `y_coord`.

`examples/column-map.json` maps a canonical name on the left to a source name on the right. The QA report records the resolved mapping and every checked file.

Output directories never contain a clock timestamp. Reproducibility comes from `run_manifest.json`, input paths, configuration, and logs rather than directory naming.
