# Stable node-pair QVDF parameter overrides

`qvdf_node_pair_overrides.npy` is the calibrated dictionary promoted from the
`auto-calibration_v2` stable-OD calibration. It contains 49,336 directed
`(from_node_id, to_node_id)` keys and period-specific PLF, QDF, N, S, CP, CD,
alpha, and beta values for AM, MD, and PM. Its SHA-256 is
`CFC0913E508D429D9DE3A9F63B8545AB1639F4225B75C6DF6085C55A52AF313F`.

The workflow first completes its normal network mapping (including the stable
network QVDF, observed PLF, observed congestion timing, and speed anchors),
then overwrites these eight QVDF fields by directed node pair. The overlay
requires complete coverage, retains `vdf_type=2` and the requested
`qvdf_profile_mode`, backs up every pre-overlay `link.csv`, and writes an
auditable manifest before assignment begins.

`source_manifest.json` records the stable-OD calibration source, complete
49,336-pair coverage, guardrail status, objective values, final relative gaps,
and per-period parameter provenance.
