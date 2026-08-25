# QVDF speed smoother subpackage

This subpackage vendors the validated computational core from
`reference-tools/QVDF-Speed-Smoother-1.0.0`. The assignment runner calls it
after every requested TAPLite period finishes and before any downstream
link-performance aggregation.

The smoother reconstructs each link across adjacent periods, enforces the
five-minute change, rolling-change, and acceleration limits, validates every
serialized profile, and atomically replaces only the `spd_mph_*` columns.
`speed_mph`, volume, demand, D/C, timing, and all other assignment outputs are
retained. By default it preserves each original period file as
`link_performance.pre-qvdf-<timestamp>.csv` and writes
`qvdf_batch_report.json` at the assignment root.

The workflow flags are `--qvdf-smoothing`,
`--qvdf-smoothing-workers`, and `--qvdf-smoothing-backup`.
