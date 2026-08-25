# NVTA Conversion Optimization Report

Date: July 24, 2026

## Scope and safety

All implementation and testing was performed in the isolated
`nvta-taplite-workflow` package. Regional profiling used
only `updated_nvta_dtalite_package/nvta_2025_base_network`, with assignment
disabled. Nothing under `DTALite_Run_07162026` was changed or used as a test
output.

The original `workflow` folder remains unchanged.

## Implemented changes

### Bounded conversion scheduling

- One flat bounded process pool per conversion stage.
- Network tasks are `period x link chunk`.
- Demand tasks are `period x mode x matrix-row chunk`.
- No nested pools or multiplication of live process counts.
- Physical-core, reserved-core, task-count, and minimum-useful-work limits.
- Optional CPU-load adaptation and automatic serial fallback.
- Conversion workers remain separate from assignment OpenMP processors.

### Faster network conversion

- Link fields are materialized as NumPy arrays once instead of repeatedly using
  pandas scalar indexing inside the link loop.
- Source Cube `ID` values are preserved rather than inferred from row position.
- Node duplicate checks use a set.
- Node creation reads only required topology arrays and does not copy unused
  link attributes to every node.
- District lookup uses vectorized `Series.map` rather than row-wise
  `DataFrame.apply`.
- `node.csv` is serialized once and copied to period folders.

### Prepared-network cache

- The reprojected network, node data, and node CSV template are cached.
- The key is a SHA-256 content fingerprint of all shapefile components plus the
  target CRS.
- Cache writes are atomic and the manifest is written last.
- A changed `.shp`, `.dbf`, `.shx`, `.prj`, or target CRS invalidates the
  cache.
- Each period gets an independent node file because optional district mapping
  may modify it.

The cache is enabled by default for normal runs and disabled by default in the
profiler so fixed-worker comparisons remain fair.

### Faster demand conversion

- Each worker reuses a read-only OMX handle instead of reopening the period OMX
  file for every modal task.
- Sparse origin, destination, and volume arrays are calculated once per chunk.
- Output can be `csv`, `binary`, or `both`.
- Binary parts are merged deterministically, so serial and parallel DTAB files
  are byte-identical.

### Native DTAB demand path

The workflow now writes the versioned sparse DTAB format already supported by
the native kernel:

- Header: magic `DTAB`, version 1, and signed 64-bit record count.
- Record: signed 32-bit origin, signed 32-bit destination, and float64 volume.
- Explicit little-endian encoding.
- Exact-size validation rejects truncated or trailing data before assignment.
- `settings.csv` selects the binary reader with `demand_format=1`.

The current TAPlite kernel maps external node, zone, and link IDs internally.
The workflow therefore sends source CSV or DTAB demand records directly to the
prepared period folder without producing a second compact-ID copy.

## Windows correctness results

- 18 unit/regression tests passed.
- Serial and parallel CSV demand files are byte-identical.
- Serial and parallel DTAB demand files are byte-identical.
- Cache hit and source-content invalidation tests passed.
- Binary preflight with source IDs and direct period-folder routing tests passed.
- The one-, two-, and four-worker NVTA AM CSV runs produced identical hashes
  for all 10 profiled CSV artifacts.
- The one-, two-, four-, and eight-worker NVTA AM binary outputs matched the
  one-worker baseline.
- Cold-cache and warm-cache AM folders matched across all 12 files, including
  readiness artifacts.
- The four-period AM/MD/PM/NT conversion completed with return code 0.

### Native kernel CSV versus DTAB parity

A separate small FFX Windows integration fixture ran two native assignment
iterations with one processor using both CSV and DTAB demand:

- Both native calls returned 0.
- Both reached a 0.802220% gap.
- Both reported total VMT 219,465.8 and VHT 5,793.2.
- Seven assignment output files matched by SHA-256, including link, OD, and
  system performance.
- Native setup time was 6.38 seconds for CSV and 6.26 seconds for DTAB on this
  small case.

This validates the complete writer, settings, native reader, and output-parity
path on Windows. Large NVTA assignment itself was intentionally not started.

## NVTA AM performance

All results below use `nvta_2025_base_network`, assignment disabled, an
otherwise idle workstation, fixed worker counts, one reserved physical core,
and cache disabled unless stated otherwise.

### CSV demand

| Workers | Total | Network | Demand | Peak RSS |
|---:|---:|---:|---:|---:|
| 1 | 84.48 s | 28.83 s | 51.91 s | 732 MB |
| 2 | 62.41 s | 27.05 s | 31.96 s | 1,115 MB |
| 4 | 45.73 s | 24.46 s | 17.73 s | 1,312 MB |

Four workers reduced the current CSV workflow by 45.9% relative to one worker.
The array-backed network implementation also removed the former dominant
pandas indexing cost: the earlier contention-affected one-worker AM run took
581.21 seconds in the network stage, versus 28.83 seconds now. That
cross-session comparison includes different machine load, so it is evidence of
the scale of improvement rather than a controlled speedup ratio.

### DTAB binary demand

| Requested workers | Total | Network | Demand | Peak RSS |
|---:|---:|---:|---:|---:|
| 1 | 35.93 s | 27.97 s | 4.49 s | 737 MB |
| 2 | 33.46 s | 25.81 s | 4.10 s | 893 MB |
| 4 | 28.17 s | 21.82 s | 3.12 s | 992 MB |
| 8 | 29.82 s | 23.33 s | 3.03 s | 1,280 MB |

Binary output reduced one-worker demand conversion by 91.4% and total AM
conversion by 57.5% relative to CSV. Four workers were the fastest tested
setting. At a request of eight workers, the network scheduler deliberately
used only six because the 49,329-link AM network could not supply eight
minimum-size chunks; the additional demand workers did not offset their
overhead.

### Prepared-network cache

Both cache tests used one worker and DTAB output.

| Cache state | Total | Network | Demand |
|---|---:|---:|---:|
| Cold miss | 38.20 s | 29.72 s | 4.92 s |
| Warm hit | 23.01 s | 14.89 s | 4.45 s |

The warm hit reduced network-stage time by 49.9% and total time by 39.8%
relative to creating the cache. The cold run is slightly slower than a
no-cache run because it also serializes the reusable payload.

## Controlled full CSV comparison with the unchanged workflow

A later controlled comparison used all four NVTA periods and CSV demand only:

- Source: `nvta_2025_base_network`.
- Assignment disabled in both runs.
- Same Python 3.11 environment and workstation.
- Legacy baseline: unchanged `workflow`, serial conversion, no cache.
- Optimized: `nvta-taplite-workflow`, four workers, verified warm cache.
- Both runs wrote to isolated benchmark directories.
- Legacy source files were staged as 16 hard links with zero file copies; the
  source scenario and `DTALite_Run_07162026` were not written.

| Measurement | Legacy serial CSV | Optimized 4-worker warm-cache CSV | Change |
|---|---:|---:|---:|
| Total wall time | 1,800.32 s (30.01 min) | 94.06 s (1.57 min) | 19.14x faster; 94.78% lower |
| Network stage | about 1,568.67 s | 24.51 s | about 64.0x faster |
| Demand stage | about 231.08 s | 65.94 s | about 3.50x faster |
| Peak workflow RSS | 1,017 MB | 2,596 MB | +1,579 MB |
| Return code | 0 | 0 | Same |

The legacy workflow does not emit per-stage timing metadata. Its network and
demand values above are reconstructed from the profiler start time and final
network/demand output timestamps; they sum to within one second of measured
wall time.

### CSV compatibility

- All 32 `node.csv`, `mode_type.csv`, and modal demand CSV files matched
  byte-for-byte across the four periods.
- All four `link.csv` files had identical byte counts, row counts, column sets,
  link order, and exact string values when aligned by column name.
- `link.csv` hashes differ only because optimized conversion emits the legacy
  passthrough DBF columns in deterministic sorted order. Required GMNS/kernel
  columns remain first.
- The native kernel reads link values by header field name, so passthrough
  column order has no effect.
- Optimized `settings.csv` preserves every legacy value and adds
  `demand_format=0`, which explicitly selects CSV demand.

Detailed results:

- Optimized:
  `performance/conversion_parallel/20260724_csv_warm_4_full/profile_report.json`
- Legacy:
  `performance/conversion_parallel/20260724_csv_legacy_1_full_rerun/profile_report.json`

## Four-period validation

The AM/MD/PM/NT run used four conversion workers, warm prepared-network cache,
DTAB demand, and assignment disabled.

| Measurement | Result |
|---|---:|
| Total wall time | 40.52 s |
| Network stage | 24.46 s |
| Demand stage | 11.02 s |
| Network tasks | 8 |
| Demand tasks | 24 |
| Peak workflow RSS | 1,741 MB |
| Minimum system-available memory | 174,660 MB |
| Return code | 0 |

Period link counts were preserved:

- AM: 49,329
- MD: 49,336
- PM: 49,329
- NT: 49,336

## Recommended settings

For normal conversion and assignment:

```powershell
python run_assignment.py "C:\path\to\nvta_2025_base_network" `
  --conversion-workers 4 `
  --conversion-reserve-cores 1 `
  --conversion-adaptive true `
  --conversion-cache true `
  --demand-output-format csv
```

These are now the parallel package defaults, so they do not need to be supplied
on a normal command. Use `--conversion-workers 0` to opt into automatic worker
selection on a shared or busy machine. Keep `--processors` for OpenMP
assignment independent of conversion worker selection.

For this workstation and current NVTA input, four fixed workers were the best
measured conversion setting. Automatic mode remains safer for shared or busy
machines because it can fall back to serial execution.

## Detailed evidence

- CSV worker sweep:
  `performance/conversion_parallel/20260724_optimized2_am_csv/profile_report.json`
- DTAB 1/2/4-worker sweep:
  `performance/conversion_parallel/20260724_optimized2_am_binary/profile_report.json`
- DTAB eight-worker result:
  `performance/conversion_parallel/20260724_optimized2_am_binary_8/profile_report.json`
- Cold cache:
  `performance/conversion_parallel/20260724_optimized2_cache_cold/profile_report.json`
- Warm cache:
  `performance/conversion_parallel/20260724_optimized2_cache_warm/profile_report.json`
- Full four-period validation:
  `performance/conversion_parallel/20260724_optimized2_full_binary/profile_report.json`
- Native FFX parity fixture:
  `nvta-taplite-workflow/.integration-artifacts/ffx_binary_parity`

## Remaining improvements

The following were not folded into this change because they alter the
workflow/kernel data contract or require a separate tuning design:

1. **Direct columnar link emission (workflow):** array access is now fast, but
   each link still becomes a Python `Link` object. Direct record generation
   could remove more allocation overhead.
2. **Minimal link export profile (workflow and kernel):** agree on the exact
   required kernel columns, then make legacy DBF passthrough columns optional.
3. **Versioned binary network input (workflow and kernel):** likely the next
   large format improvement, but it needs compatibility, endianness, schema,
   and output-parity tests similar to DTAB.
4. **Learned stage-specific worker caps (workflow):** record separate read,
   transform, and write timings and retain safe historical recommendations by
   machine/input fingerprint.
5. **Demand cache (workflow):** fingerprint OMX matrices and reuse verified
   DTAB outputs when demand sources and zone mapping have not changed.

The missing `TPBTAZ3722_TPBMod_JUR.csv` warning remains an input-package issue.
District assignment is skipped when that optional lookup is absent; conversion
and native input validation are otherwise successful.
