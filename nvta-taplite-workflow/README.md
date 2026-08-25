# NVTA TAPLite assignment workflow

This package converts the NVTA Cube network and OMX demand matrices, adds
period-specific congestion boundaries, runs TAPLite, and publishes the AM,
MD, and PM link-performance inputs used by the integrated CBI dashboard.

The package does not discover a local TAPLite source checkout. Environment
setup installs one pinned PyPI prerelease; assignment runs never select an
alternate executable or package at runtime.

When this package is used by the CBI–TAPlite monorepo, the environment variable
`NVTA_TAPLITE_RESOURCE_ROOT` identifies an isolated resource bundle generated
by the current full run.  All active QVDF, PLF, speed-anchor, observed-T2, and
completed-boundary loaders resolve through that root.  If the variable is not
set, the packaged resource directory remains the standalone default.  This
keeps the PyPI/native engine contract while allowing newly calibrated NVTA
resources to be used without editing the installed package.

## Authoritative engine

The only supported engine is the pinned Python 3.11 Windows prerelease from
PyPI:

```text
taplite4mpo-pre-release==0.4.0rc1
taplite4mpo_pre_release-0.4.0rc1-cp311-cp311-win_amd64.whl
SHA256 4ED012D7DB6446D13BDA7345523061693F540BA17087968A61BFFC112FAAFC09
```

The complete artifact and native-binary lock is recorded in
`setup/taplite_pypi_lock.json`. Every assignment verifies the distribution,
version, imported module paths, and native-module SHA256 before calling the
kernel. A conflicting `taplite4mpo` distribution is a hard failure.

This upstream build supports optional `qvdf_start_speed_mph` and
`qvdf_end_speed_mph` speed anchors plus `qvdf_profile_mode`. The observed-T2
profile mode activates QVDF reconstruction only when a valid observed T2 is
available. With mode 2 and blank T2, including on freeway link types, the
kernel reports `flat_missing_observation`, `P=0`, and `t2=0`.
Eligible observed boundary anchors are joined to the QVDF profile with the
upstream transition logic. Low anchors connect directly to the observed-T2
speed with a cubic smoothstep; other eligible anchors use the monotone-Hermite
splice. Zero assigned volume is a hard guard that keeps the profile flat and
suppresses observed-boundary fallback. When QVDF generation is skipped for a
positive-volume profile, the kernel uses its smooth observed-boundary fallback
whenever either period-boundary speed is available, retaining the assigned
period-average speed on a missing side.

`setup/install_pypi_prerelease.py` removes conflicting distributions, installs
the exact pinned PyPI release, and verifies the native binary hash.
`setup/verify_taplite_contract.py` then executes a positive-volume freeway
regression case to prove the strict mode-2 observed-T2 gate.

## Environment setup

Use 64-bit Windows and Python 3.11. From `nvta-taplite-workflow`:

```powershell
.\setup_environment.bat
conda activate dtalite_pipeline
```

The setup script creates or updates the pinned environment, installs the
verified PyPI prerelease, runs the strict mode-2 contract scenario, and then
runs the remaining setup checks. Internet access is required when the pinned
artifact or other dependencies are not already cached.

## Full NVTA run

From the `nvta-cbi-package` root:

```powershell
python nvta-taplite-workflow\run_assignment.py `
  input-data\nvta-taplite-workflow\dtalite-run-07162026 `
  --kernel-source pypi `
  --iterations 10 `
  --processors 20 `
  --route-output 0 `
  --vehicle-output 0 `
  --unit-system metric `
  --vdf-type qvdf `
  --network-conversion true `
  --demand-conversion true `
  --dtalite-assignment true `
  --conversion-workers 0 `
  --conversion-reserve-cores 1 `
  --conversion-adaptive true `
  --conversion-cache true `
  --demand-output-format csv `
  --time-periods am md pm `
  --period-times 0600_0900 0900_1500 1500_1900 `
  --output-dir outputs\nvta-taplite-workflow\<run-id>\assignment
```

Relative output and cache paths are resolved from the current working
directory. Use absolute paths when invoking the command from automation.

`--processors` controls TAPLite OpenMP threads. `--conversion-workers 0`
selects an adaptive worker count based on current free cores. Both values are
bounded by the package-wide ceiling of 20 logical cores. Conversion uses flat
process pools rather than nested pools.

This NVTA workflow enforces:

- `--kernel-source pypi`;
- `--vdf-type qvdf`;
- `--route-output 0`; and
- `--vehicle-output 0`.

Those settings prevent large route and vehicle outputs and ensure the
link-performance results use the required QVDF input.

### TAPlite unit contract

The current TAPlite4MPO GMNS schema uses a fixed, mixed set of units. The
converter always writes both the generic metric fields and the explicit
QVDF/imperial overrides; it does not treat the entire file as either metric
or imperial.

| field or resource | unit written by this workflow | kernel use |
|---|---:|---|
| Cube `DISTANCE` | mile | source network attribute |
| `length` | meter | generic GMNS input, converted to miles internally |
| `vdf_length_mi` | mile | unambiguous override used by the kernel |
| Cube speed-class value | mph | source lookup attribute |
| `free_speed` | km/h | generic GMNS input, converted to mph internally |
| `vdf_free_speed_mph` | mph | unambiguous override used by the kernel |
| `cutoff_speed` | mph | QVDF speed at capacity |
| `qvdf_start_speed_mph`, `qvdf_end_speed_mph` | mph | observed profile anchors |
| `vdf_fftt` | minute | free-flow travel time |
| `capacity` | vehicle/hour/lane | per-lane capacity |
| `t0_hour`, `t2_hour`, `t3_hour` | decimal hour | observed congestion times |
| `link_performance.csv` speed/length | mph/mile | kernel output and downstream analysis |

The legacy `--unit-system` option remains accepted so older commands continue
to run, but it no longer changes `link.csv`; the fixed schema above is always
enforced. `link_qvdf.csv` contains QVDF shape/calibration parameters rather
than link speed or length. Its observed-link PLF override is dimensionless,
while the separate observed speed-boundary lookup is explicitly mph and the
T2/boundary lookups are decimal hours.

### Authoritative QVDF mapping

Network conversion reads every QVDF parameter from
`src/dtalite4cube/resources/link_qvdf.csv`. A link uses the row whose
`vdf_code` matches its calculated `link_type`; when no exact row exists, it
uses the CSV row with `vdf_code=all`. Missing requested-period parameters are
reported as errors instead of being silently replaced by dictionary defaults.

The embedded Python QVDF dictionaries remain in the source tree only as legacy
references and are not part of the active conversion path.

For the directed best-match GMNS links of observed TMCs, conversion then
overrides only `vdf_plf` from the memory-mapped
`resources/observed_link_plf_lookup/observed_link_plf_overrides.npy` table.
The three period values follow `PLF = 1/(QDF*H)` when average-weekday
congestion was accepted and use neutral `PLF = 1` when it was not. Every
other QVDF parameter still comes from `link_qvdf.csv`.

Conversion also memory-maps
`resources/observed_link_speed_boundary_lookup/observed_link_speed_boundaries.npy`.
It is keyed by the packed directed `(from_node_id, to_node_id)` pair and stores
the speed anchor at each AM, MD, and PM start and end minute. Canonical TMC-link
winners use observed weekday-average speeds; non-canonical links use stable
assignment `speed_mph` anchors. Those values become `qvdf_start_speed_mph` and
`qvdf_end_speed_mph` in the period `link.csv`. A missing value is retained only
when a period link is absent, such as a closed reversible direction. For an
active QVDF profile, the anchors replace the two free-speed edge anchors; for
an inactive queue model, they smoothly connect across the reported period.
Neither path changes assignment costs.

`generate_hybrid_speed_boundaries.py` rebuilds this complete lookup without
running conversion or assignment. It retains existing post-QC CBI anchors,
fills the remaining canonical winners from direct regional weekday averages,
and fills the rest of the network from the stable assignment.

Conversion also memory-maps
`resources/observed_link_t2_lookup/observed_link_t2.npy`. When the matched TMC
has a representative accepted weekday-average congestion episode for the
period, the complete observed T0/T2/T3 triplet overrides the general
congestion-boundary assignment. Observed T0 and T3 may extend outside the
assignment period because TAPLite uses them only to preserve episode
asymmetry; the emitted profile is clipped to the period. A missing episode
clears all three fields, protecting the matched no-congestion observation
from spatial or ML completion.

## Time periods

| period | half-open range |
|---|---|
| AM | 06:00-09:00 |
| MD | 09:00-15:00 |
| PM | 15:00-19:00 |

The same definitions are used by CBI episode classification and the
integrated dashboard.

## Congestion-boundary assignment

During Cube-to-GMNS network conversion, each period link receives
`t0_hour`, `t2_hour`, `t3_hour`, and `qvdf_profile_mode`. The three time values
are copied from the immutable node-pair dictionaries under:

```text
src/dtalite4cube/resources/congestion_t_node_pair_lookup/
```

The three arrays are exported from the latest completed NVTA congestion-
boundary mapping run and are packaged here so a third party needs no external
workspace. Best-match observed TMC links with no accepted average-weekday
congestion are present in the lookup with all three boundary values set to
NaN; conversion deliberately writes blank `t0_hour`, `t2_hour`, and `t3_hour`
for those rows. The three time fields are preserved through the final
`link_performance.csv`. `qvdf_profile_mode` is a kernel input that accepts only
`0`, `1`, or `2`; the current conversion policy writes integer `2` for every
link while assigning the congestion-boundary fields.

## Expected outputs

For each of `am`, `md`, and `pm`, the assignment root contains:

```text
<period>/
|-- node.csv
|-- link.csv
|-- settings.csv
|-- mode_type.csv
|-- link_performance.csv
|-- od_performance.csv
`-- logs and summary files
```

The converted `link.csv` retains `t0_hour`, `t2_hour`, `t3_hour`, and
`qvdf_profile_mode`. The
TAPLite `link_performance.csv` publishes the corresponding engine fields
`t0`, `t2`, and `t3`. No `route_assignment.csv` or vehicle trajectory output
should exist.

`CONVERSION_PROFILE.json` records worker planning, cache fingerprints, task
counts, output counts, and elapsed times. Each period also has an assignment
run card and logs that record the TAPLite version and settings.

## QVDF profile smoothing before aggregation

The workflow vendors `QVDF-Speed-Smoother-1.0.0` as
`src/qvdf_speed_smoother`. After the final requested period assignment
finishes, the runner reconstructs and validates the adjacent QVDF profiles,
then atomically replaces only the `spd_mph_*` columns before any combined or
daily link-performance product is created. The original period files are
retained by default as `link_performance.pre-qvdf-<timestamp>.csv`, and the
run-level `qvdf_batch_report.json` records constraints, workers, hashes,
write-back validation, and backup paths.

Smoothing is enabled by default. Use `--qvdf-smoothing false` only for an
explicit diagnostic run. `--qvdf-smoothing-workers 0` reuses the assignment
processor count, subject to the workflow's 20-core ceiling.

## Stable post-mapping QVDF overrides

The stable mode-2 hybrid configuration also packages
`src/dtalite4cube/resources/qvdf_parameter_override_lookup/`
with a 49,336-node-pair calibrated QVDF dictionary. Network conversion first
runs every normal CBI/observed mapping stage, then overwrites period PLF, QDF,
N, S, CP, CD, alpha, and beta values by directed node pair. The overlay
requires complete coverage and verifies that `vdf_type=2` and the requested
`qvdf_profile_mode` are unchanged. Each run retains its pre-overlay network
under `qvdf_override_audit/pre-overlay-network` and writes
`qvdf_override_manifest.json` before assignment.

The packaged override is enabled by default. Use
`--qvdf-parameter-override false` for a deliberate base-mapping diagnostic,
or `--qvdf-override-dictionary <file>` for an isolated parameter experiment.

## Conversion-only and prepared-input modes

Convert without running the assignment kernel:

```powershell
python nvta-taplite-workflow\run_assignment.py <scenario> `
  --network-conversion true `
  --demand-conversion true `
  --dtalite-assignment false `
  --time-periods am md pm `
  --period-times 0600_0900 0900_1500 1500_1900 `
  --output-dir <output>
```

Run already prepared period folders:

```powershell
python nvta-taplite-workflow\run_assignment.py <prepared-scenario> `
  --network-conversion false `
  --demand-conversion false `
  --dtalite-assignment true `
  --time-periods am md pm `
  --period-times 0600_0900 0900_1500 1500_1900 `
  --output-dir <output>
```

Add `--dry-run` to stage and validate inputs without starting TAPLite.

## Parallel conversion and cache

Network tasks are `periods x chunks`; demand tasks are
`periods x modes x row chunks`. One bounded pool executes each stage.
General staging and export copies use the same ceiling. Small workloads
automatically run serially.

Assignment runs directly in each prepared period folder. TAPlite performs its
own internal ID renumbering and writes outputs using the source GMNS IDs. The
workflow does not create an `_internal` folder, rewrite IDs, copy inputs for ID
conversion, create `id_mapping.csv`, or back-map assignment outputs.

The prepared-network cache is content fingerprinted from all shapefile
components and the target CRS. Use `--conversion-cache false` for a cold run,
or `--conversion-cache-dir <path>` to place it in the output area.

For detailed command help:

```powershell
python nvta-taplite-workflow\run_assignment.py --help
python nvta-taplite-workflow\run_postprocessing.py --help
```

## Treatment-aware resource staging

`build_treatment_resource_overlay.py` merges disjoint actual and virtual PLF
lookups while retaining the actual-only daily `link_qvdf.csv` as the official
network calibration. After a first-pass assignment,
`install_final_treatment_resources.py` builds the final resources with this
audited hierarchy:

1. actual observed speed anchors;
2. virtual observed speed anchors on disjoint node pairs; and
3. first-pass assignment `speed_mph` anchors for every remaining network link.

Observed values override the assignment baseline field by field, so a partial
observed boundary keeps its usable observations and fills only missing edges.
The installer combines actual and virtual observed-T2 lookups, copies the
completed network congestion lookup, writes source hashes, and makes a full
pre-install resource backup inside the new staging directory. It refuses to
overwrite an existing staging run. Use `audit_tmc_qvdf_inputs.py` on the fresh
converted AM/MD/PM folders before assignment; it validates all speed anchors
and applies no-episode/observed-triplet rules only to the direct TMC mapping.
