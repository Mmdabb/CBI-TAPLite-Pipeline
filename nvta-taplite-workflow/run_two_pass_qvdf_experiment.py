"""Run a reproducible two-pass QVDF/anchor TAPlite experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PERIODS = ("am", "md", "pm")
DEMAND_MODES = ("apv", "com", "hov2", "hov3", "sov", "trk")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _stage_demand_links(source: Path, destination: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for period in PERIODS:
        target_period = destination / period
        target_period.mkdir(parents=True, exist_ok=True)
        for mode in DEMAND_MODES:
            name = f"{mode}_{period}.csv"
            source_path = source / period / name
            target_path = target_period / name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if target_path.exists():
                raise FileExistsError(target_path)
            os.link(source_path, target_path)
            if _sha256(source_path) != _sha256(target_path):
                raise ValueError(f"Hard-linked demand hash mismatch: {target_path}")
            records.append(
                {
                    "period": period.upper(),
                    "mode": mode,
                    "source": str(source_path.resolve()),
                    "target": str(target_path.resolve()),
                    "sha256": _sha256(source_path),
                    "storage": "verified_same-volume_hard_link",
                }
            )
    return records


def _assignment_command(
    python: Path,
    runner: Path,
    scenario: Path,
    output: Path,
    anchors: Path,
    workers: int,
    *,
    convert: bool,
    assign: bool,
    qvdf_profile_mode: int = 1,
) -> list[str]:
    return [
        str(python),
        str(runner),
        str(scenario),
        "--iterations", "10",
        "--processors", str(workers),
        "--network-conversion", str(convert).lower(),
        "--demand-conversion", "false",
        "--dtalite-assignment", str(assign).lower(),
        "--conversion-workers", str(workers),
        "--conversion-reserve-cores", "0",
        "--demand-output-format", "csv",
        "--conversion-adaptive", "false",
        "--conversion-cache", "false",
        "--observed-speed-boundary-lookup-directory", str(anchors),
        "--qvdf-profile-mode", str(qvdf_profile_mode),
        "--qvdf-parameter-override", "false",
        "--output-dir", str(output),
        "--time-periods", *PERIODS,
        "--period-times", "0600_0900", "0900_1500", "1500_1900",
        "--kernel-source", "wheel",
    ]


def _run_pass(
    *,
    package_root: Path,
    python: Path,
    cube_source: Path,
    demand_source: Path,
    qvdf_dictionary: Path,
    anchors: Path,
    run_root: Path,
    workers: int,
    env: dict[str, str],
    qvdf_profile_mode: int = 1,
) -> dict[str, object]:
    assignment = run_root / "assignment"
    resources = run_root / "resources"
    resources.mkdir(parents=True, exist_ok=False)
    runner = package_root / "nvta-taplite-workflow" / "run_assignment.py"
    validator = package_root / "nvta-taplite-workflow" / "validate_converted_qvdf_scenario.py"
    overlay = package_root / "nvta-taplite-workflow" / "apply_node_pair_qvdf_overrides.py"
    anchor_file = anchors / "observed_link_speed_boundaries.npy"

    _run(
        _assignment_command(
            python, runner, cube_source, assignment, anchors, workers,
            convert=True, assign=False,
            qvdf_profile_mode=qvdf_profile_mode,
        ),
        cwd=package_root,
        env=env,
    )
    _run(
        [
            str(python), str(validator), str(assignment), str(anchor_file),
            str(resources / "pre-overlay-network-validation.json"),
            "--expected-profile-mode", str(qvdf_profile_mode),
        ],
        cwd=package_root,
        env=env,
    )
    _run(
        [
            str(python), str(overlay), str(assignment), str(qvdf_dictionary),
            str(resources / "pre-qvdf-override-network"),
            "--qvdf-profile-mode", str(qvdf_profile_mode),
        ],
        cwd=package_root,
        env=env,
    )
    _run(
        [
            str(python), str(validator), str(assignment), str(anchor_file),
            str(resources / "post-overlay-anchor-validation.json"),
            "--expected-profile-mode", str(qvdf_profile_mode),
        ],
        cwd=package_root,
        env=env,
    )
    demand_records = _stage_demand_links(demand_source, assignment)
    (resources / "demand_link_manifest.json").write_text(
        json.dumps(demand_records, indent=2), encoding="utf-8"
    )
    _run(
        _assignment_command(
            python, runner, assignment, assignment, anchors, workers,
            convert=False, assign=True,
            qvdf_profile_mode=qvdf_profile_mode,
        ),
        cwd=package_root,
        env=env,
    )
    _run(
        [
            str(python), str(overlay), str(assignment), str(qvdf_dictionary),
            str(resources / "unused-verify-only"), "--verify-only", "--report",
            str(resources / "post-assignment-qvdf-validation.json"),
            "--qvdf-profile-mode", str(qvdf_profile_mode),
        ],
        cwd=package_root,
        env=env,
    )
    _run(
        [
            str(python), str(validator), str(assignment), str(anchor_file),
            str(resources / "post-assignment-anchor-validation.json"),
            "--expected-profile-mode", str(qvdf_profile_mode),
        ],
        cwd=package_root,
        env=env,
    )
    return {
        "run_root": str(run_root.resolve()),
        "assignment_root": str(assignment.resolve()),
        "anchor_lookup": str(anchor_file.resolve()),
        "anchor_sha256": _sha256(anchor_file),
        "qvdf_dictionary": str(qvdf_dictionary.resolve()),
        "qvdf_sha256": _sha256(qvdf_dictionary),
        "workers": workers,
        "iterations": 10,
        "qvdf_profile_mode": qvdf_profile_mode,
        "vdf_type": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--qvdf-dictionary", type=Path, required=True)
    parser.add_argument("--baseline-anchors", type=Path, required=True)
    parser.add_argument("--cube-source", type=Path, required=True)
    parser.add_argument("--demand-source", type=Path, required=True)
    parser.add_argument("--pass1-run-id", required=True)
    parser.add_argument("--final-run-id", required=True)
    parser.add_argument("--anchor-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--qvdf-profile-mode", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--anchor-policy",
        choices=("assignment-all", "preserve-actual-virtual"),
        default="assignment-all",
        help=(
            "Build stage-two anchors entirely from stage-one assignment speeds, "
            "or retain baseline actual/virtual anchors and replace only fallback links."
        ),
    )
    parser.add_argument(
        "--baseline-anchor-audit",
        type=Path,
        help=(
            "Baseline hybrid_speed_boundary_audit.csv. Required when "
            "--anchor-policy=preserve-actual-virtual."
        ),
    )
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable).resolve()
    if not 1 <= args.workers <= 20:
        raise ValueError("workers must be between 1 and 20")
    qvdf_dictionary = args.qvdf_dictionary.resolve()
    baseline_anchors = args.baseline_anchors.resolve()
    cube_source = args.cube_source.resolve()
    demand_source = args.demand_source.resolve()
    pass1_root = package_root / "outputs" / "nvta-taplite-workflow" / args.pass1_run_id
    final_root = package_root / "outputs" / "nvta-taplite-workflow" / args.final_run_id
    anchor_output = args.anchor_output.resolve()
    baseline_anchor_audit = (
        args.baseline_anchor_audit.resolve()
        if args.baseline_anchor_audit is not None
        else None
    )
    if args.anchor_policy == "preserve-actual-virtual":
        if baseline_anchor_audit is None:
            parser.error(
                "--baseline-anchor-audit is required for "
                "--anchor-policy=preserve-actual-virtual"
            )
        if not baseline_anchor_audit.is_file():
            raise FileNotFoundError(baseline_anchor_audit)
    for path in (pass1_root, final_root, anchor_output):
        if path.exists():
            raise FileExistsError(path)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(args.workers),
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "experiment": args.experiment,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(python),
        "workers": args.workers,
        "iterations_per_pass": 10,
        "qvdf_profile_mode": args.qvdf_profile_mode,
        "vdf_type": 2,
        "anchor_policy": args.anchor_policy,
        "qvdf_dictionary": str(qvdf_dictionary),
        "qvdf_sha256": _sha256(qvdf_dictionary),
        "baseline_anchors": str(baseline_anchors),
        "baseline_anchor_sha256": _sha256(
            baseline_anchors / "observed_link_speed_boundaries.npy"
        ),
        "baseline_anchor_audit": (
            str(baseline_anchor_audit) if baseline_anchor_audit is not None else None
        ),
        "baseline_anchor_audit_sha256": (
            _sha256(baseline_anchor_audit)
            if baseline_anchor_audit is not None
            else None
        ),
        "passes": {},
    }
    pass1_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = pass1_root.parent / f"{args.experiment}-two-pass-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        manifest["passes"]["pass1"] = _run_pass(
            package_root=package_root,
            python=python,
            cube_source=cube_source,
            demand_source=demand_source,
            qvdf_dictionary=qvdf_dictionary,
            anchors=baseline_anchors,
            run_root=pass1_root,
            workers=args.workers,
            env=env,
            qvdf_profile_mode=args.qvdf_profile_mode,
        )
        if args.anchor_policy == "assignment-all":
            generator = (
                package_root
                / "nvta-taplite-workflow"
                / "generate_assignment_speed_boundaries.py"
            )
            generator_command = [
                str(python),
                str(generator),
                str(pass1_root / "assignment"),
                str(anchor_output),
            ]
        else:
            generator = (
                package_root
                / "nvta-taplite-workflow"
                / "generate_treatment_preserving_speed_boundaries.py"
            )
            generator_command = [
                str(python),
                str(generator),
                "--assignment-root", str(pass1_root / "assignment"),
                "--baseline-anchors", str(baseline_anchors),
                "--baseline-audit", str(baseline_anchor_audit),
                "--output-dir", str(anchor_output),
            ]
        _run(generator_command, cwd=package_root, env=env)
        manifest["derived_anchors"] = {
            "directory": str(anchor_output),
            "lookup": str(anchor_output / "observed_link_speed_boundaries.npy"),
            "sha256": _sha256(anchor_output / "observed_link_speed_boundaries.npy"),
            "source_assignment": str((pass1_root / "assignment").resolve()),
            "policy": args.anchor_policy,
        }
        manifest["passes"]["final"] = _run_pass(
            package_root=package_root,
            python=python,
            cube_source=cube_source,
            demand_source=demand_source,
            qvdf_dictionary=qvdf_dictionary,
            anchors=anchor_output,
            run_root=final_root,
            workers=args.workers,
            env=env,
            qvdf_profile_mode=args.qvdf_profile_mode,
        )
        manifest["status"] = "PASS"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        manifest["status"] = "FAIL"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
