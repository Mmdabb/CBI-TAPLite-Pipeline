"""Run one isolated QVDF/profile-mode-1 TAPlite experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_two_pass_qvdf_experiment import _run_pass, _sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--qvdf-dictionary", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--cube-source", type=Path, required=True)
    parser.add_argument("--demand-source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable).resolve()
    if not 1 <= args.workers <= 20:
        raise ValueError("workers must be between 1 and 20")

    qvdf_dictionary = args.qvdf_dictionary.resolve()
    anchors = args.anchors.resolve()
    cube_source = args.cube_source.resolve()
    demand_source = args.demand_source.resolve()
    anchor_file = anchors / "observed_link_speed_boundaries.npy"
    run_root = package_root / "outputs" / "nvta-taplite-workflow" / args.run_id
    if run_root.exists():
        raise FileExistsError(run_root)
    for required in (qvdf_dictionary, anchor_file, cube_source, demand_source):
        if not required.exists():
            raise FileNotFoundError(required)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(args.workers),
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    manifest = {
        "status": "RUNNING",
        "experiment": args.experiment,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(python),
        "workers": args.workers,
        "iterations": 10,
        "qvdf_profile_mode": 1,
        "vdf_type": 2,
        "qvdf_dictionary": str(qvdf_dictionary),
        "qvdf_sha256": _sha256(qvdf_dictionary),
        "anchor_lookup": str(anchor_file),
        "anchor_sha256": _sha256(anchor_file),
    }
    manifest_path = (
        package_root / "outputs" / "nvta-taplite-workflow"
        / f"{args.experiment}-single-pass-manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        manifest["run"] = _run_pass(
            package_root=package_root,
            python=python,
            cube_source=cube_source,
            demand_source=demand_source,
            qvdf_dictionary=qvdf_dictionary,
            anchors=anchors,
            run_root=run_root,
            workers=args.workers,
            env=env,
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
