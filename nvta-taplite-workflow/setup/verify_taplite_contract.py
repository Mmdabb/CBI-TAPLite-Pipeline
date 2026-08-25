from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT))

from src.dtalite4cube.taplite_runtime import verify_taplite_runtime  # noqa: E402


def _write_contract_scenario(root: Path) -> None:
    (root / "node.csv").write_text(
        "node_id,zone_id,x_coord,y_coord\n1,1,0,0\n2,2,1,0\n",
        encoding="utf-8",
    )
    (root / "link.csv").write_text(
        "link_id,from_node_id,to_node_id,link_type,lanes,capacity,free_speed,"
        "vdf_free_speed_mph,length,vdf_length_mi,vdf_fftt,vdf_type,vdf_alpha,"
        "vdf_beta,vdf_plf,cutoff_speed,vdf_cp,vdf_cd,vdf_n,vdf_s,t0_hour,"
        "t2_hour,t3_hour,qvdf_profile_mode,allowed_use\n"
        "36089,1,2,301,3,2000,111.044736,69,1609,1,0.869565,2,0.15,4,1,"
        "51.75,0.5,3.487113,1,0.356351,,,,2,all\n",
        encoding="utf-8",
    )
    (root / "demand.csv").write_text(
        "o_zone_id,d_zone_id,volume\n1,2,3000\n",
        encoding="utf-8",
    )
    (root / "settings.csv").write_text(
        "number_of_iterations,number_of_processors,demand_period_starting_hours,"
        "demand_period_ending_hours,first_through_node_id,base_demand_mode,"
        "route_output,vehicle_output,log_file,odme_mode,odme_vmt,link_output,"
        "accessibility_output\n1,1,6,9,-1,0,0,0,0,0,0,2,0\n",
        encoding="utf-8",
    )


def main() -> int:
    identity = verify_taplite_runtime()
    import pytaplite

    temporary_root = Path(
        tempfile.mkdtemp(prefix="taplite-mode2-contract-", dir=WORKFLOW_ROOT / "setup")
    )
    try:
        _write_contract_scenario(temporary_root)
        result = pytaplite.assign(str(temporary_root), prefer_inproc=True)
        if result.returncode != 0:
            raise RuntimeError(f"TAPLite contract scenario failed: {result.log}")
        with (temporary_root / "link_performance.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            row = next(csv.DictReader(stream))
        expected = {
            "qvdf_profile_status": "flat_missing_observation",
            "P": 0.0,
            "t2": 0.0,
        }
        if row.get("qvdf_profile_status") != expected["qvdf_profile_status"]:
            raise RuntimeError(
                "Strict mode-2 contract failed: expected flat_missing_observation, "
                f"got {row.get('qvdf_profile_status')!r}"
            )
        for field in ("P", "t2"):
            if float(row[field]) != expected[field]:
                raise RuntimeError(
                    f"Strict mode-2 contract failed: expected {field}=0, got {row[field]}"
                )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    print(
        "Strict mode-2 contract PASS: freeway link 301 with blank observed t2 "
        "emitted P=0, t2=0, flat_missing_observation using "
        f"{identity['distribution']}=={identity['installed_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
