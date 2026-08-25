from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import PipelineConfig
from .stages import STAGES, Stage, sha256, stage_paths


LOGGER = logging.getLogger("cbi_taplite_pipeline")


def _config_sha256(config: PipelineConfig) -> str:
    return sha256(config.config_path)


def _manifest(output: Path) -> Path:
    return output / "pipeline_stage_manifest.json"


def _upstream_signature(config: PipelineConfig, index: int) -> dict[str, str]:
    paths = stage_paths(config)
    signatures: dict[str, str] = {}
    for upstream in STAGES[:index]:
        manifest = _manifest(paths[upstream.output_key])
        if manifest.is_file():
            signatures[upstream.key] = sha256(manifest)
    return signatures


def is_complete(config: PipelineConfig, index: int, stage: Stage) -> tuple[bool, str]:
    output = stage_paths(config)[stage.output_key]
    manifest_path = _manifest(output)
    if not manifest_path.is_file():
        return False, "missing stage manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable stage manifest: {exc}"
    if manifest.get("status") != "PASS":
        return False, f"manifest status is {manifest.get('status')!r}"
    if manifest.get("config_sha256") != _config_sha256(config):
        return False, "configuration changed"
    if manifest.get("upstream_manifests") != _upstream_signature(config, index):
        return False, "upstream lineage changed"
    missing = [relative for relative in stage.required_outputs if not (output / relative).exists()]
    if missing:
        return False, f"required products are missing: {missing}"
    return True, "complete and lineage-matched"


def stage_table(config: PipelineConfig) -> list[dict[str, object]]:
    paths = stage_paths(config)
    rows = []
    for index, stage in enumerate(STAGES):
        complete, reason = is_complete(config, index, stage)
        rows.append(
            {
                "number": index + 1,
                "stage": stage.key,
                "description": stage.description,
                "output": str(paths[stage.output_key]),
                "status": "PASS" if complete else "PENDING",
                "reason": reason,
            }
        )
    return rows


def _selection(from_stage: str | None, through_stage: str | None) -> tuple[int, int]:
    keys = [stage.key for stage in STAGES]
    start = keys.index(from_stage) if from_stage else 0
    end = keys.index(through_stage) if through_stage else len(STAGES) - 1
    if start > end:
        raise ValueError("--from-stage must not occur after --through-stage")
    return start, end


def run_pipeline(
    config: PipelineConfig,
    *,
    from_stage: str | None = None,
    through_stage: str | None = None,
    resume: bool = False,
    force: bool = False,
) -> dict[str, object]:
    start, end = _selection(from_stage, through_stage)
    paths = stage_paths(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    pipeline_manifest = config.output_root / "pipeline_manifest.json"
    state: dict[str, object] = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.config_path),
        "config_sha256": _config_sha256(config),
        "workers": config.workers,
        "selected_stage_range": [STAGES[start].key, STAGES[end].key],
        "stages": [],
    }
    pipeline_manifest.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    try:
        for index, stage in enumerate(STAGES):
            if index < start or index > end:
                continue
            output = paths[stage.output_key]
            complete, reason = is_complete(config, index, stage)
            if complete and resume:
                LOGGER.info("Skipping %s: %s", stage.key, reason)
                state["stages"].append({"stage": stage.key, "status": "REUSED"})
                continue
            if output.exists():
                if not force:
                    raise FileExistsError(
                        f"Stage output already exists but is not reusable: {output}. "
                        "Use --resume when lineage matches or --force to rebuild it."
                    )
                resolved = output.resolve()
                if resolved.parent != config.output_root.resolve():
                    raise ValueError(f"Refusing to remove an out-of-run stage path: {resolved}")
                shutil.rmtree(resolved)
            LOGGER.info("[%d/%d] %s", index + 1, len(STAGES), stage.description)
            upstream = _upstream_signature(config, index)
            started = datetime.now(timezone.utc)
            products = stage.runner(config, output)
            missing = [
                relative for relative in stage.required_outputs if not (output / relative).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Stage {stage.key} did not publish required products: {missing}"
                )
            completed = datetime.now(timezone.utc)
            manifest = {
                "status": "PASS",
                "stage": stage.key,
                "description": stage.description,
                "started_utc": started.isoformat(),
                "completed_utc": completed.isoformat(),
                "elapsed_seconds": (completed - started).total_seconds(),
                "config": str(config.config_path),
                "config_sha256": _config_sha256(config),
                "upstream_manifests": upstream,
                "output": str(output),
                "products": products,
                "required_outputs": list(stage.required_outputs),
            }
            output.mkdir(parents=True, exist_ok=True)
            _manifest(output).write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            state["stages"].append({"stage": stage.key, "status": "PASS"})
            pipeline_manifest.write_text(
                json.dumps(state, indent=2) + "\n", encoding="utf-8"
            )
        state["status"] = "PASS"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        state["status"] = "FAIL"
        state["failed_utc"] = datetime.now(timezone.utc).isoformat()
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        pipeline_manifest.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state

