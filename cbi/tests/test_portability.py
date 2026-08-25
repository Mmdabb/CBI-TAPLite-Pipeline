import hashlib
import json
from pathlib import Path


ACTIVE_ROOT = Path(__file__).resolve().parents[1] / "src" / "congestion_boundary_mapping"
FORBIDDEN_RUNTIME_REFERENCES = (
    'workspace_root / "t2"',
    "workspace_root / 't2'",
    '"t2/ml-experiment"',
    "'t2/ml-experiment'",
    '"t2/coverage-expansion"',
    "'t2/coverage-expansion'",
    "outputs/t2/",
    "outputs\\t2\\",
    "t2_ml_experiment",
)


def test_active_mapper_has_no_legacy_t2_runtime_dependency():
    violations = []
    for path in ACTIVE_ROOT.rglob("*.py"):
        if "tests" in path.parts or "resources" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_RUNTIME_REFERENCES:
            if fragment in text:
                violations.append((path.name, fragment))
    assert violations == []


def test_bundled_resources_match_manifest():
    root = ACTIVE_ROOT / "resources" / "ridge_completion"
    manifest = json.loads(
        (root / "resource_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.is_file(), relative
        assert path.stat().st_size == expected["size_bytes"], relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected["sha256"], relative


def test_bundled_metadata_has_no_machine_specific_user_path():
    root = ACTIVE_ROOT / "resources" / "ridge_completion"
    for path in [*root.rglob("*.json"), *root.rglob("*.md")]:
        assert "C:\\Users\\" not in path.read_text(encoding="utf-8")
