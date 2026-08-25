import re
from pathlib import Path


RUN_ID = re.compile(
    r"(?:cbi|congestion-boundary-mapping|tmc-mapmatching|nvta-taplite-workflow|"
    r"corridor-profile-measurement|t2-[a-z-]+)-20\d{2}-\d{2}-\d{2}"
)


def test_runtime_code_and_default_configs_do_not_pin_output_run_ids() -> None:
    workspace = Path(__file__).resolve().parents[1]
    candidates = []
    for relative in (
        "src",
        "examples",
    ):
        root = workspace / relative
        candidates.extend(root.rglob("*.py"))
        candidates.extend(root.rglob("*.json"))

    pinned = []
    for path in candidates:
        if not path.is_file() or "tests" in path.parts or "resources" in path.parts or "build" in path.parts:
            continue
        if RUN_ID.search(path.read_text(encoding="utf-8", errors="replace")):
            pinned.append(str(path.relative_to(workspace)))

    assert pinned == []
