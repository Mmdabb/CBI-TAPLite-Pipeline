from __future__ import annotations

import subprocess
import sys
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT))

from src.dtalite4cube.taplite_runtime import (  # noqa: E402
    load_taplite_lock,
    verify_taplite_runtime,
)


def main() -> int:
    lock = load_taplite_lock()
    requirement = f"{lock['distribution']}=={lock['version']}"
    try:
        identity = verify_taplite_runtime()
    except Exception as initial_error:
        print(f"Existing TAPLite runtime rejected: {initial_error}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "taplite4mpo",
                str(lock["distribution"]),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--only-binary=:all:",
                requirement,
            ],
            check=True,
        )
        # Verify in a fresh interpreter. A rejected native extension might
        # already be loaded in this process and cannot be safely reloaded.
        verification_code = (
            "import sys; "
            f"sys.path.insert(0, {str(WORKFLOW_ROOT)!r}); "
            "from src.dtalite4cube.taplite_runtime import verify_taplite_runtime; "
            "identity = verify_taplite_runtime(); "
            "print(identity['distribution'], identity['installed_version'], "
            "identity['native_sha256'])"
        )
        subprocess.run(
            [sys.executable, "-c", verification_code],
            check=True,
        )
        print(f"Installed and freshly verified {requirement}")
        return 0

    print(
        "Verified PyPI TAPLite runtime: "
        f"{identity['distribution']}=={identity['installed_version']} | "
        f"native SHA256 {identity['native_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
