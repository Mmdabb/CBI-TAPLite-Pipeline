from __future__ import annotations

import atexit
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class TeeTextIO:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary
        self.encoding = getattr(primary, "encoding", None)
        self.errors = getattr(primary, "errors", None)

    def write(self, text: str) -> int:
        written = self.primary.write(text)
        self.secondary.write(text)
        return written

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def fileno(self) -> int:
        return self.primary.fileno()

    def writable(self) -> bool:
        return True


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "workflow").is_dir() and (candidate / "test_networks").is_dir():
            return candidate
    return current


def install_root_log_capture(name: str, *, root: Path | None = None) -> Path:
    project_root = root or find_project_root()
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    latest_path = log_dir / f"{name}_latest.log"
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    latest_file = latest_path.open("w", encoding="utf-8", errors="replace")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeTextIO(original_stdout, log_file)  # type: ignore[assignment]
    sys.stderr = TeeTextIO(TeeTextIO(original_stderr, log_file), latest_file)  # type: ignore[assignment]
    sys.stdout = TeeTextIO(sys.stdout, latest_file)  # type: ignore[assignment]

    print(f"[INFO] Root workflow log: {log_path}")

    def close_log_files() -> None:
        print(f"[INFO] Root workflow log saved: {log_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        latest_file.close()

    atexit.register(close_log_files)
    return log_path
