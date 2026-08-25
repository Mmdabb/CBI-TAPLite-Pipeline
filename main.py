"""Repository-local entry point for the complete CBI/TAPlite pipeline."""

from pathlib import Path
import sys


# Keep ``python main.py`` usable from a fresh clone before the editable
# monorepo package has been installed.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cbi_taplite_pipeline.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
