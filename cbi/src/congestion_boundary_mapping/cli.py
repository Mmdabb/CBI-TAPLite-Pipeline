from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Sequence

from cbi.logging_utils import configure_logging

from .build_link_t2 import main as engine_main
from .build_link_t2 import parse_args


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_args(arguments)
        output = (
            parsed.output_dir.resolve()
            if parsed.output_dir is not None
            else parsed.cbi_output_root.resolve()
            / "outputs" / "congestion-boundaries" / "link-t2"
        )
        output.mkdir(parents=True, exist_ok=True)
        configure_logging(output, False)
        logging.info("Checking congestion-boundary inputs")
        detail_path = output / "logs" / "engine.log"
        with detail_path.open("w", encoding="utf-8") as detail:
            with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
                result = engine_main(arguments)
        logging.info("Congestion-boundary resource complete: %s", output)
        return result
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
