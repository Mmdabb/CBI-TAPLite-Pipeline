from .runtime import configure_numerical_threads

configure_numerical_threads(1)

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
