from corridor_measurement.cli import _parser


def test_help_describes_automatic_worker_allocation() -> None:
    help_text = _parser().format_help()

    assert "--workers WORKERS" in help_text
    assert "50% of currently free logical-core capacity" in help_text
