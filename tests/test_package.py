"""Smoke test: the package installs and exposes its version."""

import gnnbench


def test_version() -> None:
    assert gnnbench.__version__ == "0.1.0"
