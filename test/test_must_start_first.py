"""
Test that APIs require experiment to be started first.
"""

import pytest

from ml_dash import dxp


def test_params_requires_start(local_dxp):
    """Test that params.set() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.params.set(lr=0.001)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_log_requires_start(local_dxp):
    """Test that log() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.log("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_metrics_requires_start(local_dxp):
    """Test that metrics() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.metrics("train").log(step=0, value=0.5)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert (
            "not started" in str(e).lower()
            or "not open" in str(e).lower()
            or "closed" in str(e).lower()
        )


def test_files_requires_start(local_dxp):
    """Test that files() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.files().list()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_apis_work_after_start(local_dxp):
    """Test that all APIs work after start."""
    with dxp.run:
        dxp.params.set(lr=0.001)
        dxp.log("Test log")
        dxp.metrics("train").log(step=0, value=0.5)
        dxp.files().list()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
