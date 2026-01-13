"""
Test that APIs require experiment to be started first.
"""

import pytest
import tempfile

from ml_dash import Experiment


def test_params_requires_start():
    """Test that params.set() requires experiment to be started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/exp", dash_root=tmpdir)
        try:
            exp.params.set(lr=0.001)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_log_requires_start():
    """Test that log() requires experiment to be started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/exp", dash_root=tmpdir)
        try:
            exp.log("test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_metrics_requires_start():
    """Test that metrics() requires experiment to be started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/exp", dash_root=tmpdir)
        try:
            exp.metrics("train").log(loss=0.5, step=0)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert (
                "not started" in str(e).lower()
                or "not open" in str(e).lower()
                or "closed" in str(e).lower()
            )


def test_files_requires_start():
    """Test that files() requires experiment to be started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/exp", dash_root=tmpdir)
        try:
            exp.files().list()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_apis_work_after_start():
    """Test that all APIs work after start."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/exp", dash_root=tmpdir)
        with exp.run:
            exp.params.set(lr=0.001)
            exp.log("Test log")
            exp.metrics("train").log(loss=0.5, step=0)
            exp.files().list()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
