"""
Test that APIs require experiment to be started first.
"""

from ml_dash import dxp
from ml_dash.storage import LocalStorage
from pathlib import Path
import shutil
import pytest


# Setup dxp for local testing
ml_dash_dir = Path(".ml-dash-test-must-start")


def setup_module():
    """Setup dxp for local mode testing."""
    global ml_dash_dir
    if dxp._is_open:
        dxp.run.complete()
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)
    dxp._storage = LocalStorage(root_path=ml_dash_dir)
    dxp._client = None


def teardown_module():
    """Cleanup after tests."""
    if dxp._is_open:
        dxp.run.complete()
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)


def test_params_requires_start():
    """Test that params.set() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.params.set(lr=0.001)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_log_requires_start():
    """Test that log() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.log("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_metrics_requires_start():
    """Test that metrics() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.metrics("loss").append(step=0, value=0.5)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower() or "closed" in str(e).lower()


def test_files_requires_start():
    """Test that files() requires experiment to be started."""
    if dxp._is_open:
        dxp.run.complete()
    try:
        dxp.files().list()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()


def test_apis_work_after_start():
    """Test that all APIs work after start."""
    with dxp.run:
        dxp.params.set(lr=0.001)
        dxp.log("Test log")
        dxp.metrics("loss").append(step=0, value=0.5)
        dxp.files().list()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
