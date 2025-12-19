"""
Test that APIs require experiment to be started first.
"""

from ml_dash import dxp
import pytest


def test_params_requires_start():
    """Test that params.set() requires experiment to be started."""
    try:
        dxp.params.set(lr=0.001)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()
        print(f"✓ params.set() error: {e}")


def test_log_requires_start():
    """Test that log() requires experiment to be started."""
    try:
        dxp.log().info("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()
        print(f"✓ log() error: {e}")


def test_metrics_requires_start():
    """Test that metrics() requires experiment to be started."""
    try:
        dxp.metrics("loss").append(step=0, value=0.5)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower() or "closed" in str(e).lower()
        print(f"✓ metrics() error: {e}")


def test_files_requires_start():
    """Test that files() requires experiment to be started."""
    try:
        dxp.files().list()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not started" in str(e).lower() or "not open" in str(e).lower()
        print(f"✓ files() error: {e}")


def test_apis_work_after_start():
    """Test that all APIs work after start."""
    with dxp.run:
        # All should work now
        dxp.params.set(lr=0.001)
        dxp.log().info("Test log")
        dxp.metrics("loss").append(step=0, value=0.5)
        files = dxp.files().list()

    print("✓ All APIs work after start")

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
