"""
Test run.prefix setter - can only be set before initialization.
"""

import tempfile
from pathlib import Path
from ml_dash import Experiment


def test_prefix_setter_before_init():
    """Test that prefix can be set before initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="test", prefix="test", local_path=tmpdir)

        # Should be able to set prefix before initialization
        exp.run.prefix = "experiments/vision"
        assert exp.run.prefix == "experiments/vision"

        # Start the experiment
        with exp.run:
            # Verify prefix was used
            assert exp._folder_path == "experiments/vision"

    print("✓ Prefix can be set before initialization")


def test_prefix_setter_fails_after_init():
    """Test that prefix cannot be set after initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="test", prefix="test", local_path=tmpdir)

        with exp.run:
            # Try to set prefix after initialization - should fail
            try:
                exp.run.prefix = "new/prefix"
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Cannot change" in str(e) or "already" in str(e).lower()

    print("✓ Prefix cannot be set after initialization")


def test_prefix_getter():
    """Test prefix getter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            project="test",
            local_path=tmpdir,
            prefix="initial/folder"
        )

        # Getter should work before initialization
        assert exp.run.prefix == "initial/folder"

        # And after
        with exp.run:
            assert exp.run.prefix == "initial/folder"

    print("✓ Prefix getter works")


def test_dxp_prefix():
    """Test prefix setter with dxp."""
    from ml_dash import dxp
    from ml_dash.storage import LocalStorage
    import shutil

    ml_dash_dir = Path(".ml-dash-test-prefix")

    # Clean up first
    if dxp._is_open:
        dxp.run.complete()
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)

    # Configure dxp for local mode testing
    dxp._storage = LocalStorage(root_path=ml_dash_dir)
    dxp._client = None

    # Set prefix before starting
    dxp.run.prefix = "my-experiments/vision"
    assert dxp.run.prefix == "my-experiments/vision"

    # Start and use
    with dxp.run:
        assert dxp._folder_path == "my-experiments/vision"
        dxp.params.set(test="prefix_test")

    # Clean up
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)

    print("✓ dxp prefix setter works")


if __name__ == "__main__":
    # Run tests
    print("Running tests...\n")
    test_prefix_setter_before_init()
    test_prefix_setter_fails_after_init()
    test_prefix_getter()
    test_dxp_prefix()
    print("\n✅ All tests passed!")
