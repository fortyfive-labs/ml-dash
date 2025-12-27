"""
Test run.folder setter - can only be set before initialization.
"""

import tempfile
from pathlib import Path
from ml_dash import Experiment


def test_folder_setter_before_init():
    """Test that folder can be set before initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("test", project="test", local_path=tmpdir)

        # Should be able to set folder before initialization
        exp.run.folder = "experiments/vision"
        assert exp.run.folder == "experiments/vision"

        # Start the experiment
        with exp.run:
            # Verify folder was used
            assert exp.folder == "experiments/vision"

    print("✓ Folder can be set before initialization")


def test_folder_setter_fails_after_init():
    """Test that folder cannot be set after initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("test", project="test", local_path=tmpdir)

        with exp.run:
            # Try to set folder after initialization - should fail
            try:
                exp.run.folder = "new/folder"
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Cannot change folder after experiment is initialized" in str(e)

    print("✓ Folder cannot be set after initialization")


def test_folder_getter():
    """Test folder getter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name="test",
            project="test",
            local_path=tmpdir,
            prefix="initial/folder"
        )

        # Getter should work before initialization
        assert exp.run.folder == "initial/folder"

        # And after
        with exp.run:
            assert exp.run.folder == "initial/folder"

    print("✓ Folder getter works")


def test_dxp_folder():
    """Test folder setter with dxp."""
    from ml_dash import dxp

    # Set folder before starting
    dxp.run.folder = "my-experiments/vision"
    assert dxp.run.folder == "my-experiments/vision"

    # Start and use
    with dxp.run:
        assert dxp.folder == "my-experiments/vision"
        dxp.params.set(test="folder_test")

    print("✓ dxp folder setter works")


def demo_folder_usage():
    """Demo showing run.folder usage."""
    from ml_dash import dxp
    from datetime import datetime

    print("\n" + "="*60)
    print("run.folder Demo - Set Before Initialization")
    print("="*60)

    # Example 1: Set folder before starting
    print("\n1. Setting folder before initialization:")
    today = datetime.now().strftime('%Y-%m-%d')
    dxp.run.folder = f"daily-runs/{today}"
    print(f"   Folder set to: {dxp.run.folder}")

    # Example 2: Start and use
    print("\n2. Starting experiment:")
    with dxp.run:
        print(f"   Experiment initialized with folder: {dxp.folder}")
        dxp.params.set(date=today, model="resnet50")

        # Try to change folder (will fail)
        print("\n3. Attempting to change folder during runtime:")
        try:
            dxp.run.folder = "new/folder"
        except RuntimeError as e:
            print(f"   ✗ Failed (expected): {e}")

    print("\n" + "="*60)
    print("Key Points:")
    print("  • run.folder can ONLY be set before initialization")
    print("  • Set it before 'with dxp.run:' or dxp.run.start()")
    print("  • Cannot be changed once experiment is running")
    print("  • Works with both dxp and rdxp")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_folder_usage()

    # Run tests
    print("Running tests...\n")
    test_folder_setter_before_init()
    test_folder_setter_fails_after_init()
    test_folder_getter()
    test_dxp_folder()
    print("\n✅ All tests passed!")
