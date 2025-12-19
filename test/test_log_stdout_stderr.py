"""
Test that logs are mirrored to stdout/stderr.
"""

from ml_dash import dxp
import sys
from io import StringIO


def test_info_goes_to_stdout():
    """Test that INFO logs go to stdout."""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        with dxp.run:
            dxp.log().info("This is an info message")

        output = sys.stdout.getvalue()
        assert "[INFO] This is an info message" in output
        print("✓ INFO logs go to stdout", file=old_stdout)
    finally:
        sys.stdout = old_stdout


def test_warn_goes_to_stdout():
    """Test that WARN logs go to stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        with dxp.run:
            dxp.log().warn("This is a warning")

        output = sys.stdout.getvalue()
        assert "[WARN] This is a warning" in output
        print("✓ WARN logs go to stdout", file=old_stdout)
    finally:
        sys.stdout = old_stdout


def test_error_goes_to_stderr():
    """Test that ERROR logs go to stderr."""
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        with dxp.run:
            dxp.log().error("This is an error")

        output = sys.stderr.getvalue()
        assert "[ERROR] This is an error" in output
        print("✓ ERROR logs go to stderr")
    finally:
        sys.stderr = old_stderr


def test_fatal_goes_to_stderr():
    """Test that FATAL logs go to stderr."""
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        with dxp.run:
            dxp.log().fatal("This is fatal")

        output = sys.stderr.getvalue()
        assert "[FATAL] This is fatal" in output
        print("✓ FATAL logs go to stderr")
    finally:
        sys.stderr = old_stderr


def test_metadata_in_output():
    """Test that metadata is included in console output."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        with dxp.run:
            dxp.log().info("Training started", epoch=1, lr=0.001)

        output = sys.stdout.getvalue()
        assert "[INFO] Training started" in output
        assert "epoch=1" in output
        assert "lr=0.001" in output
        print("✓ Metadata appears in console output", file=old_stdout)
    finally:
        sys.stdout = old_stdout


def demo_stdout_stderr():
    """Demo showing stdout/stderr output."""
    print("\n" + "="*60)
    print("Stdout/Stderr Mirroring Demo")
    print("="*60)

    with dxp.run:
        print("\n1. INFO and DEBUG go to stdout:")
        dxp.log().info("Training started", lr=0.001, batch_size=32)
        dxp.log().debug("Memory usage", gpu_mem_mb=4096)

        print("\n2. WARN goes to stdout:")
        dxp.log().warn("High loss detected", loss=1.5)

        print("\n3. ERROR and FATAL go to stderr:")
        dxp.log().error("Checkpoint save failed", path="/models/ckpt.pt")
        dxp.log().fatal("Unrecoverable error", exit_code=1)

    print("\n" + "="*60)
    print("Features:")
    print("  • INFO, WARN, DEBUG → stdout")
    print("  • ERROR, FATAL → stderr")
    print("  • Metadata shown as [key=value, ...]")
    print("  • Format: [LEVEL] message [metadata]")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_stdout_stderr()

    # Run tests
    print("Running tests...\n")
    test_info_goes_to_stdout()
    test_warn_goes_to_stdout()
    test_error_goes_to_stderr()
    test_fatal_goes_to_stderr()
    test_metadata_in_output()
    print("\n✅ All tests passed!")
