"""
Test that logs are mirrored to stdout/stderr.
"""

from ml_dash import Experiment
import sys
from io import StringIO
import tempfile


def test_info_goes_to_stdout():
    """Test that INFO logs go to stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with Experiment(prefix="test/proj/exp", local_path=tmpdir).run as exp:
                exp.logs.info("This is an info message")

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
        with tempfile.TemporaryDirectory() as tmpdir:
            with Experiment(prefix="test/proj/exp", local_path=tmpdir).run as exp:
                exp.logs.warn("This is a warning")

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
        with tempfile.TemporaryDirectory() as tmpdir:
            with Experiment(prefix="test/proj/exp", local_path=tmpdir).run as exp:
                exp.logs.error("This is an error")

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
        with tempfile.TemporaryDirectory() as tmpdir:
            with Experiment(prefix="test/proj/exp", local_path=tmpdir).run as exp:
                exp.logs.fatal("This is fatal")

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
        with tempfile.TemporaryDirectory() as tmpdir:
            with Experiment(prefix="test/proj/exp", local_path=tmpdir).run as exp:
                exp.logs.info("Training started", epoch=1, lr=0.001)

            output = sys.stdout.getvalue()
            assert "[INFO] Training started" in output
            assert "epoch=1" in output
            assert "lr=0.001" in output
            print("✓ Metadata appears in console output", file=old_stdout)
    finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
    print("Running tests...\n")
    test_info_goes_to_stdout()
    test_warn_goes_to_stdout()
    test_error_goes_to_stderr()
    test_fatal_goes_to_stderr()
    test_metadata_in_output()
    print("\n✅ All tests passed!")
