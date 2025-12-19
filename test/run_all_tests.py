#!/usr/bin/env python3
"""
Run all test files individually and report their status.

This script runs each test file in isolation to ensure they can all
be executed standalone via their __main__ entry point.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_test_file(test_file: Path) -> Tuple[bool, str]:
    """
    Run a single test file and return (success, output).

    Args:
        test_file: Path to the test file

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=test_file.parent.parent  # Run from project root
        )

        # Check if test passed (exit code 0)
        success = result.returncode == 0
        output = result.stdout + result.stderr

        return success, output
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 60 seconds"
    except Exception as e:
        return False, f"Error running test: {str(e)}"


def main():
    """Run all test files and report results."""
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))

    # Exclude test_tom.py as requested
    test_files = [f for f in test_files if f.name != "test_tom.py"]

    print("=" * 80)
    print("Running all ML-Dash tests")
    print("=" * 80)
    print()

    results: List[Tuple[str, bool]] = []

    for test_file in test_files:
        test_name = test_file.name
        print(f"Running {test_name}...", end=" ", flush=True)

        success, output = run_test_file(test_file)
        results.append((test_name, success))

        if success:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            # Print failure details for debugging
            if "FAILED" in output or "ERROR" in output:
                # Extract relevant error lines
                lines = output.split('\n')
                error_lines = [l for l in lines if 'FAILED' in l or 'ERROR' in l]
                if error_lines:
                    for line in error_lines[:5]:  # Show first 5 error lines
                        print(f"  {line}")

    # Summary
    print()
    print("=" * 80)
    print("Test Results Summary")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    print(f"\nTotal: {len(results)} test files")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for name, success in results:
            if not success:
                print(f"  • {name}")

    print()
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
