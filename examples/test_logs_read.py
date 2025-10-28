#!/usr/bin/env python3
"""Test script for logs.read() with remote backend"""

import sys
sys.path.insert(0, '/Users/57block/PycharmProjects/vuer-dashboard/ml-logger/src')

from ml_dash import Experiment

# Create experiment with remote backend
exp = Experiment(
    namespace="test",
    workspace="logs-test",
    prefix="logs-read-test",
    remote="http://localhost:4000"
)

print("Testing logs.read() with remote backend")
print("=" * 60)

# Start a run
with exp.run():
    # Log some messages
    exp.info("First log message", step=1)
    exp.warning("Warning message", step=2, temperature=0.8)
    exp.error("Error message", step=3, error_code="E001")
    exp.debug("Debug message", step=4, details="some details")

    print("\nLogged 4 messages")
    print("Now reading logs back from server...")

    # Read logs from server
    logs = exp.logs.read()

    print(f"\nFetched {len(logs)} log entries:")
    print("-" * 60)

    for i, log in enumerate(logs, 1):
        level = log.get("level")
        message = log.get("message")
        context = log.get("context", {})
        timestamp = log.get("timestamp")

        print(f"\n{i}. [{level}] {message}")
        if context:
            print(f"   Context: {context}")
        print(f"   Timestamp: {timestamp}")

    print("\n" + "=" * 60)
    print("✓ logs.read() test completed successfully!")
