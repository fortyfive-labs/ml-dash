"""
Pre-configured experiment singleton for ML-Dash SDK.

Provides a pre-configured experiment singleton named 'dxp' in remote mode.
Requires authentication - run 'ml-dash login' first.
Requires manual start using 'with' statement or explicit start() call.

Usage:
    # First, authenticate
    # $ ml-dash login

    from ml_dash import dxp

    # Use with statement (recommended)
    with dxp.run:
        dxp.log().info("Hello from dxp!")
        dxp.params.set(lr=0.001)
        dxp.metrics("loss").append(step=0, value=0.5)
    # Automatically completes on exit from with block

    # Or start/complete manually
    dxp.run.start()
    dxp.log().info("Training...")
    dxp.run.complete()
"""

import atexit
from .experiment import Experiment

# Create pre-configured singleton experiment in remote mode
# Uses default remote server (https://api.dash.ml)
# Token is auto-loaded from storage when first used
# If not authenticated, operations will fail with AuthenticationError
dxp = Experiment(
    name="dxp",
    project="scratch",
    remote="https://api.dash.ml",
)

# Register cleanup handler to complete experiment on Python exit (if still open)
def _cleanup():
    """Complete the dxp experiment on exit if still open."""
    if dxp._is_open:
        try:
            dxp.run.complete()
        except Exception:
            # Silently ignore errors during cleanup
            pass

atexit.register(_cleanup)

__all__ = ["dxp"]
