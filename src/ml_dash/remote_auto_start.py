"""
Pre-configured remote experiment singleton for ML-Dash SDK.

Provides a pre-configured experiment singleton named 'rdxp' that uses remote mode.
Requires manual start using 'with' statement or explicit start() call.

IMPORTANT: Before using rdxp, you must authenticate with the ML-Dash server:
    # First time setup - authenticate with the server
    python -m ml_dash.cli login

Usage:
    from ml_dash import rdxp

    # Use with statement (recommended)
    with rdxp.run:
        rdxp.log().info("Hello from rdxp!")
        rdxp.params.set(lr=0.001)
        rdxp.metrics("loss").append(step=0, value=0.5)
    # Automatically completes on exit from with block

    # Or start/complete manually
    rdxp.run.start()
    rdxp.log().info("Training...")
    rdxp.run.complete()

Configuration:
    - Default server: https://api.dash.ml
    - To use a different server, set MLDASH_API_URL environment variable
    - Authentication token is auto-loaded from secure storage
"""

import atexit
from .experiment import Experiment

# Create pre-configured singleton experiment for remote mode
# Uses remote API server - token auto-loaded from storage
rdxp = Experiment(
    name="rdxp",
    project="scratch",
    remote="https://api.dash.ml"
)

# Register cleanup handler to complete experiment on Python exit (if still open)
def _cleanup():
    """Complete the rdxp experiment on exit if still open."""
    if rdxp._is_open:
        try:
            rdxp.run.complete()
        except Exception:
            # Silently ignore errors during cleanup
            pass

atexit.register(_cleanup)

__all__ = ["rdxp"]
