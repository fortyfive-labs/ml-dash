"""
Auto-start module for ML-Dash SDK.

Provides a pre-configured, auto-started experiment singleton named 'dxp'.

Usage:
    from ml_dash.auto_start import dxp

    # Ready to use immediately - no need to open/start
    dxp.log("Hello from dxp!")
    dxp.params.set(lr=0.001)
    dxp.metrics("loss").append(step=0, value=0.5)

    # Automatically closed on Python exit
"""

import atexit
from .experiment import Experiment

# Create pre-configured singleton experiment
dxp = Experiment(
    name="dxp",
    project="scratch",
    local_path=".ml-dash"
)

# Auto-start the experiment on import
dxp.run.start()

# Register cleanup handler to complete experiment on Python exit
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
