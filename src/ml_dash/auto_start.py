"""
Pre-configured experiment singleton for ML-Dash SDK.

Provides a pre-configured experiment singleton named 'dxp' in remote mode.
Requires authentication - run 'ml-dash login' first.
Requires manual start using 'with' statement or explicit start() call.

Usage:
    # First, authenticate
    # $ ml-dash login

    from ml_dash.auto_start import dxp

    # Use with statement (recommended)
    with dxp.run:
        dxp.log("Hello from dxp!", level="info")
        dxp.params.set(lr=0.001)
        dxp.metrics("train").log(loss=0.5, step=0)
    # Automatically completes on exit from with block

    # Or start/complete manually
    dxp.run.start()
    dxp.log("Training...", level="info")
    dxp.run.complete()
"""

import atexit

# Create pre-configured singleton experiment in remote mode
# Uses default remote server (https://api.dash.ml)
# Token is auto-loaded from storage when first used
# If not authenticated, operations will fail with AuthenticationError
# Prefix format: {owner}/{project}/path...
import getpass
from datetime import datetime

from .experiment import Experiment

# Get username for dxp namespace
# Note: We use userinfo for fresh data (recommended approach)
# Falls back to system username if not authenticated
try:
    from .client import userinfo
    _username = userinfo.username or getpass.getuser()
except Exception:
    # If userinfo fails (e.g., no network), fall back to system user
    _username = getpass.getuser()

_now = datetime.now()

# Create pre-configured singleton experiment in REMOTE mode
# - dash_url=True: Use default remote server (https://api.dash.ml)
# - dash_root=None: Remote-only mode (no local storage)
# - user: Uses authenticated username from userinfo (fresh from server)
# - Token is auto-loaded from storage when first used
# - If not authenticated, operations will fail with AuthenticationError
dxp = Experiment(
    user=_username,      # Use authenticated username for namespace
    dash_url=True,       # Use remote API (https://api.dash.ml)
    dash_root=None,      # Remote-only mode (no local .dash/)
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
