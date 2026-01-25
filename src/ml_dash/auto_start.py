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
# Using getpass to get current user as owner for local convenience
import getpass
from datetime import datetime

from .auth.token_storage import get_jwt_user
from .experiment import Experiment

_user = get_jwt_user()
# Fallback to system username if not authenticated
_username = _user["username"] if _user else getpass.getuser()
_now = datetime.now()

# Create pre-configured singleton experiment in REMOTE mode
# - dash_url=True: Use default remote server (https://api.dash.ml)
# - dash_root=None: Remote-only mode (no local storage)
# - user: Uses authenticated username from token for namespace
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
