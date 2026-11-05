"""
Pre-configured remote experiment singleton for ML-Dash SDK.

Provides a pre-configured experiment singleton named 'rdxp' that uses remote mode.
Requires manual start using 'with' statement or explicit start() call.

IMPORTANT: Before using rdxp, you must authenticate with the ML-Dash server:
    # First time setup - authenticate with the server.
    # The CLI ships separately, from npm:
    #   npm install -g @dreamlake/ml-dash
    $ ml-dash login

Usage:
    from ml_dash.remote_auto_start import rdxp

    # Use with statement (recommended)
    with rdxp.run:
        rdxp.log("Hello from rdxp!", level="info")
        rdxp.params.set(lr=0.001)
        rdxp.metrics("train").log(loss=0.5, step=0)
    # Automatically completes on exit from with block

    # Or start/complete manually
    rdxp.run.start()
    rdxp.log("Training...", level="info")
    rdxp.run.complete()

Configuration:
    - Default server: https://api.dash.ml
    - To use a different server, set MLDASH_API_URL environment variable
    - Authentication token is auto-loaded from secure storage
"""

import atexit

# Create pre-configured singleton experiment for remote mode
# Uses remote API server - token auto-loaded from storage
# Prefix format: {owner}/{project}/path...
import getpass

from .config import DEFAULT_API_URL
from .experiment import Experiment

_owner = getpass.getuser()
import os

if os.environ.get("ML_DASH_BACKEND") == "native":
  # DreamLake-native: the run is stored by DreamLake through the existing `dreamlake login`, in
  # ML_DASH_NATIVE_NAMESPACE (default: the login's namespace) / ML_DASH_NATIVE_PROJECT (default: scratch).
  # As ml-dash's rdxp, a missing project is created (private) on first use, where the login may create it.
  from .native import NativeConnection
  _namespace = NativeConnection.default_namespace()
  _project = os.environ.get("ML_DASH_NATIVE_PROJECT") or "scratch"
  rdxp = Experiment(prefix=f"{_namespace}/{_project}/rdxp",
                    managed=NativeConnection.from_login(_namespace, _project, auto_create_project=True))
else:
  rdxp = Experiment(prefix=f"{_owner}/scratch/rdxp", dash_url=DEFAULT_API_URL)


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
