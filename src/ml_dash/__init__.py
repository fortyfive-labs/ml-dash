"""
ML-Dash Python SDK

A simple and flexible SDK for ML experiment metricing and data storage.

Prefix format: {owner}/{project}/path.../[name]
  - owner: First segment (e.g., your username)
  - project: Second segment (e.g., project name)
  - path: Remaining segments form the folder structure
  - name: Derived from last segment (may be a seed/id)

Usage:

    from ml_dash import Experiment

    # Local mode - explicit configuration
    with Experiment(
        prefix="ge/my-project/experiments/exp1",
        dash_root=".dash"
    ).run as exp:
        exp.log("Training started")
        exp.params.set(lr=0.001)
        exp.metrics("train").log(loss=0.5, step=0)

    # Default: Remote mode (defaults to https://api.dash.ml)
    with Experiment(prefix="ge/my-project/experiments/exp1").run as exp:
        exp.log("Training started")
        exp.params.set(lr=0.001)
        exp.metrics("train").log(loss=0.5, step=0)

    # Decorator style
    from ml_dash import ml_dash_experiment

    @ml_dash_experiment(prefix="ge/my-project/experiments/exp1")
    def train_model(exp):
        exp.log("Training started")
"""

from .client import RemoteClient, userinfo
from .experiment import Experiment, OperationMode, ml_dash_experiment
from .log import LogBuilder, LogLevel
from .params import ParametersBuilder
from .run import RUN
from .storage import LocalStorage

__version__ = "0.6.10"

# Minimum version required - blocks older versions
MINIMUM_REQUIRED_VERSION = "0.6.10"


def _check_version_compatibility():
    """
    Enforce minimum version requirement.

    Raises ImportError if installed version is below minimum required version.
    This ensures users have the latest features (userinfo, namespace auto-detection, etc.)
    """
    try:
        from packaging import version
    except ImportError:
        # If packaging is not available, skip check
        # (unlikely since it's a common dependency)
        return

    current = version.parse(__version__)
    minimum = version.parse(MINIMUM_REQUIRED_VERSION)

    if current < minimum:
        raise ImportError(
            f"\n"
            f"{'=' * 80}\n"
            f"ERROR: ml-dash version {__version__} is too old!\n"
            f"{'=' * 80}\n"
            f"\n"
            f"This version of ml-dash ({__version__}) is no longer supported.\n"
            f"Minimum required version: {MINIMUM_REQUIRED_VERSION}\n"
            f"\n"
            f"Please upgrade to the latest version:\n"
            f"\n"
            f"  pip install --upgrade ml-dash\n"
            f"\n"
            f"Or install specific version:\n"
            f"\n"
            f"  pip install ml-dash>={MINIMUM_REQUIRED_VERSION}\n"
            f"\n"
            f"{'=' * 80}\n"
        )


# Enforce version check on import
_check_version_compatibility()

__all__ = [
  "Experiment",
  "ml_dash_experiment",
  "OperationMode",
  "RemoteClient",
  "LocalStorage",
  "LogLevel",
  "LogBuilder",
  "ParametersBuilder",
  "RUN",
  "userinfo",
]
