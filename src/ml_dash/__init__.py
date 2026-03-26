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
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    ExperimentError,
    MlDashError,
    NetworkError,
    StorageError,
)
from .experiment import Experiment, OperationMode, ml_dash_experiment
from .log import LogBuilder, LogLevel
from .params import ParametersBuilder
from .run import RUN
from .storage import LocalStorage

__version__ = "0.6.24"


def _check_version_compatibility():
    """
    Enforce strict version requirement by checking against PyPI.

    Raises ImportError if installed version is older than the latest on PyPI.
    This ensures all users are on the latest version with newest features and bug fixes.
    """
    try:
        from packaging import version
        import httpx
    except ImportError:
        # If packaging or httpx not available, skip check
        return

    try:
        # Check PyPI for latest version (with short timeout)
        response = httpx.get(
            "https://pypi.org/pypi/ml-dash/json",
            timeout=2.0,
            follow_redirects=True
        )

        if response.status_code == 200:
            latest_version = response.json()["info"]["version"]
            current = version.parse(__version__)
            latest = version.parse(latest_version)

            if current < latest:
                raise ImportError(
                    f"\n"
                    f"{'=' * 80}\n"
                    f"ERROR: ml-dash version {__version__} is outdated!\n"
                    f"{'=' * 80}\n"
                    f"\n"
                    f"Your installed version ({__version__}) is no longer supported.\n"
                    f"Latest version on PyPI: {latest_version}\n"
                    f"\n"
                    f"Please upgrade to the latest version:\n"
                    f"\n"
                    f"  pip install --upgrade ml-dash\n"
                    f"\n"
                    f"Or with uv:\n"
                    f"\n"
                    f"  uv pip install --upgrade ml-dash\n"
                    f"  uv sync --upgrade-package ml-dash\n"
                    f"\n"
                    f"{'=' * 80}\n"
                )
    except (httpx.TimeoutException, httpx.ConnectError, KeyError):
        # Silently skip check if PyPI is unreachable or response is malformed
        # Don't block users due to network issues
        pass


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
  # Exceptions
  "MlDashError",
  "AuthenticationError",
  "ConfigurationError",
  "ExperimentError",
  "NetworkError",
  "StorageError",
]
