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

from .client import RemoteClient
from .experiment import Experiment, OperationMode, RunManager, ml_dash_experiment
from .log import LogBuilder, LogLevel
from .params import ParametersBuilder
from .run import RUN
from .storage import LocalStorage

__version__ = "0.6.2"

__all__ = [
  "Experiment",
  "ml_dash_experiment",
  "OperationMode",
  "RunManager",
  "RemoteClient",
  "LocalStorage",
  "LogLevel",
  "LogBuilder",
  "ParametersBuilder",
  "RUN",
]
