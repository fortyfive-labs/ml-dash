"""
ML-Dash Python SDK

A simple and flexible SDK for ML experiment metricing and data storage.

Usage:

    # Remote mode (API server)
    from ml_dash import Experiment

    with Experiment(
        name="my-experiment",
        project="my-project",
        remote="http://localhost:3000",
        api_key="your-jwt-token"
    ) as experiment:
        experiment.log("Training started")
        experiment.metric("loss", {"step": 0, "value": 0.5})

    # Local mode (filesystem)
    with Experiment(
        name="my-experiment",
        project="my-project",
        local_path=".ml-dash"
    ) as experiment:
        experiment.log("Training started")

    # Decorator style
    from ml_dash import ml_dash_experiment

    @ml_dash_experiment(
        name="my-experiment",
        project="my-project",
        remote="http://localhost:3000",
        api_key="your-jwt-token"
    )
    def train_model(experiment):
        experiment.log("Training started")
"""

from .experiment import Experiment, ml_dash_experiment, OperationMode, RunManager
from .client import RemoteClient
from .storage import LocalStorage
from .log import LogLevel, LogBuilder
from .params import ParametersBuilder

__version__ = "0.1.0"

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
]
