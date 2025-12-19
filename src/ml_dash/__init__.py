"""
ML-Dash Python SDK

A simple and flexible SDK for ML experiment metricing and data storage.

Usage:

    # Quickest - dxp (pre-configured remote singleton)
    # Requires: ml-dash login
    from ml_dash import dxp

    with dxp.run:
        dxp.params.set(lr=0.001)
        dxp.log().info("Training started")
    # Auto-completes on exit from with block

    # Local mode - explicit configuration
    from ml_dash import Experiment

    with Experiment(
        name="my-experiment",
        project="my-project",
        local_path=".ml-dash"
    ) as experiment:
        experiment.log("Training started")
        experiment.params.set(lr=0.001)
        experiment.metrics("loss").append(step=0, value=0.5)

    # Remote mode - explicit configuration
    with Experiment(
        name="my-experiment",
        project="my-project",
        remote="https://api.dash.ml",
        api_key="your-jwt-token"
    ) as experiment:
        experiment.log("Training started")

    # Decorator style
    from ml_dash import ml_dash_experiment

    @ml_dash_experiment(
        name="my-experiment",
        project="my-project"
    )
    def train_model(experiment):
        experiment.log("Training started")
"""

from .experiment import Experiment, ml_dash_experiment, OperationMode, RunManager
from .client import RemoteClient
from .storage import LocalStorage
from .log import LogLevel, LogBuilder
from .params import ParametersBuilder
from .auto_start import dxp

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
    "dxp",
]

# Hidden for now - rdxp (remote auto-start singleton)
# Will be exposed in a future release
#
# # Lazy-load rdxp to avoid auto-connecting to server on package import
# _rdxp = None
#
# def __getattr__(name):
#     """Lazy-load rdxp only when accessed."""
#     if name == "rdxp":
#         global _rdxp
#         if _rdxp is None:
#             from .remote_auto_start import rdxp as _loaded_rdxp
#             _rdxp = _loaded_rdxp
#         return _rdxp
#     raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
