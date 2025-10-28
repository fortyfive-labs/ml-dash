"""Auto-configured experiment for ML-Logger.

This module provides a pre-configured global `experiment` instance that can be
imported and used immediately without manual setup.

Example:
    from ml_dash.autolog import experiment

    # No setup needed!
    experiment.params.set(learning_rate=0.001)
    experiment.metrics.log(step=0, loss=0.5)
    experiment.files.save(model.state_dict(), "checkpoint.pt")

Configuration:
    The auto-experiment is configured from environment variables:
    - ML_DASH_NAMESPACE: User/team namespace (default: "default")
    - ML_DASH_WORKSPACE: Project workspace (default: "experiments")
    - ML_DASH_PREFIX: Experiment prefix (default: auto-generated timestamp+uuid)
    - ML_DASH_REMOTE: Remote server URL (optional)

    Or from ~/.ml-logger/config.yaml:
        namespace: alice
        workspace: my-project
        remote: http://localhost:3001
"""

from .run import Experiment

# Auto-configured global experiment instance
experiment = Experiment._auto_configure()

__all__ = ["experiment"]
