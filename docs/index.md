# Welcome to ML-Dash

ML-Dash is a lightweight and flexible experiment tracking library for machine learning research and development, integrated with the visualization at dash.ml.

## Installation

To get started, install ML-Dash:

```shell
pip install ml-dash
```

ML-Dash works the best with the [dash.ml](https://dash.ml) visualization dashboard. To authenticate with dash.ml, run the following command:

```shell
ml-dash login
```

Follow the prompts to authenticate with your dash.ml account. todo: figure out how to pass access_token to workers. 

## Quick Example

Here is a simple example of logging metrics and parameters:

```python
from ml_dash import ML_Dash

# Initialize logger
logger = ML_Dash(
    namespace="your-name",
    workspace="project-name",
    prefix="my-experiment/" + __file__,
    token="logging-token"
)

# Log parameters: set and extend.
logger.params.set(
    learning_rate=0.001,
    batch_size=32,
    model="resnet50",
    # Recommend NestedDict with name-spacing.
    train=dict(
        learning_rate=0.001,
        batch_size=32,
        model="resnet50",
    )
)

# Log metrics during training
for epoch in range(100):
    train_loss = train_step()
    val_loss = validate()

    logger.metrics.log(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss
    )
```

And this should output a link to your experiment in dash.ml:
```markdown
View Experiment: https://dash.ml/your-name/project-name/my-experiment
```

todo: add link to a screenshot of dash.ml

## Key Features

- **Lightweight and performant**: Minimal overhead on your training loop
- **Flexible storage**: Local filesystem, S3, or custom backends
- **Async logging**: Non-blocking metric logging with local caching
- **Structured data**: Organize experiments with hierarchical prefixes
- **Artifact management**: Save and load models, plots, and other files
- **Open source**: MIT licensed

## Getting Started

- Check out the [Quick Start](quick_start) guide to begin using ML-Dash
- Review the [API Documentation](api/ml_dash.md) for detailed reference
- See the [CHANGE LOG](CHANGE_LOG.md) for version history

<!-- prettier-ignore-start -->

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   Quick Start <quick_start>
   Report Issues <https://github.com/fortyfive-labs/ml-dash/issues>
   CHANGE LOG <CHANGE_LOG.md>

.. toctree::
   :maxdepth: 3
   :caption: Python API
   :hidden:

   ml_dash — Core Logger <api/ml_dash.md>
   ml_dash.types — Type Interface <api/types.md>
   ml_dash.client — HTTP Client <api/client.md>
   ml_dash.cache — Local Caching <api/cache.md>

```
