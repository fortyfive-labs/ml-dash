# Welcome to ML-Dash

## Claude Skill (AI Assistants)

```
# add https://raw.githubusercontent.com/fortyfive-labs/ml-dash/main/.claude/skills/params-proto.md as a skill
```

Or in your `CLAUDE.md`:
```
@import https://raw.githubusercontent.com/fortyfive-labs/ml-dash/main/.claude/skills/params-proto.md
```

---

You can install the package with uv or pip:

```shell
uv add ml-dash==0.6.2rc1
```

or

```shell
pip install ml-dash==0.6.2rc1
```

The core of ML-Dash is the `Experiment` class. It supports logging, upload, and download of
metrics, training hyper parameters, and binary files. The following example shows how to use
it in a simple training script.

```python
from ml_dash import Experiment

exp = Experiment()

exp.run.start("You can log any message here, to mark the start of the run")

# you can log the training hyper-parameters. These will be indexed and searchable
exp.params.set(learning_rate=0.001, batch_size=32)

# log metrics
exp.metrics.append(loss=0.001, accuracy=0.5)

# and you can namespace the metrics by calling the metrics writer.
exp.metrics("eval").append(loss=0.001, accuracy=0.5)

# You can upload files with a prefix
exp.files("checkpoints").save(fname="model.pth")

# and you can mark the run as complete.
exp.run.complete("This is over!")

```

Each experiment has a current "Run", that doubles as a context manager that
automatically manages the start and end of the current execution:

```python
with exp.run("Training"):
    # training logic
    pass
```

```{admonition} Dash Experiment Run Lifecycle 🔄
:class: note    

Each experimental run has the following lifecycle stages: 
- created: when the experimental run has been registered in the zaku job queue.
- running: when the run has been pulled, hydrated, and initialized.
- on-hold: when the context recieves a pause trigger event to put it on hold.
- complete: when the run finished without error. Sometimes the job can hang here due to on-goinng file-upload in the background.
- failed: when the run failed due to an error.
- aborted: when the run was aborted by the user.
- deleted: when the run was deleted by the user (soft delete)

```

## A More Complete Example

```python
import torch.nn as nn

# We typically access the dash experiment through the singleton import
from ml_dash.auto_start import dxp


# This `auto` module creates a new experiment instance upon import.
# the default experiment path is https://dash.ml/@user/scratch/date-and-time
# and this template can be changed by setting the ml_dash.run.Run template.
def train(lr=0.001, batch_size=32, n_steps=10):
    net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    for step in range(n_steps):
        # Training logic
        loss = (lambda: 0.001)()
        accuracy = (lambda: 0.5)()

        # Logging metrics
        dxp.metrics.append(loss=loss, accuracy=accuracy, step=step)

    # then you can log the evaluation metrics.
    eval_loss, eval_accuracy = 0.001, 0.02
    dxp.metrics("eval").append(loss=eval_loss, accuracy=eval_accuracy, step=step)

    # this allows you to upload the file.
    dxp.files.save_torch(net, "model_last.pt")

```

Refer to the [Getting Started Guide](getting-started.md) and the Examples section for more detailed usage examples:

- [Basic Training Loop](basic-training.md) - Simple training loop with ML-Dash
- [Hyperparameter Search](hyperparameter-search.md) - Running parameter sweeps
- [Model Comparison](model-comparison.md) - Comparing multiple model runs
- [Complete Examples](complete-examples.md) - Full end-to-end examples

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting-started
```

```{toctree}
:maxdepth: 2
:caption: Core Concepts
:hidden:

Experiment Configuration <experiments>
The Run Life-cycle <experiments>
Parameters & Hyperparameters <parameters>
Metrics & Time Series <metrics>
Message Logging <logging>
Files & Artifacts <files>
```

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

basic-training
hyperparameter-search
model-comparison
complete-examples
```

```{toctree}
:maxdepth: 2
:caption: CLI Tools
:hidden:

Command-Line Interface <cli>
Upload Command <cli-upload>
Download Command <cli-download>
List Command <cli-list>
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

api-reference
```
