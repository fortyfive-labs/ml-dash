# Experiments

Experiments are the foundation of ML-Dash. Each experiment represents a single experiment run, containing all your logs, parameters, metrics, and files.

## Prefix Format

The prefix is a universal key that identifies your experiment:

```
{owner}/{project}/path.../[name]
```

- **owner**: First segment (e.g., your username)
- **project**: Second segment (e.g., project name)
- **path**: Remaining segments form the folder structure
- **name**: Derived from the last segment

## Three Usage Styles

**Context Manager** (recommended for most cases):

```python

from ml_dash import Experiment

# Prefix format: owner/project/experiment-name
with Experiment(prefix="alice/project/my-experiment").run as exp:
    exp.log("Training started")
    exp.params.set(learning_rate=0.001)
    # Experiment automatically closed on exit
```

**Decorator** (clean for training functions):

```python

from ml_dash import ml_dash_experiment

@ml_dash_experiment(prefix="alice/project/my-experiment")
def train_model(experiment):
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)

    for epoch in range(10):
        loss = train_epoch()
        experiment.metrics("train").log(loss=loss, epoch=epoch)

    return "Training complete!"

result = train_model()
```

**Direct** (manual control):

```python

from ml_dash import Experiment

exp = Experiment(prefix="alice/project/my-experiment")
exp.run.start()

try:
    exp.log("Training started")
    exp.params.set(learning_rate=0.001)
finally:
    exp.run.complete()
```

## Automatic Path Detection with RUN.entry

Use `RUN.entry = __file__` to automatically detect the script path and generate a meaningful experiment prefix:

```python

from ml_dash.run import RUN
from ml_dash.auto_start import dxp

# Set entry point to current file - enables automatic path detection
RUN.entry = __file__

# Now dxp will use the script's path to generate the prefix
# e.g., if running /home/user/project/experiments/train.py
# prefix becomes: user/project/experiments/train
with dxp.run:
    dxp.log("Training with automatic prefix")
    dxp.params.set(learning_rate=0.001)
```

This is useful when:
- Running multiple scripts in the same project
- You want the prefix to automatically reflect the file structure
- Organizing experiments by script location

## Local, Hybrid, and Remote Mode

Where an experiment writes depends on `dash_url` and `dash_root`:

| Mode | How to get it | Writes to |
|---|---|---|
| **Local** (default) | no `dash_url` | `.dash/` on disk. No server, no login |
| **Hybrid** | `dash_url=...` | the server **and** `.dash/` |
| **Remote** | `dash_url=...`, `dash_root=None` | the server only |

```python

# Local: zero setup, filesystem storage
with Experiment(prefix="alice/project/my-experiment").run as exp:
    exp.log("Using local storage")

# Hybrid: the server, plus a local copy
with Experiment(
    prefix="alice/project/my-experiment",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Using the server and .dash/")

# Remote only
with Experiment(
    prefix="alice/project/my-experiment",
    dash_url="https://api.dash.ml",
    dash_root=None
).run as exp:
    exp.log("Using the server only")
```

Any mode that uses the server needs a login first. See
[Authentication](reference/get-started-authentication.md). Runs recorded in local mode can
be uploaded later with [`ml-dash upload`](reference/reference-cli.md#ml-dash-upload).

## Experiment Metadata

Add a description (`readme`), tags, and bindrs for organization:

```python

with Experiment(
    prefix="alice/computer-vision/resnet50-imagenet",
    readme="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
    bindrs=["gpu-cluster", "team-a"]
).run as exp:
    exp.log("Training started")
```

**Metadata fields:**
- `readme`: Human-readable experiment description (shown as the description on the server). Note that the argument is `readme`, not `description`. Unknown keyword arguments are silently ignored.
- `tags`: List of tags for categorization (e.g., ["baseline", "production"])
- `bindrs`: List of bindrs for resource/team association (e.g., ["gpu-1", "team-ml"])

## Experiment Status Lifecycle

Experiments automatically track their status through the lifecycle:

- **RUNNING**: Automatically set when experiment opens
- **COMPLETED**: Set when experiment closes normally
- **FAILED**: Set when exception occurs during experiment
- **CANCELLED**: Can be set manually

**Automatic status management** (recommended):

```python

# Normal completion - status becomes COMPLETED
with Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Training...")
    # Status automatically set to COMPLETED on exit

# Exception handling - status becomes FAILED
with Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Training...")
    raise ValueError("Training failed!")
    # Status automatically set to FAILED on exception
```

**Manual status control**:

```python

from ml_dash import Experiment

exp = Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
)
exp.run.start()

try:
    exp.log("Training...")
    # ... training code ...
    exp.run.complete()
except KeyboardInterrupt:
    exp.run.cancel()
except Exception as e:
    exp.log(f"Error: {e}")
    exp.run.fail()
```

**Note:** Status updates only work in remote mode. Local mode doesn't track status.

## Resuming Experiments

Experiments use **upsert behavior** - reopen by using the same prefix:

```python

# First run
with Experiment(prefix="alice/ml/long-training").run as exp:
    exp.log("Starting epoch 1")
    exp.metrics("train").log(loss=0.5, epoch=1)

# Later - continues same experiment
with Experiment(prefix="alice/ml/long-training").run as exp:
    exp.log("Resuming from checkpoint")
    exp.metrics("train").log(loss=0.3, epoch=2)
```

## Report an application lifecycle to DreamLake Jobs

This optional integration requires **ml-dash 0.9.0+** and a DreamLake API that
supports the job lifecycle endpoint described below.
Dash experiment status and DreamLake job execution status are separate records.
To report application start, success, failure or cancellation, bind the reporter
to the exact existing DreamLake job and its accepted execution receipt:

```python
import os
from ml_dash import Experiment, JobLifecycle

lifecycle = JobLifecycle(
    api_url="https://api.dreamlake.ai",
    namespace="alice",
    job_id=os.environ["MY_JOB_ID"],
    attempt_id=os.environ["MY_EXECUTION_RECEIPT_ID"],
    token=os.environ["DREAMLAKE_API_KEY"],
    journal_dir="/persistent/private/job-lifecycle",
)
with Experiment(
    prefix="alice/training/run-1",
    job_lifecycle=lifecycle,
).run as experiment:
    train()
```

The caller supplies `MY_JOB_ID` and `MY_EXECUTION_RECEIPT_ID` from the existing
authenticated Jobs response (`id` and `attemptId`). These example environment
variables are not automatically injected by Nymph. Do not substitute a Dash
experiment ID, prefix, scheduler ID or worker ID. Use a DreamLake credential
authorized for this job; the Dash credential is separate. No credentials are
written to the journal.

The explicit `run.start()`, `run.complete()`, `run.fail()` and `run.cancel()`
contract drives reporting. Normal context-manager exit requests success; an
exception requests failure. Existing buffer-flush failures still downgrade a
requested completion to failure. A return value such as `False` or `None` does
not by itself mean failure. Reporting does not inspect program outputs or turn
a process exit into application success.

Each event is saved durably before transmission, with a stable event ID and the
exact API origin, namespace, job and attempt binding. Keep the private journal
on persistent storage. After a lost acknowledgment or process restart, construct
the reporter with the same binding and journal, then call `lifecycle.flush()`
to resend pending events. This sends only previously saved evidence. An API
restart returns the same accepted receipt on replay. A stale attempt or
conflicting terminal report is rejected; do not generate a replacement event
or change the attempt to make it pass. Credential renewal may retry the same
journal under the currently authorized actor.

DreamLake list/summary/detail responses expose `applicationStatus` and accepted
`applicationLifecycle` receipts. With no accepted event, the application status
is `unknown`. No terminal event is invented after a crash or forced termination.
The legacy `status` and its explicit `executionStatus` alias retain script-exit
compatibility; use `applicationStatus` to determine application lifecycle success.
An experiment timeout or stale dashboard status is not a job lifecycle receipt.

## Available Operations

Once an experiment is open, you can use all ML-Dash features:

```python

with Experiment(prefix="alice/test/demo").run as exp:
    # Logging
    exp.log("Training started", level="info")

    # Parameters
    exp.params.set(lr=0.001, batch_size=32)

    # Metrics tracking
    exp.metrics("train").log(loss=0.5, epoch=1)

    # File uploads
    exp.files("models").save("model.pth")
```

## Storage Structure

**Local mode** creates a directory structure:

```
.dash/
└── alice/                          # owner
    └── project/                    # project
        └── my-experiment/          # experiment name
            ├── logs/
            │   └── logs.jsonl
            ├── parameters.json
            ├── metrics/
            │   └── train/
            │       └── data.jsonl
            └── files/
                └── models/
                    └── 7218065541365719/
                        └── model.pth
```

Files are stored under `files/{prefix}/{snowflake_id}/{filename}` where:
- `prefix`: Logical path (e.g., "models", "configs")
- `snowflake_id`: Unique identifier for the file
- `filename`: Original filename

**Remote mode** stores data in MongoDB + S3 on your server.

---

**Next:** Learn about [Logging](reference/guides-logging.md) to track events and progress.
