---
name: dash-docs
description: ML-Dash is ML experiment tracking — a Python SDK that logs parameters, metrics, logs, files, and time-series tracks locally or to a dash.ml server, a separate `ml-dash` command line for login and moving runs, and the dash.ml dashboard. Use when: tracking ML experiments with the ml-dash Python SDK (parameters, metrics, logs, tracks); uploading or downloading files, artifacts, checkpoints, and videos; installing or using the ml-dash CLI (login, list, upload, download, update); working against a remote dash.ml server or in local mode.
---
# ML-Dash

> ML-Dash is ML experiment tracking — a Python SDK that logs parameters, metrics, logs, files, and time-series tracks locally or to a dash.ml server, a separate `ml-dash` command line for login and moving runs, and the dash.ml dashboard.

## Quickstart

```bash
pip install ml-dash                  # the Python SDK
npm install -g @dreamlake/ml-dash    # the `ml-dash` CLI (or the standalone installer)
```

```python
from ml_dash import Experiment

with Experiment("alice/my-project").run as exp:
    exp.params.set(lr=3e-4, batch_size=32)

    for epoch in range(10):
        exp.metrics("train").log(loss=loss, epoch=epoch)

    exp.files("checkpoints").save_torch(model.state_dict(), to="final.pt")
```

One `Experiment` owns one run. `params` are static key-values, `metrics` are
step-indexed scalars, `logs` are structured events, `files` are artifacts, and
`tracks` are timestamp-indexed multi-modal streams. With no `dash_url` the run
is written to `.dash/` locally; pass `dash_url="https://api.dash.ml"` (after
`ml-dash login`) to also send it to the server.

## Task routing

| Task | Read |
| --- | --- |
| Install the SDK and CLI, run a first tracked experiment | `reference/overview.md` |
| Log in; where the token lives; SDK can't find the login | `reference/get-started-authentication.md` |
| Scope a run — decorator, context manager, or explicit object | `reference/guides-experiments.md` |
| Record hyperparameters or a whole config class | `reference/guides-parameters.md` |
| Log loss/accuracy curves | `reference/guides-metrics.md` |
| Log structured events rather than numbers | `reference/guides-logging.md` |
| Save or fetch checkpoints, configs, artifacts | `reference/guides-files.md` |
| Save image frames | `reference/guides-images.md` |
| Log robot poses, sensors, per-step state | `reference/guides-tracks.md` |
| Tune batching, flushing, or throughput | `reference/guides-buffering.md` |
| Navigate the web dashboard | `reference/dashboard-overview.md` |
| Build or arrange charts | `reference/dashboard-charts.md`, `reference/dashboard-dashrc.md` |
| Compare runs side by side | `reference/dashboard-compare.md` |
| Exact signatures and return types | `reference/reference-api.md` |
| Command-line usage | `reference/reference-cli.md` |
| A complete worked script | `reference/examples.md`, then one of `reference/examples-*.md` |

## Things that are easy to get wrong

- **File annotations go on the builder, not the save call.** `save_torch(model,
  *, to)` and its siblings take only content and `to`. `description`, `tags`,
  `metadata` and `bindrs` are arguments to `files(...)`:
  `exp.files("models", tags=["best"]).save_torch(m, to="best.pt")`.
- **`tracks`, not `track`**, and `append()` is keyword-only —
  `append(x=1, y=2)`, never `append({"x": 1})`. `_ts` is optional: omitted it
  is generated from `time.time()`, and `_ts=-1` inherits the previous
  timestamp across all tracks.
- **The CLI is not in the pip package.** Since SDK 0.7.0, `pip install ml-dash`
  installs only the SDK; `ml-dash login`/`upload`/`download` come from the
  separate `@dreamlake/ml-dash` CLI.
- **`readme=`, not `description=`.** `Experiment(...)` passes unknown keywords
  through to its run config, so `description=` is silently ignored.

This skill bundles the ML-Dash documentation. Read the reference
file that matches the question; each is a self-contained markdown page.

## Reference

**Overview**

- `reference/overview.md` — ML-Dash: ML experiment tracking — install the Python SDK and the ml-dash CLI, track your first run locally, then send it to dash.ml.
- `reference/get-started-authentication.md` — Authentication: Log in with the ml-dash CLI, where the token is stored, how the Python SDK picks it up, and how the device flow works.

**Tracking**

- `reference/guides-experiments.md` — Experiments: The Experiment class owns one run's parameters, metrics, logs, files, and tracks — as a decorator, context manager, or explicit object.
- `reference/guides-parameters.md` — Parameters: Record hyperparameters and configuration as static key-value pairs, including whole config classes and nested dicts.
- `reference/guides-metrics.md` — Metrics: Log step-indexed scalars — loss, accuracy, learning rate — and read them back as summaries or series.
- `reference/guides-logging.md` — Logs: Structured event logging with levels, timestamps, and metadata, for the events that are not numbers.

**Data**

- `reference/guides-files.md` — Files: Upload and manage artifacts — checkpoints, configs, figures, blobs — with checksums, prefixes, and searchable metadata.
- `reference/guides-images.md` — Images: Save frames and image arrays, and align them with track entries for playback.
- `reference/guides-tracks.md` — Tracks: Timestamp-indexed multi-modal streams for robotics and RL — poses, sensors, per-step state — with a flexible per-topic schema.
- `reference/guides-buffering.md` — Buffering: How writes are batched and flushed in the background, and how to tune batch size and flush interval.

**Dashboard**

- `reference/dashboard-overview.md` — Dashboard: The dash.ml web dashboard: namespace/project navigation, the file tree, experiment lists, and content tabs.
- `reference/dashboard-charts.md` — Charts: Build and arrange charts over your logged metrics, and preview them live while editing.
- `reference/dashboard-compare.md` — Comparing Runs: Put runs side by side to compare parameters and metric curves across experiments.
- `reference/dashboard-dashrc.md` — The .dashrc File: Declare an experiment's dashboard layout in YAML — chart definitions, panels, and series — versioned with the run.

**Examples**

- `reference/examples.md` — Examples: Complete, runnable training scripts tracked with ml-dash — a minimal loop, PyTorch MNIST, a hyperparameter sweep, comparing runs, and debugging with logs.
- `reference/examples-simple-training.md` — Simple Training: A minimal training loop with parameters, per-epoch metrics, and a saved checkpoint.
- `reference/examples-pytorch-mnist.md` — PyTorch MNIST: A complete PyTorch MNIST run tracked end to end, from config to final model artifact.
- `reference/examples-hyperparameter-search.md` — Hyperparameter Search: Sweep hyperparameters across many runs under one project and compare the results.
- `reference/examples-experiment-comparison.md` — Comparing Experiments: Read metrics back from several runs and compare them programmatically.
- `reference/examples-logging-debugging.md` — Logging & Debugging: Use structured logs and levels to debug a run that went wrong.

**Reference**

- `reference/reference-api.md` — API Reference: Every public class, accessor, and method in the ml-dash Python SDK, with signatures and return types.
- `reference/reference-cli.md` — CLI Reference: Every ml-dash command and flag: login, profile, list, create, remove, upload, download, api, update, and version.
- `reference/reference-llm-readable.md` — LLM-Readable Docs: Every page is available as clean markdown, plus an llms.txt index, a full-corpus dump, and an importable agent skill.

## Canonical source

These docs live at https://docs.dash.ml. Each page is also fetchable as markdown
at `<page-url>.md`, and the full corpus at https://docs.dash.ml/llms-full.txt.
