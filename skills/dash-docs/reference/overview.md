# ML-Dash

ML experiment tracking and data storage. Log parameters, metrics, logs,
files, and time-series tracks from Python, keep them on disk or send them to
a dash.ml server, and browse them in the dashboard.

## Install

ML-Dash comes in two parts, installed separately:

| Part | What it does | Install |
|---|---|---|
| **Python SDK** — the `ml_dash` package | Logs from your training code | `pip install ml-dash` |
| **CLI** — the `ml-dash` command | Logs in, lists projects, uploads and downloads runs | standalone binary or npm |

You only need the SDK to track runs locally. Install the CLI as well to log in
to dash.ml.

### Python SDK

```bash
pip install ml-dash
```

Python 3.9 or newer. Two optional extras:

- `ml-dash[auth]` — reads the login token from the OS keychain. You need it on
  machines where `ml-dash login` stores the token there (see
  [Authentication](reference/get-started-authentication.md#where-the-token-lives)).
- `ml-dash[video]` — saves videos from frame arrays (`save_video`).

```bash
pip install "ml-dash[auth]"
```

### CLI

**macOS, Linux:** one self-contained binary. You don't need Node or Python.

```bash
curl -fsSL https://pub-42e1dcc7de574d4a92984865fdc95f10.r2.dev/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://pub-42e1dcc7de574d4a92984865fdc95f10.r2.dev/install.ps1 | iex
```

**If you already have Node.js 20.19 or newer:**

```bash
npm install -g @dreamlake/ml-dash
```

The npm package is scoped, but the command it installs is still `ml-dash`.
Check it worked:

```bash
ml-dash version
```

> **The CLI is no longer part of pip install:** From SDK **0.7.0**, `pip install ml-dash` installs only the Python SDK. The
> `ml-dash` command is now a separate program and is installed on its own, as
> shown above. See [Upgrading from 0.6](#upgrading-from-06).

## Your first experiment

### Track a run locally

You don't need an account or a server. By default an experiment writes to
`.dash/` in the current directory:

```python
from ml_dash import Experiment

with Experiment(prefix="alice/tutorial/first-run").run as exp:
    exp.params.set(learning_rate=0.001, batch_size=32, epochs=10)
    exp.log("Training started")

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08  # your real loss here
        exp.metrics("train").log(loss=loss, epoch=epoch)

    exp.log("Training finished")
```

The prefix is `owner/project/experiment`. The run lands on disk as:

```
.dash/
└── alice/                    # owner
    └── tutorial/             # project
        └── first-run/        # experiment
            ├── logs/logs.jsonl
            ├── parameters.json
            └── metrics/train/data.jsonl
```

### Log in

```bash
ml-dash login
```

The CLI prints a short code and a QR code, and opens your browser to approve
it. The token is saved on your machine, and the SDK reads it from there. See
[Authentication](reference/get-started-authentication.md).

### Send runs to dash.ml

Pass `dash_url`, and the same code also writes to the server:

```python
with Experiment(
    prefix="alice/tutorial/first-run",
    dash_url="https://api.dash.ml",
).run as exp:
    ...
```

When the run starts, the SDK prints a link to it on
[dash.ml](https://dash.ml). Runs you already tracked locally can be uploaded
with the CLI:

```bash
ml-dash upload                  # everything under ./.dash
ml-dash list                    # confirm it arrived
```

## How it fits together

```
training script ── ml_dash SDK ──┬──→ .dash/ on disk         (local mode)
                                 └──→ ML-Dash server ──→ dash.ml dashboard
                                        ↑    (remote mode)
ml-dash CLI ── login, list, upload, download
```

| Component | Role |
|---|---|
| **`ml_dash` (PyPI)** | Python SDK. `Experiment` with `params`, `metrics`, `logs`, `files`, and `tracks` |
| **`ml-dash` CLI** | Login, projects, bulk upload and download, raw GraphQL. Ships from npm and as a standalone binary |
| **ML-Dash server** | REST and GraphQL API at `https://api.dash.ml` that stores runs, metrics, and files |
| **Dashboard** | Browse, chart, and compare runs at [dash.ml](https://dash.ml) |

## Explore the docs

- [Experiments](reference/guides-experiments.md) — Prefixes, local, hybrid, and remote mode, and the run lifecycle.
- [Metrics](reference/guides-metrics.md) — Step-indexed scalars: loss curves, accuracy, learning rate.
- [Files](reference/guides-files.md) — Checkpoints, configs, figures, and videos, with metadata.
- [Tracks](reference/guides-tracks.md) — Timestamped multi-modal streams for robotics and RL.
- [Dashboard](reference/dashboard-overview.md) — Navigate, chart, and compare runs on dash.ml.
- [Examples](reference/examples.md) — Complete training scripts, from a minimal loop to PyTorch MNIST.
- [Python API Reference](reference/reference-api.md) — Every public class and method in the SDK.
- [CLI Reference](reference/reference-cli.md) — Every `ml-dash` command and flag.

Working with an AI agent? Every page is also available as markdown and as an
importable skill. See [LLM-Readable Docs](reference/reference-llm-readable.md).

## Install reference

The rest of this page covers maintaining an install. You don't need it to get
started.

### Updating the CLI

```bash
ml-dash update           # install the latest release
ml-dash update --check   # only report whether one exists
```

`update` uses whichever channel you installed from. An npm install runs
`npm install -g @dreamlake/ml-dash@<version>`. A standalone binary downloads the
new build, checks it against the release's sha256, runs it once, and only then
replaces itself. If any step fails, your working binary stays as it was.
`update` never downgrades.

Update the SDK with pip:

```bash
pip install -U ml-dash
```

### Pin a CLI version

```bash
curl -fsSL https://pub-42e1dcc7de574d4a92984865fdc95f10.r2.dev/install.sh | sh -s -- --version 0.1.1
```

```powershell
& ([scriptblock]::Create((irm https://pub-42e1dcc7de574d4a92984865fdc95f10.r2.dev/install.ps1))) -Version 0.1.1
```

With npm: `npm install -g @dreamlake/ml-dash@0.1.1`. An installed CLI can move
to an exact newer release with `ml-dash update --version <x.y.z>`. `update`
refuses to downgrade, so use one of the commands above to go back to an older
release.

### Where the standalone CLI installs

The installer checks every download against the sha256 in that release's
manifest before writing anything. It installs into `~/.local/bin`
(`%LOCALAPPDATA%\ml-dash\bin` on Windows), which you can change with
`--install-dir` / `-InstallDir`. It never overwrites an `ml-dash` that npm or
pip installed. It reports the conflict on `PATH` and leaves it to you.

### Supported platforms

macOS (arm64, x64), Linux (x64, arm64; glibc and musl), and Windows (x64,
arm64). The binaries bundle their own runtime. On Alpine, the musl builds need
one system library first:

```bash
apk add --no-cache libstdc++
```

### Upgrading from 0.6

SDK 0.6.27 and earlier installed an `ml-dash` command as part of
`pip install ml-dash`. That Python CLI was removed in **0.7.0**:

- `pip install -U ml-dash` removes the old `ml-dash` command. Install the new CLI
  before or right after you upgrade if your scripts call `ml-dash`.
- The command names and arguments carry over. The new CLI reads and writes the
  same keychain entry and `~/.dash/` files, so an existing login normally keeps
  working. If it doesn't, run `ml-dash login` again.
- The `ml_dash.cli` and `ml_dash.cli_commands` modules are gone. Code that
  imported them should run the `ml-dash` binary instead.
- The SDK itself (`Experiment`, `params`, `metrics`, `logs`, `files`,
  `tracks`) is unchanged.

Docs for earlier releases are in the version menu in the top bar.
