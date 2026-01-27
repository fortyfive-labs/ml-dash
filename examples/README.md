# ML-Dash Examples

Examples demonstrating ML-Dash experiment tracking with params-proto configuration management.

## 📁 Structure

```
examples/
├── experiments/sweeps/   ← Ad hoc experiments (WITH datetime)
│   ├── train.py
│   ├── launch.py
│   └── configs/
│       ├── sweep.jsonl & sweep_gen.py
│       ├── lr_sweep.jsonl & lr_sweep_gen.py
│       ├── optimizer_sweep.jsonl & optimizer_sweep_gen.py
│       └── batch_sweep.jsonl & batch_sweep_gen.py
│
└── baselines/            ← Systematic baselines (STATIC paths)
    ├── train_baseline.py
    ├── launch_baseline.py
    └── configs/
        └── resnet_baseline.jsonl & resnet_baseline_gen.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /path/to/ml-dash/examples
uv sync
```

### 2. Authenticate

```bash
ml-dash login
```

### 3. Run Your First Sweep

```bash
cd experiments/sweeps
uv run python launch.py
```

This runs 8 configurations from `configs/sweep.jsonl`, all grouped under a shared timestamp.

### 4. Run Single Training

```bash
uv run python train.py --train.learning-rate 0.01 --train.batch-size 64
```

### 5. Run Baseline Experiment

```bash
cd ../../baselines
uv run python launch_baseline.py
```

## 🔷 Two Types of Experiments

### Ad Hoc Experiments (`experiments/sweeps/`)

**Purpose**: Exploratory experiments and quick hyperparameter tests

**Path structure**: Dynamic (WITH datetime)
```
{namespace}/ml-experiments/2026/01-26/experiments/sweeps/train/16.08.04/001
                                                            ^^^^^^^^  ^^^
                                                            timestamp job#
```

**Files:**
- **`train.py`** - Training script with Train/Model/Eval configs
- **`launch.py`** - Sweep launcher (launches train.py multiple times)
- **`configs/`** - Sweep configuration files
  - `sweep.jsonl` (8 configs) - Mixed LR, batch size, optimizer
  - `lr_sweep.jsonl` (5 configs) - Learning rate variations
  - `optimizer_sweep.jsonl` (6 configs) - SGD vs Adam
  - `batch_sweep.jsonl` (5 configs) - Batch size variations

**Usage:**
```bash
cd experiments/sweeps

# Run default sweep
uv run python launch.py

# Run specific sweep
uv run python launch.py --sweep configs/lr_sweep.jsonl

# Single training run
uv run python train.py --train.learning-rate 0.01 --model.name ResNet-50

# Override RUN settings
uv run python launch.py --owner alice --project my-research

# Use local API server
uv run python launch.py --api-url http://localhost:3000

# Dry run (preview commands)
uv run python launch.py --sweep configs/optimizer_sweep.jsonl --dry-run

# Regenerate sweep files
uv run python configs/sweep_gen.py
```

**How it works:**
1. `launch.py` captures a single timestamp at start
2. Launches `train.py` as subprocess for each config in sweep file
3. Passes shared timestamp via `ML_DASH_SWEEP_TIMESTAMP` env var
4. Each run gets sequential job counter (001, 002, 003...)
5. All runs appear under same timestamp folder for easy comparison

**Dashboard organization:**
```
{namespace}/ml-experiments/2026/01-26/experiments/sweeps/train/
├── 16.08.04/          ← First sweep run
│   ├── 001/           ← lr=0.1, batch=32, SGD
│   ├── 002/           ← lr=0.01, batch=32, SGD
│   └── ...
└── 18.30.22/          ← Second sweep run
    ├── 001/
    └── ...
```

### Baseline Experiments (`baselines/`)

**Purpose**: Permanent reference baselines for benchmarks and papers

**Path structure**: STATIC (NO datetime)
```
{namespace}/ml-experiments/baselines/resnet18/001
                                     ^^^^^^^^  ^^^
                                     model    job#
```

**Files:**
- **`train_baseline.py`** - Baseline training with static paths
- **`launch_baseline.py`** - Baseline sweep launcher
- **`configs/resnet_baseline.jsonl`** - Standard ResNet configurations

**Usage:**
```bash
cd baselines

# Run single baseline
uv run python train_baseline.py --train.learning-rate 0.01

# Run baseline sweep
uv run python launch_baseline.py

# Different model (changes path)
uv run python train_baseline.py --model.name ResNet-50
# Path: .../baselines/resnet50/001

# Regenerate configs
uv run python configs/resnet_baseline_gen.py
```

**How it works:**
1. Sets static prefix without datetime: `RUN.prefix = f"{owner}/ml-experiments/baselines/{model_name}"`
2. Launcher passes `ML_DASH_JOB_COUNTER` only (no shared timestamp)
3. Each baseline always appears at the same path for permanent reference

**Dashboard organization:**
```
{namespace}/ml-experiments/baselines/
├── resnet18/
│   ├── 001/           ← Always at same path
│   ├── 002/
│   └── ...
└── resnet50/
    └── 001/
```

## 🎯 Key Features

### Multi-Config Setup with Params-Proto

Separate config classes for different components:

```python
@proto.prefix
class Train:
    learning_rate: float = 0.01
    batch_size: int = 32
    optimizer: str = "SGD"

@proto.prefix
class Model:
    name: str = "ResNet-18"
    dropout: float = 0.0

@proto.prefix
class Eval:
    metric: str = "accuracy"
    dataset: str = "CIFAR-10"

@proto.cli
def main():
    print(f"Training {Model.name} with lr={Train.learning_rate}")
```

CLI usage with namespaced arguments:
```bash
# Training parameters
uv run python train.py --train.learning-rate 0.01 --train.batch-size 64

# Model parameters
uv run python train.py --model.name ResNet-50 --model.dropout 0.3

# Evaluation parameters
uv run python train.py --eval.metric f1_score --eval.dataset ImageNet

# Mix and match
uv run python train.py \
  --train.learning-rate 0.01 \
  --train.optimizer Adam \
  --model.name ResNet-18 \
  --eval.metric accuracy
```

### Automatic Namespace Detection

Examples auto-detect your namespace using `ml_dash.userinfo`:

```python
from ml_dash import userinfo

if userinfo.username:
    RUN.owner = userinfo.username
```

No manual configuration needed! Override if needed:
```bash
uv run python train.py --run.owner alice --run.project my-research
```

### Sweep Coordination

All sweep runs share the same timestamp for grouped dashboard viewing:

```python
# In launch.py (parent process)
sweep_timestamp = datetime.now()
env["ML_DASH_SWEEP_TIMESTAMP"] = str(sweep_timestamp.timestamp())
env["ML_DASH_JOB_COUNTER"] = str(i + 1)

# In train.py (child process)
if sweep_timestamp := os.environ.get("ML_DASH_SWEEP_TIMESTAMP"):
    RUN.now = datetime.fromtimestamp(float(sweep_timestamp))
if sweep_job_counter := os.environ.get("ML_DASH_JOB_COUNTER"):
    RUN.job_counter = int(sweep_job_counter)
```

Result: All runs grouped under same timestamp with sequential counters (001, 002, 003...)

### Multiple Focused Sweep Files

Each sweep file targets a specific aspect:

| Sweep File | Focus | Configs | Generator |
|------------|-------|---------|-----------|
| `sweep.jsonl` | Mixed (LR, batch, optimizer) | 8 | `sweep_gen.py` |
| `lr_sweep.jsonl` | Learning rate variations | 5 | `lr_sweep_gen.py` |
| `optimizer_sweep.jsonl` | SGD vs Adam comparison | 6 | `optimizer_sweep_gen.py` |
| `batch_sweep.jsonl` | Batch size variations | 5 | `batch_sweep_gen.py` |

### Sweep Configuration Format

Each line in a `.jsonl` file is a JSON object with hyperparameters:

```jsonl
{"learning_rate": 0.1, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9}
{"learning_rate": 0.01, "batch_size": 64, "optimizer": "Adam", "momentum": 0.0}
{"learning_rate": 0.001, "batch_size": 128, "optimizer": "SGD", "momentum": 0.9}
```

Parameters are automatically mapped to the correct config namespace:
- `learning_rate`, `batch_size`, `optimizer` → `--train.xxx`
- `name`, `dropout`, `pretrained` → `--model.xxx`
- `metric`, `dataset` → `--eval.xxx`

## 🆚 Ad Hoc vs Baselines

| Aspect | Ad Hoc (`experiments/`) | Baselines (`baselines/`) |
|--------|------------------------|-------------------------|
| **Path** | Dynamic (WITH datetime) | STATIC (NO datetime) |
| **Purpose** | Exploratory, testing | Permanent reference |
| **Use cases** | Sweeps, quick experiments | Benchmarks, papers, comparisons |
| **Timestamp** | Shared across sweep runs | None (static path) |
| **Example path** | `.../2026/01-26/.../16.08.04/001` | `.../baselines/resnet18/001` |

**When to use ad hoc:**
- Exploring hyperparameters
- Quick experiments and testing
- Comparing multiple configurations

**When to use baselines:**
- Establishing permanent benchmarks
- Paper/publication baselines
- Model architecture comparisons
- Reproducible reference results

## 🔧 Advanced Usage

### Dry Run (Preview Commands)

```bash
uv run python launch.py --dry-run
uv run python launch_baseline.py --dry-run
```

Shows all commands that would be executed without actually running them.

### Custom Sweep Files

Create your own sweep configuration:

```python
# configs/my_custom_sweep_gen.py
import json
from pathlib import Path

configs = [
    {"learning_rate": 0.005, "batch_size": 256, "optimizer": "AdamW"},
    {"learning_rate": 0.01, "batch_size": 128, "optimizer": "SGD"},
]

output_file = Path(__file__).parent / "my_custom_sweep.jsonl"
with open(output_file, 'w') as f:
    for config in configs:
        f.write(json.dumps(config) + '\n')
```

Then run:
```bash
uv run python configs/my_custom_sweep_gen.py
uv run python launch.py --sweep configs/my_custom_sweep.jsonl
```

### Override RUN Settings

```bash
# Launcher accepts clean arguments (--owner, --project, --api-url)
uv run python launch.py --owner alice --project my-experiments
uv run python launch.py --api-url http://localhost:3000

# Training scripts use lowercase prefix (--run.owner, --run.project, --run.api-url)
uv run python train.py --run.owner alice --run.project vision-research
uv run python train.py --run.api-url http://localhost:3000
```

### Regenerate All Sweep Files

```bash
cd experiments/sweeps
uv run python configs/sweep_gen.py
uv run python configs/lr_sweep_gen.py
uv run python configs/optimizer_sweep_gen.py
uv run python configs/batch_sweep_gen.py

cd ../../baselines
uv run python configs/resnet_baseline_gen.py
```

## 💡 Tips

**View results**: Navigate to `https://dash.ml/{your_namespace}/ml-experiments/`

**Compare runs**: All sweep runs appear side-by-side in the dashboard under the same timestamp folder

**Modify configs**: Edit `.jsonl` files directly or regenerate using `*_gen.py` scripts

**Static baselines**: Use `baselines/` for experiments you want to reference permanently (papers, benchmarks)

**Quick iterations**: Use `experiments/sweeps/` for exploratory work and hyperparameter tuning

**Dry runs**: Always test with `--dry-run` first to preview what will be executed

## 📖 How It Works

### Launch → Train Relationship

`launch.py` and `train.py` work together:

1. **`launch.py`** (orchestrator):
   - Reads sweep configs from `.jsonl` file
   - Captures shared timestamp
   - **Launches `train.py` as subprocess** for each config
   - Passes parameters via CLI args
   - Coordinates via environment variables

2. **`train.py`** (worker):
   - Actual training script
   - Accepts params via `--train.xxx`, `--model.xxx`, `--eval.xxx`
   - Reads sweep coordination from env vars
   - Can run standalone or via launcher

Example flow:
```
launch.py reads configs/lr_sweep.jsonl
  ↓
Subprocess: python train.py --train.learning-rate 0.1 --sweep-index 0
Subprocess: python train.py --train.learning-rate 0.05 --sweep-index 1
Subprocess: python train.py --train.learning-rate 0.01 --sweep-index 2
  ...
```

### Environment Variables

**Ad hoc sweeps:**
- `ML_DASH_SWEEP_TIMESTAMP` - Shared timestamp (Unix timestamp)
- `ML_DASH_JOB_COUNTER` - Sequential job number (1, 2, 3...)

**Baselines:**
- `ML_DASH_JOB_COUNTER` only (no shared timestamp)

### Params-Proto CLI Integration

Uses `@proto.cli` for clean CLI generation:

```python
@proto.cli
def main(
    sweep: str = "configs/sweep.jsonl",
    dry_run: bool = False,
):
    # Auto-generates: --sweep, --dry-run
    ...
```

Parameters with underscores become kebab-case: `dry_run` → `--dry-run`

## 📚 See Also

- [ML-Dash Documentation](https://dash.ml/docs)
- [params-proto Documentation](https://github.com/geyang/params-proto)
- `experiments/sweeps/train.py` - Multi-config training implementation
- `experiments/sweeps/launch.py` - Sweep launcher implementation
- `baselines/train_baseline.py` - Baseline training with static paths
