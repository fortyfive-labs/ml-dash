# Hyperparameter Sweep Example

Simple example demonstrating how to run hyperparameter sweeps with ML-Dash.

## 📁 Files

- **`train.py`** - Training script using params-proto
- **`sweep.jsonl`** - 8 hyperparameter configurations
- **`run_sweep.py`** - Launcher that runs all configs

## 🚀 Quick Start

### 1. Authenticate

```bash
ml-dash login
```

### 2. Run Single Training

```bash
python train.py --config.learning-rate 0.01 --config.batch-size 64
```

### 3. Run Full Sweep

```bash
python run_sweep.py
```

This runs all 8 configurations from `sweep.jsonl`.

## 📊 Expected Results

All experiments will be organized under the same timestamp folder:

```
{namespace}/ml-experiments/
└── 2026/01-26/
    └── experiments/sweeps/train/
        └── HH.MM.SS/          ← Timestamp when sweep started
            ├── 001/           ← lr=0.1, batch=32, SGD
            ├── 002/           ← lr=0.01, batch=32, SGD
            ├── 003/           ← lr=0.001, batch=32, SGD
            ├── 004/           ← lr=0.01, batch=64, SGD
            ├── 005/           ← lr=0.01, batch=128, SGD
            ├── 006/           ← lr=0.01, batch=32, Adam
            ├── 007/           ← lr=0.001, batch=64, Adam
            └── 008/           ← lr=0.0001, batch=128, Adam
```

## 🎯 Key Features

### Automatic Namespace Detection

Your namespace is auto-detected from the authenticated user:

```bash
python run_sweep.py  # Uses your namespace
```

### Run for Different Namespace

```bash
python run_sweep.py --namespace zehuaw  # Run for another user
```

### Shared Timestamp

All runs in a sweep share the same timestamp, grouping them together in the dashboard.

## 💡 Tips

**View results**: Navigate to `https://dash.ml/{namespace}/ml-experiments/`

**Compare runs**: All 8 configurations appear side-by-side for easy comparison

**Modify configs**: Edit `sweep.jsonl` to test different hyperparameters

## 📝 Sweep Configuration Format

Each line in `sweep.jsonl` is a JSON object:

```jsonl
{"learning_rate": 0.01, "batch_size": 64, "optimizer": "Adam", "momentum": 0.0}
{"learning_rate": 0.001, "batch_size": 128, "optimizer": "SGD", "momentum": 0.9}
```

## 🔧 Advanced Usage

### Override Project Name

```bash
python run_sweep.py --project my-research
```

### Single Config with Custom Params

```bash
python train.py \
  --config.learning-rate 0.005 \
  --config.batch-size 256 \
  --config.optimizer Adam \
  --config.epochs 10
```

### Run for Team Member

```bash
python run_sweep.py --namespace alice --project vision-experiments
```

## 📖 How It Works

1. **`run_sweep.py`** captures a single timestamp at start
2. Launches **`train.py`** for each config in `sweep.jsonl`
3. Passes timestamp and job counter via environment variables
4. Each run gets sequential number (001, 002, ...) under same timestamp folder

This ensures all sweep runs are grouped together in the dashboard!
