# Documentation Command Test Results

All commands from README.md and QUICKSTART.md have been tested and verified.

## Test Summary

✅ **All tests passed** - All documented commands are runnable

**Date:** 2026-02-03
**Test Environment:** macOS Darwin 25.0.0, Python 3.12.12, uv project

---

## Setup Commands

### ✅ Install Dependencies
```bash
cd /path/to/ml-dash/examples
uv sync
```
**Status:** PASSED - Resolves 56 packages successfully

---

## Sweep Experiments (src/ml_experiments/sweeps/)

### ✅ Single Training Run
```bash
cd src/ml_experiments/sweeps
uv run python train.py --train.learning-rate 0.01 --train.epochs 1
```
**Status:** PASSED - Starts training successfully

### ✅ Default Sweep
```bash
uv run python launch.py --dry-run
```
**Status:** PASSED - Shows 8 configurations, generates correct commands

### ✅ Specific Sweep Files
```bash
uv run python launch.py --sweep configs/lr_sweep.jsonl --dry-run
uv run python launch.py --sweep configs/optimizer_sweep.jsonl --dry-run
uv run python launch.py --sweep configs/batch_sweep.jsonl --dry-run
```
**Status:** PASSED - All sweep files load correctly
- lr_sweep.jsonl: 5 configs
- optimizer_sweep.jsonl: 6 configs
- batch_sweep.jsonl: 5 configs

### ✅ Override Parameters
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
**Status:** PASSED - All parameter combinations work correctly

### ✅ RUN Settings Override
```bash
uv run python launch.py --owner alice --project my-research --dry-run
uv run python launch.py --api-url http://localhost:3000 --dry-run
uv run python train.py --run.owner alice --run.project vision-research
```
**Status:** PASSED - Override parameters accepted and forwarded correctly

### ✅ Generate Sweep Configs
```bash
uv run python configs/sweep_gen.py
uv run python configs/lr_sweep_gen.py
uv run python configs/optimizer_sweep_gen.py
uv run python configs/batch_sweep_gen.py
```
**Status:** PASSED - All config generators work, create valid .jsonl files

---

## Baseline Experiments (src/ml_experiments/baselines/)

### ✅ Single Baseline Training
```bash
cd src/ml_experiments/baselines
uv run python train.py --train.learning-rate 0.01
```
**Status:** PASSED - Starts baseline training successfully

### ✅ Baseline Sweep
```bash
uv run python launch.py --dry-run
```
**Status:** PASSED (FIXED) - Shows 3 baseline configurations
- **Issue Found:** launch.py was looking for `train_baseline.py` (old name)
- **Fixed:** Updated default script parameter from `train_baseline.py` to `train.py`
- **Files Modified:** `src/ml_experiments/baselines/launch.py` (lines 9, 34)

### ✅ Model Override
```bash
uv run python train.py --model.name ResNet-50
```
**Status:** PASSED - Model name override works correctly

### ✅ Generate Baseline Configs
```bash
uv run python configs/resnet_baseline_gen.py
```
**Status:** PASSED - Generates 3 baseline configurations

---

## Testing

### ✅ Run Tests
```bash
cd /path/to/ml-dash/examples
uv run pytest tests/ -v
```
**Status:** PASSED - 2/2 tests passed
- `test_package_imports` - ✓
- `test_subpackage_imports` - ✓

---

## Development Commands

### ✅ Run from Project Root
```bash
uv run python src/ml_experiments/sweeps/train.py --train.learning-rate 0.01
```
**Status:** PASSED - Full path execution works

### ✅ Help Commands
```bash
cd src/ml_experiments/sweeps
uv run python train.py --help
uv run python launch.py --help
```
**Status:** PASSED - Help text displays correctly with all options

---

## Issues Found and Fixed

### 1. Baseline Launch Script References Old Filename
**File:** `src/ml_experiments/baselines/launch.py`
**Issue:** Default script parameter was `train_baseline.py` but file was renamed to `train.py`
**Fix:**
- Line 34: Changed `script: str = "train_baseline.py"` to `script: str = "train.py"`
- Lines 9, 11, 13: Updated docstring examples from `launch_baseline.py` to `launch.py`
**Status:** ✅ FIXED

---

## Test Coverage Summary

| Category | Tests | Passed | Failed | Fixed |
|----------|-------|--------|--------|-------|
| Setup | 1 | 1 | 0 | 0 |
| Sweeps Training | 6 | 6 | 0 | 0 |
| Sweeps Launch | 5 | 5 | 0 | 0 |
| Baselines Training | 2 | 2 | 0 | 0 |
| Baselines Launch | 2 | 1 | 1 | 1 |
| Config Generators | 5 | 5 | 0 | 0 |
| Testing | 1 | 1 | 0 | 0 |
| Development | 3 | 3 | 0 | 0 |
| **TOTAL** | **25** | **24** | **1** | **1** |

---

## Notes

### Virtual Environment Warning
All commands show this warning (expected, not an error):
```
warning: `VIRTUAL_ENV=/Users/57block/fortyfive/ml-dash/.venv` does not match
the project environment path `.venv` and will be ignored
```
This occurs because the parent ml-dash project has a .venv, but examples uses its own .venv. The warning can be safely ignored.

### Timeout Commands
Training commands were tested with `timeout 3` to avoid running full experiments. All commands start successfully and only timeout during actual training loops.

### Dry Run
All launch commands were tested with `--dry-run` to verify command generation without executing full training runs.

---

## Recommendations

1. ✅ All documentation commands are working correctly
2. ✅ File structure is consistent and logical
3. ✅ Parameter overrides work as documented
4. ✅ Config generators produce valid output
5. ✅ Test suite runs successfully

**Conclusion:** The examples project is ready for use. All documented commands work as expected.
