# CLI Upload Command

Upload locally-stored ML-Dash experiments to a remote server using the command-line interface.

## Overview

The `ml-dash upload` command allows you to upload experiment data that was logged in local mode to a remote ML-Dash server. This is useful for:

- **Offline experimentation**: Run experiments without internet connectivity, then upload when ready
- **Batch uploads**: Upload multiple experiments at once
- **Data migration**: Move experiments between environments
- **Backup and sync**: Keep local and remote data synchronized

## Quick Start

### Basic Upload

Upload all experiments from the default local storage directory (`./.ml-dash`):

```bash
ml-dash upload --remote https://api.dash.ml --username your-username
```

### Upload Specific Project

Upload only experiments from a specific project:

```bash
ml-dash upload --remote https://api.dash.ml --username your-username --project my-project
```

### Dry Run

Preview what would be uploaded without actually uploading:

```bash
ml-dash upload --dry-run --verbose
```

## Installation

The CLI is included with the ML-Dash Python package:

```bash
pip install ml-dash
# or
uv pip install ml-dash
```

## Authentication

### Username-based Authentication (Recommended)

The simplest way to authenticate is using your username. The CLI will automatically generate an API key:

```bash
ml-dash upload --remote https://api.dash.ml --username john-doe
```

**How it works:**
- The CLI generates a deterministic JWT token from your username
- Same username always produces the same token
- No need to manage API keys manually
- Token is used only for the current session (not stored)

### API Key Authentication

For advanced users or production environments, you can use an explicit API key:

```bash
ml-dash upload --remote https://api.dash.ml --api-key your-jwt-token
```

### Configuration File

Store your credentials in `~/.ml-dash/config.json` to avoid passing them every time:

```json
{
  "remote_url": "https://api.dash.ml",
  "api_key": "your-jwt-token-here"
}
```

Then simply run:

```bash
ml-dash upload
```

## Command Reference

### Positional Arguments

- `path` - Local storage directory to upload from (default: `./.ml-dash`)

### Remote Configuration

- `--remote URL` - Remote server URL (required unless set in config)
- `--api-key TOKEN` - JWT token for authentication
- `--username NAME` - Username for automatic authentication

### Filtering Options

- `--project NAME` - Upload only experiments from this project
- `--experiment NAME` - Upload only this specific experiment (requires `--project`)

### Data Filtering

- `--skip-logs` - Don't upload logs
- `--skip-metrics` - Don't upload metrics
- `--skip-files` - Don't upload files
- `--skip-params` - Don't upload parameters

### Behavior Control

- `--dry-run` - Show what would be uploaded without uploading
- `--strict` - Fail on any validation error (default: skip invalid data)
- `-v, --verbose` - Show detailed progress
- `--batch-size N` - Batch size for logs/metrics (default: 100)

### Resume Functionality

- `--resume` - Resume previous interrupted upload
- `--state-file PATH` - Path to state file (default: `.ml-dash-upload-state.json`)

## Usage Examples

### Example 1: Upload All Experiments with Progress

```bash
ml-dash upload \
  --remote https://api.dash.ml \
  --username john-doe \
  --verbose
```

**Output:**
```
Scanning local storage: /path/to/.ml-dash
Found 3 experiment(s)

Validating experiments...
3 experiment(s) ready to upload

Uploading to: https://api.dash.ml
[1/3] my-project/experiment-1 ━━━━━━━━━━━━━━━━━━ 100%
  ✓ Uploaded (5 params, 100 logs, 2 metrics)
[2/3] my-project/experiment-2 ━━━━━━━━━━━━━━━━━━ 100%
  ✓ Uploaded (3 params, 50 logs, 1 metrics, 2 files)
[3/3] my-project/experiment-3 ━━━━━━━━━━━━━━━━━━ 100%
  ✓ Uploaded (metadata only)

┏━━━━━━━━━━━━┳━━━━━━━┓
┃ Status     ┃ Count ┃
┡━━━━━━━━━━━━╇━━━━━━━┩
│ Successful │ 3/3   │
└────────────┴───────┘

┏━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Type   ┃ Count        ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Logs   │ 150 entries  │
│ Metrics│ 3 metrics    │
│ Files  │ 2 files      │
└────────┴──────────────┘
```

### Example 2: Upload Specific Project with Filters

Upload only parameters and metrics (skip logs and files):

```bash
ml-dash upload \
  --remote https://api.dash.ml \
  --username john-doe \
  --project deep-learning \
  --skip-logs \
  --skip-files
```

### Example 3: Dry Run Before Upload

Preview what will be uploaded:

```bash
ml-dash upload --dry-run --verbose

# Output:
# Discovered experiments:
#   • project1/exp1 (logs, params, 2 metrics, 5 files)
#   • project1/exp2 (logs, params, 1 metrics)
#   • project2/exp1 (params only)
#
# DRY RUN - No data will be uploaded
# Run without --dry-run to proceed with upload.
```

### Example 4: Resume Interrupted Upload

If an upload is interrupted (network issue, etc.), resume it:

```bash
ml-dash upload --resume --verbose

# Output:
# Resuming previous upload from 2024-12-03T10:30:00
#   Already completed: 5 experiments
#   Failed: 0 experiments
# Skipping 5 already completed experiment(s)
# Found 3 experiment(s) to upload
# ...continues with remaining experiments...
```

### Example 5: Upload from Custom Directory

```bash
ml-dash upload /path/to/custom/.ml-dash \
  --remote https://api.dash.ml \
  --username john-doe
```

### Example 6: Upload with Strict Validation

Fail if any data validation errors occur:

```bash
ml-dash upload \
  --remote https://api.dash.ml \
  --username john-doe \
  --strict
```

## Data Validation

The CLI automatically validates experiment data before uploading. Validation is fail-graceful by default:

### What Gets Validated

1. **Experiment Metadata** (required)
   - Must have valid `experiment.json`
   - Must contain `name` and `project` fields

2. **Parameters** (optional)
   - Must be valid JSON
   - Must be a dictionary/object

3. **Logs** (optional)
   - Each log entry must have a `message` field
   - Invalid lines are skipped with a warning

4. **Metrics** (optional)
   - Each data point must have a `data` field
   - Invalid lines are skipped with a warning

5. **Files** (optional)
   - Files referenced in metadata must exist on disk
   - Missing files are skipped with a warning

### Validation Modes

#### Lenient Mode (Default)

Skips invalid data and continues:

```bash
ml-dash upload --remote https://api.dash.ml --username john-doe
```

**Behavior:**
- Invalid log entries are skipped
- Invalid metric data points are skipped
- Missing files are skipped
- Warnings are displayed
- Upload continues with valid data

#### Strict Mode

Fails on any validation error:

```bash
ml-dash upload --strict --remote https://api.dash.ml --username john-doe
```

**Behavior:**
- Any validation error stops the upload
- Warnings become errors
- No data is uploaded if validation fails
- Useful for production pipelines where data integrity is critical

### Validation Output

**Lenient mode output:**
```
Validating experiments...
  ⚠ project1/exp1:
      logs.jsonl has 3 invalid lines (e.g., [10, 15, 20]...) - will skip these
3 experiment(s) ready to upload
```

**Strict mode output:**
```
Validating experiments...
  ✗ project1/exp1:
      logs.jsonl has 3 invalid lines (e.g., [10, 15, 20]...)
Error: Validation failed in --strict mode
```

## Upload Process

### Upload Order

Data is uploaded in this specific order to maintain referential integrity:

1. **Experiment metadata** - Creates the experiment on the server
2. **Parameters** - Uploads experiment parameters
3. **Logs** - Uploads log entries in batches
4. **Metrics** - Uploads metric data points in batches
5. **Files** - Uploads files one by one

### Batch Processing

Logs and metrics are uploaded in batches for efficiency:

```bash
# Default batch size (100)
ml-dash upload --remote https://api.dash.ml --username john-doe

# Custom batch size
ml-dash upload --batch-size 500 --remote https://api.dash.ml --username john-doe
```

**Performance considerations:**
- Larger batches = fewer API calls = faster uploads
- Smaller batches = more resilient to errors
- Default (100) is a good balance for most cases

### Error Handling

The CLI handles errors gracefully:

- **Network errors**: Experiment marked as failed, saved to state file
- **Invalid data**: Skipped with warning (lenient) or fails (strict)
- **Server errors**: Experiment marked as failed, error message displayed

Failed experiments can be retried with `--resume`:

```bash
ml-dash upload --resume
```

## Resume Functionality

Interrupted uploads can be resumed without re-uploading completed experiments.

### How Resume Works

1. During upload, state is saved to `.ml-dash-upload-state.json` after each experiment
2. State includes: completed experiments, failed experiments, timestamp
3. Use `--resume` to continue from where you left off

### Resume Example

**First attempt (interrupted):**
```bash
ml-dash upload --remote https://api.dash.ml --username john-doe
# ...uploads 3 experiments successfully...
# ...network error on experiment 4...
# State saved to .ml-dash-upload-state.json. Use --resume to retry failed uploads.
```

**Resume:**
```bash
ml-dash upload --resume
# Resuming previous upload from 2024-12-03T10:30:00
#   Already completed: 3 experiments
#   Failed: 0 experiments
# Skipping 3 already completed experiment(s)
# Found 2 experiment(s) to upload
# ...continues with remaining experiments...
```

### Custom State File

Use a custom state file location:

```bash
ml-dash upload --state-file /path/to/my-state.json
ml-dash upload --resume --state-file /path/to/my-state.json
```

### State File Format

The state file is a JSON file:

```json
{
  "local_path": "/path/to/.ml-dash",
  "remote_url": "https://api.dash.ml",
  "completed_experiments": [
    "project1/exp1",
    "project1/exp2",
    "project2/exp1"
  ],
  "failed_experiments": [],
  "in_progress_experiment": null,
  "timestamp": "2024-12-03T10:30:00"
}
```

**Note:** The state file is automatically deleted after a successful upload.

## Common Workflows

### Workflow 1: Offline Training with Upload

Train models offline, then upload when connected:

```python
# During training (offline, no internet needed)
from ml_dash import Experiment

with Experiment(
    name="resnet-training",
    project="image-classification",
    local_path="./.ml-dash"
).run as exp:
    exp.params.set(
        model="resnet50",
        lr=0.001,
        batch_size=32
    )

    for epoch in range(10):
        # Training loop...
        exp.metrics("loss").append(value=loss, epoch=epoch)
        exp.metrics("accuracy").append(value=acc, epoch=epoch)
```

```bash
# Later, when online - upload all experiments
ml-dash upload --remote https://api.dash.ml --username john-doe
```

### Workflow 2: Selective Upload

Upload only successful experiments:

```bash
# Upload only the best experiment
ml-dash upload \
  --project image-classification \
  --experiment resnet-training-final \
  --remote https://api.dash.ml \
  --username john-doe
```

### Workflow 3: Data Migration

Move experiments from one server to another:

```bash
# Step 1: Download experiments from old server (using local mode)
# (This would be done during normal experiment logging)

# Step 2: Upload to new server
ml-dash upload \
  --remote http://new-server.com \
  --username john-doe
```

### Workflow 4: Backup Strategy

Regular backups from local to remote:

```bash
#!/bin/bash
# backup-experiments.sh

# Upload all new/modified experiments
ml-dash upload \
  --remote http://backup-server.com \
  --username john-doe \
  --verbose

# The CLI automatically skips already-uploaded experiments
```

### Workflow 5: CI/CD Integration

Upload experiment results from CI pipeline:

```yaml
# .github/workflows/train.yml
name: Train Model
on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: pip install ml-dash

      - name: Train model
        run: python train.py

      - name: Upload results
        env:
          ML_DASH_REMOTE: ${{ secrets.ML_DASH_REMOTE }}
          ML_DASH_API_KEY: ${{ secrets.ML_DASH_API_KEY }}
        run: |
          ml-dash upload \
            --remote $ML_DASH_REMOTE \
            --api-key $ML_DASH_API_KEY \
            --strict
```

## Troubleshooting

### Authentication Errors

**Error:** `Error: --api-key or --username is required`

**Solution:** Provide authentication credentials:
```bash
ml-dash upload --username your-username --remote https://api.dash.ml
```

Or create a config file at `~/.ml-dash/config.json`:
```json
{
  "remote_url": "https://api.dash.ml",
  "api_key": "your-jwt-token"
}
```

### Connection Errors

**Error:** Connection refused to `https://api.dash.ml`

**Solutions:**
- Verify the server is running: `curl https://api.dash.ml/health`
- Check the remote URL is correct
- Ensure no firewall is blocking the connection
- Try with verbose mode: `ml-dash upload -v`

### Invalid Data Errors

**Error:** Experiments fail validation

**Solutions:**

1. Use verbose mode to see detailed errors:
   ```bash
   ml-dash upload --verbose
   ```

2. Use lenient mode (default) to skip invalid data:
   ```bash
   ml-dash upload  # automatically skips invalid data
   ```

3. Check the specific validation errors and fix the source data

### No Experiments Found

**Error:** `No experiments found in local storage`

**Solutions:**
- Verify the path: `ls ./.ml-dash`
- Check if experiments exist: `ls ./.ml-dash/*/*/experiment.json`
- Specify correct path: `ml-dash upload /path/to/.ml-dash`
- Ensure experiments were created in local mode

### Slow Uploads

**Solutions:**

1. Increase batch size for logs/metrics:
   ```bash
   ml-dash upload --batch-size 500
   ```

2. Skip large files during initial upload:
   ```bash
   ml-dash upload --skip-files
   ```

3. Upload specific projects one at a time:
   ```bash
   ml-dash upload --project project1
   ml-dash upload --project project2
   ```

### Interrupted Uploads

**Error:** Upload interrupted due to network issues

**Solution:** Use resume functionality:
```bash
ml-dash upload --resume
```

The CLI automatically saves progress and skips already-uploaded experiments.

### State File Issues

**Error:** `State file local path doesn't match`

**Solution:** The state file is for a different local directory. Either:
- Start fresh (delete `.ml-dash-upload-state.json`)
- Use the correct local directory
- Specify a different state file: `--state-file my-state.json`

## Advanced Usage

### Custom Validation

For advanced use cases, you can pre-validate data before upload:

```python
from ml_dash.cli_commands.upload import discover_experiments, ExperimentValidator
from pathlib import Path

# Discover experiments
experiments = discover_experiments(Path("./.ml-dash"))

# Validate
validator = ExperimentValidator(strict=True)
for exp in experiments:
    result = validator.validate_experiment(exp)
    if not result.is_valid:
        print(f"Invalid: {exp.project}/{exp.experiment}")
        for error in result.errors:
            print(f"  - {error}")
```

### Programmatic Upload

Use the upload functionality in Python scripts:

```python
from ml_dash.cli_commands.upload import cmd_upload
import argparse

args = argparse.Namespace(
    path="./.ml-dash",
    remote="https://api.dash.ml",
    api_key=None,
    user_name="john-doe",
    project=None,
    experiment=None,
    dry_run=False,
    strict=False,
    verbose=True,
    batch_size=100,
    skip_logs=False,
    skip_metrics=False,
    skip_files=False,
    skip_params=False,
    resume=False,
    state_file=".ml-dash-upload-state.json",
)

exit_code = cmd_upload(args)
print(f"Upload {'succeeded' if exit_code == 0 else 'failed'}")
```

### Monitoring Progress

The CLI provides real-time progress feedback:

- **Progress bars** show overall upload progress
- **Spinners** indicate current operation
- **Colored output** highlights success/failure
- **Summary tables** show final statistics

Enable verbose mode for maximum detail:
```bash
ml-dash upload --verbose
```

## Best Practices

### 1. Use Dry Run First

Always preview uploads before running them:

```bash
ml-dash upload --dry-run --verbose
```

### 2. Store Credentials in Config

Avoid passing credentials on command line:

```json
// ~/.ml-dash/config.json
{
  "remote_url": "https://api.dash.ml",
  "api_key": "your-token"
}
```

### 3. Use Resume for Large Uploads

For many experiments or large files:

```bash
ml-dash upload  # initial attempt
ml-dash upload --resume  # if interrupted
```

### 4. Filter by Project

Upload projects incrementally:

```bash
ml-dash upload --project project1
ml-dash upload --project project2
```

### 5. Validate Strictly in CI/CD

Use strict mode in automated pipelines:

```bash
ml-dash upload --strict
```

### 6. Skip Large Files Initially

Upload metadata/metrics first, files later:

```bash
ml-dash upload --skip-files  # fast initial upload
ml-dash upload  # upload everything (including files)
```

### 7. Monitor with Verbose Mode

For debugging or monitoring:

```bash
ml-dash upload --verbose 2>&1 | tee upload.log
```

## See Also

- [Getting Started](getting-started.md) - Initial setup and basic usage
- [Experiments](experiments.md) - Working with experiments
- [API Reference](api-reference.md) - Complete API documentation
- [Local vs Remote Mode](getting-started.md#operation-modes) - Understanding operation modes

## Support

For issues or questions:
- GitHub Issues: [https://github.com/anthropics/ml-dash/issues](https://github.com/anthropics/ml-dash/issues)
- Documentation: [https://ml-dash.readthedocs.io](https://ml-dash.readthedocs.io)
