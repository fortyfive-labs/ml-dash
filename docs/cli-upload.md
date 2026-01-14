# CLI Upload Command

Upload locally-stored ML-Dash experiments to a remote server using the command-line interface.

## Overview

The `ml-dash upload` command allows you to upload experiment data that was logged in local mode to a remote ML-Dash server. This is useful for:

- **Offline experimentation**: Run experiments without internet connectivity, then upload when ready
- **Batch uploads**: Upload multiple experiments at once
- **Data migration**: Move experiments between environments
- **Backup and sync**: Keep local and remote data synchronized
- **Glob pattern matching**: Upload experiments using powerful glob patterns

## Quick Start

### Basic Upload

Upload all experiments from the default local storage directory (`./.dash`):

```bash
ml-dash upload --dash-url https://api.dash.ml
```

### Upload Specific Project

Upload only experiments from a specific project using the `-p` alias:

```bash
ml-dash upload --dash-url https://api.dash.ml -p my-project
```

### Upload with Glob Patterns

Upload experiments matching a glob pattern:

```bash
# All projects starting with "test"
ml-dash upload -p "test*"

# Specific experiments in a project
ml-dash upload -p "alice/*/baseline*"
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

### OAuth2 Device Flow (Recommended)

The recommended way to authenticate is using the OAuth2 device flow:

```bash
# Login once (opens browser for authentication)
ml-dash login --dash-url https://api.dash.ml

# Then upload without providing credentials
ml-dash upload
```

**Benefits:**
- Secure OAuth2 authentication
- Token stored in system keychain
- No need to manage API keys manually
- Token auto-loaded for all CLI commands

### API Key Authentication

For advanced users or production environments, you can use an explicit API key:

```bash
ml-dash upload --dash-url https://api.dash.ml --api-key your-jwt-token
```

### Configuration File

Store your configuration in `~/.dash/config.json` to avoid passing them every time:

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

- `path` - Local storage directory to upload from (default: `./.dash`)

### Remote Configuration

- `--dash-url URL` - ML-Dash server URL (defaults to config or https://api.dash.ml)
- `--api-key TOKEN` - JWT token for authentication (auto-loaded from login if not provided)

### Filtering Options

- `-p`, `--pref`, `--prefix`, `--proj`, `--project` PATTERN - Filter experiments by prefix pattern
  - Supports glob patterns: `'tom/*/exp*'`, `'alice/project-?/baseline'`
  - Simple project names: `my-project`
  - Full paths: `alice/my-project/experiment-1`

- `-t`, `--target` PREFIX - Target prefix on server (like `scp` destination)
  - Upload to different location: `alice/shared-project`
  - Combine with source pattern for remapping

### Data Filtering

- `--skip-logs` - Don't upload logs
- `--skip-metrics` - Don't upload metrics
- `--skip-files` - Don't upload files
- `--skip-params` - Don't upload parameters

### Behavior Control

- `--dry-run` - Show what would be uploaded without uploading
- `--strict` - Fail on any validation error (default: skip invalid data)
- `-v`, `--verbose` - Show detailed progress
- `--batch-size N` - Batch size for logs/metrics (default: 100)

### Resume Functionality

- `--resume` - Resume previous interrupted upload
- `--state-file PATH` - Path to state file (default: `.dash-upload-state.json`)

## Usage Examples

### Example 1: Upload All Experiments with Progress

```bash
ml-dash upload --dash-url https://api.dash.ml --verbose
```

**Output:**
```
Scanning local storage: /path/to/.dash
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

### Example 2: Upload with Glob Patterns

Upload all experiments from projects starting with "test":

```bash
ml-dash upload -p "test*" --verbose
```

Upload specific experiment pattern across all projects:

```bash
ml-dash upload -p "*/deep-learning/baseline-*"
```

### Example 3: Upload with Target Prefix

Upload experiments to a different location on the server (like `scp`):

```bash
# Upload from local alice/experiments/* to remote shared/team-experiments/
ml-dash upload \
  --prefix "alice/experiments/*" \
  --target "shared/team-experiments" \
  --verbose
```

### Example 4: Upload Specific Project with Filters

Upload only parameters and metrics (skip logs and files):

```bash
ml-dash upload \
  --dash-url https://api.dash.ml \
  -p deep-learning \
  --skip-logs \
  --skip-files
```

### Example 5: Dry Run Before Upload

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

### Example 6: Resume Interrupted Upload

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

### Example 7: Upload from Custom Directory

```bash
ml-dash upload /path/to/custom/.dash --dash-url https://api.dash.ml
```

### Example 8: Upload with Strict Validation

Fail if any data validation errors occur:

```bash
ml-dash upload --dash-url https://api.dash.ml --strict
```

## Glob Pattern Support

The upload command supports powerful glob pattern matching for filtering experiments:

### Pattern Syntax

- `*` - Matches any characters (including /)
- `?` - Matches any single character
- `[seq]` - Matches any character in seq
- `[!seq]` - Matches any character not in seq

### Pattern Examples

```bash
# All projects starting with "test"
ml-dash upload -p "test*"

# Projects matching pattern
ml-dash upload -p "alice/project-[0-9]*"

# Specific experiments
ml-dash upload -p "*/deep-learning/baseline-?"

# Complex patterns
ml-dash upload -p "tom*/tutorials/hyperparameter-*"
```

### Combined with Target

```bash
# Upload matching experiments to a different prefix
ml-dash upload \
  --prefix "alice/*/experiment-*" \
  --target "archive/2024"
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
ml-dash upload --dash-url https://api.dash.ml
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
ml-dash upload --strict --dash-url https://api.dash.ml
```

**Behavior:**
- Any validation error stops the upload
- Warnings become errors
- No data is uploaded if validation fails
- Useful for production pipelines where data integrity is critical

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
ml-dash upload --dash-url https://api.dash.ml

# Custom batch size
ml-dash upload --batch-size 500 --dash-url https://api.dash.ml
```

**Performance considerations:**
- Larger batches = fewer API calls = faster uploads
- Smaller batches = more resilient to errors
- Default (100) is a good balance for most cases

## Resume Functionality

Interrupted uploads can be resumed without re-uploading completed experiments.

### How Resume Works

1. During upload, state is saved to `.dash-upload-state.json` after each experiment
2. State includes: completed experiments, failed experiments, timestamp
3. Use `--resume` to continue from where you left off

### Resume Example

**First attempt (interrupted):**
```bash
ml-dash upload --dash-url https://api.dash.ml
# ...uploads 3 experiments successfully...
# ...network error on experiment 4...
# State saved to .dash-upload-state.json. Use --resume to retry failed uploads.
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

## Common Workflows

### Workflow 1: Offline Training with Upload

Train models offline, then upload when connected:

```python
# During training (offline, no internet needed)
from ml_dash import Experiment

with Experiment(
    prefix="resnet-training",
    project="image-classification",
    dash_root=".dash"
).run as exp:
    exp.params.set(
        model="resnet50",
        lr=0.001,
        batch_size=32
    )

    for epoch in range(10):
        # Simulated training
        loss = 1.0 / (epoch + 1)
        acc = 0.5 + epoch * 0.05
        exp.metrics("train").log(loss=loss, accuracy=acc)
```

```bash
# Later, when online - upload all experiments
ml-dash upload --dash-url https://api.dash.ml
```

### Workflow 2: Selective Upload

Upload only successful experiments:

```bash
# Upload only the best experiment
ml-dash upload \
  -p image-classification/resnet-training-final \
  --dash-url https://api.dash.ml
```

### Workflow 3: Data Migration

Move experiments from one server to another:

```bash
# Step 1: Download from old server
ml-dash download ./backup --dash-url https://old-server.com

# Step 2: Upload to new server
ml-dash upload ./backup --dash-url https://new-server.com
```

### Workflow 4: CI/CD Integration

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

      - name: Login
        env:
          ML_DASH_URL: ${{ secrets.ML_DASH_URL }}
        run: ml-dash login --dash-url $ML_DASH_URL

      - name: Train model
        run: python train.py

      - name: Upload results
        run: ml-dash upload --strict
```

## Troubleshooting

### Authentication Errors

**Error:** `Not authenticated. Run 'ml-dash login' to authenticate`

**Solution:** Login first:
```bash
ml-dash login --dash-url https://api.dash.ml
```

### Connection Errors

**Error:** Connection refused to `https://api.dash.ml`

**Solutions:**
- Verify the server is running: `curl https://api.dash.ml/health`
- Check the remote URL is correct
- Ensure no firewall is blocking the connection
- Try with verbose mode: `ml-dash upload -v`

### No Experiments Found

**Error:** `No experiments found in local storage`

**Solutions:**
- Verify the path: `ls ./.dash`
- Check if experiments exist: `ls ./.dash/*/*/experiment.json`
- Specify correct path: `ml-dash upload /path/to/.dash`
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
   ml-dash upload -p project1
   ml-dash upload -p project2
   ```

## Best Practices

### 1. Login Once

Use OAuth2 device flow for secure authentication:

```bash
ml-dash login --dash-url https://api.dash.ml
```

### 2. Use Dry Run First

Always preview uploads before running them:

```bash
ml-dash upload --dry-run --verbose
```

### 3. Use Glob Patterns for Filtering

Upload specific experiments efficiently:

```bash
ml-dash upload -p "*/production/*"
```

### 4. Use Resume for Large Uploads

For many experiments or large files:

```bash
ml-dash upload  # initial attempt
ml-dash upload --resume  # if interrupted
```

### 5. Validate Strictly in CI/CD

Use strict mode in automated pipelines:

```bash
ml-dash upload --strict
```

## See Also

- [Getting Started](getting-started.md) - Initial setup and basic usage
- [CLI Download](cli-download.md) - Download experiments from remote
- [CLI List](cli-list.md) - Browse available experiments
- [Experiments](experiments.md) - Working with experiments

## Support

For issues or questions:
- GitHub Issues: [https://github.com/anthropics/ml-dash/issues](https://github.com/anthropics/ml-dash/issues)
