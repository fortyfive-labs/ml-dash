# CLI List Command

Discover and browse projects and experiments available on a remote ML-Dash server.

## Overview

The `ml-dash list` command allows you to explore what experiments are available on a remote ML-Dash server before downloading them. This is useful for:

- **Discovery**: Find available projects and experiments
- **Filtering**: Search for experiments by status, tags, or project
- **Planning**: Decide what to download before initiating transfers
- **Automation**: Programmatically query experiments with JSON output
- **Monitoring**: Check experiment status and completion

## Quick Start

### List All Experiments

See all experiments available to you:

```bash
ml-dash list --dash-url https://api.dash.ml
```

### List Experiments by Project Pattern

See experiments matching a project pattern:

```bash
ml-dash list --dash-url https://api.dash.ml -p my-project
```

### List with Glob Patterns

Use glob patterns to filter experiments:

```bash
# All projects starting with "test"
ml-dash list -p "test*"

# Specific experiments
ml-dash list -p "alice/*/baseline-*"
```

### Filter by Status

Show only completed experiments:

```bash
ml-dash list --dash-url https://api.dash.ml -p my-project --status COMPLETED
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

# Then list without providing credentials
ml-dash list
```

**Benefits:**
- Secure OAuth2 authentication
- Token stored in system keychain
- No need to manage API keys manually
- Token auto-loaded for all CLI commands

### API Key Authentication

For advanced users or production environments, you can use an explicit API key:

```bash
ml-dash list --dash-url https://api.dash.ml --api-key your-jwt-token
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
ml-dash list
```

## Command Reference

### Remote Configuration

- `--dash-url URL` - ML-Dash server URL (defaults to config or https://api.dash.ml)
- `--api-key TOKEN` - JWT token for authentication (auto-loaded from login if not provided)

### Filtering Options

- `-p`, `--pref`, `--prefix`, `--proj`, `--project` PATTERN - Filter experiments by project or pattern
  - Supports glob patterns: `'test*'`, `'tom*/tutorials/*'`
  - Simple project names: `my-project`
  - Full paths with wildcards: `*/deep-learning/*`

- `--status {COMPLETED,RUNNING,FAILED,ARCHIVED}` - Filter experiments by status
- `--tags TAGS` - Filter experiments by tags (comma-separated)

### Output Options

- `--json` - Output as JSON (for scripting and automation)
- `--detailed` - Show detailed information
- `-v`, `--verbose` - Verbose output

## Usage Examples

### Example 1: List All Experiments

Show all experiments available to you:

```bash
ml-dash list --dash-url https://api.dash.ml
```

**Output:**
```
🔬 Experiments:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Experiment                         ┃ Status     ┃ Created               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ tom/vision-models/resnet-50        │ COMPLETED  │ 2025-12-08 09:23:45   │
│ tom/nlp-experiments/bert-base      │ COMPLETED  │ 2025-12-07 18:42:10   │
│ alice/deep-learning/vit-base       │ RUNNING    │ 2025-12-06 14:15:33   │
└────────────────────────────────────┴────────────┴───────────────────────┘

Total: 3 experiments
```

### Example 2: List Experiments by Project

Show all experiments within a specific project:

```bash
ml-dash list --dash-url https://api.dash.ml -p vision-models
```

**Output:**
```
🔬 Experiments in project 'vision-models':

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Experiment            ┃ Status     ┃ Created               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ resnet-50-baseline    │ COMPLETED  │ 2025-12-08 08:00:00   │
│ resnet-101-v2         │ COMPLETED  │ 2025-12-08 06:30:15   │
│ vit-base-experiment   │ RUNNING    │ 2025-12-08 09:00:00   │
│ efficientnet-b0       │ FAILED     │ 2025-12-07 22:15:45   │
└───────────────────────┴────────────┴───────────────────────┘

Total: 4 experiments
```

### Example 3: List with Glob Patterns

Use glob patterns to filter experiments:

```bash
# All projects starting with "test"
ml-dash list --dash-url https://api.dash.ml -p "test*"

# Specific experiments across all projects
ml-dash list --dash-url https://api.dash.ml -p "*/deep-learning/baseline-*"

# Complex patterns
ml-dash list -p "tom*/tutorials/hyperparameter-*"
```

### Example 4: Filter by Status

Show only completed experiments:

```bash
ml-dash list --dash-url https://api.dash.ml -p vision-models --status COMPLETED
```

**Output:**
```
🔬 Experiments in project 'vision-models' (status: COMPLETED):

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Experiment            ┃ Status     ┃ Created               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ resnet-50-baseline    │ COMPLETED  │ 2025-12-08 08:00:00   │
│ resnet-101-v2         │ COMPLETED  │ 2025-12-08 06:30:15   │
└───────────────────────┴────────────┴───────────────────────┘

Total: 2 experiments
```

**Available Status Values:**
- `COMPLETED` - Experiment finished successfully
- `RUNNING` - Experiment currently in progress
- `FAILED` - Experiment failed with an error
- `ARCHIVED` - Experiment archived for reference

### Example 5: Filter by Tags

Show experiments with specific tags:

```bash
ml-dash list --dash-url https://api.dash.ml -p vision-models --tags "production,verified"
```

**Output:**
```
🔬 Experiments in project 'vision-models' (tags: production, verified):

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Experiment            ┃ Status     ┃ Tags                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ resnet-50-baseline    │ COMPLETED  │ production, verified, stable │
└───────────────────────┴────────────┴──────────────────────────────┘

Total: 1 experiment
```

### Example 6: Detailed Information

Show comprehensive details about experiments:

```bash
ml-dash list --dash-url https://api.dash.ml -p vision-models --detailed
```

**Output:**
```
🔬 Experiments in project 'vision-models' (detailed):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment: resnet-50-baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: COMPLETED
Created: 2025-12-08 08:00:00
Updated: 2025-12-08 08:45:23
Duration: 45 minutes

Parameters:
  • learning_rate: 0.001
  • batch_size: 32
  • epochs: 100
  • optimizer: adam

Metrics:
  • train_loss: 2,500 data points
  • train_accuracy: 2,500 data points
  • val_loss: 100 data points
  • val_accuracy: 100 data points

Files: 12 artifacts (1.2 GB)

Tags: production, verified, stable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Example 7: JSON Output for Scripting

Output as JSON for automated workflows:

```bash
ml-dash list --dash-url https://api.dash.ml -p vision-models --json
```

**Output:**
```json
{
  "namespace": "tom",
  "project": "vision-models",
  "experiments": [
    {
      "id": "272226743312449536",
      "name": "resnet-50-baseline",
      "status": "COMPLETED",
      "createdAt": "2025-12-08T08:00:00.000Z",
      "updatedAt": "2025-12-08T08:45:23.000Z",
      "tags": ["production", "verified", "stable"],
      "parameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
      },
      "metricsCount": 4,
      "filesCount": 12,
      "totalSize": 1258291200
    },
    {
      "id": "272226743312449537",
      "name": "resnet-101-v2",
      "status": "COMPLETED",
      "createdAt": "2025-12-08T06:30:15.000Z",
      "updatedAt": "2025-12-08T07:15:42.000Z",
      "tags": ["experimental"],
      "parameters": {
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 150,
        "optimizer": "sgd"
      },
      "metricsCount": 4,
      "filesCount": 8,
      "totalSize": 945678320
    }
  ],
  "total": 2
}
```

## Glob Pattern Support

The list command supports powerful glob pattern matching for filtering experiments:

### Pattern Syntax

- `*` - Matches any characters (including /)
- `?` - Matches any single character
- `[seq]` - Matches any character in seq
- `[!seq]` - Matches any character not in seq

### Pattern Examples

```bash
# All projects starting with "test"
ml-dash list -p "test*"

# Projects matching pattern
ml-dash list -p "alice/project-[0-9]*"

# Specific experiments
ml-dash list -p "*/deep-learning/baseline-?"

# Complex patterns
ml-dash list -p "tom*/tutorials/hyperparameter-*"

# All baseline experiments across all projects
ml-dash list -p "*/*/baseline*"
```

## Common Workflows

### Workflow 1: Discover and Download

Find experiments, then download specific ones:

```bash
# Step 1: List all experiments
ml-dash list --dash-url https://api.dash.ml

# Step 2: List experiments in a specific project
ml-dash list --dash-url https://api.dash.ml -p vision-models

# Step 3: Download matching experiments
ml-dash download ./data --dash-url https://api.dash.ml -p vision-models
```

### Workflow 2: Find Completed Experiments

Filter for successful experiments only:

```bash
# List all completed experiments
ml-dash list --dash-url https://api.dash.ml -p my-project --status COMPLETED

# Download all completed experiments
ml-dash download ./completed --dash-url https://api.dash.ml -p my-project
```

### Workflow 3: Automated Monitoring

Monitor running experiments with a script:

```bash
#!/bin/bash

# Get running experiments as JSON
RUNNING=$(ml-dash list \
  --dash-url https://api.dash.ml \
  -p my-project \
  --status RUNNING \
  --json)

# Count running experiments
COUNT=$(echo "$RUNNING" | jq '.total')

echo "Currently running experiments: $COUNT"

# Check for failed experiments
FAILED=$(ml-dash list \
  --dash-url https://api.dash.ml \
  -p my-project \
  --status FAILED \
  --json | jq '.total')

if [ "$FAILED" -gt 0 ]; then
  echo "⚠️  Warning: $FAILED failed experiments found!"
fi
```

### Workflow 4: Tag-Based Filtering

Find production-ready experiments:

```bash
# List experiments tagged as production
ml-dash list --dash-url https://api.dash.ml -p my-project --tags "production" --detailed

# Download production experiments for deployment
ml-dash download ./production --dash-url https://api.dash.ml -p my-project
```

### Workflow 5: Glob Pattern Search

Use glob patterns to find specific experiments:

```bash
# Find all baseline experiments across projects
ml-dash list -p "*/*/baseline-*" --json

# Find all test experiments
ml-dash list -p "test*" --detailed

# Find experiments in deep-learning projects
ml-dash list -p "*/deep-learning/*"
```

### Workflow 6: Export Experiment Catalog

Create a catalog of all experiments:

```bash
# Export all experiments to JSON
ml-dash list --dash-url https://api.dash.ml --json > catalog.json

# Generate markdown report
cat catalog.json | jq -r '
  "# Experiment Catalog\n\n" +
  (.experiments[] |
    "## \(.name)\n" +
    "- Status: \(.status)\n" +
    "- Created: \(.createdAt)\n" +
    "- Metrics: \(.metricsCount)\n" +
    "- Files: \(.filesCount)\n\n"
  )
' > CATALOG.md
```

## Output Formats

### Standard Table Format

Default human-readable table output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Experiment            ┃ Status     ┃ Created               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ resnet-50-baseline    │ COMPLETED  │ 2025-12-08 08:00:00   │
└───────────────────────┴────────────┴───────────────────────┘
```

### JSON Format

Structured data for automation and scripting:

```json
{
  "namespace": "tom",
  "project": "vision-models",
  "experiments": [...],
  "total": 24
}
```

### Detailed Format

Comprehensive information with all metadata:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment: resnet-50-baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: COMPLETED
Created: 2025-12-08 08:00:00
...
```

## Troubleshooting

### Authentication Errors

**Error:** `Not authenticated. Run 'ml-dash login' to authenticate`

**Solution:** Login first:
```bash
ml-dash login --dash-url https://api.dash.ml
```

### Issue: No Experiments Found

**Symptoms:**
- "No experiments found" message
- Empty list output

**Solutions:**

1. **Check authentication**: Ensure you're logged in
   ```bash
   ml-dash login --dash-url https://api.dash.ml
   ```

2. **Verify the pattern**: Try listing without filters first
   ```bash
   ml-dash list --dash-url https://api.dash.ml
   ```

3. **Check glob pattern syntax**:
   ```bash
   # Make sure pattern is quoted
   ml-dash list -p "test*"  # Correct
   ml-dash list -p test*     # May not work (shell expansion)
   ```

### Issue: Connection Errors

**Symptoms:**
- "Connection refused" error
- "Failed to connect to remote server"

**Solutions:**

1. **Verify server is running**:
   ```bash
   curl https://api.dash.ml/health
   ```

2. **Check the URL**:
   ```bash
   ml-dash list --dash-url https://api.dash.ml  # Correct production URL
   # Or for local development:
   ml-dash list --dash-url http://localhost:3000
   ```

3. **Test with verbose output**:
   ```bash
   ml-dash list --verbose
   ```

### Issue: Status Filter Not Working

**Symptoms:**
- Filter returns no results
- Unexpected experiments in results

**Solution:**

Use exact status values (case-sensitive):

```bash
# Correct
ml-dash list --status COMPLETED

# Wrong
ml-dash list --status completed  # Won't work
ml-dash list --status Complete   # Won't work
```

**Valid status values:**
- `COMPLETED`
- `RUNNING`
- `FAILED`
- `ARCHIVED`

## Advanced Usage

### Combining Filters

Use multiple filters together:

```bash
ml-dash list \
  --dash-url https://api.dash.ml \
  -p vision-models \
  --status COMPLETED \
  --tags "production" \
  --detailed
```

### Scripting with jq

Process JSON output with `jq`:

```bash
# Get experiment IDs only
ml-dash list --json | jq -r '.experiments[].id'

# Get experiments with high accuracy
ml-dash list --json | jq '.experiments[] | select(.parameters.accuracy > 0.95)'

# Export to CSV
ml-dash list --json | jq -r '
  ["Name","Status","Created"],
  (.experiments[] | [.name, .status, .createdAt])
  | @csv
' > experiments.csv
```

### Pagination (Future Enhancement)

For very large result sets, results may be paginated:

```bash
# Note: Pagination not yet implemented
# Future versions may support --page and --per-page flags
```

## Best Practices

### 1. Login Once

Use OAuth2 device flow for secure authentication:

```bash
ml-dash login --dash-url https://api.dash.ml
```

### 2. Use List Before Download

Always explore before downloading to avoid unnecessary transfers:

```bash
# First: see what's available
ml-dash list --dash-url https://api.dash.ml

# Then: download selectively
ml-dash download -p specific-project
```

### 3. Use Glob Patterns for Filtering

Narrow down results efficiently:

```bash
# Find specific experiments
ml-dash list -p "*/production/*"

# Combine with other filters
ml-dash list \
  -p "my-project/*" \
  --status COMPLETED \
  --tags "verified,production"
```

### 4. Use JSON for Automation

Programmatic access is easier with JSON:

```bash
# Count experiments
TOTAL=$(ml-dash list --json | jq '.total')

# Check for failures
FAILED=$(ml-dash list --status FAILED --json | jq '.total')
```

### 5. Save Output for Reference

Keep a record of experiments:

```bash
# Save full catalog
ml-dash list --json > experiments-$(date +%Y-%m-%d).json

# Save detailed report
ml-dash list --detailed > experiments-report.txt
```

### 6. Monitor Running Experiments

Set up regular checks for running experiments:

```bash
# Add to crontab (check every hour)
0 * * * * ml-dash list --status RUNNING --json | jq '.' > ~/experiment-status.json
```

## Next Steps

- [Download Command](cli-download.md) - Download experiments from remote server
- [Upload Command](cli-upload.md) - Upload experiments to remote server
- [CLI Overview](cli.md) - General CLI documentation
- [Getting Started](getting-started.md) - Learn ML-Dash fundamentals
