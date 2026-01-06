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

### List All Projects

See all projects in your namespace:

```bash
ml-dash list --remote https://api.dash.ml --username your-username
```

### List Experiments in a Project

See all experiments within a specific project:

```bash
ml-dash list --remote https://api.dash.ml --username your-username --project my-project
```

### Filter by Status

Show only completed experiments:

```bash
ml-dash list --remote https://api.dash.ml --username your-username \
  --project my-project --status COMPLETED
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
ml-dash list --remote https://api.dash.ml --username john-doe
```

**How it works:**
- The CLI generates a deterministic JWT token from your username
- Same username always produces the same token
- No need to manage API keys manually
- Token is used only for the current session (not stored)

### API Key Authentication

For advanced users or production environments, you can use an explicit API key:

```bash
ml-dash list --remote https://api.dash.ml --api-key your-jwt-token
```

### Configuration File

Store your credentials in `~/.dash/config.json` to avoid passing them every time:

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

- `--remote URL` - Remote server URL (required unless set in config)
- `--api-key TOKEN` - JWT token for authentication
- `--username USERNAME` - Username for auto-generating API key
- `--namespace NAMESPACE` - Namespace slug (defaults to username)

### Filtering Options

- `--project PROJECT` - List experiments in this project
- `--status {COMPLETED,RUNNING,FAILED,ARCHIVED}` - Filter experiments by status
- `--tags TAGS` - Filter experiments by tags (comma-separated)

### Output Options

- `--json` - Output as JSON (for scripting and automation)
- `--detailed` - Show detailed information
- `-v, --verbose` - Verbose output

## Usage Examples

### Example 1: List All Projects

Show all projects in your namespace:

```bash
ml-dash list --remote https://api.dash.ml --username tom
```

**Output:**
```
📚 Projects in namespace 'tom':

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Project            ┃ Experiments  ┃ Last Updated          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ vision-models      │ 24           │ 2025-12-08 09:23:45   │
│ nlp-experiments    │ 15           │ 2025-12-07 18:42:10   │
│ reinforcement      │ 8            │ 2025-12-06 14:15:33   │
└────────────────────┴──────────────┴───────────────────────┘

Total: 3 projects, 47 experiments
```

### Example 2: List Experiments in a Project

Show all experiments within a specific project:

```bash
ml-dash list --remote https://api.dash.ml --username tom --project vision-models
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

### Example 3: Filter by Status

Show only completed experiments:

```bash
ml-dash list --remote https://api.dash.ml --username tom \
  --project vision-models --status COMPLETED
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

### Example 4: Filter by Tags

Show experiments with specific tags:

```bash
ml-dash list --remote https://api.dash.ml --username tom \
  --project vision-models --tags "production,verified"
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

### Example 5: Detailed Information

Show comprehensive details about experiments:

```bash
ml-dash list --remote https://api.dash.ml --username tom \
  --project vision-models --detailed
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

### Example 6: JSON Output for Scripting

Output as JSON for automated workflows:

```bash
ml-dash list --remote https://api.dash.ml --username tom \
  --project vision-models --json
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

## Common Workflows

### Workflow 1: Discover and Download

Find experiments, then download specific ones:

```bash
# Step 1: List all projects
ml-dash list --remote https://api.dash.ml --username tom

# Step 2: List experiments in interesting project
ml-dash list --remote https://api.dash.ml --username tom --project vision-models

# Step 3: Download specific experiment
ml-dash download ./data --remote https://api.dash.ml --username tom \
  --project vision-models --experiment resnet-50-baseline
```

### Workflow 2: Find Completed Experiments

Filter for successful experiments only:

```bash
# List all completed experiments
ml-dash list --remote https://api.dash.ml --username tom \
  --project my-project --status COMPLETED

# Download all completed experiments
ml-dash download ./completed --remote https://api.dash.ml --username tom \
  --project my-project
```

### Workflow 3: Automated Monitoring

Monitor running experiments with a script:

```bash
#!/bin/bash

# Get running experiments as JSON
RUNNING=$(ml-dash list \
  --remote https://api.dash.ml \
  --username tom \
  --project my-project \
  --status RUNNING \
  --json)

# Count running experiments
COUNT=$(echo "$RUNNING" | jq '.total')

echo "Currently running experiments: $COUNT"

# Check for failed experiments
FAILED=$(ml-dash list \
  --remote https://api.dash.ml \
  --username tom \
  --project my-project \
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
ml-dash list --remote https://api.dash.ml --username tom \
  --project my-project --tags "production" --detailed

# Download production experiments for deployment
ml-dash download ./production --remote https://api.dash.ml --username tom \
  --project my-project  # Then filter locally by tags
```

### Workflow 5: Export Experiment Catalog

Create a catalog of all experiments:

```bash
# Export all projects to JSON
ml-dash list --remote https://api.dash.ml --username tom --json > catalog.json

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

### Issue: No Projects Found

**Symptoms:**
- "No projects found" message
- Empty list output

**Solutions:**

1. **Verify namespace**: Make sure you're using the correct username/namespace
   ```bash
   ml-dash list --namespace correct-namespace
   ```

2. **Check authentication**: Ensure you have permission to view projects
   ```bash
   ml-dash list --username your-username --verbose
   ```

3. **Try explicit API key**:
   ```bash
   ml-dash list --api-key "eyJhbGc..."
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

2. **Check remote URL**:
   ```bash
   ml-dash list --remote https://api.dash.ml  # Correct production URL
   # Or for local development:
   ml-dash list --remote http://localhost:3000
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
  --remote https://api.dash.ml \
  --username tom \
  --project vision-models \
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

### 1. Use List Before Download

Always explore before downloading to avoid unnecessary transfers:

```bash
# First: see what's available
ml-dash list --remote https://api.dash.ml --username tom

# Then: download selectively
ml-dash download --project specific-project
```

### 2. Filter Aggressively

Narrow down results to find exactly what you need:

```bash
ml-dash list \
  --project my-project \
  --status COMPLETED \
  --tags "verified,production"
```

### 3. Use JSON for Automation

Programmatic access is easier with JSON:

```bash
# Count experiments
TOTAL=$(ml-dash list --json | jq '.total')

# Check for failures
FAILED=$(ml-dash list --status FAILED --json | jq '.total')
```

### 4. Save Output for Reference

Keep a record of experiments:

```bash
# Save full catalog
ml-dash list --json > experiments-$(date +%Y-%m-%d).json

# Save detailed report
ml-dash list --detailed > experiments-report.txt
```

### 5. Monitor Running Experiments

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
