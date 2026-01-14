# CLI Download Command

Download experiments from a remote ML-Dash server to local storage using the command-line interface.

## Overview

The `ml-dash download` command allows you to download experiment data from a remote ML-Dash server to your local machine. This is useful for:

- **Offline analysis**: Download experiments to analyze without internet connectivity
- **Backup and archival**: Create local backups of important experiments
- **Data migration**: Move experiments between environments
- **Collaboration**: Share experiments by downloading and redistributing
- **High-performance downloads**: Chunk-aware parallel downloads (218x faster than sequential)
- **Glob pattern matching**: Download experiments using powerful glob patterns

## Quick Start

### Basic Download

Download all experiments from the remote server:

```bash
ml-dash download ./data --dash-url https://api.dash.ml
```

### Download Specific Project

Download only experiments from a specific project:

```bash
ml-dash download ./data --dash-url https://api.dash.ml -p my-project
```

### Download with Glob Patterns

Download experiments matching a glob pattern:

```bash
# All projects starting with "test"
ml-dash download -p "test*"

# Specific experiments
ml-dash download -p "alice/*/baseline-*"
```

### Dry Run

Preview what would be downloaded without actually downloading:

```bash
ml-dash download ./data --dry-run --verbose
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

# Then download without providing credentials
ml-dash download ./data
```

**Benefits:**
- Secure OAuth2 authentication
- Token stored in system keychain
- No need to manage API keys manually
- Token auto-loaded for all CLI commands

### API Key Authentication

For advanced users or production environments, you can use an explicit API key:

```bash
ml-dash download ./data --dash-url https://api.dash.ml --api-key your-jwt-token
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
ml-dash download ./data
```

## Command Reference

### Positional Arguments

- `path` - Local storage directory to download to (default: `./.dash`)

### Remote Configuration

- `--dash-url URL` - ML-Dash server URL (defaults to config or https://api.dash.ml)
- `--api-key TOKEN` - JWT token for authentication (auto-loaded from login if not provided)

### Filtering Options

- `-p`, `--pref`, `--prefix`, `--proj`, `--project` PATTERN - Filter experiments by project or pattern
  - Supports glob patterns: `'tut*'`, `'tom*/tutorials/*'`
  - Simple project names: `my-project`
  - Full paths with wildcards: `*/deep-learning/*`

- `--experiment NAME` - Download specific experiment (requires exact project name)

### Content Selection

- `--skip-logs` - Don't download logs
- `--skip-metrics` - Don't download metrics
- `--skip-files` - Don't download files
- `--skip-params` - Don't download parameters

### Download Behavior

- `--dry-run` - Preview without downloading
- `--overwrite` - Overwrite existing experiments (default: skip existing)
- `--resume` - Resume interrupted download
- `--state-file FILE` - State file path for resume (default: `.dash-download-state.json`)

### Performance Options

- `--batch-size SIZE` - Batch size for logs/metrics (default: 1000, max: 10000)
- `--max-concurrent-metrics N` - Parallel metric downloads (default: 5)
- `--max-concurrent-files N` - Parallel file downloads (default: 3)

### Output Options

- `-v`, `--verbose` - Detailed progress output

## Usage Examples

### Example 1: Download All Experiments

Download all experiments to a local directory:

```bash
ml-dash download ./experiments --dash-url https://api.dash.ml --verbose
```

**Output:**
```
Discovering experiments on remote server...
Found 12 experiment(s)

Downloading 12 experiment(s)...

[1/12] my-project/experiment-1
  ✓ Downloaded successfully

...

Download Summary
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric            ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Experiments │ 12       │
│ Successful        │ 12       │
│ Failed            │ 0        │
│ Total Data        │ 2.3 GB   │
│ Total Time        │ 3m 45s   │
│ Avg Speed         │ 10 MB/s  │
└───────────────────┴──────────┘
```

### Example 2: Download with Glob Patterns

Download all experiments from projects starting with "training":

```bash
ml-dash download ./data -p "training*" --verbose
```

Download specific experiment pattern across all projects:

```bash
ml-dash download ./data -p "*/vision/resnet-*"
```

### Example 3: Download Without Large Files

Download experiments but skip large artifact files:

```bash
ml-dash download ./lightweight --dash-url https://api.dash.ml --skip-files
```

This is useful for:
- Quick parameter/metric analysis without downloading model checkpoints
- Bandwidth-constrained environments
- CI/CD pipelines that only need metadata

### Example 4: Download Single Experiment

Download a specific experiment by name:

```bash
ml-dash download ./specific \
  --dash-url https://api.dash.ml \
  -p vision-models \
  --experiment resnet-50-baseline
```

### Example 5: High-Performance Download

Download with optimized settings for large datasets:

```bash
ml-dash download ./large-dataset \
  --dash-url https://api.dash.ml \
  --batch-size 10000 \
  --max-concurrent-metrics 10 \
  --max-concurrent-files 5 \
  --verbose
```

**Performance benefits:**
- **Chunk-aware downloads**: Automatically downloads entire metric chunks in parallel (218x faster)
- **Large batches**: Reduces API calls by fetching more data per request
- **High concurrency**: Multiple metrics and files downloaded simultaneously

### Example 6: Resume Interrupted Download

If a download is interrupted, resume from where it left off:

```bash
# Start download
ml-dash download ./data --resume --verbose

# ... download interrupted ...

# Resume from checkpoint
ml-dash download ./data --resume --verbose
```

## Glob Pattern Support

The download command supports powerful glob pattern matching for filtering experiments:

### Pattern Syntax

- `*` - Matches any characters (including /)
- `?` - Matches any single character
- `[seq]` - Matches any character in seq
- `[!seq]` - Matches any character not in seq

### Pattern Examples

```bash
# All projects starting with "test"
ml-dash download -p "test*"

# Projects matching pattern
ml-dash download -p "alice/project-[0-9]*"

# Specific experiments
ml-dash download -p "*/deep-learning/baseline-?"

# Complex patterns
ml-dash download -p "tom*/tutorials/hyperparameter-*"
```

## Download Process

The download command follows this workflow:

### 1. Discovery Phase

Connects to remote server and discovers experiments:
- Authenticates using provided credentials
- Queries available projects and experiments
- Applies filters (glob patterns, project, experiment)
- Shows summary of what will be downloaded

### 2. Download Phase

For each experiment:

1. **Download Parameters** (unless `--skip-params`)
   - Fetches experiment metadata and hyperparameters
   - Saves to `parameters.json`

2. **Download Metrics** (unless `--skip-metrics`)
   - **Chunk-aware strategy** for optimal performance:
     - Gets metric metadata (total chunks, buffer size)
     - Downloads all chunks in parallel (10 workers by default)
     - Downloads buffer data separately
     - Merges and sorts data locally
   - **Fallback to pagination** if chunk download fails
   - Saves to `metrics/{metric-name}/data.jsonl`

3. **Download Logs** (unless `--skip-logs`)
   - Fetches logs in batches (configurable batch size)
   - Saves to `logs/logs.jsonl`

4. **Download Files** (unless `--skip-files`)
   - Downloads files in parallel (3 workers by default)
   - Preserves folder structure
   - Validates checksums
   - Saves to `files/{prefix}/{file-id}/{filename}`

### 3. Summary Phase

Shows:
- Number of experiments downloaded
- Total data size downloaded
- Total time elapsed
- Any warnings or errors

## Performance Features

### Chunk-Aware Metric Downloads

**218x faster than sequential downloads!**

The download command uses an intelligent chunk-aware strategy:

**Traditional Approach (Slow):**
- Paginate through merged data (1,000 records at a time)
- Each API call downloads entire 100k chunk from S3
- Parses all 100k records, returns only 1k
- For 1M records: 1,000 API calls × 4 seconds = 68 minutes

**Chunk-Aware Approach (Fast):**
- Get metric metadata to know total chunks
- Download all chunks in parallel (10 workers)
- Download buffer data separately
- Merge and sort locally
- For 1M records: 10 chunks downloaded in parallel = 19 seconds

**Performance Comparison:**
```
Dataset: 1,000,000 metrics
Traditional: 4,075.70 seconds (~68 minutes)
Chunk-aware: 18.69 seconds
Speedup: 218x faster! 🚀
```

### Parallel Processing

Multiple operations run concurrently:

- **Metric downloads**: Default 5 concurrent, configurable up to 10
- **File downloads**: Default 3 concurrent, configurable up to 10
- **Chunk downloads**: Fixed 10 workers for optimal S3 performance

## Common Workflows

### Workflow 1: Backup Production Experiments

Create a local backup of all production experiments:

```bash
# Download everything to backup directory
ml-dash download ./backups/$(date +%Y-%m-%d) \
  --dash-url https://prod.example.com \
  --verbose

# Verify download
ls -lh ./backups/$(date +%Y-%m-%d)
```

### Workflow 2: Offline Analysis

Download experiments for offline analysis:

```bash
# Download only metrics and parameters (skip large files)
ml-dash download ./analysis \
  --dash-url https://dash.example.com \
  -p deep-learning \
  --skip-files

# Run analysis locally
cd analysis
python analyze_metrics.py
```

### Workflow 3: Selective Download

Download only what you need:

```bash
# Only parameters for hyperparameter analysis
ml-dash download ./params-only \
  --skip-logs --skip-metrics --skip-files

# Only metrics for visualization
ml-dash download ./metrics-only \
  --skip-logs --skip-files --skip-params
```

### Workflow 4: Migration Between Servers

Migrate experiments from one server to another:

```bash
# Step 1: Download from old server
ml-dash download ./migration --dash-url https://old-server.com

# Step 2: Upload to new server
ml-dash upload ./migration --dash-url https://new-server.com
```

### Workflow 5: Resume Large Downloads

For very large datasets, use resume to handle interruptions:

```bash
# Start download with resume enabled
ml-dash download ./large-dataset \
  --dash-url https://dash.example.com \
  --resume \
  --verbose

# If interrupted (network issue, etc.), just run again
ml-dash download ./large-dataset --resume --verbose
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
- Check the URL is correct
- Ensure no firewall is blocking the connection
- Try with verbose mode: `ml-dash download -v`

### Slow Downloads

**Solutions:**

1. Increase batch size (up to 10,000):
   ```bash
   ml-dash download --batch-size 10000
   ```

2. Increase concurrent downloads:
   ```bash
   ml-dash download --max-concurrent-metrics 10 --max-concurrent-files 5
   ```

### Download Interrupted

**Solution:** Use `--resume` to continue from checkpoint:

```bash
ml-dash download ./data --resume --verbose
```

### Disk Space

**Solutions:**

1. Check available space before download:
   ```bash
   df -h .
   ```

2. Use `--dry-run` to preview size:
   ```bash
   ml-dash download --dry-run --verbose
   ```

3. Download selectively:
   ```bash
   ml-dash download --skip-files
   ```

## Best Practices

### 1. Login Once

Use OAuth2 device flow for secure authentication:

```bash
ml-dash login --dash-url https://api.dash.ml
```

### 2. Use Dry Run First

Always preview before downloading large datasets:

```bash
ml-dash download --dry-run --verbose
```

### 3. Enable Resume for Large Downloads

For downloads over 1GB or unreliable networks:

```bash
ml-dash download --resume
```

### 4. Use Glob Patterns for Filtering

Download specific experiments efficiently:

```bash
ml-dash download -p "*/production/*"
```

### 5. Skip Unnecessary Data

Save bandwidth and time:

```bash
# Analysis only: skip large model files
ml-dash download --skip-files

# Debugging: skip everything except logs
ml-dash download --skip-metrics --skip-files --skip-params
```

## Storage Format

Downloaded experiments use the same local storage format as experiments logged in local mode:

```
{path}/
└── {namespace}/
    └── {project}/
        └── {experiment-name}/
            ├── experiment.json         # Metadata
            ├── parameters.json         # Hyperparameters
            ├── logs/
            │   └── logs.jsonl         # Event logs
            ├── metrics/
            │   └── {metric-name}/
            │       └── data.jsonl     # Time-series data
            └── files/
                └── {prefix}/
                    └── {file-id}/
                        └── {filename}  # Original filename
```

**File Formats:**
- **JSONL (JSON Lines)**: Each line is a valid JSON object
- **Metrics**: `{"index": 0, "data": {"loss": 0.5}, "createdAt": "..."}`
- **Logs**: `{"level": "INFO", "message": "...", "timestamp": "..."}`

## See Also

- [CLI Upload](cli-upload.md) - Upload experiments to remote server
- [CLI List](cli-list.md) - Browse available experiments
- [Getting Started](getting-started.md) - Learn ML-Dash fundamentals

## Support

For issues or questions:
- GitHub Issues: [https://github.com/anthropics/ml-dash/issues](https://github.com/anthropics/ml-dash/issues)
