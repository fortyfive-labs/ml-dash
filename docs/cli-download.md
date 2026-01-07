# CLI Download Command

Download experiments from a remote ML-Dash server to local storage using the command-line interface.

## Overview

The `ml-dash download` command allows you to download experiment data from a remote ML-Dash server to your local machine. This is useful for:

- **Offline analysis**: Download experiments to analyze without internet connectivity
- **Backup and archival**: Create local backups of important experiments
- **Data migration**: Move experiments between environments
- **Collaboration**: Share experiments by downloading and redistributing
- **High-performance downloads**: Chunk-aware parallel downloads (218x faster than sequential)

## Quick Start

### Basic Download

Download all experiments from the remote server:

```bash
ml-dash download ./data --remote https://api.dash.ml --username your-username
```

### Download Specific Project

Download only experiments from a specific project:

```bash
ml-dash download ./data --remote https://api.dash.ml --username your-username --project my-project
```

### Download Specific Experiment

Download a single experiment:

```bash
ml-dash download ./data --remote https://api.dash.ml --username your-username \
  --project my-project --experiment my-experiment
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

### Username-based Authentication (Recommended)

The simplest way to authenticate is using your username. The CLI will automatically generate an API key:

```bash
ml-dash download ./data --remote https://api.dash.ml --username john-doe
```

**How it works:**
- The CLI generates a deterministic JWT token from your username
- Same username always produces the same token
- No need to manage API keys manually
- Token is used only for the current session (not stored)

### API Key Authentication

For advanced users or production environments, you can use an explicit API key:

```bash
ml-dash download ./data --remote https://api.dash.ml --api-key your-jwt-token
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
ml-dash download ./data
```

## Command Reference

### Positional Arguments

- `path` - Local storage directory to download to (default: `./.dash`)

### Remote Configuration

- `--remote URL` - Remote server URL (required unless set in config)
- `--api-key TOKEN` - JWT token for authentication
- `--username USERNAME` - Username for auto-generating API key
- `--namespace NAMESPACE` - Namespace slug (defaults to username)

### Filtering Options

- `--project PROJECT` - Download only this project
- `--experiment EXPERIMENT` - Download specific experiment (requires `--project`)

### Content Selection

- `--skip-logs` - Don't download logs
- `--skip-metrics` - Don't download metrics
- `--skip-files` - Don't download files
- `--skip-params` - Don't download parameters

### Download Behavior

- `--dry-run` - Preview without downloading
- `--overwrite` - Overwrite existing experiments (default: skip existing)
- `--resume` - Resume interrupted download
- `--state-file STATE_FILE` - State file path for resume (default: `.dash-download-state.json`)

### Performance Options

- `--batch-size SIZE` - Batch size for logs/metrics (default: 1000, max: 10000)
- `--max-concurrent-metrics N` - Parallel metric downloads (default: 5)
- `--max-concurrent-files N` - Parallel file downloads (default: 3)

### Output Options

- `-v, --verbose` - Detailed progress output

## Usage Examples

### Example 1: Download All Experiments

Download all experiments to a local directory:

```bash
ml-dash download ./experiments --remote https://api.dash.ml --username tom --verbose
```

**Output:**
```
🔍 Discovering experiments...
Found 3 projects with 12 experiments

📥 Downloading experiments...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 12/12

✅ Download Summary:
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Category      ┃ Count     ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Projects      │ 3         │
│ Experiments   │ 12        │
│ Metrics       │ 48        │
│ Files         │ 156       │
│ Total Size    │ 2.3 GB    │
└───────────────┴───────────┘

Download completed in 3m 45s
```

### Example 2: Download Specific Project

Download only experiments from the "training-runs" project:

```bash
ml-dash download ./training-data \
  --remote https://api.dash.ml \
  --username tom \
  --project training-runs
```

### Example 3: Download Without Large Files

Download experiments but skip large artifact files (parameters and metrics only):

```bash
ml-dash download ./lightweight \
  --remote https://api.dash.ml \
  --username tom \
  --skip-files
```

This is useful for:
- Quick parameter/metric analysis without downloading model checkpoints
- Bandwidth-constrained environments
- CI/CD pipelines that only need metadata

### Example 4: Download Single Experiment

Download a specific experiment by name:

```bash
ml-dash download ./specific \
  --remote https://api.dash.ml \
  --username tom \
  --project vision-models \
  --experiment resnet-50-baseline
```

### Example 5: High-Performance Download

Download with optimized settings for large datasets:

```bash
ml-dash download ./large-dataset \
  --remote https://api.dash.ml \
  --username tom \
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
ml-dash download ./data --remote https://api.dash.ml --username tom --resume

# ... download interrupted ...

# Resume from checkpoint
ml-dash download ./data --remote https://api.dash.ml --username tom --resume
```

The download state is saved to `.dash-download-state.json` and includes:
- Which experiments have been downloaded
- Which metrics/files within each experiment are complete
- Progress tracking for resuming exactly where you left off

## Download Process

The download command follows this workflow:

### 1. Discovery Phase

```
🔍 Discovering experiments...
```

- Connects to remote server
- Authenticates using provided credentials
- Queries available projects and experiments
- Applies filters (project, experiment, namespace)
- Shows summary of what will be downloaded

### 2. Download Phase

```
📥 Downloading experiments...
```

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

```
✅ Download Summary:
```

Shows:
- Number of projects, experiments, metrics, files
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

### Configurable Batch Sizes

Adjust batch size based on your network and data:

```bash
# Small batches for unstable connections
ml-dash download --batch-size 100

# Large batches for high-bandwidth connections
ml-dash download --batch-size 10000
```

## Common Workflows

### Workflow 1: Backup Production Experiments

Create a local backup of all production experiments:

```bash
# Download everything to backup directory
ml-dash download ./backups/$(date +%Y-%m-%d) \
  --remote https://prod.example.com \
  --username production-user \
  --verbose

# Verify download
ls -lh ./backups/$(date +%Y-%m-%d)
```

### Workflow 2: Offline Analysis

Download experiments for offline analysis:

```bash
# Download only metrics and parameters (skip large files)
ml-dash download ./analysis \
  --remote https://dash.example.com \
  --username analyst \
  --project deep-learning \
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
ml-dash download ./migration \
  --remote https://old-server.com \
  --username john

# Step 2: Upload to new server
ml-dash upload ./migration \
  --remote https://new-server.com \
  --username john
```

### Workflow 5: Resume Large Downloads

For very large datasets, use resume to handle interruptions:

```bash
# Start download with resume enabled
ml-dash download ./large-dataset \
  --remote https://dash.example.com \
  --username john \
  --resume \
  --verbose

# If interrupted (network issue, etc.), just run again
# It will automatically continue from where it left off
ml-dash download ./large-dataset \
  --remote https://dash.example.com \
  --username john \
  --resume \
  --verbose
```

## Troubleshooting

### Issue: Slow Downloads

**Symptoms:**
- Downloads taking much longer than expected
- Low network utilization

**Solutions:**

1. **Increase batch size** (up to 10,000):
   ```bash
   ml-dash download --batch-size 10000
   ```

2. **Increase concurrent downloads**:
   ```bash
   ml-dash download --max-concurrent-metrics 10 --max-concurrent-files 5
   ```

3. **Check server performance**:
   ```bash
   curl -w "@curl-format.txt" https://api.dash.ml/health
   ```

### Issue: Download Interrupted

**Symptoms:**
- Network error mid-download
- Partial data downloaded

**Solution:**

Use `--resume` to continue from checkpoint:

```bash
ml-dash download ./data --resume --verbose
```

State file (`.dash-download-state.json`) tracks:
- Completed experiments
- Completed metrics within each experiment
- Completed files

### Issue: Authentication Failed

**Symptoms:**
- "401 Unauthorized" error
- "Invalid token" message

**Solutions:**

1. **Verify username matches server**:
   ```bash
   # Check what users exist on server
   ml-dash list --remote https://api.dash.ml --username test-user
   ```

2. **Use explicit API key**:
   ```bash
   ml-dash download --api-key "eyJhbGc..."
   ```

3. **Check config file** (`~/.dash/config.json`):
   ```json
   {
     "remote_url": "https://api.dash.ml",
     "api_key": "valid-token-here"
   }
   ```

### Issue: Disk Space

**Symptoms:**
- "No space left on device" error
- Download stops mid-transfer

**Solutions:**

1. **Check available space before download**:
   ```bash
   df -h .
   ```

2. **Use `--dry-run` to preview size**:
   ```bash
   ml-dash download --dry-run --verbose
   ```

3. **Download selectively**:
   ```bash
   # Skip large files
   ml-dash download --skip-files

   # Download only specific project
   ml-dash download --project small-project
   ```

### Issue: Overwrite Protection

**Symptoms:**
- Experiments skipped because they already exist
- "Experiment already exists, skipping" message

**Solution:**

Use `--overwrite` to replace existing experiments:

```bash
ml-dash download ./data --overwrite
```

**Warning:** This will completely replace existing data. Make backups first!

## Advanced Usage

### Custom State File Location

Specify a custom state file for resume functionality:

```bash
ml-dash download ./data \
  --resume \
  --state-file ~/my-downloads/custom-state.json
```

Useful for:
- Multiple parallel downloads to different directories
- Shared network drives with custom paths
- CI/CD pipelines with specific state file locations

### Dry Run with Verbose Output

Preview exactly what will be downloaded:

```bash
ml-dash download ./data --dry-run --verbose
```

**Output includes:**
- List of projects and experiments to download
- Number of metrics, files, and logs per experiment
- Estimated total size
- Download order and strategy

### JSON Output for Scripting

Combine with `ml-dash list --json` for automated workflows:

```bash
# Get experiment list as JSON
ml-dash list --project my-project --json > experiments.json

# Process with jq
cat experiments.json | jq '.experiments[] | select(.status == "COMPLETED") | .name'

# Download completed experiments
for exp in $(cat experiments.json | jq -r '.experiments[] | select(.status == "COMPLETED") | .name'); do
  ml-dash download ./completed \
    --project my-project \
    --experiment "$exp"
done
```

## Storage Format

Downloaded experiments use the same local storage format as experiments logged in local mode:

```
{path}/
└── {project}/
    └── {experiment-name}/
        ├── parameters.json         # Hyperparameters and metadata
        ├── logs/
        │   └── logs.jsonl         # Event logs (JSONL format)
        ├── metrics/
        │   └── {metric-name}/
        │       └── data.jsonl     # Time-series data (JSONL format)
        └── files/
            └── {prefix}/
                └── {file-id}/
                    └── {filename}  # Original filename preserved
```

**File Formats:**

- **JSONL (JSON Lines)**: Each line is a valid JSON object
- **Metrics**: `{"index": 0, "data": {"loss": 0.5, "acc": 0.9}, "createdAt": "..."}`
- **Logs**: `{"level": "INFO", "message": "...", "timestamp": "..."}`
- **Parameters**: Single JSON object with all parameters

This format is:
- Human-readable (can inspect with text editor)
- Line-by-line processable (stream processing)
- Git-friendly (meaningful diffs)
- Language-agnostic (any tool can parse JSON)

## Best Practices

### 1. Use Dry Run First

Always preview before downloading large datasets:

```bash
ml-dash download --dry-run --verbose
```

### 2. Enable Resume for Large Downloads

For downloads over 1GB or unreliable networks:

```bash
ml-dash download --resume
```

### 3. Optimize Concurrency

Tune based on your network and server capacity:

```bash
# High-bandwidth, powerful server
ml-dash download --max-concurrent-metrics 10 --max-concurrent-files 5

# Low-bandwidth or rate-limited server
ml-dash download --max-concurrent-metrics 2 --max-concurrent-files 1
```

### 4. Skip Unnecessary Data

Save bandwidth and time:

```bash
# Analysis only: skip large model files
ml-dash download --skip-files

# Debugging: skip everything except logs
ml-dash download --skip-metrics --skip-files --skip-params
```

### 5. Regular Backups

Automate regular backups with cron:

```bash
# Add to crontab (daily backup at 2 AM)
0 2 * * * ml-dash download ~/backups/$(date +\%Y-\%m-\%d) --remote https://dash.example.com --username backup-user
```

### 6. Verify Downloads

After downloading, verify data integrity:

```bash
# Check experiment structure
tree .dash/my-project/my-experiment

# Validate JSONL files
cat .dash/my-project/my-experiment/metrics/loss/data.jsonl | jq '.'

# Check file count
find .dash -type f | wc -l
```

## Next Steps

- [Upload Command](cli-upload.md) - Upload experiments to remote server
- [List Command](cli-list.md) - Discover available experiments
- [CLI Overview](cli.md) - General CLI documentation
- [Getting Started](getting-started.md) - Learn ML-Dash fundamentals
