# Command-Line Interface (CLI)

ML-Dash provides a powerful command-line interface for managing experiments between local and remote storage. The CLI enables offline experimentation, batch operations, and seamless data synchronization.

## Overview

The ML-Dash CLI includes four main commands:

- **`login`** - Authenticate with ML-Dash server using OAuth2 device flow
- **`logout`** - Clear stored authentication credentials
- **`upload`** - Upload local experiments to a remote server
- **`download`** - Download experiments from a remote server to local storage
- **`list`** - Discover and browse projects and experiments on a remote server

## Installation

The CLI is included with the ML-Dash Python package:

```bash
pip install ml-dash
# or
uv pip install ml-dash
```

Verify installation:

```bash
ml-dash --help
```

## Quick Start

### 1. Authenticate

Before using CLI commands, authenticate with the ML-Dash server:

```bash
ml-dash login
```

This opens your browser for secure OAuth2 authentication. Your credentials are stored securely in your system keychain.

### 2. Upload Experiments

Upload all local experiments to the remote server:

```bash
ml-dash upload
```

### 3. Download Experiments

Download experiments from the remote server:

```bash
ml-dash download ./data --project my-project
```

### 4. List Available Experiments

Discover what's available on the remote server:

```bash
ml-dash list
```

## Authentication

ML-Dash uses secure OAuth2 device flow for authentication.

### Login (One-time Setup)

```bash
ml-dash login
```

This command:
1. Opens your browser for authentication
2. Displays a verification code if browser doesn't open
3. Stores your authentication token securely in your system keychain
4. Sets the default remote URL to https://api.dash.ml

**Custom Server:**

```bash
ml-dash login --dash-url https://your-server.com
```

### Logout

Clear stored credentials:

```bash
ml-dash logout
```

### Alternative: API Key Authentication

For CI/CD or when you have an explicit API key:

```bash
ml-dash upload --api-key your-jwt-token
```

Or store it in `~/.dash/config.json`:

```json
{
  "remote_url": "https://api.dash.ml",
  "api_key": "your-jwt-token-here"
}
```

## Common Workflows

### Offline Development → Upload

Work offline, then sync when ready:

```bash
# 1. Authenticate once (stores credentials)
ml-dash login

# 2. Run experiments locally (no internet needed)
python train.py  # Uses local ML-Dash storage

# 3. Upload when internet is available
ml-dash upload
```

### Download → Analyze → Re-upload

Download experiments, run additional analysis, then upload results:

```bash
# 1. Download existing experiments
ml-dash download ./analysis --project training-runs

# 2. Run analysis (adds metrics/files to experiments)
python analyze.py

# 3. Upload updated experiments
ml-dash upload ./analysis
```

### Backup and Migration

Backup experiments or migrate between servers:

```bash
# Backup from production
ml-dash login --dash-url https://prod.example.com
ml-dash download ./backup

# Restore to development
ml-dash login --dash-url https://api.dash.ml
ml-dash upload ./backup
```

### Discovery and Selective Download

Find what you need, then download it:

```bash
# 1. List all projects
ml-dash list

# 2. List experiments in a project
ml-dash list --project vision-models

# 3. Download specific experiment
ml-dash download ./data --project vision-models --experiment resnet-50-baseline
```

## Performance Features

### Parallel Downloads

The download command uses parallel processing for optimal performance:

- **Chunk-aware metric downloads**: Downloads entire metric chunks in parallel (218x faster than sequential)
- **Concurrent file downloads**: Multiple files downloaded simultaneously
- **Configurable workers**: Adjust parallelism based on your network

Example with custom parallelism:

```bash
ml-dash download ./data --max-concurrent-metrics 10 --max-concurrent-files 5
```

### Resume Interrupted Transfers

Both upload and download support resuming interrupted operations:

```bash
# Upload with resume
ml-dash upload --resume

# Download with resume
ml-dash download ./data --resume
```

State is automatically saved to `.dash-upload-state.json` or `.dash-download-state.json`.

## Command Reference

### [`ml-dash upload`](cli-upload.md)

Upload local experiments to a remote server.

**Key Features:**
- Batch uploads with progress tracking
- Resume interrupted uploads
- Dry-run mode for preview
- Selective upload (filter by project/experiment)
- Data validation (lenient/strict modes)

[View full upload documentation →](cli-upload.md)

### [`ml-dash download`](cli-download.md)

Download experiments from a remote server to local storage.

**Key Features:**
- High-performance chunk-aware downloads (218x faster)
- Parallel metric and file downloads
- Resume interrupted downloads
- Selective download (skip logs/metrics/files/params)
- Dry-run mode for preview

[View full download documentation →](cli-download.md)

### [`ml-dash list`](cli-list.md)

Discover and browse projects and experiments on a remote server.

**Key Features:**
- List all projects in your namespace
- List experiments in a specific project
- Filter by status (COMPLETED, RUNNING, FAILED, ARCHIVED)
- Filter by tags
- JSON output for scripting
- Detailed mode for comprehensive information

[View full list documentation →](cli-list.md)

## Global Options

All commands support these common options:

- `--dash-url URL` - Remote server URL (defaults to https://api.dash.ml)
- `--api-key TOKEN` - JWT authentication token (optional if logged in)
- `-v, --verbose` - Detailed progress output
- `--help` - Show command-specific help

## Tips and Best Practices

### 1. Authenticate Once

Run `ml-dash login` once to authenticate. Credentials are securely stored and reused:

```bash
ml-dash login
```

### 2. Use API Keys for CI/CD

For automated environments, use API keys instead of interactive login:

```bash
# Set environment variable
export ML_DASH_API_KEY="your-jwt-token"
ml-dash upload

# Or use config file
echo '{"api_key": "your-jwt-token"}' > ~/.dash/config.json
```

### 3. Use Dry-Run Before Large Operations

Preview what will happen before uploading or downloading:

```bash
ml-dash upload --dry-run --verbose
ml-dash download ./data --dry-run --verbose
```

### 4. Resume Long Transfers

For large uploads/downloads, use resume to recover from interruptions:

```bash
# Start upload
ml-dash upload --resume

# If interrupted, run again with --resume to continue
ml-dash upload --resume
```

### 5. Use List to Explore Before Downloading

Discover what's available before downloading:

```bash
# See all projects
ml-dash list

# See experiments in a project
ml-dash list --project my-project

# Get detailed info
ml-dash list --project my-project --detailed
```

### 6. Selective Downloads to Save Bandwidth

Skip unnecessary data types:

```bash
# Download only parameters and metrics (skip large files)
ml-dash download ./data --skip-files

# Download only metrics for analysis
ml-dash download ./data --skip-logs --skip-files --skip-params
```

## Troubleshooting

### Command Not Found

If `ml-dash` command is not found after installation:

```bash
# Try running via Python module
python -m ml_dash.cli --help

# Or with uv
uv run python -m ml_dash.cli --help
```

### Authentication Errors

If you get authentication errors:

1. Try logging in again: `ml-dash login`
2. Check that the remote URL is correct
3. Ensure the server is running and accessible
4. Verify your authentication token hasn't expired

```bash
# Check if you're logged in
ml-dash list

# Re-authenticate if needed
ml-dash logout && ml-dash login
```

### Slow Uploads/Downloads

For slow transfers:

1. Increase batch size (up to 10,000):
   ```bash
   ml-dash download --batch-size 10000
   ```

2. Adjust concurrent workers:
   ```bash
   ml-dash download --max-concurrent-metrics 10 --max-concurrent-files 5
   ```

3. Check network connectivity and server performance

### Resume Not Working

If resume doesn't work:

1. Check that the state file exists (`.dash-upload-state.json` or `.dash-download-state.json`)
2. Ensure you're running from the same directory
3. Use `--state-file` to specify a custom state file path

## Next Steps

- [Upload Command Reference](cli-upload.md) - Detailed upload documentation
- [Download Command Reference](cli-download.md) - Detailed download documentation
- [List Command Reference](cli-list.md) - Detailed list documentation
- [Getting Started Guide](getting-started.md) - Learn about ML-Dash fundamentals
