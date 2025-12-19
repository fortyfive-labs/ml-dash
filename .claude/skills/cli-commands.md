---
description: ML-Dash CLI commands for authentication, uploading, downloading, and listing experiments
globs:
  - "**/*.sh"
  - "**/*.bash"
  - "**/Makefile"
  - "**/.github/**"
keywords:
  - ml-dash
  - cli
  - command line
  - login
  - logout
  - upload
  - download
  - list
  - authentication
  - OAuth
  - api-key
  - sync
  - batch
---

# ML-Dash CLI Commands

## Installation Verification

```bash
ml-dash --help
```

## Authentication

### Login (One-time Setup)
```bash
ml-dash login
```
Opens browser for OAuth2 authentication. Token stored in system keychain.

### Custom Server
```bash
ml-dash login --remote https://your-server.com
```

### Logout
```bash
ml-dash logout
```

### API Key (CI/CD)
```bash
# Environment variable
export ML_DASH_API_KEY="your-jwt-token"
ml-dash upload

# Or command line
ml-dash upload --api-key your-jwt-token

# Or config file ~/.ml-dash/config.json
{"remote_url": "https://api.dash.ml", "api_key": "your-jwt-token"}
```

---

## Upload Command

Upload local experiments to remote server.

```bash
# Upload all from default .ml-dash directory
ml-dash upload

# Upload from specific directory
ml-dash upload ./experiments

# Filter by project
ml-dash upload --project my-project

# Filter by experiment
ml-dash upload --project my-project --experiment specific-run

# Dry run (preview only)
ml-dash upload --dry-run --verbose

# Resume interrupted upload
ml-dash upload --resume
```

### Key Options
- `--project`: Filter by project name
- `--experiment`: Filter by experiment name
- `--dry-run`: Preview without uploading
- `--resume`: Continue from last state
- `-v, --verbose`: Detailed output

---

## Download Command

Download experiments from remote server.

```bash
# Download to directory
ml-dash download ./data

# Download specific project
ml-dash download ./data --project my-project

# Download specific experiment
ml-dash download ./data --project my-project --experiment run-123

# Skip certain data types
ml-dash download ./data --skip-files
ml-dash download ./data --skip-logs --skip-files --skip-params

# Dry run
ml-dash download ./data --dry-run --verbose

# Resume interrupted download
ml-dash download ./data --resume
```

### Performance Options
```bash
# Adjust parallelism
ml-dash download --max-concurrent-metrics 10 --max-concurrent-files 5

# Increase batch size (up to 10,000)
ml-dash download --batch-size 10000
```

---

## List Command

Discover projects and experiments on remote server.

```bash
# List all projects
ml-dash list

# List experiments in project
ml-dash list --project vision-models

# Filter by status
ml-dash list --project ml --status COMPLETED
ml-dash list --project ml --status RUNNING
ml-dash list --project ml --status FAILED

# Filter by tags
ml-dash list --project ml --tags baseline

# Detailed output
ml-dash list --project ml --detailed

# JSON output (for scripting)
ml-dash list --project ml --json
```

---

## Common Workflows

### Offline Development -> Upload
```bash
ml-dash login                # One-time auth
python train.py              # Run experiments locally
ml-dash upload               # Sync when internet available
```

### Download -> Analyze -> Re-upload
```bash
ml-dash download ./analysis --project training-runs
python analyze.py            # Add metrics/files
ml-dash upload ./analysis
```

### Backup and Migration
```bash
# Backup from production
ml-dash login --remote https://prod.example.com
ml-dash download ./backup

# Restore to development
ml-dash login --remote https://api.dash.ml
ml-dash upload ./backup
```

### Discovery and Selective Download
```bash
ml-dash list                                    # See all projects
ml-dash list --project vision-models            # See experiments
ml-dash download ./data --project vision-models --experiment resnet-50
```

---

## Global Options

All commands support:
- `--remote URL`: Remote server URL (default: https://api.dash.ml)
- `--api-key TOKEN`: JWT auth token
- `-v, --verbose`: Detailed output
- `--help`: Command help

---

## Troubleshooting

### Command Not Found
```bash
python -m ml_dash.cli --help
# or with uv
uv run python -m ml_dash.cli --help
```

### Authentication Errors
```bash
ml-dash logout && ml-dash login
```

### Resume Not Working
Check state files exist:
- `.ml-dash-upload-state.json`
- `.ml-dash-download-state.json`
