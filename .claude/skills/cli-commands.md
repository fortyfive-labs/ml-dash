---
description: ML-Dash CLI commands for authentication, creating projects, uploading, downloading, and listing experiments
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
  - create
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
ml-dash login --dash-url https://your-server.com
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

# Or config file ~/.dash/config.json
{"remote_url": "https://api.dash.ml", "api_key": "your-jwt-token"}
```

---

## Create Command

Create projects on the remote server.

```bash
# Create a project in current user's namespace
ml-dash create -p new-project

# Create a project in a specific namespace
ml-dash create -p geyang/new-project

# Create with description
ml-dash create -p geyang/tutorials -d "ML tutorials and examples"
```

---

## Upload Command

Upload local experiments to remote server.

```bash
# Upload all from default .dash directory
ml-dash upload

# Upload from specific directory
ml-dash upload ./experiments

# Filter by project or pattern (supports glob patterns)
ml-dash upload -p my-project
ml-dash upload -p "test*"
ml-dash upload -p "alice/*/baseline-*"

# Upload to different location on server
ml-dash upload -p "alice/experiments/*" -t "shared/team-experiments"

# Dry run (preview only)
ml-dash upload --dry-run --verbose

# Resume interrupted upload
ml-dash upload --resume
```

### Key Options
- `-p, --pref, --prefix, --proj, --project`: Filter by pattern (supports globs)
- `-t, --target`: Target prefix on server
- `--dry-run`: Preview without uploading
- `--resume`: Continue from last state
- `-v, --verbose`: Detailed output

---

## Download Command

Download experiments from remote server.

```bash
# Download to directory
ml-dash download ./data

# Download by project or pattern (supports glob patterns)
ml-dash download ./data -p my-project
ml-dash download ./data -p "test*"
ml-dash download ./data -p "*/deep-learning/baseline-*"

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
# List all experiments
ml-dash list

# List by project or pattern (supports glob patterns)
ml-dash list -p vision-models
ml-dash list -p "test*"
ml-dash list -p "*/deep-learning/*"

# Filter by status
ml-dash list -p ml --status COMPLETED
ml-dash list -p ml --status RUNNING
ml-dash list -p ml --status FAILED

# Filter by tags
ml-dash list -p ml --tags baseline

# Detailed output
ml-dash list -p ml --detailed

# JSON output (for scripting)
ml-dash list -p ml --json
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
ml-dash download ./analysis -p training-runs
python analyze.py            # Add metrics/files
ml-dash upload ./analysis
```

### Backup and Migration
```bash
# Backup from production
ml-dash login --dash-url https://prod.example.com
ml-dash download ./backup

# Restore to development
ml-dash login --dash-url https://api.dash.ml
ml-dash upload ./backup
```

### Discovery and Selective Download
```bash
ml-dash list                              # See all experiments
ml-dash list -p vision-models             # See specific project
ml-dash download ./data -p "vision-models/resnet-50"
```

### Create Project Then Run Experiment
```bash
# Create project first
ml-dash create -p geyang/new-research

# Then run experiments
python train.py  # Uses prefix="geyang/new-research/exp-name"
```

---

## Global Options

All commands support:
- `--dash-url URL`: Remote server URL (default: https://api.dash.ml)
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
- `.dash-upload-state.json`
- `.dash-download-state.json`
