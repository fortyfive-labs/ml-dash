# ml-dash Examples

Comprehensive examples covering every feature of the `ml-dash` SDK.

## Files

| File | What it covers |
|------|----------------|
| `01_quickstart.py` | Three usage styles: context manager, decorator, manual start/stop |
| `02_metrics.py` | All metric patterns: named metrics, nested-dict, buffer+log_summary, SummaryCache, read, stats |
| `03_params.py` | Hyperparameter storage: flat, nested, dataclass, read-back |
| `04_logs.py` | Structured logging: info, warn, error, debug, fatal, extra metadata |
| `05_files.py` | File management: upload, save_text/json/blob/image/fig/torch/pkl/video, list, download, delete |
| `06_tracks.py` | Time-stamped track data: append, flush, read (JSON/mocap), slice, list_entries |
| `07_experiment_modes.py` | Local, remote-only, hybrid, default remote, tags/readme/metadata, env vars |
| `08_advanced_patterns.py` | Full training loop, SummaryCache, buffer stats, run states, flush, file duplicate |
| `09_auto_start_singleton.py` | `dxp` singleton, `RUN` global config object |

## Quick start

```bash
# Local mode — no server needed
python 01_quickstart.py

# Remote mode — requires `ml-dash login` first
# ml-dash login
python 09_auto_start_singleton.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_DASH_PREFIX` | — | Full experiment prefix (`owner/project/name`) |
| `ML_DASH_ROOT` | `.dash` | Local storage directory |
| `ML_DASH_URL` | — | Remote server URL |
| `ML_DASH_USER` | `$USER` | Owner / namespace |
| `ML_DASH_BUFFER_ENABLED` | `true` | Enable background buffering |
