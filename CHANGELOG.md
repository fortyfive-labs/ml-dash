# Changelog

## 0.9.0

- Add explicit DreamLake Jobs application lifecycle reporting bound to an exact job and execution receipt.
- Persist immutable lifecycle events and acknowledgments in a private durable journal for retry and process-restart recovery.
- Keep application outcomes separate from process exit and Dash experiment status; no outcome is inferred after a crash.

All notable changes to ML-Dash will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Native metric names follow ml-dash's rule (requires a DreamLake server with the matching native routes): any name of 1-256 characters without `/` (not `.`/`..`), sent percent-encoded as one path segment, so names such as `训练` or `training loss` work as they did with ml-dash. The SDK previously refused anything outside `[A-Za-z0-9_.-]{1,128}`.
- Native writers (workload credentials, requires a DreamLake server listing `workload-credentials`): `NativeConnection` writes through a writer as `ManagedConnection` does. It registers one per client and renews by id, or uses `idempotency_key`, `workload_id` (CI with an explicit `DREAMLAKE_API_KEY`) or `run_id` (only that run, opened by id, never by key). A retired writer is an error and never replaced. Add `list_credentials`, `get_credential`, `create_credential`, `rotate_credential` and `retire_credential` (metadata only, with your login).
- Native run paths, metadata and protection (requires a DreamLake server with native run paths and protected creation): `Experiment(prefix="ns/project/folders.../name")` opens the run at that path, creating its folders, with Unicode names; `metadata=` is stored with the run; `_write_protected=True` creates it protected, with its metadata, in one server transaction, as a person's action over a plain session (an explicit `workload_id`/`idempotency_key`/`run_id` writer is refused before any request). bindrs stay unsupported.

## [0.14.0] - 2026-11-05

- Tracks on native runs, whichever store holds the run (requires a DreamLake server with native tracks): `exp.tracks(topic)` appends (single and batched `flush()`), `read()`/`list_entries()` with inclusive `start_timestamp`/`end_timestamp` and `columns`, `slice()`/`findByTime()`, and the client's `create_track`, `list_tracks` (all, an exact topic, or `prefix/*`) and `get_track_data`. Entries are flattened to dot keys (arrays kept) and returned in timestamp order. Formats: `json` (dict), and as bytes `jsonl`, `parquet` (an Arrow IPC file, as Dash: `timestamp` first, objects and arrays as JSON text; a column mixing value types is refused) and `mcap` (one topic, or every topic under a prefix; camera topics embed the run's images). A numeric `timestamp` field in an entry's data is that entry's timestamp; a non-numeric one is refused.
- Platform jobs on DreamLake-native runs (`Experiment()` inside a job, through the local job broker) can now use the native run data methods the job capability permits: tracks (`exp.tracks(...)` append/read in json, jsonl, parquet and mcap, `create_track`, `list_tracks`), metric descriptors, `stats()`, `list_metrics`, `read_rows()` (typed cells, paged past 1000), `update_metric`, `query_logs`, `delete_parameters`, and file `get_file`/`update()`/`delete()`. They go only through the job's local broker session (`NativeRemoteClient`) and only for the job's run; another run is refused before any request. Run creation, status, metadata and credentials stay platform- or human-controlled; managed (non-native) jobs keep their previous refusals.
- Not yet on native runs: nested folders, bindrs, run metadata and write protection. The existing `Experiment` and Dash client interfaces are unchanged.

## [0.13.0] - 2026-11-05

- Add `NativeConnection` for interactive DreamLake-native runs: `Experiment(prefix="ns/project/run", managed=NativeConnection.from_login("ns", "project"))` uses the existing DreamLake login to mint short-lived project write sessions and writes only to the server's `/native-experiments` routes. It creates or reopens a run, sets or clears its description and tags (`readme=""`, `tags=[]` clear), writes parameters, logs and metrics, sets the final status, and uploads, lists and downloads files (sha256-verified).
- Native run data parity with the Dash API (requires a DreamLake server with native run-data edits): a metric's description, tags and metadata are stored with its first points in one request (later appends never change them; `update_metric` edits them); `stats()`, `list_all()`, and `read_rows()`/`iter_row_blocks()` with the canonical metric id, limit 1-10000, an exact cursor and typed cells (nested values are refused, `unsupported_cell`); file `update()` and `delete()`; `query_logs()` filters by level, time range, substring and order; `delete_parameters()`.
- Not yet on native runs: tracks (`NATIVE_UNSUPPORTED`), nested folders, bindrs, run metadata and write protection.

## [0.12.0] - 2026-11-01

- Route Nymph job experiment logging to the DreamLake-native project API when the broker selects the native provider, while retaining the existing Dash job destination for older tasks.
- Support native job parameters, logs, metrics, and file upload and download through the pinned run session. Production compute execution still requires an explicitly configured launch server and remains unverified.

## [0.11.0] - 2026-10-16

- Add memory-only worker broker logging for managed compute tasks. `Experiment()` reuses the platform run without user login or local fallback.
- Pin destination, run, workload and generation across renewals; validate exact run links and keep task imports offline.
- SDK scope close flushes data; only the control plane execution receipt controls task completion/cancellation.
- Requires matching BFF, Dash, control plane and opt-in Nymph releases. Production compute acceptance is deferred; existing workers are not upgraded automatically.

## [0.10.0] - 2026-10-16

### Added
- `ManagedConnection.from_login(namespace, project)` reuses the selected DreamLake
  login for `Experiment(..., managed=connection)`, with project-scoped, short-lived
  reporting credentials renewed in memory. Matching managed server releases are
  required. Independent Dash credentials are never a fallback.
- Managed runs support parameters, buffered metrics, logs, status, committed metric
  reads and root-level file uploads/downloads up to 8 MiB. Run links use the
  server-verified DreamLake page URL. Nested folders and run/metric/file metadata
  operations are explicitly unsupported in managed mode; standalone behavior is
  unchanged.
- Managed setup retains the reviewed operation identity across interrupted calls.
  Credential renewal pins the service instance, project and endpoint, and rejects
  a changed destination. Revoked DreamLake grants prevent renewal; already-issued
  capabilities expire within 65 seconds including allowed clock skew.

## [0.8.0] - 2026-09-28

Remote reads of committed metric rows need an ML-Dash server that serves
`GET /api/experiments/:expId/metrics/:metricName/rows`. Without that route,
`read_rows()` fails with `MetricRowsError` (404).

### ⚠️ BREAKING CHANGES
- **Index-based remote metric reads now raise `ConfigurationError`** without
  contacting the server. The server no longer assigns point indices or chunk
  numbers, so these reads cannot be mapped to stored rows. This affects
  `exp.metrics(name).read()` on a remote-only experiment and
  `RemoteClient.read_metric_data()`, `get_metric_data()` and
  `download_metric_chunk()`.
  - **Migration**: read committed rows by cursor with
    `exp.metrics(name).read_rows()` / `iter_row_blocks()` or
    `RemoteClient.read_metric_rows()`. `read()` on local and hybrid
    experiments works as before.
- **Remote metrics reject NaN and inf** when they are logged, with a
  `ValueError`, instead of failing later in a background flush. Local storage
  still accepts them, and `buffer()` / `summary_cache.store()` still take NaN.

### Added
- `MetricBuilder.read_rows(limit, cursor)` and `iter_row_blocks(limit)` read a
  metric's committed rows in storage order, in blocks with their own typed
  columns (`ml-dash.metric-rows.v1`). New public types: `MetricRowsPage`,
  `MetricRowBlock`, `MetricColumn`, and `MetricRowsError` (a `NetworkError`
  with `status_code` and the server's `code`, e.g. `snapshot_changed`).
- `ML_DASH_NO_VERSION_CHECK=1` skips the PyPI version check at import.

### Changed
- The SDK reads tokens from every store the CLI writes, in the CLI's order:
  the OS keyring, then `tokens.encrypted`, then `tokens.json`. An unreadable
  store raises `StorageError`, and the message names the file and error type
  only, never the stored contents.
- Buffered sends are retried on transport errors, 429 and 5xx, honouring
  `Retry-After`. Batches that could not be sent are kept until they are sent
  instead of being dropped. `flush()` and closing the experiment raise
  `NetworkError` when data is still unsent or was lost.
- If buffered data cannot all be sent, closing the experiment records it as
  `FAILED` instead of `COMPLETED`.
- An exception raised inside `with exp.run:` is no longer replaced by an error
  from marking the experiment failed. That error is logged, and attached as a
  note on Python 3.11+.
- Closing an experiment logs a warning for values stored in `summary_cache`
  that no summary included.
- The import-time version check can no longer break the import, whatever the
  failure.

## [0.7.0] - 2026-09-24

### ⚠️ BREAKING CHANGES
- **The `ml-dash` command line is no longer part of this package.** It has been
  rewritten in TypeScript and ships separately; from 0.7.0, `pip install ml-dash`
  installs the SDK only and no longer puts an `ml-dash` executable on your PATH.
  0.6.27 and every earlier release still carry the old Python CLI.
  - **Migration**: install the CLI on its own. The npm package is
    `@dreamlake/ml-dash` (the command it installs is still `ml-dash`):

    ```bash
    npm install -g @dreamlake/ml-dash    # needs Node.js >= 20.19
    ```

    or, with neither Node.js nor Python, the standalone installer:

    ```bash
    curl -fsSL https://pub-42e1dcc7de574d4a92984865fdc95f10.r2.dev/install.sh | sh   # macOS / Linux
    irm https://pub-42e1dcc7de574d4a92984865fdc95f10.r2.dev/install.ps1 | iex        # Windows PowerShell
    ```

    On Alpine the musl builds also need `apk add --no-cache libstdc++`.
    Upgrading `ml-dash` with pip removes the old executable, so install the new
    CLI before (or right after) upgrading if scripts call `ml-dash`.
  - Command names and arguments carry over, and the CLI writes its token to the
    same keyring entry and `~/.dash/` files the SDK reads, so a valid login is
    normally reused — re-run `ml-dash login` if it is not. Per-command
    behaviour is not identical; see the CLI's own release notes.
  - Removed the `ml_dash.cli` module and the `ml_dash.cli_commands` package.
    Code that imported them directly must shell out to the binary instead.
  - Dropped the `qrcode` dependency from the `auth` extra; only the CLI's login
    command drew a QR code. `keyring` stays — the SDK reads tokens through it.

### Changed
- The `dev` extra now declares `pyjwt` and `numpy`, which the test suite
  imports directly.

### Unchanged
- The SDK itself: `Experiment`, `params`, `metrics`, `log`, `files`, `tracks`,
  `storage`, and `ml_dash.auth` are all untouched, as is remote/local mode.

## [0.6.10] - 2026-01-26

### ⚠️ BREAKING CHANGES
- **Removed `get_jwt_user()` function**: This function returned outdated cached data. Use `userinfo` singleton instead.
  - **Migration**: Replace `from ml_dash.auth.token_storage import get_jwt_user` with `from ml_dash import userinfo`
  - **Old**: `user = get_jwt_user(); username = user["username"] if user else None`
  - **New**: `username = userinfo.username`
- **Hard version enforcement**: ml-dash now blocks versions older than 0.6.10 on import
  - Users with old versions will see clear error message with upgrade instructions
  - Ensures all users have latest features (userinfo, namespace auto-detection)

### Added
- **`userinfo` Singleton**: Fresh user data from API server
  - `from ml_dash import userinfo` - Lazy-loaded singleton
  - `userinfo.username` - Namespace (e.g., "tom_tao_e4c2c9")
  - `userinfo.email` - User email
  - `userinfo.name` - Full name
  - `userinfo.given_name`, `userinfo.family_name` - Name components
  - Fetches fresh data from server (not outdated JWT token)
  - Automatic lazy loading on first access

- **`RemoteClient.get_current_user()` Method**: API method to fetch fresh user profile
  - Queries GraphQL `me` endpoint for current user data
  - Used internally by `userinfo` singleton

- **Namespace Auto-Detection**: Automatic namespace resolution from authenticated user
  - `dxp` singleton now uses `userinfo.username` for namespace
  - Examples show namespace auto-detection in action
  - Easy override via CLI: `--config.namespace zehuaw`

- **Profile Command Enhancement**: Now fetches fresh data by default
  - **Changed default behavior**: Fetches from server by default (was: cached token)
  - New `--cached` flag to use cached token data (was: `--refresh` to fetch fresh)
  - Shows data source: "Server (Fresh)" vs "Token (Cached)"
  - Uses new `get_current_user()` method internally

### Changed
- **auto_start.py**: Now uses `userinfo.username` instead of `get_jwt_user()`
  - `dxp` singleton has fresh namespace on every import
  - Falls back to system username if userinfo fails

- **Profile command**: Reversed flag logic
  - **Old**: Default=cached, `--refresh` for fresh data
  - **New**: Default=fresh data, `--cached` for cached
  - Aligns with deprecation of cached token usage

### Removed
- **`get_jwt_user()` function** (`auth/token_storage.py`)
  - Completely removed (no deprecation period)
  - Replaced by `userinfo` singleton
  - All internal code updated to use `userinfo`

### Documentation
- **NAMESPACE_EXAMPLES.md**: Comprehensive guide for running experiments with different namespaces
  - Examples: Personal, team collaboration, shared projects, multi-user comparison
  - Use case: Running experiments for user "zehuaw"
  - Best practices and troubleshooting

- **Updated sweep examples**:
  - `train.py`: Shows namespace auto-detection and manual override
  - `run_sweep.py`: Added `--namespace` and `--project` CLI arguments
  - `README.md`: Enhanced with namespace examples
  - `QUICK_START.md`: Practical examples with zehuaw namespace

### Examples
- **Namespace override examples**:
  ```bash
  # Auto-detect namespace
  python3 train.py

  # Run for another user
  python3 train.py --config.namespace zehuaw

  # Full sweep for another user
  python3 run_sweep.py --namespace zehuaw --project research-2026
  ```

### Migration Guide

**If you were using `get_jwt_user()`:**

```python
# ❌ Old (no longer works)
from ml_dash.auth.token_storage import get_jwt_user
user = get_jwt_user()
username = user["username"] if user else None

# ✅ New (required)
from ml_dash import userinfo
username = userinfo.username  # Fresh from server
email = userinfo.email
```

**If you're on an old version:**

```bash
# You'll see this error on import:
# ImportError: ml-dash version 0.6.4 is too old!

# Solution: Upgrade immediately
pip install --upgrade ml-dash
```

## [0.6.9] - 2026-01-24

### Fixed
- **Profile command**: Now fetches fresh user data from server with `--refresh` flag
  - Fixed issue where profile showed outdated information from cached JWT token
  - Added token expiration checking and warnings
  - Shows data source indicator (cached token vs fresh server)
  - Better error messages for expired tokens

### Added
- **Profile command**: `--refresh` flag to fetch latest data from API server
  - `ml-dash profile` - Fast, shows cached token data
  - `ml-dash profile --refresh` - Accurate, fetches from server

### Documentation
- Added comprehensive guides for v0.6.7 features
- Created examples for robotics, vision, and RL use cases
- Reorganized documentation structure

## [0.6.8] - 2026-01-23

### Documentation
- Added Claude Code plugin and skills for ML-Dash
- Added marketplace.json for plugin distribution
- Enhanced tracks documentation with realistic examples

## [0.6.7] - 2026-01-23

### Added
- **Unified Background Buffering System**: Non-blocking I/O operations with automatic batching
  - Background thread orchestrates all buffering with time-based (5s) and size-based (100 items) flush triggers
  - Resource-specific queues for logs, metrics, tracks, and files
  - Graceful error handling without crashing training
  - Progress messages show flush status after training
  - Configurable via environment variables

- **Track API**: New time-series data tracking
  - `experiment.track("topic").append(data)` for tracking time-series data
  - Timestamp-based entry merging for efficient storage
  - Integrated with buffering system for non-blocking writes
  - Perfect for robotics, RL, and sequential data

- **Numpy Image Support**: Save numpy arrays as images
  - `experiment.files("path").save_image(array, to="file.png")` method
  - Auto-detection in `save()` method
  - PNG and JPEG format support with quality control
  - Automatic RGBA to RGB conversion for JPEG
  - Handles uint8 and normalized float arrays

- **Enhanced File Operations**: All save methods work seamlessly with buffering
  - `save_json()`, `save_torch()`, `save_pickle()`, `save_fig()`, `save_video()`
  - Proper temp file cleanup with buffered uploads
  - Parallel file uploads via ThreadPoolExecutor

### Changed
- Experiment lifecycle now waits indefinitely for all data to flush on close (no timeout limits)
- Buffer manager automatically cleans up temp files after upload
- All write operations (`log()`, `metric()`, `track()`, `files()`) are now non-blocking by default

### Configuration
- `ML_DASH_BUFFER_ENABLED`: Enable/disable buffering (default: true)
- `ML_DASH_FLUSH_INTERVAL`: Flush interval in seconds (default: 5.0)
- `ML_DASH_LOG_BATCH_SIZE`: Logs per batch (default: 100)
- `ML_DASH_METRIC_BATCH_SIZE`: Metric points per batch (default: 100)
- `ML_DASH_TRACK_BATCH_SIZE`: Track entries per batch (default: 100)

## [0.6.6] - 2026-01-05

### Changed
- Updated documentation structure
- Moved legacy docs to archived folder

## [0.6.2rc1] - 2025-12-XX

### Added
- Initial release with experiment tracking
- OAuth2 authentication
- Remote and local storage modes
- File upload/download
- Parameters, metrics, and logging
- CLI commands

[0.6.10]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.9...v0.6.10
[0.6.9]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.8...v0.6.9
[0.6.8]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.7...v0.6.8
[0.6.7]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/fortyfive-labs/ml-dash/releases/tag/v0.6.6
[0.6.2rc1]: https://github.com/fortyfive-labs/ml-dash/releases/tag/v0.6.2rc1
