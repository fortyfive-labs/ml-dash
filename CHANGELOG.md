# Changelog

All notable changes to ML-Dash will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.6.7]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/fortyfive-labs/ml-dash/releases/tag/v0.6.6
[0.6.2rc1]: https://github.com/fortyfive-labs/ml-dash/releases/tag/v0.6.2rc1
