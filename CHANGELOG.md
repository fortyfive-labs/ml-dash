# Changelog

All notable changes to ML-Dash will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.6.9]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.8...v0.6.9
[0.6.8]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.7...v0.6.8
[0.6.7]: https://github.com/fortyfive-labs/ml-dash/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/fortyfive-labs/ml-dash/releases/tag/v0.6.6
[0.6.2rc1]: https://github.com/fortyfive-labs/ml-dash/releases/tag/v0.6.2rc1
