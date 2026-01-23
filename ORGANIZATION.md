# Project Organization

This document describes the organization of the ML-Dash project.

## Directory Structure

```
ml-dash/
├── docs/                       # Documentation
│   ├── buffering.md           # Background buffering guide (NEW)
│   ├── tracks.md              # Track API documentation (NEW)
│   ├── images.md              # Image saving guide (NEW)
│   ├── index.md               # Documentation index
│   ├── examples/              # Code examples (NEW)
│   │   ├── robotics/         # Robotics examples
│   │   │   └── mujoco_tracking.py
│   │   ├── vision/           # Computer vision examples
│   │   └── rl/               # Reinforcement learning examples
│   ├── archived/              # Legacy documentation
│   └── _build/                # Built documentation
│
├── src/ml_dash/               # Source code
│   ├── __init__.py
│   ├── experiment.py          # Core experiment class
│   ├── buffer.py              # Background buffering system (NEW)
│   ├── track.py               # Track API (NEW)
│   ├── files.py               # File operations with image support
│   ├── client.py              # API client
│   ├── storage.py             # Local storage
│   ├── metric.py              # Metrics tracking
│   ├── log.py                 # Logging
│   └── cli_commands/          # CLI commands
│
├── test/                      # Test files
│   ├── test_buffering.py      # Buffer system tests (NEW)
│   ├── test_experiment.py     # Core experiment tests
│   ├── test_files.py          # File operation tests
│   ├── test_integration.py    # Integration tests
│   └── ...                    # Other test files
│
├── CHANGELOG.md               # Version history (NEW)
├── ORGANIZATION.md            # This file (NEW)
├── README.md                  # Project README
├── pyproject.toml             # Project configuration
└── LICENSE                    # MIT License
```

## Documentation

### New Features Documentation (v0.6.7)

Three new comprehensive guides have been added:

1. **buffering.md**: Complete guide to the background buffering system
   - How it works
   - Configuration options
   - Performance benefits
   - Best practices

2. **tracks.md**: Track API for time-series data
   - Basic usage
   - MuJoCo integration
   - RL examples
   - Aligning with frames

3. **images.md**: Numpy array image saving
   - Supported formats (PNG/JPEG)
   - Quality control
   - Array type handling
   - Integration examples

### Core Documentation

Legacy documentation remains in `docs/archived/` for reference:
- Getting started
- Experiments
- Parameters
- Metrics
- Logging
- Files
- API reference
- CLI commands

## Source Code Organization

### New Files

- **src/ml_dash/buffer.py**: Background buffering implementation
  - BufferConfig class
  - BackgroundBufferManager class
  - Thread-safe queuing
  - Automatic flushing

- **src/ml_dash/track.py**: Track API implementation
  - Track class
  - TrackBuilder class
  - Timestamp-based merging

### Enhanced Files

- **src/ml_dash/files.py**: Added numpy image support
  - `save_image()` method
  - PNG/JPEG format handling
  - Auto-detection in `save()`
  - Temp file cleanup for buffering

- **src/ml_dash/experiment.py**: Integrated buffering
  - Buffer manager lifecycle
  - Non-blocking writes
  - Manual flush support

## Examples

### New Examples Directory

Created `docs/examples/` with organized subdirectories:

- **robotics/**: MuJoCo and robotics examples
  - `mujoco_tracking.py`: Complete tracking example

- **vision/**: Computer vision examples (placeholder)

- **rl/**: Reinforcement learning examples (placeholder)

## Tests

### New Test Files

- **test/test_buffering.py**: Comprehensive buffering tests
  - 27 test cases
  - Unit and integration tests
  - Performance tests
  - Concurrency tests

### Test Organization

Tests are organized by feature area:
- `test_experiment.py`: Core experiment functionality
- `test_files.py`: File operations
- `test_buffering.py`: Background buffering
- `test_integration.py`: End-to-end workflows
- `test_*.py`: Other feature-specific tests

## Configuration Files

### Updated Files

- **pyproject.toml**: Project metadata and dependencies
  - Version: 0.6.7
  - Dependencies include Pillow for image support

- **CHANGELOG.md**: Version history
  - Detailed release notes for v0.6.7
  - Links to GitHub releases

## Documentation Build

To build the documentation:

```bash
# Install dependencies
uv sync --extra dev

# Build HTML docs
cd docs
make html

# Or use sphinx directly
uv run python -m sphinx -b html docs docs/_build/html

# Live preview (auto-rebuild on changes)
uv run sphinx-autobuild docs docs/_build/html
```

## Future Organization Plans

### Planned Improvements

1. **More Examples**:
   - Add vision examples (CV tasks)
   - Add RL examples (Gym integration)
   - Add multi-agent examples

2. **Tutorial Series**:
   - Beginner tutorials
   - Advanced patterns
   - Best practices guide

3. **API Documentation**:
   - Auto-generated from docstrings
   - API reference reorganization

4. **Test Organization**:
   - Group tests by feature
   - Add performance benchmarks
   - Integration test suite

## Contributing

When adding new features:

1. **Add documentation** in `docs/` with the feature name
2. **Add examples** in `docs/examples/` under appropriate category
3. **Add tests** in `test/test_<feature>.py`
4. **Update CHANGELOG.md** with changes
5. **Update README.md** if it's a major feature

## Questions?

For questions about project organization:
- Check the documentation in `docs/`
- See examples in `docs/examples/`
- Review tests in `test/`
- Read the CHANGELOG for version history
