# ML-Dash Test Suite

## Overview

This directory contains the comprehensive test suite for the ML-Dash Python SDK. All test files follow a consistent pattern and can be run individually or as a suite.

## Running Tests

### Run Individual Test Files

Each test file can be executed standalone:

```bash
python3 test/test_auth.py
python3 test/test_folder_templates.py
python3 test/test_parameters.py
```

### Run All Tests

Run all tests at once using the test runner:

```bash
python3 test/run_all_tests.py
```

### Run with Pytest

You can also use pytest directly:

```bash
# Run specific test file
pytest test/test_auth.py -v

# Run all tests (334 tests)
pytest test/ -v

# Run only local tests (skip remote) - recommended for CI/development
pytest test/ -m "not remote"

# Run only remote tests (requires server at localhost:3000)
pytest test/ -m "remote"

# Run specific test
pytest test/test_logging.py::test_basic_logging
```

## Test File Pattern

All test files follow this pattern:

1. **File naming**: `test_*.py` (except `test_tom.py` which is excluded)
2. **Entry point**: Each file has `if __name__ == "__main__":` block
3. **Execution**: Can be run directly with `python3 test/test_<name>.py`

## Test Categories

### ✅ Local Tests (No Server Required)

These tests work without a running server:

- `test_auth.py` - Authentication and token storage (32 passed)
- `test_auto_start.py` - Auto-start singleton tests (8 passed)
- `test_basic_session.py` - Basic experiment operations (7 passed)
- `test_folder_fix_complete.py` - Folder path consistency (3 passed)
- `test_folder_templates.py` - Template variable resolution (6 passed)
- `test_log_stdout_stderr.py` - Logging with stdout/stderr (5 passed)
- `test_must_start_first.py` - Manual start requirements (5 passed)
- `test_params_class_objects.py` - Parameter class objects (4 passed)
- `test_params_log_simple.py` - params.log() alias (3 passed)
- `test_run_folder_setter.py` - run.folder setter (4 passed)
- `test_summary_cache.py` - Metrics summary cache (7 passed)

### ⚠️ Remote Tests (Server Required)

These tests require a running ML-Dash server at `http://localhost:3000` with:
- Valid JWT secret configured
- Test user with namespace in database
- GraphQL and REST API endpoints available

Remote test files (marked with `@pytest.mark.remote`):
- `test_cli_upload.py` - CLI upload functionality
- `test_duplicate.py` - Experiment duplication
- `test_experiment.py` - Remote experiment operations
- `test_file_upload_download.py` - File upload/download
- `test_files.py` - File operations
- `test_integration.py` - End-to-end workflows
- `test_logging.py` - Remote logging
- `test_optional_metric_name.py` - Metric naming
- `test_parameters.py` - Remote parameters
- `test_save_video.py` - Video saving
- `test_summary_cache_comprehensive.py` - Remote cache tests
- `test_tracks.py` - Remote metrics tracking

### ⏱️ Long-Running Tests

- `test_performance.py` - Performance benchmarks (takes several minutes)

**Total: 334 tests, 334 passing (100%)** ✅
- Local tests: 278 passing
- Remote tests: 56 passing

## Test Coverage

The test suite covers:

- ✅ Experiment creation and lifecycle management
- ✅ Context manager usage
- ✅ Manual open/close operations
- ✅ Experiment metadata (description, tags, folder)
- ✅ Log levels (debug, info, warn, error, fatal)
- ✅ Structured logging with metadata
- ✅ Simple and nested parameter logging
- ✅ Parameter flattening
- ✅ Parameter updates
- ✅ Time-series metrics logging
- ✅ Batch data appending
- ✅ Metric statistics and metadata
- ✅ File uploads with metadata
- ✅ File checksums
- ✅ File tagging and prefixes
- ✅ Complete ML workflow integration
- ✅ Hyperparameter search workflows
- ✅ Multi-experiment pipelines
- ✅ Error handling
- ✅ Authentication and token management
- ✅ CLI upload functionality
- ✅ Remote server integration

## Fixtures

### `temp_project`
Provides a temporary directory for test data (uses pytest's `tmp_path`). Automatically cleaned up after tests.

### `local_experiment`
Factory function to create local-mode experiments with default configuration.

Usage:
```python
def test_example(local_experiment, temp_project):
    with local_experiment(name="my-experiment", project="test").run as experiment:
        experiment.log("Test message")
```

### `remote_experiment`
Factory function to create remote-mode experiments. Connects to `http://localhost:3000` with test JWT token. Generates unique experiment names using timestamps to avoid collisions.

Usage:
```python
@pytest.mark.remote
def test_remote_example(remote_experiment):
    with remote_experiment(name="my-experiment", project="test").run as experiment:
        experiment.log("Remote test message")
```

### `sample_files`
Creates sample files for testing file uploads:
- `model.txt` - Simulated model weights
- `config.json` - Configuration file
- `results.csv` - CSV results
- `test_image.png` - Small binary file
- `large_file.bin` - 100KB binary file

### `sample_data`
Sample data for testing metrics and parameters:
- `simple_params` - Basic parameter dictionary
- `nested_params` - Nested parameter structures
- `metric_data` - Time-series metric data
- `multi_metric_data` - Multiple metrics per epoch

## Configuration

Test configuration is defined in `conftest.py`:

- **Remote server URL**: `http://localhost:3000`
- **Test user**: Real user from database (configured in conftest.py)
- **Authentication**: Auto-generated JWT token using OIDC standard claims
- **JWT Secret**: Must match server's `JWT_SECRET` environment variable
- **Storage structure**: `root_path/owner/project/prefix`

### Remote Test Setup

To run remote tests, ensure:

1. **Server running** at `http://localhost:3000`
2. **JWT_SECRET** configured (default: `your-secret-key-change-this-in-production`)
3. **Test user exists** in database with a namespace
4. **Token storage mocked** via `mock_remote_token` fixture

The test suite automatically:
- Generates JWT tokens for authentication
- Mocks token storage to return test credentials
- Creates unique experiment names to avoid collisions

## Notes

- All tests use temporary directories and are automatically cleaned up
- Tests use pytest fixtures for dependency injection
- Test isolation: Remote experiments use timestamp-based unique names
- Excluded: `test_tom.py` (development/debugging file)
- Storage paths follow: `root_path/owner/project/prefix` structure
- Owner is derived from system username for local tests
