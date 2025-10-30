# ML-Dash Python SDK Tests

This directory contains the pytest test suite for the ML-Dash Python SDK.

## Test Files

- **`conftest.py`** - Pytest configuration and fixtures
- **`test_basic_experiment.py`** - Basic experiment operations (7 tests)
- **`test_logging.py`** - Logging functionality (7 tests)
- **`test_parameters.py`** - Parameter metricing (7 tests)
- **`test_metrics.py`** - Time-series metrics metricing (10 tests)
- **`test_files.py`** - File upload operations (10 tests)
- **`test_integration.py`** - End-to-end integration workflows (6 tests)

**Total: 47 tests**

## Running Tests

### Run all tests
```bash
uv run pytest test/
```

### Run specific test file
```bash
uv run pytest test/test_basic_experiment.py
```

### Run with verbose output
```bash
uv run pytest test/ -v
```

### Run specific test
```bash
uv run pytest test/test_logging.py::test_basic_logging
```

## Test Coverage

The test suite covers:

-  Experiment creation and lifecycle management
-  Context manager usage
-  Manual open/close operations
-  Experiment metadata (description, tags, folder)
-  Log levels (debug, info, warn, error, fatal)
-  Structured logging with metadata
-  Simple and nested parameter metricing
-  Parameter flattening
-  Parameter updates
-  Time-series metrics metricing
-  Batch data appending
-  Metric statistics and metadata
-  File uploads with metadata
-  File checksums
-  File tagging and prefixes
-  Complete ML workflow integration
-  Hyperparameter search workflows
-  Multi-experiment pipelines
-  Error handling

## Fixtures

### `temp_project`
Provides a temporary directory for test data (uses pytest's `tmp_path`).

### `local_experiment`
Factory function to create local-mode experiments with default configuration.

Usage:
```python
def test_example(local_experiment, temp_project):
    with local_experiment(name="my-experiment", project="test") as experiment:
        experiment.log("Test message")
```

### `sample_files`
Creates sample files for testing file uploads:
- `model.txt` - Simulated model weights
- `config.json` - Configuration file
- `results.txt` - CSV-like results

### `remote_experiment` (Optional)
Factory function to create remote-mode experiments. Requires `ML_DASH_SERVER_URL` env variable.

## Environment Variables

Optional environment variables for remote testing:

- `ML_DASH_SERVER_URL` - Remote server URL (default: skip remote tests)
- `ML_DASH_TEST_USER` - Test username (default: "test-user")
- `ML_DASH_API_KEY` - API key (optional, auto-generated from username)

## Notes

- All tests use temporary directories and are automatically cleaned up
- Tests use simple function notation (no test classes)
- Remote tests are skipped if server URL is not configured
- All 47 tests currently passing (as of last run)
