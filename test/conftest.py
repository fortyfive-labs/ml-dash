"""Pytest configuration and fixtures for ML-Dash tests."""
import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_dash import Experiment


# Configuration
REMOTE_SERVER_URL = "http://localhost:3000"
TEST_USERNAME = "test-user"

# Generate test JWT token for remote experiments
def _generate_test_api_key():
    """Generate a test JWT token for testing using OIDC standard claims."""
    import jwt

    # Use OIDC standard claims matching vuer-auth token format
    payload = {
        "sub": "test-user-id-12345",
        "email": "test@example.com",
        "name": "Test User",
        "username": TEST_USERNAME,
        "given_name": "Test",
        "family_name": "User",
    }

    # Test secret key (matches server test configuration)
    secret = "your-secret-key-change-this-in-production"
    return jwt.encode(payload, secret, algorithm="HS256")

TEST_API_KEY = _generate_test_api_key()


@pytest.fixture
def temp_project(tmp_path):
    """
    Temporary project directory for tests.

    Uses pytest's tmp_path fixture which creates a unique temporary directory
    that is automatically cleaned up after the test.
    """
    return tmp_path


@pytest.fixture
def local_experiment(temp_project):
    """
    Create a test experiment in local mode.

    Returns a function that creates experiments with default config but allows overrides.
    """
    def _create_experiment(name="test-experiment", project="test-project", **kwargs):
        defaults = {
            "local_path": str(temp_project),
        }
        defaults.update(kwargs)
        return Experiment(name=name, project=project, **defaults)

    return _create_experiment


@pytest.fixture
def remote_experiment():
    """
    Create a test experiment in remote mode.

    Returns a function that creates remote experiments with localhost:3000.
    Use the @pytest.mark.remote marker for tests that require a running server.
    Uses a pre-generated test JWT token for authentication.

    Generates unique experiment names using timestamps to avoid collisions between test runs.
    """
    import time

    def _create_experiment(name="test-experiment", project="test-project", **kwargs):
        # Add timestamp suffix to experiment name to make it unique
        # This prevents collisions from previous test runs
        timestamp_suffix = str(int(time.time() * 1000000))  # microsecond precision
        unique_name = f"{name}-{timestamp_suffix}"

        defaults = {
            "remote": REMOTE_SERVER_URL,
            "api_key": TEST_API_KEY,
        }
        defaults.update(kwargs)
        return Experiment(name=unique_name, project=project, **defaults)

    return _create_experiment


@pytest.fixture(params=["local", "remote"])
def any_experiment(request, local_experiment, remote_experiment):
    """
    Parametrized fixture that runs tests with both local and remote experiments.

    Tests using this fixture will run twice: once with local mode and once with remote mode.
    Remote tests will be skipped if the server is not available.
    """
    if request.param == "local":
        return local_experiment
    else:
        # Check if remote server is available before running remote tests
        if request.node.get_closest_marker("skip_remote"):
            pytest.skip("Test explicitly skips remote mode")
        return remote_experiment


@pytest.fixture
def sample_files(tmp_path):
    """
    Create sample files for file upload tests.

    Returns a dict with paths to created files:
    - model: model.txt (simulated model weights)
    - config: config.json (configuration file)
    - results: results.csv (CSV results)
    - image: test_image.png (small binary file)
    - large: large_file.bin (larger binary file)
    """
    files_dir = tmp_path / "sample_files"
    files_dir.mkdir()

    # Create model file
    model_file = files_dir / "model.txt"
    model_file.write_text("Simulated model weights\n" * 10)

    # Create config file
    config_file = files_dir / "config.json"
    config_file.write_text('{"model": "resnet50", "lr": 0.001, "batch_size": 32}')

    # Create results file
    results_file = files_dir / "results.csv"
    results_file.write_text("epoch,loss,accuracy\n1,0.5,0.85\n2,0.3,0.90\n3,0.2,0.93\n")

    # Create a small binary file (simulating an image)
    image_file = files_dir / "test_image.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Create a larger binary file
    large_file = files_dir / "large_file.bin"
    large_file.write_bytes(b"\x00" * (1024 * 100))  # 100 KB

    return {
        "model": str(model_file),
        "config": str(config_file),
        "results": str(results_file),
        "image": str(image_file),
        "large": str(large_file),
        "dir": str(files_dir),
    }


@pytest.fixture
def sample_data():
    """
    Sample data for testing metrics and parameters.

    Returns a dict with various test data structures.
    """
    return {
        "simple_params": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        },
        "nested_params": {
            "model": {
                "architecture": "resnet50",
                "pretrained": True,
                "layers": {
                    "conv1": {"filters": 64, "kernel": 3},
                    "conv2": {"filters": 128, "kernel": 3},
                }
            },
            "optimizer": {
                "type": "adam",
                "beta1": 0.9,
                "beta2": 0.999,
                "lr": 0.001,
            }
        },
        "metric_data": [
            {"value": 0.5, "epoch": 0, "step": 0},
            {"value": 0.4, "epoch": 1, "step": 100},
            {"value": 0.3, "epoch": 2, "step": 200},
            {"value": 0.25, "epoch": 3, "step": 300},
            {"value": 0.2, "epoch": 4, "step": 400},
        ],
        "multi_metric_data": [
            {"epoch": 0, "train_loss": 0.5, "val_loss": 0.6, "accuracy": 0.7},
            {"epoch": 1, "train_loss": 0.4, "val_loss": 0.5, "accuracy": 0.75},
            {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4, "accuracy": 0.8},
        ]
    }


@pytest.fixture
def check_remote_available():
    """
    Check if remote server is available.

    Returns True if the server responds, False otherwise.
    Useful for conditional test skipping.
    """
    import httpx
    try:
        response = httpx.get(f"{REMOTE_SERVER_URL}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "remote: mark test as requiring remote server (will attempt to connect to localhost:3000)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "skip_remote: mark test to skip remote mode in parametrized tests"
    )
    config.addinivalue_line(
        "markers",
        "local_only: mark test to run only in local mode"
    )
    config.addinivalue_line(
        "markers",
        "remote_only: mark test to run only in remote mode"
    )


def pytest_collection_modifyitems(items):
    """
    Modify test collection to handle remote tests gracefully.

    Adds skip markers to remote tests if the server is not available.
    """
    # Check if remote server is available
    import httpx
    server_available = False
    try:
        response = httpx.get(f"{REMOTE_SERVER_URL}/health", timeout=2.0)
        server_available = response.status_code == 200
    except Exception:
        pass

    skip_remote = pytest.mark.skip(reason="Remote server not available at localhost:3000")

    for item in items:
        # Skip remote-only tests if server not available
        if "remote_only" in item.keywords and not server_available:
            item.add_marker(skip_remote)

        # Skip remote tests if marked and server not available
        if "remote" in item.keywords and not server_available:
            item.add_marker(skip_remote)
