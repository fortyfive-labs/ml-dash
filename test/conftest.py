"""Pytest configuration and fixtures for ML-Dash tests."""

import os
import sys
import tempfile
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

  # Use real user from database (Tom Tao)
  # This user exists in the ml-dash server database with a namespace
  payload = {
    "sub": "4bhvoR725vyRx2lla91uUnNIWAYWKYWR",
    "email": "tom.tao@57blocks.com",
    "name": "Tom Tao",
    "username": "tom_tao_34833x",
    "given_name": "Tom",
    "family_name": "Tao",
  }

  # Server JWT secret (from ml-dash-server/.env)
  secret = "your-secret-key-change-this-in-production"
  return jwt.encode(payload, secret, algorithm="HS256")


TEST_API_KEY = _generate_test_api_key()


@pytest.fixture
def tmp_proj(tmp_path):
  """
  Temporary project directory for tests.

  Uses pytest's tmp_path fixture which creates a unique temporary directory
  that is automatically cleaned up after the test.
  """
  return tmp_path


@pytest.fixture
def local_experiment(tmp_proj):
  """
  Create a test experiment in local mode.

  Returns a function that creates experiments with default config but allows overrides.
  Prefix format: {owner}/{project}/path.../[name]

  Usage:
    local_experiment("test-user/test-project/exp-1")
    local_experiment("test-user/my-proj/experiments/baseline")
  """
  import getpass

  # Use current username for test isolation
  owner = getpass.getuser()
  DEFAULT_PREFIX = f"{owner}/test-project/test-experiment"

  def _create_experiment(prefix=DEFAULT_PREFIX, **kwargs):
    defaults = {
      "local_path": str(tmp_proj),
      "prefix": prefix,
    }
    defaults.update(kwargs)
    return Experiment(**defaults)

  return _create_experiment


@pytest.fixture
def tmp_wd(tmp_path):
  """
  Change CWD to tmp_path for test isolation.

  Useful for tests where operations default to CWD (e.g., downloads without dest path).
  """
  original_cwd = os.getcwd()
  os.chdir(tmp_path)
  yield tmp_path
  os.chdir(original_cwd)


@pytest.fixture
def mock_remote_token(monkeypatch):
  """
  Mock token storage to return TEST_API_KEY for remote tests.

  This allows RemoteClient to auto-load the test token without
  explicitly passing api_key parameter.
  """
  from unittest.mock import MagicMock

  # Create mock token storage
  mock_storage = MagicMock()
  mock_storage.load.return_value = TEST_API_KEY

  # Mock get_token_storage to return our mock
  def mock_get_token_storage():
    return mock_storage

  monkeypatch.setattr(
    "ml_dash.auth.token_storage.get_token_storage", mock_get_token_storage
  )

  return mock_storage


@pytest.fixture
def remote_experiment(mock_remote_token):
  """
  Create a test experiment in remote mode.

  Returns a function that creates remote experiments with localhost:3000.
  Use the @pytest.mark.remote marker for tests that require a running server.
  Uses a pre-generated test JWT token for authentication (auto-loaded via mock).

  Generates unique experiment names using timestamps to avoid collisions between test runs.
  Prefix format: {owner}/{project}/path.../[name]

  Usage:
    remote_experiment("test-user/test-project/exp-1")
    remote_experiment("test-user/my-proj/experiments/baseline")
  """
  import time

  def _create_experiment(prefix="test-user/test-project/test-experiment", **kwargs):
    # Add timestamp suffix to make it unique
    timestamp_suffix = str(int(time.time() * 1000000))  # microsecond precision

    # Parse prefix and add timestamp to the name part
    parts = prefix.rstrip("/").split("/")
    if len(parts) >= 2:
      parts[-1] = f"{parts[-1]}-{timestamp_suffix}"
    unique_prefix = "/".join(parts)

    defaults = {
      "remote": REMOTE_SERVER_URL,
      "prefix": unique_prefix,
      "local_path": None,  # Remote only
    }
    defaults.update(kwargs)
    return Experiment(**defaults)

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
        },
      },
      "optimizer": {
        "type": "adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 0.001,
      },
    },
    "metric_data": [
      {"loss": 0.5, "epoch": 0, "step": 0},
      {"loss": 0.4, "epoch": 1, "step": 100},
      {"loss": 0.3, "epoch": 2, "step": 200},
      {"loss": 0.25, "epoch": 3, "step": 300},
      {"loss": 0.2, "epoch": 4, "step": 400},
    ],
    "multi_metric_data": [
      {"epoch": 0, "train_loss": 0.5, "val_loss": 0.6, "accuracy": 0.7},
      {"epoch": 1, "train_loss": 0.4, "val_loss": 0.5, "accuracy": 0.75},
      {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4, "accuracy": 0.8},
    ],
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
    "remote: mark test as requiring remote server (will attempt to connect to localhost:3000)",
  )
  config.addinivalue_line("markers", "slow: mark test as slow running")
  config.addinivalue_line(
    "markers", "skip_remote: mark test to skip remote mode in parametrized tests"
  )
  config.addinivalue_line("markers", "local_only: mark test to run only in local mode")
  config.addinivalue_line(
    "markers", "remote_only: mark test to run only in remote mode"
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
