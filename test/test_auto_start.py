"""Tests for auto_start module with dxp singleton."""

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cleanup_dxp():
  """Clean up .dash directory and reset dxp state before and after each test."""
  import getpass
  from ml_dash.auto_start import dxp
  from ml_dash.storage import LocalStorage

  ml_dash_dir = Path(".dash")
  owner = getpass.getuser()

  # Close dxp and clean up before test
  if dxp._is_open:
    dxp.run.complete()
  if ml_dash_dir.exists():
    shutil.rmtree(ml_dash_dir)

  # Configure dxp for local mode testing
  # Use the full prefix format: owner/project/name
  dxp._storage = LocalStorage(root_path=ml_dash_dir)
  dxp._client = None  # Disable remote
  dxp._folder_path = f"{owner}/scratch/dxp"  # Set full prefix for local mode

  # Reopen dxp for the test
  dxp.run.start()

  yield

  # Close dxp and clean up after test
  if dxp._is_open:
    dxp.run.complete()
  if ml_dash_dir.exists():
    shutil.rmtree(ml_dash_dir)


def test_dxp_is_auto_started():
  """Test that dxp is automatically started on import."""
  from ml_dash.auto_start import dxp

  # Should be open immediately after import
  assert dxp._is_open
  assert dxp.name == "dxp"
  assert dxp.project == "scratch"


def test_dxp_logging():
  """Test that dxp can log messages."""
  import getpass

  from ml_dash.auto_start import dxp

  dxp.log("Test message from dxp")
  dxp.log("Another message", level="info")

  # Verify logs exist in .dash directory
  # New structure: root / owner / project / prefix
  owner = getpass.getuser()
  experiment_dir = Path(".dash") / owner / "scratch/dxp"
  logs_file = experiment_dir / "logs/logs.jsonl"

  assert logs_file.exists()

  # Read logs
  with open(logs_file, "r") as f:
    logs = [json.loads(line) for line in f]

  # Check that our test messages are in the logs
  messages = [log["message"] for log in logs]
  assert "Test message from dxp" in messages
  assert "Another message" in messages


def test_dxp_parameters():
  """Test that dxp can set and get parameters."""
  import getpass

  from ml_dash.auto_start import dxp

  # Set parameters
  dxp.params.set(lr=0.001, batch_size=32)
  dxp.params.set(model={"name": "resnet50", "layers": 50})

  # Get parameters
  params = dxp.params.get()

  assert params["lr"] == 0.001
  assert params["batch_size"] == 32
  assert params["model.name"] == "resnet50"
  assert params["model.layers"] == 50

  # Verify parameters file exists
  # New structure: root / owner / project / prefix
  owner = getpass.getuser()
  experiment_dir = Path(".dash") / owner / "scratch/dxp"
  params_file = experiment_dir / "parameters.json"
  assert params_file.exists()


def test_dxp_metrics():
  """Test that dxp can log metrics."""
  import getpass

  from ml_dash.auto_start import dxp

  # Log metrics using the correct API
  dxp.metrics("train").log(step=0, loss=0.5)
  dxp.metrics("train").log(step=1, loss=0.4)
  dxp.metrics("eval").log(step=0, accuracy=0.8)

  # Close and reopen dxp to flush metric buffers
  dxp.run.complete()
  dxp.run.start()

  # Verify metrics exist (stored as metrics/<name>/data.jsonl)
  # New structure: root / owner / project / prefix
  owner = getpass.getuser()
  experiment_dir = Path(".dash") / owner / "scratch/dxp"
  metrics_dir = experiment_dir / "metrics"

  assert metrics_dir.exists()
  assert (metrics_dir / "train/data.jsonl").exists()
  assert (metrics_dir / "eval/data.jsonl").exists()

  # Read train metrics
  with open(metrics_dir / "train/data.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

  assert len(lines) == 2
  assert lines[0]["data"]["loss"] == 0.5
  assert lines[1]["data"]["loss"] == 0.4


def test_dxp_files(tmp_path):
  """Test that dxp can upload files."""
  import getpass

  from ml_dash.auto_start import dxp

  # Create a temporary file
  test_file = tmp_path / "test.txt"
  test_file.write_text("Test content")

  # Upload file
  result = dxp.files("tests").upload(str(test_file))

  assert result["filename"] == "test.txt"
  assert result["path"] == "/tests"
  file_id = result["id"]

  # Verify file exists in experiment (files are stored as files/<prefix>/<file_id>/<filename>)
  # New structure: root / owner / project / prefix
  owner = getpass.getuser()
  experiment_dir = Path(".dash") / owner / "scratch/dxp"
  uploaded_file = experiment_dir / "files/tests" / file_id / "test.txt"
  assert uploaded_file.exists()
  assert uploaded_file.read_text() == "Test content"


def test_dxp_singleton_behavior():
  """Test that dxp is a singleton - same instance across imports."""
  from ml_dash.auto_start import dxp as dxp1
  from ml_dash.auto_start import dxp as dxp2

  # Should be the same object
  assert dxp1 is dxp2


def test_dxp_works_like_normal_experiment():
  """Test that dxp works exactly like a normal experiment."""
  import getpass

  from ml_dash.auto_start import dxp

  # Should have all experiment properties
  assert hasattr(dxp, "name")
  assert hasattr(dxp, "project")
  assert hasattr(dxp, "run")
  assert hasattr(dxp, "params")
  assert hasattr(dxp, "log")
  assert hasattr(dxp, "metrics")
  assert hasattr(dxp, "files")

  # Should be able to use all methods
  dxp.log("Test log")
  dxp.params.set(test="value")
  dxp.metrics("test_metric").log(step=0, value=1.0)

  # Verify data was saved
  # New structure: root / owner / project / prefix
  owner = getpass.getuser()
  experiment_dir = Path(".dash") / owner / "scratch/dxp"
  assert experiment_dir.exists()
  assert (experiment_dir / "experiment.json").exists()


def test_dxp_immutable_config():
  """Test that dxp config cannot be changed after initialization."""
  from ml_dash.auto_start import dxp

  # Should not be able to change name, project
  # (These are read-only properties set during initialization)
  assert dxp.name == "dxp"
  assert dxp.project == "scratch"

  # The experiment is already started, so it uses local storage
  assert dxp._storage is not None


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
