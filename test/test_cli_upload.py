"""Tests for CLI upload command."""

import argparse
import getpass
from pathlib import Path

import pytest
from conftest import TEST_API_KEY  # Import the test API key

from ml_dash.cli_commands.upload import (
  ExperimentValidator,
  UploadState,
  cmd_upload,
  discover_experiments,
)


class TestExperimentDiscovery:
  """Tests for experiment discovery functionality."""

  def test_discover_single_experiment(self, local_experiment):
    """Test discovering a single experiment."""
    exp = local_experiment("57block/proj1/exp1")
    with exp.run as e:
      e.log("Test log message")
      e.params.set(**{"lr": 0.001})

    local_path = Path(exp._storage.root_path)
    experiments = discover_experiments(local_path)

    assert len(experiments) == 1
    assert experiments[0].project == "proj1"
    assert experiments[0].experiment == "exp1"
    assert experiments[0].has_logs is True
    assert experiments[0].has_params is True

  def test_discover_multiple_experiments(self, local_experiment):
    """Test discovering multiple experiments."""
    exp1 = local_experiment("57block/proj1/exp1")
    with exp1.run as e:
      e.log("Test")

    exp2 = local_experiment("57block/proj1/exp2")
    with exp2.run as e:
      e.log("Test")

    exp3 = local_experiment("57block/proj2/exp1")
    with exp3.run as e:
      e.log("Test")

    local_path = Path(exp1._storage.root_path)
    experiments = discover_experiments(local_path)

    assert len(experiments) == 3
    project_names = [e.project for e in experiments]
    assert "proj1" in project_names
    assert "proj2" in project_names

  def test_discover_with_project_filter(self, local_experiment):
    """Test discovering experiments with project filter."""
    exp1 = local_experiment("57block/proj1/exp1")
    with exp1.run as e:
      e.log("Test")

    exp2 = local_experiment("57block/proj2/exp2")
    with exp2.run as e:
      e.log("Test")

    local_path = Path(exp1._storage.root_path)
    experiments = discover_experiments(local_path, project_filter="proj1")

    assert len(experiments) == 1
    assert experiments[0].project == "proj1"

  def test_discover_with_experiment_filter(self, local_experiment):
    """Test discovering experiments with experiment and project filter."""
    exp1 = local_experiment("57block/proj1/exp1")
    with exp1.run as e:
      e.log("Test")

    exp2 = local_experiment("57block/proj1/exp2")
    with exp2.run as e:
      e.log("Test")

    local_path = Path(exp1._storage.root_path)
    experiments = discover_experiments(
      local_path, project_filter="proj1", experiment_filter="exp1"
    )

    assert len(experiments) == 1
    assert experiments[0].experiment == "exp1"

  def test_discover_with_metrics(self, local_experiment):
    """Test discovering experiments with metrics."""
    exp = local_experiment("57block/proj1/exp1")
    with exp.run as e:
      e.metrics("train").log(loss=0.5)
      e.metrics("eval").log(loss=0.85)

    local_path = Path(exp._storage.root_path)
    experiments = discover_experiments(local_path)

    assert len(experiments) == 1
    assert "train" in experiments[0].metric_names
    assert "eval" in experiments[0].metric_names

  def test_discover_empty_directory(self, tmp_proj):
    """Test discovering experiments in empty directory."""
    experiments = discover_experiments(tmp_proj)
    assert len(experiments) == 0

  def test_discover_nonexistent_directory(self):
    """Test discovering experiments in nonexistent directory."""
    experiments = discover_experiments(Path("/nonexistent/path"))
    assert len(experiments) == 0


class TestExperimentValidator:
  """Tests for experiment validation functionality."""

  def test_validate_valid_experiment(self, local_experiment):
    """Test validating a valid experiment."""
    exp = local_experiment("57block/proj1/exp1")
    with exp.run as e:
      e.log("Test message")
      e.params.set(**{"lr": 0.001})

    local_path = Path(exp._storage.root_path)
    experiments = discover_experiments(local_path)

    validator = ExperimentValidator(strict=False)
    result = validator.validate_experiment(experiments[0])

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert "metadata" in result.valid_data

  def test_validate_missing_metadata(self, tmp_proj):
    """Test validating experiment with missing metadata."""
    # Create experiment directory without experiment.json
    exp_dir = tmp_proj / getpass.getuser() / "proj1/exp1"
    exp_dir.mkdir(parents=True)

    from ml_dash.cli_commands.upload import ExperimentInfo

    exp_info = ExperimentInfo(
      project="proj1",
      experiment="exp1",
      path=exp_dir,
    )

    validator = ExperimentValidator(strict=False)
    result = validator.validate_experiment(exp_info)

    assert result.is_valid is False
    assert len(result.errors) > 0

  def test_validate_invalid_json(self, tmp_proj):
    """Test validating experiment with invalid JSON."""
    exp_dir = tmp_proj / getpass.getuser() / "proj1/exp1"
    exp_dir.mkdir(parents=True)

    # Write invalid JSON
    exp_json = exp_dir / "experiment.json"
    exp_json.write_text("{invalid json")

    from ml_dash.cli_commands.upload import ExperimentInfo

    exp_info = ExperimentInfo(
      project="proj1",
      experiment="exp1",
      path=exp_dir,
    )

    validator = ExperimentValidator(strict=False)
    result = validator.validate_experiment(exp_info)

    assert result.is_valid is False
    assert "Invalid JSON" in str(result.errors)

  def test_validate_with_warnings(self, local_experiment):
    """Test validating experiment with warnings (non-fatal issues)."""
    exp = local_experiment("57block/proj1/exp1")
    with exp.run as e:
      e.log("Test")

    # Add invalid log line
    import getpass

    owner = getpass.getuser()
    local_path = Path(exp._storage.root_path)
    logs_dir = local_path / owner / "proj1/exp1/logs"
    logs_file = logs_dir / "logs.jsonl"

    # Append invalid JSON line
    with open(logs_file, "a") as f:
      f.write("invalid json line\n")

    experiments = discover_experiments(local_path)
    validator = ExperimentValidator(strict=False)
    result = validator.validate_experiment(experiments[0])

    # Should still be valid but with warnings
    assert result.is_valid is True
    assert len(result.warnings) > 0

  def test_validate_strict_mode(self, local_experiment):
    """Test validating experiment in strict mode (warnings become errors)."""
    exp = local_experiment("57block/proj1/exp1")
    with exp.run as e:
      e.log("Test")

    # Add invalid log line
    import getpass

    owner = getpass.getuser()
    local_path = Path(exp._storage.root_path)
    logs_dir = local_path / owner / "proj1/exp1/logs"
    logs_file = logs_dir / "logs.jsonl"

    with open(logs_file, "a") as f:
      f.write("invalid json line\n")

    experiments = discover_experiments(local_path)
    validator = ExperimentValidator(strict=True)
    result = validator.validate_experiment(experiments[0])

    # In strict mode, warnings become errors
    assert result.is_valid is False


class TestUploadState:
  """Tests for upload state tracking."""

  def test_save_and_load_state(self, tmp_proj):
    """Test saving and loading upload state."""
    state_file = tmp_proj / "state.json"

    state = UploadState(
      dash_root="/path/to/local",
      remote_url="http://localhost:3000",
      completed_experiments=["proj1/exp1", "proj1/exp2"],
      failed_experiments=["proj2/exp1"],
    )

    state.save(state_file)
    assert state_file.exists()

    loaded_state = UploadState.load(state_file)
    assert loaded_state is not None
    assert loaded_state.local_path == state.local_path
    assert loaded_state.remote_url == state.remote_url
    assert len(loaded_state.completed_experiments) == 2
    assert len(loaded_state.failed_experiments) == 1

  def test_load_nonexistent_state(self, tmp_proj):
    """Test loading state from nonexistent file."""
    state_file = tmp_proj / "nonexistent.json"
    loaded_state = UploadState.load(state_file)
    assert loaded_state is None

  def test_load_invalid_state_file(self, tmp_proj):
    """Test loading state from invalid JSON file."""
    state_file = tmp_proj / "invalid.json"
    state_file.write_text("{invalid json")

    loaded_state = UploadState.load(state_file)
    assert loaded_state is None

  def test_state_to_dict(self):
    """Test converting state to dictionary."""
    state = UploadState(
      dash_root="/path/to/local",
      remote_url="http://localhost:3000",
      completed_experiments=["proj1/exp1"],
    )

    state_dict = state.to_dict()
    assert state_dict["local_path"] == "/path/to/local"
    assert state_dict["remote_url"] == "http://localhost:3000"
    assert "completed_experiments" in state_dict

  def test_state_from_dict(self):
    """Test creating state from dictionary."""
    state_dict = {
      "local_path": "/path/to/local",
      "remote_url": "http://localhost:3000",
      "completed_experiments": ["proj1/exp1"],
      "failed_experiments": [],
      "in_progress_experiment": None,
      "timestamp": "2024-01-01T00:00:00",
    }

    state = UploadState.from_dict(state_dict)
    assert state.local_path == "/path/to/local"
    assert state.remote_url == "http://localhost:3000"
    assert len(state.completed_experiments) == 1


@pytest.mark.remote
class TestUploadIntegration:
  """Integration tests for upload functionality (requires running server)."""

  def test_upload_single_experiment(self, local_experiment):
    """Test uploading a single experiment to remote server."""
    # Create local experiment
    exp = local_experiment("57block/cli-test-proj/cli-test-exp1")
    with exp.run as e:
      e.log("Test log message")
      e.params.set(**{"lr": 0.001, "batch_size": 32})
      e.metrics("train").log(loss=0.5)

    local_path = Path(exp._storage.root_path)

    # Create mock arguments
    args = argparse.Namespace(
      path=str(local_path),
      dash_url="http://localhost:3000",
      api_key=TEST_API_KEY,  # Use test API key for authentication
      user_name="test-cli-user",
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=str(local_path / ".test-state.json"),
    )

    # Run upload
    result = cmd_upload(args)

    assert result == 0  # Success

  def test_upload_dry_run(self, local_experiment):
    """Test dry run mode (no actual upload)."""
    exp = local_experiment("57block/cli-test-proj/cli-test-exp2")
    with exp.run as e:
      e.log("Test message")

    local_path = Path(exp._storage.root_path)

    args = argparse.Namespace(
      path=str(local_path),
      dash_url="http://localhost:3000",
      api_key=None,
      user_name="test-cli-user",
      project=None,
      experiment=None,
      dry_run=True,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=str(local_path / ".test-state.json"),
    )

    result = cmd_upload(args)
    assert result == 0

  def test_upload_with_project_filter(self, local_experiment):
    """Test uploading with project filter."""
    exp1 = local_experiment("57block/proj1/exp1")
    with exp1.run as e:
      e.log("Test")

    exp2 = local_experiment("57block/proj2/exp2")
    with exp2.run as e:
      e.log("Test")

    local_path = Path(exp1._storage.root_path)

    args = argparse.Namespace(
      path=str(local_path),
      dash_url="http://localhost:3000",
      api_key=None,
      user_name="test-cli-user",
      project="proj1",
      experiment=None,
      dry_run=True,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=str(local_path / ".test-state.json"),
    )

    result = cmd_upload(args)
    assert result == 0

  def test_upload_skip_options(self, local_experiment):
    """Test uploading with skip options."""
    exp = local_experiment("57block/cli-test-proj/cli-test-exp3")
    with exp.run as e:
      e.log("Test message")
      e.params.set(**{"lr": 0.001})
      e.metrics("train").log(loss=0.5)

    local_path = Path(exp._storage.root_path)

    args = argparse.Namespace(
      path=str(local_path),
      dash_url="http://localhost:3000",
      api_key=TEST_API_KEY,  # Use test API key for authentication
      user_name="test-cli-user",
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=True,
      batch_size=100,
      skip_logs=True,  # Skip logs
      skip_metrics=True,  # Skip metrics
      skip_files=True,  # Skip files
      skip_params=False,  # Upload params only
      resume=False,
      state_file=str(local_path / ".test-state.json"),
    )

    result = cmd_upload(args)
    assert result == 0

  def test_upload_resume_functionality(self, local_experiment):
    """Test resume functionality after failed upload."""
    # Create multiple experiments
    exp1 = local_experiment("57block/resume-proj/resume-exp1")
    with exp1.run as e:
      e.log("Test")

    exp2 = local_experiment("57block/resume-proj/resume-exp2")
    with exp2.run as e:
      e.log("Test")

    local_path = Path(exp1._storage.root_path)
    state_file = local_path / ".test-resume-state.json"

    # Create state file simulating partial upload
    state = UploadState(
      dash_root=str(local_path.absolute()),
      remote_url="http://localhost:3000",
      completed_experiments=["resume-proj/resume-exp1"],
      failed_experiments=[],
    )
    state.save(state_file)

    args = argparse.Namespace(
      path=str(local_path),
      dash_url="http://localhost:3000",
      api_key=TEST_API_KEY,  # Use test API key for authentication
      user_name="test-cli-user",
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=True,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=True,
      state_file=str(state_file),
    )

    result = cmd_upload(args)
    assert result == 0

    # Clean up state file
    if state_file.exists():
      state_file.unlink()


class TestCLIErrors:
  """Tests for CLI error handling."""

  def test_missing_remote_url(self, tmp_proj):
    """Test error when remote URL is missing."""
    args = argparse.Namespace(
      path=str(tmp_proj),
      dash_url=None,
      api_key=None,
      user_name=None,
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=".state.json",
    )

    result = cmd_upload(args)
    assert result == 1  # Error

  def test_missing_authentication(self, tmp_proj):
    """Test error when authentication is missing."""
    args = argparse.Namespace(
      path=str(tmp_proj),
      dash_url="http://localhost:3000",
      api_key=None,
      user_name=None,  # No auth provided
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=".state.json",
    )

    result = cmd_upload(args)
    assert result == 1  # Error

  def test_nonexistent_local_path(self):
    """Test error when local path doesn't exist."""
    args = argparse.Namespace(
      path="/nonexistent/path",
      dash_url="http://localhost:3000",
      api_key=None,
      user_name="testuser",
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=".state.json",
    )

    result = cmd_upload(args)
    assert result == 1  # Error

  def test_empty_local_storage(self, tmp_proj):
    """Test error when local storage has no experiments."""
    args = argparse.Namespace(
      path=str(tmp_proj),
      dash_url="http://localhost:3000",
      api_key=None,
      user_name="testuser",
      project=None,
      experiment=None,
      dry_run=False,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=".state.json",
    )

    result = cmd_upload(args)
    assert result == 1  # Error

  def test_experiment_filter_requires_project(self, tmp_proj):
    """Test error when experiment filter is used without project."""
    args = argparse.Namespace(
      path=str(tmp_proj),
      dash_url="http://localhost:3000",
      api_key=None,
      user_name="testuser",
      project=None,
      experiment="exp1",  # Experiment without project
      dry_run=False,
      strict=False,
      verbose=False,
      batch_size=100,
      skip_logs=False,
      skip_metrics=False,
      skip_files=False,
      skip_params=False,
      resume=False,
      state_file=".state.json",
    )

    result = cmd_upload(args)
    assert result == 1  # Error


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
