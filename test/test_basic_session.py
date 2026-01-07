"""Tests for basic experiment operations."""

import getpass
import json

import pytest


def test_experiment_creation_with_context_manager(local_experiment, tmp_proj):
  """Test basic experiment creation using context manager."""
  with local_experiment(name="hello-ml-dash", project="tutorials").run as experiment:
    experiment.log("Hello from ML-Dash!", level="info")
    experiment.params.set(message="Hello World")

  # Verify experiment directory was created
  # Local storage uses owner/project/prefix structure
  owner = getpass.getuser()
  experiment_dir = tmp_proj / owner / "tutorials/hello-ml-dash"
  assert experiment_dir.exists()
  assert (experiment_dir / "experiment.json").exists()


def test_experiment_with_metadata(local_experiment, tmp_proj):
  """Test experiment creation with description, tags, and prefix."""
  with local_experiment(
    name="experiments/mnist/mnist-baseline",  # Full prefix
    project="computer-vision",
    description="Baseline CNN for MNIST classification",
    tags=["mnist", "cnn", "baseline"],
  ).run as experiment:
    experiment.log("Experiment created with metadata")

  # Verify metadata was saved
  # Files go to: root_path / owner / project / prefix
  owner = getpass.getuser()
  experiment_dir = tmp_proj / owner / "computer-vision/experiments/mnist/mnist-baseline"
  experiment_file = experiment_dir / "experiment.json"

  assert experiment_file.exists()
  with open(experiment_file) as f:
    metadata = json.load(f)
    assert metadata["name"] == "mnist-baseline"
    assert metadata["project"] == "computer-vision"
    assert metadata["description"] == "Baseline CNN for MNIST classification"
    assert "mnist" in metadata["tags"]
    assert "cnn" in metadata["tags"]
    assert metadata["prefix"] == "experiments/mnist/mnist-baseline"


def test_experiment_manual_open_close(local_experiment, tmp_proj):
  """Test manual experiment lifecycle management."""
  experiment = local_experiment(name="manual-experiment", project="test")

  # The experiment is not initially open
  assert not experiment._is_open

  # Open experiment
  experiment.run.start()
  assert experiment._is_open

  # Do work
  experiment.log("Working...")

  # Close experiment
  experiment.run.complete()
  assert not experiment._is_open

  # Verify data was saved
  owner = getpass.getuser()
  experiment_dir = tmp_proj / owner / "test/manual-experiment"
  assert experiment_dir.exists()


def test_experiment_auto_close_on_context_exit(local_experiment):
  """Test that experiment is automatically closed when exiting context manager."""
  with local_experiment(name="auto-close", project="test").run as experiment:
    assert experiment._is_open
    experiment.log("Working...")

  # After exiting context, the experiment should be closed
  assert not experiment._is_open


def test_experiments_same_project(local_experiment, tmp_proj):
  """Test experiments in the same project."""
  # Create first experiment
  with local_experiment(name="experiment-1", project="shared").run as experiment:
    experiment.log("Experiment 1")

  # Create second experiment
  with local_experiment(name="experiment-2", project="shared").run as experiment:
    experiment.log("Experiment 2")

  # Verify both experiments exist
  owner = getpass.getuser()
  project_dir = tmp_proj / owner / "shared"
  assert (project_dir / "experiment-1").exists()
  assert (project_dir / "experiment-2").exists()


def test_experiment_name_and_project_properties(local_experiment):
  """Test that experiment properties are accessible."""
  with local_experiment(name="my-experiment", project="my-project").run as experiment:
    assert experiment.name == "my-experiment"
    assert experiment.project == "my-project"


def test_experiment_error_handling(local_experiment, tmp_proj):
  """Test that experiment handles errors gracefully."""
  try:
    with local_experiment(name="error-test", project="test").run as experiment:
      experiment.log("Starting work")
      experiment.params.set(param="value")
      raise ValueError("Simulated error")
  except ValueError:
    pass

  # Experiment directory should be created even if error occurs
  owner = getpass.getuser()
  experiment_dir = tmp_proj / owner / "test/error-test"
  assert experiment_dir.exists(), "Experiment directory should exist even after error"


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
