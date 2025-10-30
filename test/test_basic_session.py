"""Tests for basic experiment operations."""
import json
from pathlib import Path


def test_experiment_creation_with_context_manager(local_experiment, temp_project):
    """Test basic experiment creation using context manager."""
    with local_experiment(name="hello-ml-dash", project="tutorials") as experiment:
        experiment.log("Hello from ML-Dash!", level="info")
        experiment.parameters().set(message="Hello World")

    # Verify experiment directory was created
    experiment_dir = temp_project / "tutorials" / "hello-ml-dash"
    assert experiment_dir.exists()
    assert (experiment_dir / "experiment.json").exists()


def test_experiment_with_metadata(local_experiment, temp_project):
    """Test experiment creation with description, tags, and folder."""
    with local_experiment(
        name="mnist-baseline",
        project="computer-vision",
        description="Baseline CNN for MNIST classification",
        tags=["mnist", "cnn", "baseline"],
        folder="/experiments/mnist",
    ) as experiment:
        experiment.log("Experiment created with metadata")

    # Verify metadata was saved
    experiment_dir = temp_project / "computer-vision" / "mnist-baseline"
    experiment_file = experiment_dir / "experiment.json"

    assert experiment_file.exists()
    with open(experiment_file) as f:
        metadata = json.load(f)
        assert metadata["name"] == "mnist-baseline"
        assert metadata["project"] == "computer-vision"
        assert metadata["description"] == "Baseline CNN for MNIST classification"
        assert "mnist" in metadata["tags"]
        assert "cnn" in metadata["tags"]
        assert metadata["folder"] == "/experiments/mnist"


def test_experiment_manual_open_close(local_experiment, temp_project):
    """Test manual experiment lifecycle management."""
    experiment = local_experiment(name="manual-experiment", project="test")

    # The experiment is not initially open
    assert not experiment._is_open

    # Open experiment
    experiment.open()
    assert experiment._is_open

    # Do work
    experiment.log("Working...")

    # Close experiment
    experiment.close()
    assert not experiment._is_open

    # Verify data was saved
    experiment_dir = temp_project / "test" / "manual-experiment"
    assert experiment_dir.exists()


def test_experiment_auto_close_on_context_exit(local_experiment):
    """Test that experiment is automatically closed when exiting context manager."""
    with local_experiment(name="auto-close", project="test") as experiment:
        assert experiment._is_open
        experiment.log("Working...")

    # After exiting context, the experiment should be closed
    assert not experiment._is_open


def test_experiments_same_project(local_experiment, temp_project):
    """Test experiments in the same project."""
    # Create first experiment
    with local_experiment(name="experiment-1", project="shared") as experiment:
        experiment.log("Experiment 1")

    # Create second experiment
    with local_experiment(name="experiment-2", project="shared") as experiment:
        experiment.log("Experiment 2")

    # Verify both experiments exist
    project_dir = temp_project / "shared"
    assert (project_dir / "experiment-1").exists()
    assert (project_dir / "experiment-2").exists()


def test_experiment_name_and_project_properties(local_experiment):
    """Test that experiment properties are accessible."""
    with local_experiment(name="my-experiment", project="my-project") as experiment:
        assert experiment.name == "my-experiment"
        assert experiment.project == "my-project"


def test_experiment_error_handling(local_experiment, temp_project):
    """Test that experiment handles errors gracefully and still saves data."""
    try:
        with local_experiment(name="error-test", project="test") as experiment:
            experiment.log("Starting work")
            experiment.parameters().set(param="value")
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Experiment should still be closed and data saved
    experiment_dir = temp_project / "test" / "error-test"
    assert experiment_dir.exists()
    assert (experiment_dir / "logs" / "logs.jsonl").exists()
    assert (experiment_dir / "parameters.json").exists()
