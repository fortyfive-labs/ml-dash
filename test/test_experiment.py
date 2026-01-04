"""Comprehensive tests for Experiment operations in both local and remote modes."""
import json
import pytest
from pathlib import Path


class TestExperimentCreation:
    """Tests for basic experiment creation and lifecycle."""

    def test_context_manager_local(self, local_experiment, temp_project):
        """Test experiment creation using context manager in local mode."""
        with local_experiment(name="test-ctx", project="test-ws").run as experiment:
            assert experiment._is_open
            assert experiment.name == "test-ctx"
            assert experiment.project == "test-ws"
            experiment.log("Test message")

        assert not experiment._is_open
        # New structure: root / owner / project / prefix
        experiment_dir = temp_project /  "test-ws" / "test-ctx"
        assert experiment_dir.exists()
        assert (experiment_dir / "experiment.json").exists()

    @pytest.mark.remote
    def test_context_manager_remote(self, remote_experiment):
        """Test experiment creation using context manager in remote mode."""
        with remote_experiment(name="test-ctx-remote", project="test-ws-remote").run as experiment:
            assert experiment._is_open
            # Name has timestamp suffix for uniqueness
            assert experiment.name.startswith("test-ctx-remote-")
            assert experiment.project == "test-ws-remote"
            experiment.log("Test message from remote")

        assert not experiment._is_open

    def test_manual_open_close_local(self, local_experiment, temp_project):
        """Test manual experiment lifecycle management in local mode."""
        experiment = local_experiment(name="manual-test", project="test-ws")
        assert not experiment._is_open

        experiment.run.start()
        assert experiment._is_open

        experiment.log("Working...")
        experiment.params.set(test_param="test_value")

        experiment.run.complete()
        assert not experiment._is_open

        # Verify data was saved
        # New structure: root / owner / project / prefix
        experiment_dir = temp_project /  "test-ws" / "manual-test"
        assert experiment_dir.exists()
        assert (experiment_dir / "logs" / "logs.jsonl").exists()

    @pytest.mark.remote
    def test_manual_open_close_remote(self, remote_experiment):
        """Test manual experiment lifecycle management in remote mode."""
        experiment = remote_experiment(name="manual-test-remote", project="test-ws")
        assert not experiment._is_open

        experiment.run.start()
        assert experiment._is_open

        experiment.log("Working remotely...")
        experiment.params.set(test_param="remote_value")

        experiment.run.complete()
        assert not experiment._is_open

    def test_experiment_with_metadata_local(self, local_experiment, temp_project):
        """Test experiment with description, tags, and prefix in local mode."""
        with local_experiment(
            name="meta-experiment",
            project="meta-ws",
            description="Test experiment with metadata",
            tags=["test", "metadata", "local"],
            prefix="experiments/meta/meta-experiment",
        ).run as experiment:
            experiment.log("Experiment with metadata")

        # Verify metadata
        # New structure: root / owner / project / prefix
        experiment_file = temp_project /  "meta-ws" / "experiments" / "meta" / "meta-experiment" / "experiment.json"
        assert experiment_file.exists()

        with open(experiment_file) as f:
            metadata = json.load(f)
            assert metadata["name"] == "meta-experiment"
            assert metadata["project"] == "meta-ws"
            assert metadata["description"] == "Test experiment with metadata"
            assert "test" in metadata["tags"]
            assert "metadata" in metadata["tags"]
            assert metadata["prefix"] == "experiments/meta/meta-experiment"

    @pytest.mark.remote
    def test_experiment_with_metadata_remote(self, remote_experiment):
        """Test experiment with description, tags, and folder in remote mode."""
        with remote_experiment(
            name="meta-experiment-remote",
            project="meta-ws-remote",
            description="Remote test experiment with metadata",
            tags=["test", "metadata", "remote"],
            folder="/experiments/remote",
        ).run as experiment:
            experiment.log("Remote experiment with metadata")
            # In remote mode, metadata is sent to server


class TestExperimentProperties:
    """Tests for experiment properties and attributes."""

    def test_experiment_properties_local(self, local_experiment):
        """Test accessing experiment properties in local mode."""
        with local_experiment(name="props-test", project="props-ws").run as experiment:
            assert experiment.name == "props-test"
            assert experiment.project == "props-ws"
            assert experiment._is_open

    @pytest.mark.remote
    def test_experiment_properties_remote(self, remote_experiment):
        """Test accessing experiment properties in remote mode."""
        with remote_experiment(name="props-test-remote", project="props-ws-remote").run as experiment:
            # Name has timestamp suffix for uniqueness
            assert experiment.name.startswith("props-test-remote-")
            assert experiment.project == "props-ws-remote"
            assert experiment._is_open


class TestMultipleExperiments:
    """Test experiments."""

    def test_experiments_same_project_local(self, local_experiment, temp_project):
        """Test experiments in the same project."""
        with local_experiment(name="experiment-1", project="shared-ws").run as experiment:
            experiment.log("Experiment 1")
            experiment.params.set(experiment_id=1)

        with local_experiment(name="experiment-2", project="shared-ws").run as experiment:
            experiment.log("Experiment 2")
            experiment.params.set(experiment_id=2)

        with local_experiment(name="experiment-3", project="shared-ws").run as experiment:
            experiment.log("Experiment 3")
            experiment.params.set(experiment_id=3)

        # Verify all experiments exist
        # New structure: root / owner / project / prefix
        project_dir = temp_project /  "shared-ws"
        assert (project_dir / "experiment-1").exists()
        assert (project_dir / "experiment-2").exists()
        assert (project_dir / "experiment-3").exists()

    @pytest.mark.remote
    def test_experiments_same_project_remote(self, remote_experiment):
        """Test experiments in the same project in remote mode."""
        with remote_experiment(name="remote-experiment-1", project="shared-ws-remote").run as experiment:
            experiment.log("Remote Experiment 1")
            experiment.params.set(experiment_id=1)

        with remote_experiment(name="remote-experiment-2", project="shared-ws-remote").run as experiment:
            experiment.log("Remote Experiment 2")
            experiment.params.set(experiment_id=2)

    def test_experiments_different_projects_local(self, local_experiment, temp_project):
        """Test experiments in different projects."""
        with local_experiment(name="experiment-a", project="project-1").run as experiment:
            experiment.log("Experiment A in project 1")

        with local_experiment(name="experiment-b", project="project-2").run as experiment:
            experiment.log("Experiment B in project 2")

        with local_experiment(name="experiment-c", project="project-3").run as experiment:
            experiment.log("Experiment C in project 3")

        # Verify all projects and experiments exist
        # New structure: root / owner / project / prefix
        assert (temp_project /  "project-1" / "experiment-a").exists()
        assert (temp_project /  "project-2" / "experiment-b").exists()
        assert (temp_project /  "project-3" / "experiment-c").exists()

    def test_experiments_local(self, local_experiment):
        """Test experiments sequentially."""
        experiments = []
        for i in range(5):
            with local_experiment(name=f"seq-experiment-{i}", project="sequential").run as experiment:
                experiment.log(f"Sequential experiment {i}")
                experiment.params.set(index=i)
                experiments.append(experiment)

        # All experiments should be closed
        for experiment in experiments:
            assert not experiment._is_open


class TestExperimentErrorHandling:
    """Test experiments."""

    def test_experiment_error_still_saves_data_local(self, local_experiment, temp_project):
        """Test that experiment saves data even when errors occur."""
        try:
            with local_experiment(name="error-test", project="error-ws").run as experiment:
                experiment.log("Starting work")
                experiment.params.set(param="value")
                experiment.metrics("metric").append(value=0.5, step=0)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Data should still be saved
        # New structure: root / owner / project / prefix
        experiment_dir = temp_project /  "error-ws" / "error-test"
        assert experiment_dir.exists()
        assert (experiment_dir / "logs" / "logs.jsonl").exists()
        assert (experiment_dir / "parameters.json").exists()

    @pytest.mark.remote
    def test_experiment_error_still_saves_data_remote(self, remote_experiment):
        """Test that remote experiment handles errors gracefully."""
        try:
            with remote_experiment(name="error-test-remote", project="error-ws-remote").run as experiment:
                experiment.log("Starting remote work")
                experiment.params.set(param="remote_value")
                raise ValueError("Simulated remote error")
        except ValueError:
            pass
        # Experiment should be closed properly

    def test_experiment_local(self, local_experiment, temp_project):
        """Test experiment handling multiple errors."""
        with local_experiment(name="multi-error", project="error-ws").run as experiment:
            try:
                experiment.log("Attempt 1")
                raise ValueError("Error 1")
            except ValueError:
                experiment.log("Caught error 1", level="error")

            try:
                experiment.log("Attempt 2")
                raise RuntimeError("Error 2")
            except RuntimeError:
                experiment.log("Caught error 2", level="error")

            experiment.log("Continuing after errors")

        # Experiment should have all logs
        # New structure: root / owner / project / prefix
        logs_file = temp_project /  "error-ws" / "multi-error" / "logs" / "logs.jsonl"
        assert logs_file.exists()

        with open(logs_file) as f:
            logs = [json.loads(line) for line in f]

        assert len(logs) >= 3


class TestExperimentReuse:
    """Test experiments."""

    def test_experiment_local(self, local_experiment, temp_project):
        """Test reopening an existing experiment (upsert behavior)."""
        # Create initial experiment
        with local_experiment(name="reuse-experiment", project="reuse-ws").run as experiment:
            experiment.log("Initial experiment")
            experiment.params.set(version=1)

        # Reopen same experiment
        with local_experiment(name="reuse-experiment", project="reuse-ws").run as experiment:
            experiment.log("Reopened experiment")
            experiment.params.set(version=2, new_param="added")

        # Verify both operations are recorded
        # New structure: root / owner / project / prefix
        experiment_dir = temp_project /  "reuse-ws" / "reuse-experiment"
        logs_file = experiment_dir / "logs" / "logs.jsonl"

        with open(logs_file) as f:
            logs = [json.loads(line) for line in f]

        assert len(logs) >= 2

    @pytest.mark.remote
    def test_experiment_remote(self, remote_experiment):
        """Test reopening an existing experiment in remote mode."""
        # Create initial experiment
        with remote_experiment(name="reuse-experiment-remote", project="reuse-ws-remote").run as experiment:
            experiment.log("Initial remote experiment")
            experiment.params.set(version=1)

        # Reopen same experiment
        with remote_experiment(name="reuse-experiment-remote", project="reuse-ws-remote").run as experiment:
            experiment.log("Reopened remote experiment")
            experiment.params.set(version=2)


class TestExperimentEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_experiment_local(self, local_experiment, temp_project):
        """Test experiment with no operations."""
        with local_experiment(name="empty-experiment", project="empty-ws").run as experiment:
            pass  # Do nothing

        # Experiment directory should still be created
        # New structure: root / owner / project / prefix
        experiment_dir = temp_project /  "empty-ws" / "empty-experiment"
        assert experiment_dir.exists()

    def test_experiment_with_special_characters_local(self, local_experiment, temp_project):
        """Test experiment names with special characters."""
        with local_experiment(name="test-experiment_v1.0", project="special-ws").run as experiment:
            experiment.log("Experiment with special chars in name")

        # New structure: root / owner / project / prefix
        experiment_dir = temp_project /  "special-ws" / "test-experiment_v1.0"
        assert experiment_dir.exists()

    def test_experiment_with_long_name_local(self, local_experiment):
        """Test experiment with very long name."""
        long_name = "a" * 200
        with local_experiment(name=long_name, project="long-ws").run as experiment:
            experiment.log("Experiment with long name")

    def test_deeply_nested_prefix_local(self, local_experiment, temp_project):
        """Test experiment with deeply nested prefix structure."""
        with local_experiment(
            name="nested-experiment",
            project="nested-ws",
            prefix="a/b/c/d/e/f/g/h/nested-experiment",
        ).run as experiment:
            experiment.log("Deeply nested experiment")

        # New structure: root / owner / project / prefix
        experiment_file = temp_project /  "nested-ws" / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h" / "nested-experiment" / "experiment.json"
        with open(experiment_file) as f:
            metadata = json.load(f)
            assert metadata["prefix"] == "a/b/c/d/e/f/g/h/nested-experiment"

    def test_experiment_with_many_tags_local(self, local_experiment, temp_project):
        """Test experiment with many tags."""
        tags = [f"tag-{i}" for i in range(50)]
        with local_experiment(
            name="many-tags",
            project="tags-ws",
            tags=tags,
        ).run as experiment:
            experiment.log("Experiment with many tags")

        # New structure: root / owner / project / prefix
        experiment_file = temp_project /  "tags-ws" / "many-tags" / "experiment.json"
        with open(experiment_file) as f:
            metadata = json.load(f)
            assert len(metadata["tags"]) == 50

    def test_experiment_double_close_local(self, local_experiment):
        """Test that closing a experiment twice doesn't cause issues."""
        experiment = local_experiment(name="double-close", project="test-ws")
        experiment.run.start()
        experiment.run.complete()
        experiment.run.complete()  # Should not raise error
        assert not experiment._is_open

    def test_operations_before_open_local(self, local_experiment):
        """Test that operations before open are handled gracefully."""
        experiment = local_experiment(name="not-opened", project="test-ws")
        # Attempting operations before opening should handle gracefully
        # The actual behavior depends on implementation

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
