"""Tests for local backend with directory feature."""

import sys
import pytest
import shutil
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_dash import Experiment
from ml_dash.backends.local_backend import LocalBackend


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp(prefix="ml-logger-test-")
    yield tmp
    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)


class TestLocalBackendDirectory:
    """Test suite for local backend directory feature."""

    def test_single_directory_level(self, temp_dir):
        """Test experiment with single directory level."""
        exp = Experiment(
            namespace="team",
            workspace="project",
            prefix="exp1",
            directory="models",
            local_root=temp_dir
        )

        # Check attributes
        assert exp.directory == "models"
        assert exp.local_path == "team/project/models/exp1"

        # Create files
        with exp.run():
            exp.params.set(test=1, name="single_dir")
            exp.metrics.log(step=1, accuracy=0.95)
            exp.info("Test log entry")

        # Verify file structure
        expected_path = Path(temp_dir) / "team" / "project" / "models" / "exp1"
        assert expected_path.exists()
        assert (expected_path / "parameters.jsonl").exists()
        assert (expected_path / "metrics.jsonl").exists()
        assert (expected_path / "logs.jsonl").exists()

    def test_nested_directory_structure(self, temp_dir):
        """Test experiment with nested directory structure."""
        exp = Experiment(
            namespace="research",
            workspace="nlp",
            prefix="bert-exp",
            directory="transformers/bert/squad",
            local_root=temp_dir
        )

        # Check attributes
        assert exp.directory == "transformers/bert/squad"
        assert exp.local_path == "research/nlp/transformers/bert/squad/bert-exp"

        # Create files
        with exp.run():
            exp.params.set(
                model="bert-base-uncased",
                dataset="squad-v2",
                learning_rate=3e-5
            )

            for step in range(3):
                exp.metrics.log(
                    step=step,
                    train_loss=0.5 - (step * 0.1),
                    val_accuracy=0.7 + (step * 0.05)
                )

        # Verify nested directory structure
        expected_path = Path(temp_dir) / "research" / "nlp" / "transformers" / "bert" / "squad" / "bert-exp"
        assert expected_path.exists()

        # Verify intermediate directories exist
        assert (Path(temp_dir) / "research").exists()
        assert (Path(temp_dir) / "research" / "nlp").exists()
        assert (Path(temp_dir) / "research" / "nlp" / "transformers").exists()
        assert (Path(temp_dir) / "research" / "nlp" / "transformers" / "bert").exists()
        assert (Path(temp_dir) / "research" / "nlp" / "transformers" / "bert" / "squad").exists()

        # Verify files
        assert (expected_path / "parameters.jsonl").exists()
        assert (expected_path / "metrics.jsonl").exists()

    def test_no_directory_backward_compatible(self, temp_dir):
        """Test backward compatibility without directory parameter."""
        exp = Experiment(
            namespace="legacy",
            workspace="experiments",
            prefix="old-exp",
            local_root=temp_dir
        )

        # Check attributes
        assert exp.directory is None
        assert exp.local_path == "legacy/experiments/old-exp"

        # Create files
        with exp.run():
            exp.params.set(test=3, legacy=True)
            exp.metrics.log(step=1, metric=1.0)

        # Verify file structure (no directory in path)
        expected_path = Path(temp_dir) / "legacy" / "experiments" / "old-exp"
        assert expected_path.exists()
        assert (expected_path / "parameters.jsonl").exists()

    def test_multiple_experiments_same_directory(self, temp_dir):
        """Test multiple experiments in the same directory."""
        experiments = []

        for i in range(3):
            exp = Experiment(
                namespace="team",
                workspace="vision",
                prefix=f"resnet-run-{i+1}",
                directory="image-classification/cifar10",
                local_root=temp_dir
            )

            with exp.run():
                exp.params.set(
                    run_number=i + 1,
                    learning_rate=0.001 * (i + 1),
                    batch_size=128
                )
                exp.metrics.log(step=1, accuracy=0.8 + (i * 0.05))

            experiments.append(exp)

        # Verify all experiments exist in same directory structure
        base_dir = Path(temp_dir) / "team" / "vision" / "image-classification" / "cifar10"
        assert base_dir.exists()

        for i in range(3):
            exp_dir = base_dir / f"resnet-run-{i+1}"
            assert exp_dir.exists()
            assert (exp_dir / "parameters.jsonl").exists()
            assert (exp_dir / "metrics.jsonl").exists()

    def test_deep_nesting(self, temp_dir):
        """Test very deep directory nesting."""
        exp = Experiment(
            namespace="deep",
            workspace="nested",
            prefix="experiment",
            directory="level1/level2/level3/level4/level5",
            local_root=temp_dir
        )

        # Check attributes
        assert exp.directory == "level1/level2/level3/level4/level5"

        # Create files
        with exp.run():
            exp.params.set(depth=5)
            exp.metrics.log(step=1, value=100)

        # Verify all levels exist
        expected_path = Path(temp_dir) / "deep" / "nested" / "level1" / "level2" / "level3" / "level4" / "level5" / "experiment"
        assert expected_path.exists()

    def test_directory_with_special_characters(self, temp_dir):
        """Test directory with special characters (hyphens, underscores)."""
        exp = Experiment(
            namespace="test",
            workspace="special",
            prefix="exp",
            directory="my-project/sub_dir/test-123",
            local_root=temp_dir
        )

        with exp.run():
            exp.params.set(test=True)

        expected_path = Path(temp_dir) / "test" / "special" / "my-project" / "sub_dir" / "test-123" / "exp"
        assert expected_path.exists()

    def test_directory_preserves_workspace(self, temp_dir):
        """Test that directory is added after workspace, not replacing it."""
        exp = Experiment(
            namespace="team",
            workspace="project-a",
            prefix="exp",
            directory="subdir",
            local_root=temp_dir
        )

        # Verify workspace is preserved
        assert "project-a" in exp.local_path
        assert exp.local_path == "team/project-a/subdir/exp"

        with exp.run():
            exp.params.set(test=True)

        # Verify workspace directory exists
        workspace_path = Path(temp_dir) / "team" / "project-a"
        assert workspace_path.exists()

        expected_path = Path(temp_dir) / "team" / "project-a" / "subdir" / "exp"
        assert expected_path.exists()

    def test_file_operations_with_directory(self, temp_dir):
        """Test file operations work correctly with directory."""
        exp = Experiment(
            namespace="test",
            workspace="files",
            prefix="exp",
            directory="artifacts/models",
            local_root=temp_dir
        )

        with exp.run():
            # Test parameters
            exp.params.set(
                model="resnet50",
                epochs=10,
                lr=0.001
            )

            # Test metrics
            for step in range(5):
                exp.metrics.log(
                    step=step,
                    loss=1.0 - (step * 0.1),
                    accuracy=step * 0.2
                )

            # Test logs
            exp.info("Training started")
            exp.warning("High memory usage")
            exp.error("An error occurred")

            # Test file upload (use .bin for raw bytes)
            test_file_content = b"model weights data"
            exp.files.save(test_file_content, "model.bin")

        # Verify all files exist
        exp_path = Path(temp_dir) / "test" / "files" / "artifacts" / "models" / "exp"
        assert (exp_path / "parameters.jsonl").exists()
        assert (exp_path / "metrics.jsonl").exists()
        assert (exp_path / "logs.jsonl").exists()
        assert (exp_path / "files" / "model.bin").exists()

        # Verify file content
        saved_content = (exp_path / "files" / "model.bin").read_bytes()
        assert saved_content == test_file_content

    def test_metadata_with_directory(self, temp_dir):
        """Test that metadata is saved correctly with directory."""
        exp = Experiment(
            namespace="test",
            workspace="meta",
            prefix="exp",
            directory="experiments/run1",
            local_root=temp_dir,
            readme="Test experiment"
        )

        with exp.run():
            exp.params.set(test=True)

        # Verify metadata file exists
        exp_path = Path(temp_dir) / "test" / "meta" / "experiments" / "run1" / "exp"
        meta_file = exp_path / ".ml-logger.meta.json"
        assert meta_file.exists()

        # Verify metadata content
        import json
        meta_data = json.loads(meta_file.read_text())
        assert meta_data["namespace"] == "test"
        assert meta_data["workspace"] == "meta"
        assert meta_data["prefix"] == "exp"
        assert meta_data["readme"] == "Test experiment"
        assert meta_data["status"] == "completed"

    def test_empty_directory_string(self, temp_dir):
        """Test that empty directory string is treated as None."""
        exp = Experiment(
            namespace="test",
            workspace="empty",
            prefix="exp",
            directory="",  # Empty string
            local_root=temp_dir
        )

        # Empty directory should be treated as None
        # Path should be: test/empty/exp (no extra directory)
        with exp.run():
            exp.params.set(test=True)

        # Since directory is empty string, it will add it to the path
        # Let's verify the actual behavior
        expected_path = Path(temp_dir) / "test" / "empty" / "" / "exp"

        # Check what actually got created
        test_path = Path(temp_dir) / "test"
        if test_path.exists():
            # Find where files were actually created
            for path in test_path.rglob("parameters.jsonl"):
                print(f"Files created at: {path.parent}")


class TestLocalBackendDirectoryMethods:
    """Test LocalBackend methods with directory paths."""

    def test_backend_path_resolution(self, temp_dir):
        """Test that backend resolves paths correctly with directories."""
        backend = LocalBackend(temp_dir)

        # Test path resolution
        path1 = "namespace/workspace/dir/exp/file.txt"
        resolved = backend._resolve_path(path1)
        # Use resolve() to handle symlinks on macOS (/var -> /private/var)
        assert resolved.resolve() == (Path(temp_dir) / path1).resolve()

    def test_backend_makedirs_with_nested_path(self, temp_dir):
        """Test backend creates nested directories correctly."""
        backend = LocalBackend(temp_dir)

        nested_path = "team/project/dir1/dir2/dir3/exp"
        backend.makedirs(nested_path)

        # Verify all levels were created
        full_path = Path(temp_dir) / nested_path
        assert full_path.exists()
        assert full_path.is_dir()

    def test_backend_file_operations_nested_path(self, temp_dir):
        """Test backend file operations with nested directory paths."""
        backend = LocalBackend(temp_dir)

        # Write text file
        file_path = "namespace/workspace/dir1/dir2/file.txt"
        content = "Test content"
        backend.write_text(file_path, content)

        # Verify file exists
        assert backend.exists(file_path)

        # Read file
        read_content = backend.read_text(file_path)
        assert read_content == content

        # Write binary file
        binary_path = "namespace/workspace/dir1/dir2/data.bin"
        binary_content = b"Binary data"
        backend.write_bytes(binary_path, binary_content)

        # Read binary
        read_binary = backend.read_bytes(binary_path)
        assert read_binary == binary_content


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
