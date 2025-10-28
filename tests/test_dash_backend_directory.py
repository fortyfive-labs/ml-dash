"""Tests for dash backend with directory feature.

These tests require a running ml-dash-server and MongoDB instance.

Prerequisites:
1. Start MongoDB: docker-compose -f ../../docker/docker-compose.yml up -d
2. Start ml-dash-server: cd ../../ml-dash/ml-dash-server && pnpm dev
3. Run tests: pytest tests/test_dash_backend_directory.py -v

Set SKIP_INTEGRATION_TESTS=1 to skip these tests if server is not running.
"""

import sys
import os
import pytest
import time
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_dash import Experiment
from ml_dash.backends.dash_backend import DashBackend

# Server configuration
SERVER_URL = os.environ.get("ML_LOGGER_SERVER_URL", "https://qwqdug4btp.us-east-1.awsapprunner.com")
SKIP_IF_NO_SERVER = os.environ.get("SKIP_INTEGRATION_TESTS", "0") == "1"


def check_server_health():
    """Check if ml-dash-server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


# Skip all tests if server is not running
pytestmark = pytest.mark.skipif(
    SKIP_IF_NO_SERVER or not check_server_health(),
    reason="ml-dash-server not running or SKIP_INTEGRATION_TESTS=1"
)


@pytest.fixture
def unique_namespace():
    """Generate unique namespace for each test to avoid conflicts."""
    return f"test-{int(time.time() * 1000)}"


class TestDashBackendDirectory:
    """Test suite for dash backend directory feature."""

    def test_server_health(self):
        """Verify server is running."""
        assert check_server_health(), f"Server at {SERVER_URL} is not responding"

    def test_single_directory_level(self, unique_namespace):
        """Test experiment with single directory level."""
        exp = Experiment(
            namespace=unique_namespace,
            workspace="ml-experiments",
            prefix="test-single-dir",
            directory="models",
            remote=SERVER_URL,
            readme="Testing single directory level"
        )

        # Check attributes
        assert exp.directory == "models"
        assert exp.experiment_id is not None
        # namespace_id is only available on the backend
        assert isinstance(exp.backend, DashBackend)
        assert exp.backend.namespace_id is not None

        # Create run and log data
        with exp.run():
            exp.params.set(test="single_directory", level=1)
            exp.metrics.log(step=1, accuracy=0.95)
            exp.info("Single directory test completed")
            logs = exp.logs.read()
            for log in logs:
                print(f"[{log['level']}] {log['message']}")

        assert exp.run_id is not None

    def test_nested_directory_structure(self, unique_namespace):
        """Test experiment with nested directory structure."""
        exp = Experiment(
            namespace=unique_namespace,
            workspace="nlp-projects",
            prefix="bert-experiment",
            directory="transformers/bert/squad",
            remote=SERVER_URL,
            readme="Testing nested directory structure"
        )

        # Check attributes
        assert exp.directory == "transformers/bert/squad"
        assert exp.experiment_id is not None

        # Create run and log data
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

            exp.info("Nested directory test completed")

        assert exp.run_id is not None

        # Verify we can query the experiment via GraphQL
        query = f"""
        query {{
            experiment(id: "{exp.experiment_id}") {{
                id
                name
                namespaceId
                directoryId
            }}
        }}
        """

        response = requests.post(
            f"{SERVER_URL}/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["experiment"]["directoryId"] is not None

    def test_no_directory_backward_compatible(self, unique_namespace):
        """Test backward compatibility without directory parameter."""
        exp = Experiment(
            namespace=unique_namespace,
            workspace="old-experiments",
            prefix="no-directory-test",
            remote=SERVER_URL,
        )

        # Check attributes
        assert exp.directory is None
        assert exp.experiment_id is not None

        # Create run
        with exp.run():
            exp.params.set(test="no_directory")
            exp.metrics.log(step=1, metric=1.0)

        assert exp.run_id is not None

        # Query experiment - directoryId should be null
        query = f"""
        query {{
            experiment(id: "{exp.experiment_id}") {{
                id
                directoryId
            }}
        }}
        """

        response = requests.post(
            f"{SERVER_URL}/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )

        data = response.json()
        # directoryId should be null or not present
        exp_data = data.get("data", {}).get("experiment", {})
        assert exp_data.get("directoryId") is None

    def test_multiple_experiments_same_directory(self, unique_namespace):
        """Test multiple experiments in the same directory."""
        experiments = []

        for i in range(3):
            exp = Experiment(
                namespace=unique_namespace,
                workspace="vision",
                prefix=f"resnet-run-{i+1}",
                directory="image-classification/cifar10",
                remote=SERVER_URL,
                tags=["resnet", "cifar10", f"run-{i+1}"]
            )

            with exp.run():
                exp.params.set(
                    run_number=i + 1,
                    learning_rate=0.001 * (i + 1),
                    batch_size=128
                )
                exp.metrics.log(step=1, accuracy=0.8 + (i * 0.05))

            experiments.append(exp)

        # Verify all experiments were created
        assert len(experiments) == 3
        assert all(exp.experiment_id is not None for exp in experiments)
        assert all(exp.run_id is not None for exp in experiments)

        # Verify they all have different IDs
        exp_ids = [exp.experiment_id for exp in experiments]
        assert len(set(exp_ids)) == 3, "Experiments should have unique IDs"

    def test_deep_nesting(self, unique_namespace):
        """Test very deep directory nesting."""
        exp = Experiment(
            namespace=unique_namespace,
            workspace="nested-work",
            prefix="deep-experiment",
            directory="level1/level2/level3/level4/level5",
            remote=SERVER_URL,
        )

        assert exp.directory == "level1/level2/level3/level4/level5"
        assert exp.experiment_id is not None

        with exp.run():
            exp.params.set(depth=5)
            exp.metrics.log(step=1, value=100)

        assert exp.run_id is not None

    def test_directory_reuse(self, unique_namespace):
        """Test that creating multiple experiments in same directory reuses directory records."""
        # Create first experiment
        exp1 = Experiment(
            namespace=unique_namespace,
            workspace="shared",
            prefix="exp1",
            directory="shared-dir/subdir",
            remote=SERVER_URL,
        )

        with exp1.run():
            exp1.params.set(test=1)

        # Create second experiment in same directory
        exp2 = Experiment(
            namespace=unique_namespace,
            workspace="shared",
            prefix="exp2",
            directory="shared-dir/subdir",
            remote=SERVER_URL,
        )

        with exp2.run():
            exp2.params.set(test=2)

        # Both should have experiment IDs and run IDs
        assert exp1.experiment_id is not None
        assert exp2.experiment_id is not None
        assert exp1.run_id is not None
        assert exp2.run_id is not None

        # Experiments should be different
        assert exp1.experiment_id != exp2.experiment_id

    def test_file_upload_with_directory(self, unique_namespace):
        """Test file upload works with directory."""
        exp = Experiment(
            namespace=unique_namespace,
            workspace="files-test",
            prefix="upload-exp",
            directory="artifacts/models",
            remote=SERVER_URL,
        )

        with exp.run():
            exp.params.set(test="file_upload")

            # Upload a file
            test_file_content = b"model checkpoint data"
            exp.files.save(test_file_content, "checkpoint.bin")
            f = exp.files.list()
            assert len(f) == 1
        assert exp.run_id is not None

        # Query artifacts via GraphQL
        query = f"""
        query {{
            artifactsByRun(runId: "{exp.run_id}") {{
                id
                path
                type
                blob
                size
            }}
        }}
        """

        response = requests.post(
            f"{SERVER_URL}/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )

        data = response.json()
        artifacts = data.get("data", {}).get("artifactsByRun", [])

        assert len(artifacts) > 0
        artifact = artifacts[0]
        assert artifact["path"] == "checkpoint.bin"
        assert artifact["size"] == len(test_file_content)
        assert "s3://" in artifact["blob"]

    def test_experiment_name_uniqueness_within_directory(self, unique_namespace):
        """Test that experiment names are unique within a directory."""
        # Create first experiment
        exp1 = Experiment(
            namespace=unique_namespace,
            workspace="unique-test",
            prefix="duplicate-name",
            directory="test-dir",
            remote=SERVER_URL,
        )

        with exp1.run():
            exp1.params.set(version=1)

        exp1_id = exp1.experiment_id

        # Try to create another experiment with same name in same directory
        # Should return the existing experiment
        exp2 = Experiment(
            namespace=unique_namespace,
            workspace="unique-test",
            prefix="duplicate-name",
            directory="test-dir",
            remote=SERVER_URL,
        )

        # Should get the same experiment ID
        assert exp2.experiment_id == exp1_id

    def test_experiment_name_uniqueness_different_directories(self, unique_namespace):
        """Test that same experiment name can exist in different directories."""
        # Create experiment in dir1
        exp1 = Experiment(
            namespace=unique_namespace,
            workspace="multi-dir",
            prefix="same-name",
            directory="dir1",
            remote=SERVER_URL,
        )

        with exp1.run():
            exp1.params.set(location="dir1")

        # Create experiment with same name in dir2
        exp2 = Experiment(
            namespace=unique_namespace,
            workspace="multi-dir",
            prefix="same-name",
            directory="dir2",
            remote=SERVER_URL,
        )

        with exp2.run():
            exp2.params.set(location="dir2")

        # Should have different experiment IDs
        assert exp1.experiment_id != exp2.experiment_id

    def test_all_operations_with_directory(self, unique_namespace):
        """Test all major operations work with directory."""
        exp = Experiment(
            namespace=unique_namespace,
            workspace="full-test",
            prefix="all-operations",
            directory="complete/test/path",
            remote=SERVER_URL,
            readme="Testing all operations",
            tags=["comprehensive", "test"]
        )

        with exp.run():
            # Parameters
            exp.params.set(
                model="resnet50",
                epochs=10,
                batch_size=128,
                lr=0.001
            )

            # Metrics
            for step in range(5):
                exp.metrics.log(
                    step=step,
                    loss=1.0 - (step * 0.15),
                    accuracy=0.5 + (step * 0.1),
                    lr=0.001 - (step * 0.0001)
                )

            # Logs
            exp.info("Training started")
            exp.info(f"Epoch 1 completed", epoch=1, loss=0.5)
            exp.warning("Memory usage high", usage_gb=15.2)

            # File upload
            config_data = {"model": "resnet50", "layers": 50}
            exp.files.save(config_data, "config.json")

        # Verify everything was created
        assert exp.experiment_id is not None
        assert exp.run_id is not None

        # Query via GraphQL to verify
        query = f"""
        query {{
            experiment(id: "{exp.experiment_id}") {{
                id
                name
                description
                tags
                directoryId
                run {{
                    id
                    status
                }}
            }}
            artifactsByRun(runId: "{exp.run_id}") {{
                id
                path
            }}
        }}
        """

        response = requests.post(
            f"{SERVER_URL}/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )

        data = response.json()
        exp_data = data["data"]["experiment"]

        assert exp_data["name"] == "all-operations"
        assert exp_data["description"] == "Testing all operations"
        assert "comprehensive" in exp_data["tags"]
        assert "test" in exp_data["tags"]
        assert exp_data["directoryId"] is not None
        assert exp_data["run"]["status"] == "COMPLETED"

        artifacts = data["data"]["artifactsByRun"]
        assert len(artifacts) > 0


class TestDashBackendDirectoryMethods:
    """Test DashBackend methods directly."""

    def test_backend_initialization(self, unique_namespace):
        """Test DashBackend initializes correctly with directory."""
        backend = DashBackend(
            server_url=SERVER_URL,
            namespace=unique_namespace,
            workspace="backend-test",
            experiment_name="test-exp",
            directory="test/directory"
        )

        assert backend.directory == "test/directory"
        assert backend.server_url == SERVER_URL

    def test_initialize_experiment_with_directory(self, unique_namespace):
        """Test initialize_experiment sends directory to server."""
        backend = DashBackend(
            server_url=SERVER_URL,
            namespace=unique_namespace,
            workspace="backend-test",
            experiment_name="test-init",
            directory="backend/test/dir"
        )

        experiment = backend.initialize_experiment(
            description="Test initialization",
            tags=["backend", "test"]
        )

        assert experiment["id"] is not None
        assert backend.experiment_id is not None
        assert backend.namespace_id is not None

    def test_create_run_after_initialize(self, unique_namespace):
        """Test creating run after initializing experiment."""
        backend = DashBackend(
            server_url=SERVER_URL,
            namespace=unique_namespace,
            workspace="backend-test",
            experiment_name="test-run",
            directory="run/test"
        )

        # Initialize experiment
        backend.initialize_experiment()

        # Create run
        run = backend.create_run(
            name="test-run-1",
            tags=["test"],
            metadata={"version": "1.0"}
        )

        assert run["id"] is not None
        assert backend.run_id is not None

    def test_log_metrics_with_directory(self, unique_namespace):
        """Test logging metrics works with directory."""
        backend = DashBackend(
            server_url=SERVER_URL,
            namespace=unique_namespace,
            workspace="metrics-test",
            experiment_name="test-metrics",
            directory="metrics/test"
        )

        backend.initialize_experiment()
        backend.create_run()

        # Log metrics
        result = backend.log_metrics({
            "loss": [
                {"step": 1, "timestamp": time.time() * 1000, "value": 0.5},
                {"step": 2, "timestamp": time.time() * 1000, "value": 0.3},
            ],
            "accuracy": [
                {"step": 1, "timestamp": time.time() * 1000, "value": 0.8},
                {"step": 2, "timestamp": time.time() * 1000, "value": 0.9},
            ]
        })

        assert result["success"] is True

    def test_log_parameters_with_directory(self, unique_namespace):
        """Test logging parameters works with directory."""
        backend = DashBackend(
            server_url=SERVER_URL,
            namespace=unique_namespace,
            workspace="params-test",
            experiment_name="test-params",
            directory="params/test"
        )

        backend.initialize_experiment()

        # Log parameters
        result = backend.log_parameters(
            parameters={
                "model": "resnet50",
                "learning_rate": 0.001,
                "batch_size": 128
            },
            operation="set"
        )

        assert result["success"] is True

    def test_upload_file_with_directory(self, unique_namespace):
        """Test file upload works with directory."""
        backend = DashBackend(
            server_url=SERVER_URL,
            namespace=unique_namespace,
            workspace="upload-test",
            experiment_name="test-upload",
            directory="upload/test"
        )

        backend.initialize_experiment()
        backend.create_run()

        # Upload file
        file_data = b"test file content for directory test"
        result = backend.upload_file(
            name="test-file.txt",
            file_data=file_data,
            artifact_type="OTHER",
            mime_type="text/plain",
            metadata={"test": True}
        )

        assert result["success"] is True
        assert result["artifact"]["path"] == "test-file.txt"
        assert result["artifact"]["size"] == len(file_data)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
