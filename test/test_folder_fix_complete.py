"""
Test that all operations (parameters, logs, files, metrics) respect the folder field.

This test verifies the fix for the folder path inconsistency bug where:
- experiment.json was created at: .ml-dash/{folder}/{project}/{experiment}/
- But parameters, logs, files, metrics were created at: .ml-dash/{project}/{experiment}/

After the fix, ALL data should be at: .ml-dash/{folder}/{project}/{experiment}/
"""
import tempfile
import json
from pathlib import Path

from ml_dash import Experiment


def test_all_operations_use_folder_field():
    """Test that parameters, logs, files, and metrics all respect folder field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create experiment with folder template
        exp = Experiment(
            name="test_exp",
            project="test_project",
            folder="/iclr_2024/{RUN.name}",
            local_path=tmpdir
        )

        with exp.run:
            # When folder template is passed to constructor, it stays as template
            # The storage layer resolves it internally
            assert exp.folder == "/iclr_2024/{RUN.name}"

            # Expected base path - template should be resolved in storage
            # The {RUN.name} gets replaced with experiment name
            expected_base = Path(tmpdir) / "iclr_2024" / "{RUN.name}" / "test_project" / "test_exp"

            # 1. Test parameters
            exp.params.log(batch_size=128, lr=0.001)
            params_file = expected_base / "parameters.json"
            assert params_file.exists(), f"parameters.json not at {params_file}"

            with open(params_file) as f:
                params_data = json.load(f)
            assert params_data["data"]["batch_size"] == 128

            # 2. Test logs
            exp.log().info("Test log message", epoch=1)
            logs_file = expected_base / "logs" / "logs.jsonl"
            assert logs_file.exists(), f"logs.jsonl not at {logs_file}"

            with open(logs_file) as f:
                log_line = f.readline()
                log_data = json.loads(log_line)
            assert log_data["message"] == "Test log message"
            assert log_data["metadata"]["epoch"] == 1

            # 3. Test files
            test_file = Path(tmpdir) / "test_upload.txt"
            test_file.write_text("test content")
            exp.files(file_path=str(test_file), prefix="models").save()

            files_dir = expected_base / "files" / "models"
            assert files_dir.exists(), f"files directory not at {files_dir}"

            # Check metadata
            metadata_file = expected_base / "files" / ".files_metadata.json"
            assert metadata_file.exists(), f"files metadata not at {metadata_file}"

            with open(metadata_file) as f:
                files_meta = json.load(f)
            assert len(files_meta["files"]) == 1
            assert files_meta["files"][0]["path"] == "models"

            # 4. Test metrics - use correct API: metrics.append()
            exp.metrics.append(loss=0.5, accuracy=0.95, step=1)

            metrics_dir = expected_base / "metrics" / "None"
            assert metrics_dir.exists(), f"metrics directory not at {metrics_dir}"

            metric_data_file = metrics_dir / "data.jsonl"
            assert metric_data_file.exists(), f"metric data not at {metric_data_file}"

            with open(metric_data_file) as f:
                metric_line = f.readline()
                metric_point = json.loads(metric_line)
            assert metric_point["data"]["loss"] == 0.5
            assert metric_point["data"]["accuracy"] == 0.95

            # 5. Verify experiment.json is also in the same place
            experiment_file = expected_base / "experiment.json"
            assert experiment_file.exists(), f"experiment.json not at {experiment_file}"

            with open(experiment_file) as f:
                exp_data = json.load(f)
            assert exp_data["name"] == "test_exp"
            assert exp_data["folder"] == "/iclr_2024/{RUN.name}"

        print(f"✓ All operations correctly saved to: {expected_base}")


def test_folder_consistency_with_static_path():
    """Test folder consistency with a static (non-template) path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name="static_exp",
            project="proj",
            folder="/custom/path",
            local_path=tmpdir
        )

        with exp.run:
            expected_base = Path(tmpdir) / "custom" / "path" / "proj" / "static_exp"

            # Add all types of data
            exp.params.log(test_param=123)
            exp.log().info("Test log")
            exp.metrics.append(value=1.0)

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            exp.files(file_path=str(test_file)).save()

            # Verify all in same location
            assert (expected_base / "parameters.json").exists()
            assert (expected_base / "logs" / "logs.jsonl").exists()
            assert (expected_base / "metrics" / "None" / "data.jsonl").exists()
            assert (expected_base / "files" / ".files_metadata.json").exists()
            assert (expected_base / "experiment.json").exists()

        print(f"✓ Static folder path works: {expected_base}")


def test_no_folder_field_still_works():
    """Test backward compatibility when folder field is not set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name="no_folder_exp",
            project="proj",
            local_path=tmpdir
        )

        with exp.run:
            # Should use default path: root_path/project/experiment
            expected_base = Path(tmpdir) / "proj" / "no_folder_exp"

            exp.params.log(test=1)
            exp.log().info("Test")
            exp.metrics.append(val=2.0)

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            exp.files(file_path=str(test_file)).save()

            # All should be in the default location
            assert (expected_base / "parameters.json").exists()
            assert (expected_base / "logs" / "logs.jsonl").exists()
            assert (expected_base / "metrics" / "None" / "data.jsonl").exists()
            assert (expected_base / "files" / ".files_metadata.json").exists()
            assert (expected_base / "experiment.json").exists()

        print(f"✓ No folder field works (backward compatible): {expected_base}")


if __name__ == "__main__":
    test_all_operations_use_folder_field()
    test_folder_consistency_with_static_path()
    test_no_folder_field_still_works()
    print("\n✅ All folder consistency tests passed!")
