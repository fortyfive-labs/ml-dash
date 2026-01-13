"""
Test that all operations (parameters, logs, files, metrics) respect the prefix field.

This test verifies the storage path structure:
- All data is stored at: root / prefix
- Prefix format: owner/project/path.../name
"""

import json
import tempfile
from pathlib import Path

from ml_dash import Experiment


def test_all_operations_use_folder_field():
  """Test that parameters, logs, files, and metrics all respect prefix field."""
  with tempfile.TemporaryDirectory() as tmpdir:
    # Create experiment with prefix (new format: owner/project/path/name)
    exp = Experiment(
      prefix="test-user/test_project/iclr_2024/test_exp", dash_root=tmpdir
    )

    with exp.run:
      # Expected base path: root / prefix
      expected_base = Path(tmpdir) / "test-user/test_project/iclr_2024/test_exp"

      # 1. Test parameters
      exp.params.set(batch_size=128, lr=0.001)
      params_file = expected_base / "parameters.json"
      assert params_file.exists(), f"parameters.json not at {params_file}"

      with open(params_file) as f:
        params_data = json.load(f)
      assert params_data["data"]["batch_size"] == 128

      # 2. Test logs
      exp.log("Test log message", epoch=1)
      logs_file = expected_base / "logs/logs.jsonl"
      assert logs_file.exists(), f"logs.jsonl not at {logs_file}"

      with open(logs_file) as f:
        log_line = f.readline()
        log_data = json.loads(log_line)
      assert log_data["message"] == "Test log message"
      assert log_data["metadata"]["epoch"] == 1

      # 3. Test files
      test_file = Path(tmpdir) / "test_upload.txt"
      test_file.write_text("test content")
      exp.files(dir="models").upload(str(test_file))

      files_dir = expected_base / "files/models"
      assert files_dir.exists(), f"files directory not at {files_dir}"

      # Check metadata
      metadata_file = expected_base / "files/.files_metadata.json"
      assert metadata_file.exists(), f"files metadata not at {metadata_file}"

      with open(metadata_file) as f:
        files_meta = json.load(f)
      assert len(files_meta["files"]) == 1

      # 4. Test metrics - use correct API: metrics("name").log()
      exp.metrics("train").log(loss=0.5, accuracy=0.95, step=1)

      metrics_dir = expected_base / "metrics/train"
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

    print(f"✓ All operations correctly saved to: {expected_base}")


def test_folder_consistency_with_static_path():
  """Test folder consistency with a static path."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test-user/proj/custom/path/static_exp", dash_root=tmpdir)

    with exp.run:
      # Expected: root / prefix
      expected_base = Path(tmpdir) / "test-user/proj/custom/path/static_exp"

      # Add all types of data
      exp.params.set(test_param=123)
      exp.log("Test log")
      exp.metrics("train").log(loss=1.0)

      test_file = Path(tmpdir) / "test.txt"
      test_file.write_text("test")
      exp.files().upload(str(test_file))

      # Verify all in same location
      assert (expected_base / "parameters.json").exists()
      assert (expected_base / "logs/logs.jsonl").exists()
      assert (expected_base / "metrics/train/data.jsonl").exists()
      assert (expected_base / "files/.files_metadata.json").exists()
      assert (expected_base / "experiment.json").exists()

    print(f"✓ Static folder path works: {expected_base}")


def test_no_folder_field_still_works():
  """Test when prefix is just owner/project/name (minimal path)."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test-user/proj/no_folder_exp", dash_root=tmpdir)

    with exp.run:
      # Should use path: root / prefix
      expected_base = Path(tmpdir) / "test-user/proj/no_folder_exp"

      exp.params.set(test=1)
      exp.log("Test")
      exp.metrics("train").log(val=2.0)

      test_file = Path(tmpdir) / "test.txt"
      test_file.write_text("test")
      exp.files().upload(str(test_file))

      # All should be in the expected location
      assert (expected_base / "parameters.json").exists()
      assert (expected_base / "logs/logs.jsonl").exists()
      assert (expected_base / "metrics/train/data.jsonl").exists()
      assert (expected_base / "files/.files_metadata.json").exists()
      assert (expected_base / "experiment.json").exists()

    print(f"✓ Simple prefix works: {expected_base}")


if __name__ == "__main__":
  test_all_operations_use_folder_field()
  test_folder_consistency_with_static_path()
  test_no_folder_field_still_works()
  print("\n✅ All folder consistency tests passed!")
