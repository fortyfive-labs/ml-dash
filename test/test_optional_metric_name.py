"""Test optional metric name functionality."""

import getpass
import json

import pytest


class TestOptionalMetricName:
  """Tests for optional metric name parameter."""

  def test_append_without_name_local(self, local_experiment, tmp_proj):
    """Test appending metric without specifying name (uses None)."""
    with local_experiment("57block/test/no-name-metric").run as experiment:
      experiment.metrics.log(loss=0.5, step=1)
      experiment.metrics.log(loss=0.4, step=2)
      experiment.metrics.log(loss=0.3, step=3)

    # Should create a "None" metric
    metric_file = (
      tmp_proj / getpass.getuser() / "test/no-name-metric/metrics/None/data.jsonl"
    )
    assert metric_file.exists()

    with open(metric_file) as f:
      data_points = [json.loads(line) for line in f]

    assert len(data_points) == 3
    assert data_points[0]["data"]["loss"] == 0.5
    assert data_points[1]["data"]["loss"] == 0.4
    assert data_points[2]["data"]["loss"] == 0.3

  def test_log_with_explicit_name_still_works(self, local_experiment, tmp_proj):
    """Test that explicit name via metrics() call works."""
    with local_experiment("57block/test/explicit-name").run as experiment:
      experiment.metrics("train").log(loss=0.5, step=1)
      experiment.metrics("train").log(loss=0.4, step=2)

    metric_file = (
      tmp_proj / getpass.getuser() / "test/explicit-name/metrics/train/data.jsonl"
    )
    assert metric_file.exists()

    with open(metric_file) as f:
      data_points = [json.loads(line) for line in f]

    assert len(data_points) == 2

  def test_multiple_logs_without_name_local(self, local_experiment, tmp_proj):
    """Test multiple log calls without name."""
    with local_experiment("57block/test/multi-no-name").run as experiment:
      experiment.metrics.log(loss=0.5, step=1)
      experiment.metrics.log(loss=0.4, step=2)
      experiment.metrics.log(loss=0.3, step=3)

    metric_file = (
      tmp_proj / getpass.getuser() / "test/multi-no-name/metrics/None/data.jsonl"
    )
    assert metric_file.exists()

    with open(metric_file) as f:
      points = [json.loads(line) for line in f]

    assert len(points) == 3

  def test_mixed_named_and_none_metrics(self, local_experiment, tmp_proj):
    """Test using both named and None metrics in same experiment."""
    with local_experiment("57block/test/mixed").run as experiment:
      # Log to None metric (unnamed)
      experiment.metrics.log(loss=1.0, step=0)

      # Log to named metric
      experiment.metrics("train").log(loss=0.5, step=0)

      # Log to None again
      experiment.metrics.log(loss=0.9, step=1)

    # Check both metrics exist
    none_file = tmp_proj / getpass.getuser() / "test/mixed/metrics/None/data.jsonl"
    loss_file = tmp_proj / getpass.getuser() / "test/mixed/metrics/train/data.jsonl"

    assert none_file.exists()
    assert loss_file.exists()

    with open(none_file) as f:
      none_points = [json.loads(line) for line in f]
    assert len(none_points) == 2

    with open(loss_file) as f:
      loss_points = [json.loads(line) for line in f]
    assert len(loss_points) == 1

  @pytest.mark.remote
  def test_append_without_name_remote(self, remote_experiment):
    """Test appending without name in remote mode."""
    with remote_experiment("57block/test/no-name-remote").run as experiment:
      experiment.metrics.log(loss=0.5, step=1)
      experiment.metrics.log(loss=0.4, step=2)

      # Verify stats show "None" metric (converted to string by server)
      stats = experiment.metrics(None).stats()
      assert stats["name"] == "None"

  @pytest.mark.remote
  def test_multiple_logs_without_name_remote(self, remote_experiment):
    """Test multiple log calls without name in remote mode."""
    with remote_experiment(
      "test-user/test/multi-no-name-remote"
    ).run as experiment:
      for i in range(10):
        experiment.metrics.log(loss=i * 0.1, step=i)

      # Verify it went to "None" metric (converted to string by server)
      stats = experiment.metrics(None).stats()
      assert stats["name"] == "None"


class TestLogAndFlushMethods:
  """Tests for log() and flush() methods (alias for append with method chaining)."""

  def test_metrics_log_method_local(self, local_experiment, tmp_proj):
    """Test metrics().log() as alias for append()."""
    with local_experiment("57block/test/log-method").run as experiment:
      experiment.metrics("train").log(loss=0.5, accuracy=0.8)
      experiment.metrics("train").log(loss=0.4, accuracy=0.85)

    metric_file = (
      tmp_proj / getpass.getuser() / "test/log-method/metrics/train/data.jsonl"
    )
    assert metric_file.exists()

    with open(metric_file) as f:
      data_points = [json.loads(line) for line in f]

    assert len(data_points) == 2
    assert data_points[0]["data"]["loss"] == 0.5
    assert data_points[0]["data"]["accuracy"] == 0.8

  def test_metrics_log_with_flush_chain_local(self, local_experiment, tmp_proj):
    """Test metrics.log(epoch=n).flush() chaining pattern."""
    with local_experiment("57block/test/log-flush-chain").run as experiment:
      experiment.metrics("train").log(loss=0.5, accuracy=0.8)
      experiment.metrics.log(epoch=1).flush()

      experiment.metrics("train").log(loss=0.4, accuracy=0.85)
      experiment.metrics.log(epoch=2).flush()

    # Check train metrics
    train_file = (
      tmp_proj / getpass.getuser() / "test/log-flush-chain/metrics/train/data.jsonl"
    )
    assert train_file.exists()

    with open(train_file) as f:
      train_points = [json.loads(line) for line in f]
    assert len(train_points) == 2

    # Check None metrics (epoch logging)
    none_file = (
      tmp_proj / getpass.getuser() / "test/log-flush-chain/metrics/None/data.jsonl"
    )
    assert none_file.exists()

    with open(none_file) as f:
      none_points = [json.loads(line) for line in f]
    assert len(none_points) == 2
    assert none_points[0]["data"]["epoch"] == 1
    assert none_points[1]["data"]["epoch"] == 2

  def test_metrics_log_nested_dict_pattern(self, local_experiment, tmp_proj):
    """Test metrics.log() with nested dict pattern."""
    with local_experiment("57block/test/log-nested").run as experiment:
      experiment.metrics.log(
        epoch=100,
        train=dict(loss=0.142, accuracy=0.80),
        eval=dict(loss=0.201, accuracy=0.76),
      )

    # Check train metrics
    train_file = (
      tmp_proj / getpass.getuser() / "test/log-nested/metrics/train/data.jsonl"
    )
    assert train_file.exists()

    with open(train_file) as f:
      train_points = [json.loads(line) for line in f]
    assert len(train_points) == 1
    assert train_points[0]["data"]["loss"] == 0.142
    assert train_points[0]["data"]["accuracy"] == 0.80
    assert train_points[0]["data"]["epoch"] == 100  # epoch included

    # Check eval metrics
    eval_file = tmp_proj / getpass.getuser() / "test/log-nested/metrics/eval/data.jsonl"
    assert eval_file.exists()

    with open(eval_file) as f:
      eval_points = [json.loads(line) for line in f]
    assert len(eval_points) == 1
    assert eval_points[0]["data"]["loss"] == 0.201
    assert eval_points[0]["data"]["accuracy"] == 0.76
    assert eval_points[0]["data"]["epoch"] == 100  # epoch included

  def test_metrics_log_with_flush_param(self, local_experiment, tmp_proj):
    """Test metrics.log(_flush=True) parameter."""
    with local_experiment("57block/test/log-flush-param").run as experiment:
      experiment.metrics("train").log(loss=0.5, accuracy=0.8)
      experiment.metrics.log(epoch=1, _flush=True)

    # Check None metrics
    none_file = (
      tmp_proj / getpass.getuser() / "test/log-flush-param/metrics/None/data.jsonl"
    )
    assert none_file.exists()

    with open(none_file) as f:
      none_points = [json.loads(line) for line in f]
    assert len(none_points) == 1
    assert none_points[0]["data"]["epoch"] == 1

  def test_fluent_method_chaining(self, local_experiment, tmp_proj):
    """Test that log() and flush() return self for chaining."""
    with local_experiment("57block/test/fluent-chain").run as experiment:
      # Test that these all return chainable objects
      result1 = experiment.metrics("train").log(loss=0.5)
      assert hasattr(result1, "log")  # Should return MetricBuilder
      assert hasattr(result1, "flush")

      result2 = experiment.metrics.log(epoch=1)
      assert hasattr(result2, "log")  # Should return MetricsManager
      assert hasattr(result2, "flush")

      result3 = experiment.metrics.flush()
      assert hasattr(result3, "log")  # Should return MetricsManager
      assert hasattr(result3, "flush")

  def test_complete_documented_pattern(self, local_experiment, tmp_proj):
    """Test the complete pattern from documentation:
    experiment.metrics("train").log(loss=train_loss, accuracy=train_acc)
    experiment.metrics("eval").log(loss=eval_loss, accuracy=eval_acc)
    experiment.metrics.log(epoch=epoch).flush()
    """
    with local_experiment("57block/test/complete-pattern").run as experiment:
      for epoch in range(3):
        train_loss = 1.0 - epoch * 0.1
        train_acc = 0.5 + epoch * 0.1
        eval_loss = 1.1 - epoch * 0.1
        eval_acc = 0.45 + epoch * 0.1

        experiment.metrics("train").log(loss=train_loss, accuracy=train_acc)
        experiment.metrics("eval").log(loss=eval_loss, accuracy=eval_acc)
        experiment.metrics.log(epoch=epoch).flush()

    # Check train metrics
    train_file = (
      tmp_proj / getpass.getuser() / "test/complete-pattern/metrics/train/data.jsonl"
    )
    with open(train_file) as f:
      train_points = [json.loads(line) for line in f]
    assert len(train_points) == 3

    # Check eval metrics
    eval_file = (
      tmp_proj / getpass.getuser() / "test/complete-pattern/metrics/eval/data.jsonl"
    )
    with open(eval_file) as f:
      eval_points = [json.loads(line) for line in f]
    assert len(eval_points) == 3

    # Check epoch context
    none_file = (
      tmp_proj / getpass.getuser() / "test/complete-pattern/metrics/None/data.jsonl"
    )
    with open(none_file) as f:
      none_points = [json.loads(line) for line in f]
    assert len(none_points) == 3
    assert none_points[0]["data"]["epoch"] == 0
    assert none_points[1]["data"]["epoch"] == 1
    assert none_points[2]["data"]["epoch"] == 2


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
