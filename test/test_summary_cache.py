"""Tests for summary cache functionality."""

import shutil
from pathlib import Path

import pytest

from ml_dash import Experiment


@pytest.fixture
def experiment():
  """Create a local experiment for testing."""
  exp = Experiment(prefix="test-user/test_project/test_exp", dash_root=".dash-test")
  exp.run.start()
  yield exp
  exp.run.complete()
  # Cleanup
  if Path(".dash-test").exists():
    shutil.rmtree(".dash-test")


def test_summary_cache_store_and_summarize(experiment):
  """Test basic store and summarize workflow."""
  metric = experiment.metrics("train")

  # Store 10 loss values
  for i in range(10):
    metric.summary_cache.store(loss=0.5 - i * 0.01)

  # Summarize
  metric.summary_cache.summarize()

  # Verify statistics were appended
  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert len(data["data"]) == 1
  assert "loss.mean" in data["data"][0]["data"]
  assert "loss.min" in data["data"][0]["data"]
  assert "loss.max" in data["data"][0]["data"]
  assert "loss.std" in data["data"][0]["data"]
  assert "loss.count" in data["data"][0]["data"]
  assert data["data"][0]["data"]["loss.count"] == 10


def test_summary_cache_rolling_window(experiment):
  """Test that clear=True (default) creates rolling window."""
  metric = experiment.metrics("train")

  # First window
  for i in range(5):
    metric.summary_cache.store(loss=0.5)
  metric.summary_cache.summarize()  # clear=True by default

  # Second window
  for i in range(5):
    metric.summary_cache.store(loss=0.3)
  metric.summary_cache.summarize()

  # Verify two separate summaries
  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert len(data["data"]) == 2
  assert data["data"][0]["data"]["loss.mean"] == pytest.approx(0.5)
  assert data["data"][1]["data"]["loss.mean"] == pytest.approx(0.3)


def test_summary_cache_cumulative(experiment):
  """Test that clear=False creates cumulative statistics."""
  metric = experiment.metrics("train")

  # First batch
  metric.summary_cache.store(loss=0.5)
  metric.summary_cache.summarize(clear=False)

  # Second batch (cumulative)
  metric.summary_cache.store(loss=0.3)
  metric.summary_cache.summarize(clear=False)

  # Verify cumulative statistics
  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert len(data["data"]) == 2
  assert data["data"][0]["data"]["loss.count"] == 1
  assert data["data"][1]["data"]["loss.count"] == 2  # Cumulative


def test_summary_cache_set_metadata(experiment):
  """Test set() for metadata that doesn't need aggregation."""
  metric = experiment.metrics("train")

  # Store values and set metadata
  for i in range(5):
    metric.summary_cache.store(loss=0.5 - i * 0.01)

  metric.summary_cache.set(lr=0.001, epoch=1)

  # Summarize combines metadata with statistics
  metric.summary_cache.summarize()

  # Verify metadata and stats are in same data point
  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert len(data["data"]) == 1
  assert data["data"][0]["data"]["lr"] == 0.001
  assert data["data"][0]["data"]["epoch"] == 1
  assert "loss.mean" in data["data"][0]["data"]
  assert data["data"][0]["data"]["loss.count"] == 5

  # Second interval - metadata should be replaced
  for i in range(3):
    metric.summary_cache.store(loss=0.3)

  metric.summary_cache.set(lr=0.0005, epoch=2)  # Updated values
  metric.summary_cache.summarize()

  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert len(data["data"]) == 2
  assert data["data"][1]["data"]["lr"] == 0.0005  # Replaced
  assert data["data"][1]["data"]["epoch"] == 2  # Replaced
  assert data["data"][1]["data"]["loss.count"] == 3


def test_summary_cache_multiple_metrics(experiment):
  """Test storing multiple metrics simultaneously."""
  metric = experiment.metrics("train")

  # Store multiple metrics
  for i in range(5):
    metric.summary_cache.store(loss=0.5 - i * 0.01, accuracy=0.8 + i * 0.01)

  metric.summary_cache.summarize()

  # Verify both metrics summarized
  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert len(data["data"]) == 1
  assert "loss.mean" in data["data"][0]["data"]
  assert "accuracy.mean" in data["data"][0]["data"]


def test_summary_cache_peek(experiment):
  """Test non-destructive peek operation (ml-logger pattern)."""
  metric = experiment.metrics("train")

  # Store some values
  for i in range(10):
    metric.summary_cache.store(loss=0.5 - i * 0.01)

  # Peek at last 3 values
  peeked = metric.summary_cache.peek("loss", limit=3)
  assert "loss" in peeked
  assert len(peeked["loss"]) == 3
  # Use approximate comparison for floating point
  assert peeked["loss"][0] == pytest.approx(0.43)
  assert peeked["loss"][1] == pytest.approx(0.42)
  assert peeked["loss"][2] == pytest.approx(0.41)

  # Peek doesn't affect the buffer
  metric.summary_cache.summarize()
  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  assert data["data"][0]["data"]["loss.count"] == 10  # All 10 values still there


def test_summary_cache_nan_handling(experiment):
  """Test that None and NaN values are handled gracefully."""
  metric = experiment.metrics("train")

  # Store values with None
  metric.summary_cache.store(loss=0.5, accuracy=None)
  metric.summary_cache.store(loss=None, accuracy=0.9)
  metric.summary_cache.store(loss=0.4, accuracy=0.85)

  metric.summary_cache.summarize()

  experiment.run.complete()
  experiment.run.start()

  data = metric.read()
  # Only non-None values should be counted
  assert data["data"][0]["data"]["loss.count"] == 2
  assert data["data"][0]["data"]["accuracy.count"] == 2
  assert data["data"][0]["data"]["loss.mean"] == pytest.approx(0.45)
  assert data["data"][0]["data"]["accuracy.mean"] == pytest.approx(0.875)


# =============================================================================
# New Buffer API Tests
# =============================================================================


def test_buffer_basic_usage(experiment):
  """Test basic buffer() and log_summary() workflow."""
  # Accumulate values with prefix
  for i in range(10):
    experiment.metrics("train").buffer(loss=0.5 - i * 0.01, accuracy=0.8 + i * 0.01)

  # Log summary (default: mean)
  experiment.metrics.buffer.log_summary()

  # Verify statistics were logged
  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  assert len(data["data"]) == 1
  assert "loss.mean" in data["data"][0]["data"]
  assert "accuracy.mean" in data["data"][0]["data"]
  assert data["data"][0]["data"]["loss.mean"] == pytest.approx(0.455)
  assert data["data"][0]["data"]["accuracy.mean"] == pytest.approx(0.845)


def test_buffer_multiple_aggs(experiment):
  """Test log_summary() with multiple aggregations."""
  for i in range(10):
    experiment.metrics("train").buffer(loss=0.5 - i * 0.01)

  # Log with multiple aggregations
  experiment.metrics.buffer.log_summary("mean", "std", "min", "max", "count")

  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  assert len(data["data"]) == 1
  assert "loss.mean" in data["data"][0]["data"]
  assert "loss.std" in data["data"][0]["data"]
  assert "loss.min" in data["data"][0]["data"]
  assert "loss.max" in data["data"][0]["data"]
  assert "loss.count" in data["data"][0]["data"]
  assert data["data"][0]["data"]["loss.count"] == 10


def test_buffer_multiple_prefixes(experiment):
  """Test buffering across multiple prefixes."""
  # Buffer to different prefixes
  for i in range(5):
    experiment.metrics("train").buffer(loss=0.5 - i * 0.01)
    experiment.metrics("val").buffer(loss=0.6 - i * 0.01)

  # Single log_summary() flushes all prefixes
  experiment.metrics.buffer.log_summary()

  experiment.run.complete()
  experiment.run.start()

  train_data = experiment.metrics("train").read()
  val_data = experiment.metrics("val").read()

  assert len(train_data["data"]) == 1
  assert len(val_data["data"]) == 1
  assert train_data["data"][0]["data"]["loss.mean"] == pytest.approx(0.48)
  assert val_data["data"][0]["data"]["loss.mean"] == pytest.approx(0.58)


def test_buffer_with_direct_log(experiment):
  """Test combining buffer with direct log."""
  # Buffer training metrics
  for i in range(5):
    experiment.metrics("train").buffer(loss=0.5)

  # Log summary
  experiment.metrics.buffer.log_summary()

  # Direct log (non-buffered)
  experiment.metrics.log(epoch=1, lr=0.001)

  experiment.run.complete()
  experiment.run.start()

  train_data = experiment.metrics("train").read()
  assert len(train_data["data"]) == 1
  assert "loss.mean" in train_data["data"][0]["data"]


def test_buffer_percentiles(experiment):
  """Test percentile aggregations."""
  # Create a range of values
  for i in range(100):
    experiment.metrics("train").buffer(loss=float(i) / 100)

  experiment.metrics.buffer.log_summary("p50", "p90", "p95", "p99")

  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  assert "loss.p50" in data["data"][0]["data"]
  assert "loss.p90" in data["data"][0]["data"]
  assert "loss.p95" in data["data"][0]["data"]
  assert "loss.p99" in data["data"][0]["data"]
  # p50 should be around 0.5
  assert data["data"][0]["data"]["loss.p50"] == pytest.approx(0.495, abs=0.01)


def test_buffer_first_last(experiment):
  """Test first and last aggregations."""
  experiment.metrics("train").buffer(loss=0.5)
  experiment.metrics("train").buffer(loss=0.4)
  experiment.metrics("train").buffer(loss=0.3)

  experiment.metrics.buffer.log_summary("first", "last")

  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  assert data["data"][0]["data"]["loss.first"] == pytest.approx(0.5)
  assert data["data"][0]["data"]["loss.last"] == pytest.approx(0.3)


def test_buffer_clears_after_log_summary(experiment):
  """Test that buffer is cleared after log_summary."""
  # First batch
  for i in range(5):
    experiment.metrics("train").buffer(loss=0.5)
  experiment.metrics.buffer.log_summary()

  # Second batch (should start fresh)
  for i in range(5):
    experiment.metrics("train").buffer(loss=0.3)
  experiment.metrics.buffer.log_summary()

  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  assert len(data["data"]) == 2
  assert data["data"][0]["data"]["loss.mean"] == pytest.approx(0.5)
  assert data["data"][1]["data"]["loss.mean"] == pytest.approx(0.3)


def test_buffer_invalid_agg_raises_error(experiment):
  """Test that invalid aggregation raises ValueError."""
  experiment.metrics("train").buffer(loss=0.5)

  with pytest.raises(ValueError, match="Unsupported aggregation"):
    experiment.metrics.buffer.log_summary("invalid_agg")


def test_buffer_nan_handling(experiment):
  """Test that None values are handled gracefully in buffer."""
  experiment.metrics("train").buffer(loss=0.5, accuracy=None)
  experiment.metrics("train").buffer(loss=None, accuracy=0.9)
  experiment.metrics("train").buffer(loss=0.4, accuracy=0.85)

  experiment.metrics.buffer.log_summary("mean", "count")

  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  # Only non-None values should be counted
  assert data["data"][0]["data"]["loss.count"] == 2
  assert data["data"][0]["data"]["accuracy.count"] == 2
  assert data["data"][0]["data"]["loss.mean"] == pytest.approx(0.45)


def test_buffer_method_chaining(experiment):
  """Test that buffer() returns self for method chaining."""
  # This should work without errors
  result = experiment.metrics("train").buffer(loss=0.5).buffer(loss=0.4)
  assert result._name == "train"

  experiment.metrics.buffer.log_summary()

  experiment.run.complete()
  experiment.run.start()

  data = experiment.metrics("train").read()
  assert data["data"][0]["data"]["loss.mean"] == pytest.approx(0.45)


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
