"""Tests for summary cache functionality."""
import pytest
from ml_dash import Experiment
from pathlib import Path
import json
import shutil


@pytest.fixture
def experiment():
    """Create a local experiment for testing."""
    exp = Experiment("test_exp", project="test_project", local_path=".ml-dash-test")
    exp.run.start()
    yield exp
    exp.run.complete()
    # Cleanup
    if Path(".ml-dash-test").exists():
        shutil.rmtree(".ml-dash-test")


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
        metric.summary_cache.store(
            loss=0.5 - i * 0.01,
            accuracy=0.8 + i * 0.01
        )

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
    peeked = metric.summary_cache.peek('loss', limit=3)
    assert 'loss' in peeked
    assert len(peeked['loss']) == 3
    # Use approximate comparison for floating point
    assert peeked['loss'][0] == pytest.approx(0.43)
    assert peeked['loss'][1] == pytest.approx(0.42)
    assert peeked['loss'][2] == pytest.approx(0.41)

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

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
