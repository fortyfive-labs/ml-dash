"""Performance tests for ML-Dash - stress testing with large data volumes."""

import time

import pytest


@pytest.mark.skip(reason="Skipped for now")
class TestMetricPerformance:
  """Performance tests for metric operations."""

  @pytest.mark.slow
  def test_million_metrics_local(self, local_experiment, tmp_proj):
    """Test creating and storing one million metric data points."""
    total_metrics = 1_000_000

    print(f"\nStarting test to create {total_metrics:,} metrics...")
    start_time = time.time()

    with local_experiment(
      name="million-metrics", project="perf-test"
    ).run as experiment:
      metric = experiment.metrics("train")
      # Track progress
      for step in range(total_metrics):
        metric.log(
          step=step,
          loss=1.0 / (step + 1),  # Decreasing value simulating loss
          epoch=step // 1000,
        )

        # Progress update every 100k
        if (step + 1) % 100_000 == 0:
          elapsed = time.time() - start_time
          rate = (step + 1) / elapsed
          print(
            f"  Progress: {step + 1:,}/{total_metrics:,} ({(step + 1) / total_metrics * 100:.1f}%) - "
            f"{rate:.0f} metrics/sec - ETA: {(total_metrics - step - 1) / rate:.1f}s"
          )

      # Verify the metrics were created (before context closes)
      stats = experiment.metrics("train").stats()

    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    rate = total_metrics / total_time

    print(f"\n✓ Successfully created {total_metrics:,} metrics")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average rate: {rate:.0f} metrics/second")
    print(f"  Time per metric: {total_time / total_metrics * 1000:.3f} ms")
    print(f"  Verified: {stats['totalDataPoints']} data points stored")
    print(
      f"  Buffered: {stats['bufferedDataPoints']}, Chunked: {stats['chunkedDataPoints']}"
    )

    assert int(stats["totalDataPoints"]) == total_metrics

  @pytest.mark.slow
  @pytest.mark.remote
  def test_million_metrics_remote(self, remote_experiment):
    """Test creating one million metrics on remote server."""
    total_metrics = 1_080_000

    print(f"\nStarting remote test to create {total_metrics:,} metrics...")
    start_time = time.time()

    with remote_experiment(
      name="million-metrics-remote", project="perf-test"
    ).run as experiment:
      metric = experiment.metrics("train")
      for step in range(total_metrics):
        metric.log(
          step=step,
          loss=1.0 / (step + 1),
          epoch=step // 1000,
        )

        if (step + 1) % 100_000 == 0:
          elapsed = time.time() - start_time
          rate = (step + 1) / elapsed
          print(
            f"  Progress: {step + 1:,}/{total_metrics:,} ({(step + 1) / total_metrics * 100:.1f}%) - "
            f"{rate:.0f} metrics/sec"
          )

    end_time = time.time()
    total_time = end_time - start_time
    rate = total_metrics / total_time

    print(f"\n✓ Successfully created {total_metrics:,} remote metrics")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average rate: {rate:.0f} metrics/second")

  @pytest.mark.slow
  def test_hundred_thousand_metrics_individual(self, local_experiment, tmp_proj):
    """Test creating 100k metrics individually."""
    total_metrics = 100_000

    print(f"\nStarting test to create {total_metrics:,} metrics individually...")
    start_time = time.time()

    with local_experiment(
      name="hundred-k-individual", project="perf-test"
    ).run as experiment:
      metric = experiment.metrics("train")

      for i in range(total_metrics):
        metric.log(step=i, loss=1.0 / (i + 1), epoch=i // 1000)

        # Progress update every 10k
        if (i + 1) % 10_000 == 0:
          elapsed = time.time() - start_time
          rate = (i + 1) / elapsed
          print(
            f"  Progress: {i + 1:,}/{total_metrics:,} ({(i + 1) / total_metrics * 100:.1f}%) - "
            f"{rate:.0f} metrics/sec"
          )

      # Verify (before context closes)
      stats = experiment.metrics("train").stats()

    end_time = time.time()
    total_time = end_time - start_time
    rate = total_metrics / total_time

    print(f"\n✓ Successfully created {total_metrics:,} individual metrics")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average rate: {rate:.0f} metrics/second")

    assert int(stats["totalDataPoints"]) == total_metrics

  @pytest.mark.slow
  def test_multiple_metrics_concurrent(self, local_experiment, tmp_proj):
    """Test creating multiple metric streams concurrently."""
    num_metrics = 10
    points_per_metric = 100_000
    total_points = num_metrics * points_per_metric

    print(
      f"\nStarting test with {num_metrics} concurrent metric streams, "
      f"{points_per_metric:,} points each ({total_points:,} total)..."
    )
    start_time = time.time()

    with local_experiment(
      name="concurrent-metrics", project="perf-test"
    ).run as experiment:
      for metric_idx in range(num_metrics):
        metric_name = f"metric_{metric_idx}"
        metric = experiment.metrics(metric_name)

        for step in range(points_per_metric):
          metric.log(
            step=step,
            loss=(metric_idx + 1) * 0.1 / (step + 1),
          )

        # Progress after each metric
        elapsed = time.time() - start_time
        processed = (metric_idx + 1) * points_per_metric
        rate = processed / elapsed
        print(
          f"  Completed metric {metric_idx + 1}/{num_metrics}: "
          f"{processed:,}/{total_points:,} total points - {rate:.0f} pts/sec"
        )

    end_time = time.time()
    total_time = end_time - start_time
    rate = total_points / total_time

    print(
      f"\n✓ Successfully created {num_metrics} metrics with {total_points:,} total points"
    )
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average rate: {rate:.0f} metrics/second")

  @pytest.mark.slow
  def test_large_batch_sizes(self, local_experiment, tmp_proj):
    """Test performance with high-frequency logging."""
    points_per_test = 100_000

    print(f"\nTesting logging performance with {points_per_test:,} points...")

    with local_experiment(name="logging-perf", project="perf-test").run as experiment:
      start_time = time.time()
      metric = experiment.metrics("train")

      for step in range(points_per_test):
        metric.log(step=step, loss=step * 0.001)

      elapsed = time.time() - start_time
      rate = points_per_test / elapsed

      print(
        f"  Logged {points_per_test:,} points: {elapsed:6.2f}s, {rate:8.0f} pts/sec"
      )

    print("\n✓ Completed logging performance test")


class TestMetricReadPerformance:
  """Performance tests for reading large metric datasets."""

  @pytest.mark.slow
  def test_read_million_metrics(self, local_experiment, tmp_proj):
    """Test reading from a large metric dataset."""
    total_metrics = 50_000  # Create smaller dataset for read test

    print(f"\nCreating {total_metrics:,} metrics for read performance test...")

    with local_experiment(name="read-perf", project="perf-test").run as experiment:
      # Create data
      metric = experiment.metrics("train")
      for step in range(total_metrics):
        metric.log(step=step, loss=step * 0.001)

      # Test reading different ranges
      print("\nTesting read performance...")

      # Read first 1000
      start = time.time()
      result = experiment.metrics("train").read(start_index=0, limit=1000)
      elapsed = time.time() - start
      print(f"  Read first 1,000 points: {elapsed * 1000:.2f}ms")
      assert len(result["data"]) == 1000

      # Read middle 1000
      start = time.time()
      middle_index = total_metrics // 2
      result = experiment.metrics("train").read(start_index=middle_index, limit=1000)
      elapsed = time.time() - start
      print(f"  Read middle 1,000 points: {elapsed * 1000:.2f}ms")
      assert len(result["data"]) == 1000

      # Read last 1000
      start = time.time()
      result = experiment.metrics("train").read(
        start_index=total_metrics - 1000, limit=1000
      )
      elapsed = time.time() - start
      print(f"  Read last 1,000 points: {elapsed * 1000:.2f}ms")
      assert len(result["data"]) == 1000

      # Read large range (10k)
      start = time.time()
      result = experiment.metrics("train").read(start_index=0, limit=10000)
      elapsed = time.time() - start
      print(f"  Read 10,000 points: {elapsed * 1000:.2f}ms")
      assert len(result["data"]) == 10000

      # Get stats
      start = time.time()
      stats = experiment.metrics("train").stats()
      elapsed = time.time() - start
      print(f"  Get stats: {elapsed * 1000:.2f}ms")

    # Verify total data points (after context closes)
    assert int(stats["totalDataPoints"]) == total_metrics


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
