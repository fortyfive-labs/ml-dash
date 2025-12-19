"""Comprehensive tests for summary cache in both local and remote modes."""
import pytest


class TestSummaryCacheLocal:
    """Test summary cache in local mode."""

    def test_basic_store_and_summarize(self, local_experiment):
        """Test basic store and summarize workflow."""
        exp = local_experiment(name="test_basic", project="test")
        exp.run.start()
        metric = exp.metrics("train")

        # Store 10 loss values
        for i in range(10):
            metric.summary_cache.store(loss=0.5 - i * 0.01)

        metric.summary_cache.summarize()

        # Verify
        exp.run.complete()
        exp.run.start()
        data = metric.read()

        assert len(data["data"]) == 1
        assert data["data"][0]["data"]["loss/count"] == 10
        assert "loss/mean" in data["data"][0]["data"]
        assert "loss/std" in data["data"][0]["data"]

        exp.run.complete()

    def test_multiple_calls_to_metrics(self, local_experiment):
        """Test that multiple calls to metrics() return cached instance."""
        exp = local_experiment(name="test_cache", project="test")
        exp.run.start()

        # Store using multiple calls (natural usage pattern)
        for i in range(10):
            exp.metrics("train").summary_cache.store(loss=0.5 - i * 0.01)

        exp.metrics("train").summary_cache.set(lr=0.001, epoch=1)
        exp.metrics("train").summary_cache.summarize()

        # Verify all data was accumulated
        exp.run.complete()
        exp.run.start()
        data = exp.metrics("train").read()

        assert len(data["data"]) == 1
        assert data["data"][0]["data"]["loss/count"] == 10
        assert data["data"][0]["data"]["lr"] == 0.001

        exp.run.complete()

    def test_rolling_window(self, local_experiment):
        """Test rolling window behavior (clear=True)."""
        exp = local_experiment(name="test_rolling", project="test")
        exp.run.start()
        metric = exp.metrics("train")

        # First window
        for i in range(5):
            metric.summary_cache.store(loss=0.5)
        metric.summary_cache.summarize()

        # Second window
        for i in range(5):
            metric.summary_cache.store(loss=0.3)
        metric.summary_cache.summarize()

        # Verify two separate summaries
        exp.run.complete()
        exp.run.start()
        data = metric.read()

        assert len(data["data"]) == 2
        assert data["data"][0]["data"]["loss/mean"] == pytest.approx(0.5)
        assert data["data"][1]["data"]["loss/mean"] == pytest.approx(0.3)

        exp.run.complete()

    def test_training_loop_pattern(self, local_experiment):
        """Test realistic training loop pattern."""
        exp = local_experiment(name="test_loop", project="test")
        exp.run.start()
        train_metric = exp.metrics("train")
        log_interval = 10

        # Simulate 50 batches
        for batch_idx in range(50):
            loss = 0.5 - batch_idx * 0.01
            train_metric.summary_cache.store(loss=loss)

            if batch_idx % log_interval == 0 and batch_idx > 0:
                train_metric.summary_cache.set(
                    lr=0.001 * (0.9 ** (batch_idx // log_interval)),
                    epoch=batch_idx // log_interval
                )
                train_metric.summary_cache.summarize()

        # Final summarize
        train_metric.summary_cache.summarize()

        # Verify results
        exp.run.complete()
        exp.run.start()
        data = train_metric.read()

        assert len(data["data"]) == 5
        assert data["data"][0]["data"]["loss/count"] == 11  # 0-10 inclusive
        assert data["data"][4]["data"]["loss/count"] == 9   # 41-49

        exp.run.complete()


@pytest.mark.remote
class TestSummaryCacheRemote:
    """Test summary cache in remote mode."""

    def test_basic_store_and_summarize(self, remote_experiment):
        """Test basic store and summarize workflow in remote mode."""
        exp = remote_experiment(name="test_basic_remote", project="test_summary")
        exp.run.start()
        metric = exp.metrics("train")

        # Store 10 loss values
        for i in range(10):
            metric.summary_cache.store(loss=0.5 - i * 0.01)

        metric.summary_cache.summarize()

        # Verify
        data = metric.read()
        assert len(data["data"]) == 1
        assert data["data"][0]["data"]["loss/count"] == 10

        exp.run.complete()

    def test_multiple_calls_to_metrics(self, remote_experiment):
        """Test that multiple calls to metrics() return cached instance in remote mode."""
        exp = remote_experiment(name="test_cache_remote", project="test_summary")
        exp.run.start()

        # Store using multiple calls
        for i in range(10):
            exp.metrics("train").summary_cache.store(loss=0.5 - i * 0.01)

        exp.metrics("train").summary_cache.set(lr=0.001, epoch=1)
        exp.metrics("train").summary_cache.summarize()

        # Verify
        data = exp.metrics("train").read()
        assert len(data["data"]) == 1
        assert data["data"][0]["data"]["loss/count"] == 10
        assert data["data"][0]["data"]["lr"] == 0.001

        exp.run.complete()

    def test_rolling_window(self, remote_experiment):
        """Test rolling window behavior in remote mode."""
        exp = remote_experiment(name="test_rolling_remote", project="test_summary")
        exp.run.start()
        metric = exp.metrics("train")

        # First window
        for i in range(5):
            metric.summary_cache.store(loss=0.5)
        metric.summary_cache.summarize()

        # Second window
        for i in range(5):
            metric.summary_cache.store(loss=0.3)
        metric.summary_cache.summarize()

        # Verify
        data = metric.read()
        assert len(data["data"]) == 2
        assert data["data"][0]["data"]["loss/mean"] == pytest.approx(0.5)
        assert data["data"][1]["data"]["loss/mean"] == pytest.approx(0.3)

        exp.run.complete()

    def test_training_loop_pattern(self, remote_experiment):
        """Test realistic training loop pattern in remote mode."""
        exp = remote_experiment(name="test_loop_remote", project="test_summary")
        exp.run.start()
        train_metric = exp.metrics("train")
        log_interval = 10

        # Simulate 50 batches
        for batch_idx in range(50):
            loss = 0.5 - batch_idx * 0.01
            train_metric.summary_cache.store(loss=loss)

            if batch_idx % log_interval == 0 and batch_idx > 0:
                train_metric.summary_cache.set(
                    lr=0.001 * (0.9 ** (batch_idx // log_interval)),
                    epoch=batch_idx // log_interval
                )
                train_metric.summary_cache.summarize()

        # Final summarize
        train_metric.summary_cache.summarize()

        # Verify
        data = train_metric.read()
        assert len(data["data"]) == 5
        assert data["data"][0]["data"]["loss/count"] == 11

        exp.run.complete()

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
