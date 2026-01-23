"""
Comprehensive test suite for buffering system.

Tests background buffer manager for logs, metrics, and files.
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_dash import Experiment
from ml_dash.buffer import BackgroundBufferManager, BufferConfig


@pytest.fixture
def buffer_config():
    """Create a test buffer configuration with short intervals."""
    return BufferConfig(
        flush_interval=0.5,  # Short interval for testing
        log_batch_size=5,
        metric_batch_size=5,
        file_upload_workers=2,
        buffer_enabled=True,
    )


@pytest.fixture
def buffered_experiment(local_experiment, buffer_config, monkeypatch):
    """
    Create a test experiment with buffering enabled.

    Returns a function that creates experiments with buffering.
    """
    def _create_experiment(prefix="test-user/test-project/buffer-test", **kwargs):
        # Patch BufferConfig.from_env to return our test config
        monkeypatch.setattr(
            "ml_dash.experiment.BufferConfig.from_env",
            lambda: buffer_config
        )
        return local_experiment(prefix, **kwargs)

    return _create_experiment


@pytest.fixture
def unbuffered_experiment(local_experiment, monkeypatch):
    """
    Create a test experiment with buffering disabled.
    """
    def _create_experiment(prefix="test-user/test-project/unbuffered-test", **kwargs):
        # Patch BufferConfig.from_env to disable buffering
        disabled_config = BufferConfig(buffer_enabled=False)
        monkeypatch.setattr(
            "ml_dash.experiment.BufferConfig.from_env",
            lambda: disabled_config
        )
        return local_experiment(prefix, **kwargs)

    return _create_experiment


class TestBufferConfig:
    """Test BufferConfig initialization and environment loading."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BufferConfig()
        assert config.flush_interval == 5.0
        assert config.log_batch_size == 100
        assert config.metric_batch_size == 100
        assert config.file_upload_workers == 4
        assert config.buffer_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BufferConfig(
            flush_interval=2.0,
            log_batch_size=50,
            metric_batch_size=75,
            file_upload_workers=2,
            buffer_enabled=False,
        )
        assert config.flush_interval == 2.0
        assert config.log_batch_size == 50
        assert config.metric_batch_size == 75
        assert config.file_upload_workers == 2
        assert config.buffer_enabled is False

    def test_config_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("ML_DASH_FLUSH_INTERVAL", "3.0")
        monkeypatch.setenv("ML_DASH_LOG_BATCH_SIZE", "200")
        monkeypatch.setenv("ML_DASH_METRIC_BATCH_SIZE", "150")
        monkeypatch.setenv("ML_DASH_FILE_UPLOAD_WORKERS", "8")
        monkeypatch.setenv("ML_DASH_BUFFER_ENABLED", "false")

        config = BufferConfig.from_env()
        assert config.flush_interval == 3.0
        assert config.log_batch_size == 200
        assert config.metric_batch_size == 150
        assert config.file_upload_workers == 8
        assert config.buffer_enabled is False


class TestLogBuffering:
    """Test log buffering behavior."""

    def test_buffer_batches_logs(self, buffered_experiment):
        """Verify logs are batched before sending."""
        with buffered_experiment().run as exp:
            # Write multiple logs quickly
            for i in range(10):
                exp.logs.info(f"Log message {i}")

            # Verify buffer has logs queued
            assert not exp._buffer_manager._log_queue.empty()

            # Manual flush
            exp.flush()
            time.sleep(0.1)  # Wait for flush to complete

            # Queue should be empty after flush
            assert exp._buffer_manager._log_queue.empty()

    def test_buffer_size_based_flush(self, buffered_experiment):
        """Verify size-based flush trigger works."""
        with buffered_experiment().run as exp:
            # Write exactly batch_size logs (5 in test config)
            for i in range(5):
                exp.logs.info(f"Log message {i}")

            # Wait for background thread to flush
            time.sleep(0.2)

            # Queue should be flushed
            assert exp._buffer_manager._log_queue.qsize() < 5

    def test_buffer_time_based_flush(self, buffered_experiment):
        """Verify time-based flush trigger works."""
        with buffered_experiment().run as exp:
            # Write a few logs
            exp.logs.info("Message 1")
            exp.logs.info("Message 2")

            # Wait for flush interval (0.5s in test config)
            time.sleep(0.6)

            # Queue should be flushed
            assert exp._buffer_manager._log_queue.empty()

    def test_logs_printed_immediately(self, buffered_experiment, capsys):
        """Verify logs are printed to console immediately despite buffering."""
        with buffered_experiment().run as exp:
            exp.logs.info("Test message")

            # Check stdout immediately (before flush)
            captured = capsys.readouterr()
            assert "Test message" in captured.out

    def test_buffer_disabled_fallback(self, unbuffered_experiment):
        """Verify immediate writes when buffering disabled."""
        with unbuffered_experiment().run as exp:
            # No buffer manager should exist
            assert exp._buffer_manager is None

            # Logs should still work
            exp.logs.info("Immediate log")
            exp.logs.error("Immediate error")


class TestMetricBuffering:
    """Test metric buffering behavior."""

    def test_buffer_batches_metrics(self, buffered_experiment):
        """Verify metrics are batched before sending."""
        with buffered_experiment().run as exp:
            # Write multiple metric points quickly
            for i in range(10):
                exp.metrics("train").log(loss=0.5 - i * 0.01, step=i)

            # Verify buffer has metrics queued
            assert "train" in exp._buffer_manager._metric_queues
            assert not exp._buffer_manager._metric_queues["train"].empty()

            # Manual flush
            exp.flush()
            time.sleep(0.1)  # Wait for flush to complete

            # Queue should be empty after flush
            assert exp._buffer_manager._metric_queues["train"].empty()

    def test_buffer_multiple_metrics(self, buffered_experiment):
        """Verify multiple metrics are buffered independently."""
        with buffered_experiment().run as exp:
            # Write to different metrics
            for i in range(5):
                exp.metrics("train").log(loss=0.5 - i * 0.01)
                exp.metrics("eval").log(loss=0.6 - i * 0.01)

            # Both metrics should have queues
            assert "train" in exp._buffer_manager._metric_queues
            assert "eval" in exp._buffer_manager._metric_queues

            # Manual flush
            exp.flush()
            time.sleep(0.1)

            # Both queues should be empty
            assert exp._buffer_manager._metric_queues["train"].empty()
            assert exp._buffer_manager._metric_queues["eval"].empty()

    def test_unnamed_metric_buffering(self, buffered_experiment):
        """Verify unnamed metrics are buffered correctly."""
        with buffered_experiment().run as exp:
            # Write unnamed metrics
            for i in range(5):
                exp.metrics.log(epoch=i, loss=0.5 - i * 0.01)

            # Unnamed metrics use None as key
            assert None in exp._buffer_manager._metric_queues

            exp.flush()
            time.sleep(0.1)

            assert exp._buffer_manager._metric_queues[None].empty()

    def test_metric_size_based_flush(self, buffered_experiment):
        """Verify metric size-based flush trigger works."""
        with buffered_experiment().run as exp:
            # Write exactly batch_size metrics (5 in test config)
            for i in range(5):
                exp.metrics("train").log(loss=0.5 - i * 0.01)

            # Wait for background thread to flush
            time.sleep(0.2)

            # Queue should be flushed
            queue = exp._buffer_manager._metric_queues["train"]
            assert queue.qsize() < 5


class TestFileBuffering:
    """Test file upload buffering behavior."""

    def test_buffer_file_upload(self, buffered_experiment, sample_files):
        """Verify file uploads are queued."""
        with buffered_experiment().run as exp:
            # Queue file upload
            result = exp.files("models").upload(sample_files["model"])

            # Should return pending status
            assert result["id"] == "pending"
            assert result["status"] == "queued"

            # Verify file is in queue
            assert not exp._buffer_manager._file_queue.empty()

            # Manual flush
            exp.flush()
            time.sleep(0.5)  # Wait for upload to complete

            # Queue should be empty
            assert exp._buffer_manager._file_queue.empty()

    def test_multiple_file_uploads(self, buffered_experiment, sample_files):
        """Verify multiple files are uploaded in parallel."""
        with buffered_experiment().run as exp:
            # Queue multiple files
            exp.files("models").upload(sample_files["model"])
            exp.files("configs").upload(sample_files["config"])
            exp.files("results").upload(sample_files["results"])

            # All files should be queued
            assert exp._buffer_manager._file_queue.qsize() >= 3

            # Manual flush
            exp.flush()
            time.sleep(1.0)  # Wait for parallel uploads

            # Queue should be empty
            assert exp._buffer_manager._file_queue.empty()


class TestBufferLifecycle:
    """Test buffer manager lifecycle and thread safety."""

    def test_buffer_starts_with_experiment(self, buffered_experiment):
        """Verify buffer manager starts when experiment opens."""
        exp = buffered_experiment()

        # Buffer should not be started yet
        assert exp._buffer_manager is None

        with exp.run as exp:
            # Buffer should be started
            assert exp._buffer_manager is not None
            assert exp._buffer_manager._thread is not None
            assert exp._buffer_manager._thread.is_alive()

    def test_buffer_stops_with_experiment(self, buffered_experiment):
        """Verify buffer manager stops when experiment closes."""
        exp = buffered_experiment()

        with exp.run as exp:
            buffer_manager = exp._buffer_manager
            thread = buffer_manager._thread

            # Write some data
            exp.logs.info("Test log")
            exp.metrics("train").log(loss=0.5)

        # After context exit, buffer should be stopped
        assert not thread.is_alive()

    def test_buffer_flushes_on_close(self, buffered_experiment):
        """Verify all buffered data is flushed when experiment closes."""
        with buffered_experiment().run as exp:
            # Write data without manual flush
            for i in range(20):
                exp.logs.info(f"Log {i}")
                exp.metrics("train").log(loss=0.5 - i * 0.01)

        # All data should be flushed by close
        # (verified by no exceptions and clean exit)

    def test_manual_flush(self, buffered_experiment):
        """Verify manual flush works correctly without errors."""
        with buffered_experiment().run as exp:
            # Write data
            for i in range(10):
                exp.logs.info(f"Log {i}")

            # Verify data is queued initially
            initial_size = exp._buffer_manager._log_queue.qsize()
            assert initial_size > 0

            # Manual flush (synchronous - should empty queues)
            exp.flush()

            # Queue size should have decreased after flush
            final_size = exp._buffer_manager._log_queue.qsize()
            assert final_size < initial_size

        # Context exit should flush any remaining data without errors


class TestBufferErrorHandling:
    """Test graceful error handling during buffering."""

    def test_buffer_continues_on_network_error(self, buffered_experiment):
        """Verify training continues despite network errors."""
        with buffered_experiment().run as exp:
            # Mock client to raise exception
            if exp.run._client:
                original_method = exp.run._client.create_log_entries
                exp.run._client.create_log_entries = MagicMock(
                    side_effect=Exception("Network error")
                )

            # Write logs
            for i in range(10):
                exp.logs.info(f"Log {i}")

            # Manual flush (should not crash)
            exp.flush()
            time.sleep(0.2)

            # Experiment should still be running
            assert exp._is_open

    def test_buffer_handles_storage_error(self, buffered_experiment):
        """Verify graceful handling of storage errors."""
        with buffered_experiment().run as exp:
            # Mock storage to raise exception
            if exp.run._storage:
                original_method = exp.run._storage.write_log
                exp.run._storage.write_log = MagicMock(
                    side_effect=Exception("Storage error")
                )

            # Write logs
            exp.logs.info("Test log")

            # Manual flush (should not crash)
            exp.flush()
            time.sleep(0.2)

            # Experiment should still be running
            assert exp._is_open


class TestBufferPerformance:
    """Test buffering performance improvements."""

    def test_buffering_performance_improvement(self, buffered_experiment, unbuffered_experiment):
        """Benchmark speedup with buffering vs immediate writes."""
        import timeit

        # Test with buffering
        def buffered_test():
            with buffered_experiment().run as exp:
                for i in range(100):
                    exp.logs.info(f"Log {i}")
                exp.flush()

        # Test without buffering
        def unbuffered_test():
            with unbuffered_experiment().run as exp:
                for i in range(100):
                    exp.logs.info(f"Log {i}")

        # Note: This is a local-only test, so speedup may be minimal
        # The real benefit is with remote mode where each write is an HTTP request
        buffered_time = timeit.timeit(buffered_test, number=1)
        unbuffered_time = timeit.timeit(unbuffered_test, number=1)

        # Buffering should not be slower (may be similar for local storage)
        assert buffered_time <= unbuffered_time * 2  # Allow 2x margin for test overhead


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_code_works_with_buffering(self, buffered_experiment):
        """Verify existing experiment code works with buffering enabled."""
        with buffered_experiment().run as exp:
            # Set parameters
            exp.params.set(lr=0.001, batch_size=32)

            # Log messages
            exp.logs.info("Training started")
            exp.logs.warn("GPU memory low")

            # Log metrics
            for epoch in range(5):
                exp.metrics("train").log(
                    epoch=epoch,
                    loss=0.5 - epoch * 0.05,
                    accuracy=0.7 + epoch * 0.05
                )

        # No errors should occur

    def test_can_disable_buffering(self, unbuffered_experiment):
        """Verify buffering can be disabled for backward compatibility."""
        with unbuffered_experiment().run as exp:
            # Buffer manager should not be created
            assert exp._buffer_manager is None

            # Everything should work as before
            exp.logs.info("Test log")
            exp.metrics("train").log(loss=0.5)


class TestConcurrentWrites:
    """Test thread safety of concurrent writes."""

    def test_concurrent_log_writes(self, buffered_experiment):
        """Verify buffer handles concurrent log writes without errors."""
        import threading

        errors = []

        with buffered_experiment().run as exp:
            def write_logs(thread_id):
                try:
                    for i in range(10):
                        exp.logs.info(f"Thread {thread_id} - Log {i}")
                except Exception as e:
                    errors.append(e)

            # Create multiple threads writing concurrently
            threads = [
                threading.Thread(target=write_logs, args=(i,))
                for i in range(5)
            ]

            # Start all threads
            for t in threads:
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

        # No errors should occur during concurrent writes
        assert len(errors) == 0

        # All data should be flushed on context exit (checked by no exceptions)

    def test_concurrent_metric_writes(self, buffered_experiment):
        """Verify buffer handles concurrent metric writes."""
        import threading

        with buffered_experiment().run as exp:
            def write_metrics(thread_id):
                for i in range(10):
                    exp.metrics(f"metric_{thread_id}").log(
                        value=i,
                        loss=0.5 - i * 0.01
                    )

            # Create multiple threads writing concurrently
            threads = [
                threading.Thread(target=write_metrics, args=(i,))
                for i in range(5)
            ]

            # Start all threads
            for t in threads:
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

            # Flush all metrics
            exp.flush()
            time.sleep(0.5)

            # All queues should be empty
            for queue in exp._buffer_manager._metric_queues.values():
                assert queue.empty()


class TestIntegration:
    """End-to-end integration tests."""

    def test_end_to_end_training_loop(self, buffered_experiment):
        """Simulate real training workflow with buffering."""
        with buffered_experiment().run as exp:
            # Set parameters
            exp.params.set(lr=0.001, epochs=10, batch_size=32)

            # Training loop
            for epoch in range(10):
                # Log epoch start
                exp.logs.info(f"Starting epoch {epoch}")

                # Simulate batch training
                for batch in range(100):
                    exp.metrics("train").log(
                        epoch=epoch,
                        batch=batch,
                        loss=0.5 - (epoch * 100 + batch) * 0.00001
                    )

                # Log epoch end
                exp.logs.info(f"Epoch {epoch} complete")

                # Periodic flush
                if epoch % 5 == 0:
                    exp.flush()

        # Everything should complete successfully

    def test_context_manager_auto_flush(self, buffered_experiment, sample_files):
        """Verify context manager automatically flushes on exit."""
        with buffered_experiment().run as exp:
            # Write lots of data
            for i in range(50):
                exp.logs.info(f"Log {i}")
                exp.metrics("train").log(loss=0.5 - i * 0.01)

            # Upload file
            exp.files("models").upload(sample_files["model"])

            # Don't manually flush

        # All data should be flushed automatically on context exit
