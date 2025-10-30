"""Comprehensive tests for metric (time-series) functionality in both local and remote modes."""
import json
import pytest
from pathlib import Path


class TestBasicMetrics:
    """Tests for basic metric operations."""

    def test_single_metric_append_local(self, local_experiment, temp_project):
        """Test appending single data points to a metric."""
        with local_experiment(name="metric-test", project="test") as experiment:
            for i in range(5):
                experiment.metric("loss").append(value=1.0 / (i + 1), epoch=i)

        metric_file = temp_project / "test" / "metric-test" / "metrics" / "loss" / "data.jsonl"
        assert metric_file.exists()

        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 5
        assert data_points[0]["data"]["value"] == 1.0
        assert data_points[0]["data"]["epoch"] == 0

    @pytest.mark.remote
    def test_single_metric_append_remote(self, remote_experiment):
        """Test appending data points in remote mode."""
        with remote_experiment(name="metric-test-remote", project="test") as experiment:
            for i in range(10):
                experiment.metric("loss").append(value=0.5 - i * 0.05, epoch=i)

    def test_multiple_metrics_local(self, local_experiment, temp_project):
        """Test metricing multiple different metrics."""
        with local_experiment(name="multi-metric", project="test") as experiment:
            for epoch in range(5):
                experiment.metric("train_loss").append(value=0.5 - epoch * 0.1, epoch=epoch)
                experiment.metric("val_loss").append(value=0.6 - epoch * 0.1, epoch=epoch)
                experiment.metric("accuracy").append(value=0.7 + epoch * 0.05, epoch=epoch)

        metrics_dir = temp_project / "test" / "multi-metric" / "metrics"
        assert (metrics_dir / "train_loss" / "data.jsonl").exists()
        assert (metrics_dir / "val_loss" / "data.jsonl").exists()
        assert (metrics_dir / "accuracy" / "data.jsonl").exists()

    @pytest.mark.remote
    def test_multiple_metrics_remote(self, remote_experiment):
        """Test metricing multiple metrics in remote mode."""
        with remote_experiment(name="multi-metric-remote", project="test") as experiment:
            for epoch in range(3):
                experiment.metric("train_loss").append(value=0.4 - epoch * 0.1, epoch=epoch)
                experiment.metric("val_loss").append(value=0.5 - epoch * 0.1, epoch=epoch)


class TestBatchAppend:
    """Tests for batch appending metric data."""

    def test_batch_append_local(self, local_experiment, temp_project, sample_data):
        """Test batch appending multiple data points at once."""
        with local_experiment(name="batch-metric", project="test") as experiment:
            result = experiment.metric("loss").append_batch(sample_data["metric_data"])
            assert result["count"] == 5

        metric_file = temp_project / "test" / "batch-metric" / "metrics" / "loss" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 5
        assert data_points[0]["data"]["value"] == 0.5
        assert data_points[4]["data"]["value"] == 0.2

    @pytest.mark.remote
    def test_batch_append_remote(self, remote_experiment, sample_data):
        """Test batch appending in remote mode."""
        with remote_experiment(name="batch-metric-remote", project="test") as experiment:
            result = experiment.metric("metrics").append_batch(sample_data["metric_data"])
            assert result["count"] == 5

    def test_large_batch_append_local(self, local_experiment, temp_project):
        """Test appending a large batch of data."""
        batch_data = [{"value": i * 0.01, "step": i} for i in range(1000)]

        with local_experiment(name="large-batch", project="test") as experiment:
            result = experiment.metric("metric").append_batch(batch_data)
            assert result["count"] == 1000

        metric_file = temp_project / "test" / "large-batch" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 1000


class TestFlexibleSchema:
    """Tests for flexible metric schema with multiple fields."""

    def test_multi_field_metricing_local(self, local_experiment, temp_project):
        """Test metrics with multiple fields per data point."""
        with local_experiment(name="multi-field", project="test") as experiment:
            experiment.metric("all_metrics").append(
                epoch=5,
                train_loss=0.3,
                val_loss=0.35,
                train_acc=0.85,
                val_acc=0.82,
                learning_rate=0.001
            )

        metric_file = temp_project / "test" / "multi-field" / "metrics" / "all_metrics" / "data.jsonl"
        with open(metric_file) as f:
            data_point = json.loads(f.readline())

        assert data_point["data"]["epoch"] == 5
        assert data_point["data"]["train_loss"] == 0.3
        assert data_point["data"]["val_loss"] == 0.35
        assert data_point["data"]["train_acc"] == 0.85

    @pytest.mark.remote
    def test_multi_field_metricing_remote(self, remote_experiment, sample_data):
        """Test multi-field metricing in remote mode."""
        with remote_experiment(name="multi-field-remote", project="test") as experiment:
            for data in sample_data["multi_metric_data"]:
                experiment.metric("combined").append(**data)

    def test_varying_schemas_local(self, local_experiment, temp_project):
        """Test that schema can vary between data points."""
        with local_experiment(name="varying-schema", project="test") as experiment:
            experiment.metric("flexible").append(field_a=1, field_b=2)
            experiment.metric("flexible").append(field_a=3, field_c=4)
            experiment.metric("flexible").append(field_a=5, field_b=6, field_c=7)

        metric_file = temp_project / "test" / "varying-schema" / "metrics" / "flexible" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 3
        assert "field_b" in data_points[0]["data"]
        assert "field_c" in data_points[1]["data"]
        assert "field_c" in data_points[2]["data"]


class TestMetricMetadata:
    """Tests for metric metadata."""

    def test_metric_metadata_creation_local(self, local_experiment, temp_project):
        """Test that metric metadata is created."""
        with local_experiment(name="metric-meta", project="test") as experiment:
            for i in range(15):
                experiment.metric("metric").append(value=i * 0.1, step=i)

        metadata_file = temp_project / "test" / "metric-meta" / "metrics" / "metric" / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["name"] == "metric"
        assert metadata["totalDataPoints"] == 15

    def test_metric_stats_local(self, local_experiment):
        """Test getting metric statistics."""
        with local_experiment(name="metric-stats", project="test") as experiment:
            for i in range(20):
                experiment.metric("accuracy").append(value=0.5 + i * 0.02, step=i)

            stats = experiment.metric("accuracy").stats()

        assert stats["name"] == "accuracy"
        assert int(stats["totalDataPoints"]) == 20

    @pytest.mark.remote
    def test_metric_stats_remote(self, remote_experiment):
        """Test getting metric stats in remote mode."""
        with remote_experiment(name="metric-stats-remote", project="test") as experiment:
            for i in range(10):
                experiment.metric("loss").append(value=1.0 / (i + 1), step=i)

            stats = experiment.metric("loss").stats()
            assert stats["name"] == "loss"


class TestMetricRead:
    """Tests for reading metric data."""

    def test_read_metric_data_local(self, local_experiment):
        """Test reading metric data."""
        with local_experiment(name="metric-read", project="test") as experiment:
            # Write data
            for i in range(20):
                experiment.metric("metric").append(value=i * 0.1, step=i)

            # Read data
            result = experiment.metric("metric").read(start_index=0, limit=10)

        assert result["total"] >= 10
        assert len(result["data"]) == 10
        assert result["data"][0]["data"]["step"] == 0

    def test_read_with_pagination_local(self, local_experiment):
        """Test reading metric data with pagination."""
        with local_experiment(name="metric-page", project="test") as experiment:
            # Write 100 data points
            for i in range(100):
                experiment.metric("metric").append(value=i, step=i)

            # Read first page
            page1 = experiment.metric("metric").read(start_index=0, limit=25)
            assert len(page1["data"]) == 25

            # Read second page
            page2 = experiment.metric("metric").read(start_index=25, limit=25)
            assert len(page2["data"]) == 25
            assert page2["data"][0]["data"]["step"] == 25

    @pytest.mark.remote
    def test_read_metric_data_remote(self, remote_experiment):
        """Test reading metric data in remote mode."""
        with remote_experiment(name="metric-read-remote", project="test") as experiment:
            for i in range(15):
                experiment.metric("metric").append(value=i * 0.05, step=i)

            result = experiment.metric("metric").read(start_index=0, limit=5)
            assert len(result["data"]) <= 15


class TestListMetrics:
    """Tests for listing all metrics."""

    def test_list_all_metrics_local(self, local_experiment):
        """Test listing all metrics in a experiment."""
        with local_experiment(name="metric-list", project="test") as experiment:
            experiment.metric("loss").append(value=0.5, step=0)
            experiment.metric("accuracy").append(value=0.8, step=0)
            experiment.metric("learning_rate").append(value=0.001, step=0)

            metrics = experiment.metric("loss").list_all()

        assert len(metrics) == 3
        metric_names = [t["name"] for t in metrics]
        assert "loss" in metric_names
        assert "accuracy" in metric_names
        assert "learning_rate" in metric_names

    @pytest.mark.remote
    def test_list_all_metrics_remote(self, remote_experiment):
        """Test listing metrics in remote mode."""
        with remote_experiment(name="metric-list-remote", project="test") as experiment:
            experiment.metric("metric1").append(value=1.0, step=0)
            experiment.metric("metric2").append(value=2.0, step=0)

            metrics = experiment.metric("metric1").list_all()
            assert len(metrics) >= 2


class TestMetricIndexing:
    """Tests for metric data indexing."""

    def test_metric_sequential_indices_local(self, local_experiment, temp_project):
        """Test that metric data points have sequential indices."""
        with local_experiment(name="metric-index", project="test") as experiment:
            for i in range(10):
                experiment.metric("metric").append(value=i * 10)

        metric_file = temp_project / "test" / "metric-index" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        for i, point in enumerate(data_points):
            assert point["index"] == i

    def test_metric_indices_with_batch_local(self, local_experiment, temp_project):
        """Test indices with batch append."""
        with local_experiment(name="batch-index", project="test") as experiment:
            batch1 = [{"value": i} for i in range(5)]
            batch2 = [{"value": i + 5} for i in range(5)]

            experiment.metric("metric").append_batch(batch1)
            experiment.metric("metric").append_batch(batch2)

        metric_file = temp_project / "test" / "batch-index" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 10
        for i, point in enumerate(data_points):
            assert point["index"] == i


class TestMetricEdgeCases:
    """Tests for edge cases in metric operations."""

    def test_empty_metric_local(self, local_experiment, temp_project):
        """Test experiment with no metrics."""
        with local_experiment(name="no-metrics", project="test") as experiment:
            experiment.log("No metrics created")

        metrics_dir = temp_project / "test" / "no-metrics" / "metrics"
        assert metrics_dir.exists()
        subdirs = [d for d in metrics_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 0

    def test_metric_with_null_values_local(self, local_experiment, temp_project):
        """Test metricing data with null values."""
        with local_experiment(name="null-metric", project="test") as experiment:
            experiment.metric("metric").append(value=None, step=0, status="pending")
            experiment.metric("metric").append(value=0.5, step=1, status="complete")

        metric_file = temp_project / "test" / "null-metric" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert data_points[0]["data"]["value"] is None
        assert data_points[1]["data"]["value"] == 0.5

    def test_metric_with_special_characters_local(self, local_experiment, temp_project):
        """Test metric names with special characters."""
        with local_experiment(name="special-metric", project="test") as experiment:
            experiment.metric("metric_1").append(value=1.0)
            experiment.metric("metric-2").append(value=2.0)
            experiment.metric("metric.3").append(value=3.0)

        metrics_dir = temp_project / "test" / "special-metric" / "metrics"
        # Check that metrics were created (names may be sanitized)
        assert metrics_dir.exists()

    def test_very_frequent_metricing_local(self, local_experiment, temp_project):
        """Test rapid, frequent metricing."""
        with local_experiment(name="frequent-metric", project="test") as experiment:
            for i in range(1000):
                experiment.metric("metric").append(value=i * 0.001, step=i)

        metric_file = temp_project / "test" / "frequent-metric" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 1000

    @pytest.mark.remote
    def test_frequent_metricing_remote(self, remote_experiment):
        """Test rapid metricing in remote mode."""
        with remote_experiment(name="frequent-metric-remote", project="test") as experiment:
            for i in range(100):
                experiment.metric("metric").append(value=i * 0.01, step=i)

    def test_metric_with_large_values_local(self, local_experiment, temp_project):
        """Test metricing with very large numeric values."""
        with local_experiment(name="large-values", project="test") as experiment:
            experiment.metric("metric").append(
                huge_int=999999999999999,
                huge_float=1.23e100,
                tiny_float=1.23e-100
            )

        metric_file = temp_project / "test" / "large-values" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_point = json.loads(f.readline())

        assert data_point["data"]["huge_int"] == 999999999999999

    def test_metric_with_nested_data_local(self, local_experiment, temp_project):
        """Test metricing with nested data structures."""
        with local_experiment(name="nested-metric", project="test") as experiment:
            experiment.metric("metric").append(
                epoch=1,
                metrics={
                    "train": {"loss": 0.5, "acc": 0.8},
                    "val": {"loss": 0.6, "acc": 0.75}
                }
            )

        metric_file = temp_project / "test" / "nested-metric" / "metrics" / "metric" / "data.jsonl"
        with open(metric_file) as f:
            data_point = json.loads(f.readline())

        assert data_point["data"]["epoch"] == 1
        assert isinstance(data_point["data"]["metrics"], dict)

    def test_metric_name_collision_local(self, local_experiment, temp_project):
        """Test multiple appends to same metric name."""
        with local_experiment(name="collision", project="test") as experiment:
            experiment.metric("loss").append(value=1.0, epoch=0)
            experiment.metric("loss").append(value=0.9, epoch=1)
            experiment.metric("loss").append(value=0.8, epoch=2)

        metric_file = temp_project / "test" / "collision" / "metrics" / "loss" / "data.jsonl"
        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 3
        assert data_points[0]["data"]["value"] == 1.0
        assert data_points[2]["data"]["value"] == 0.8
