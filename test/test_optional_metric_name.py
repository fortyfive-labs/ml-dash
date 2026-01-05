"""Test optional metric name functionality."""
import json
import pytest


class TestOptionalMetricName:
    """Tests for optional metric name parameter."""

    def test_append_without_name_local(self, local_experiment, temp_project):
        """Test appending metric without specifying name (uses None)."""
        with local_experiment(name="no-name-metric", project="test").run as experiment:
            experiment.metrics.append(value=0.5, step=1)
            experiment.metrics.append(value=0.4, step=2)
            experiment.metrics.append(value=0.3, step=3)

        # Should create a "None" metric
        metric_file = temp_project /  "test" / "no-name-metric" / "metrics" / "None" / "data.jsonl"
        assert metric_file.exists()

        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 3
        assert data_points[0]["data"]["value"] == 0.5
        assert data_points[1]["data"]["value"] == 0.4
        assert data_points[2]["data"]["value"] == 0.3

    def test_append_with_explicit_name_still_works(self, local_experiment, temp_project):
        """Test that explicit name parameter still works."""
        with local_experiment(name="explicit-name", project="test").run as experiment:
            experiment.metrics.append(name="loss", value=0.5, step=1)
            experiment.metrics.append(name="loss", value=0.4, step=2)

        metric_file = temp_project /  "test" / "explicit-name" / "metrics" / "loss" / "data.jsonl"
        assert metric_file.exists()

        with open(metric_file) as f:
            data_points = [json.loads(line) for line in f]

        assert len(data_points) == 2

    def test_append_batch_without_name_local(self, local_experiment, temp_project):
        """Test batch append without name."""
        with local_experiment(name="batch-no-name", project="test").run as experiment:
            data_points = [
                {"value": 0.5, "step": 1},
                {"value": 0.4, "step": 2},
                {"value": 0.3, "step": 3}
            ]
            result = experiment.metrics.append_batch(data_points=data_points)
            assert result["count"] == 3

        metric_file = temp_project /  "test" / "batch-no-name" / "metrics" / "None" / "data.jsonl"
        assert metric_file.exists()

        with open(metric_file) as f:
            points = [json.loads(line) for line in f]

        assert len(points) == 3

    def test_mixed_named_and_none_metrics(self, local_experiment, temp_project):
        """Test using both named and None metrics in same experiment."""
        with local_experiment(name="mixed", project="test").run as experiment:
            # Append to None metric
            experiment.metrics.append(value=1.0, step=0)

            # Append to named metric
            experiment.metrics.append(name="loss", value=0.5, step=0)

            # Append to None again
            experiment.metrics.append(value=0.9, step=1)

        # Check both metrics exist
        none_file = temp_project /  "test" / "mixed" / "metrics" / "None" / "data.jsonl"
        loss_file = temp_project /  "test" / "mixed" / "metrics" / "loss" / "data.jsonl"

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
        with remote_experiment(name="no-name-remote", project="test").run as experiment:
            experiment.metrics.append(value=0.5, step=1)
            experiment.metrics.append(value=0.4, step=2)

            # Verify stats show "None" metric (converted to string by server)
            stats = experiment.metrics(None).stats()
            assert stats["name"] == "None"

    @pytest.mark.remote
    def test_append_batch_without_name_remote(self, remote_experiment):
        """Test batch append without name in remote mode."""
        with remote_experiment(name="batch-no-name-remote", project="test").run as experiment:
            data_points = [
                {"value": i * 0.1, "step": i}
                for i in range(10)
            ]
            result = experiment.metrics.append_batch(data_points=data_points)
            assert result["count"] == 10

            # Verify it went to "None" metric (converted to string by server)
            stats = experiment.metrics(None).stats()
            assert stats["name"] == "None"

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
