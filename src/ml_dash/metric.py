"""
Metric API - Time-series data metricing for ML experiments.

Metrics are used for storing continuous data series like training metrics,
validation losses, system measurements, etc.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import defaultdict
import statistics

if TYPE_CHECKING:
    from .experiment import Experiment


class SummaryCache:
    """
    Buffer for collecting metric values and computing statistics periodically.

    Inspired by ml-logger's SummaryCache design:
    - Lazy computation: Store raw values, compute stats on demand
    - Hierarchical naming: Stats get suffixes (loss.mean, loss.std)
    - Robust handling: Converts None → NaN, filters before stats
    """

    def __init__(self, metric_builder: 'MetricBuilder'):
        """
        Initialize SummaryCache.

        Args:
            metric_builder: Parent MetricBuilder instance
        """
        self._metric_builder = metric_builder
        self._buffer: Dict[str, List[float]] = defaultdict(list)
        self._metadata: Dict[str, Any] = {}  # For set() metadata

    def store(self, **kwargs) -> None:
        """
        Store values in buffer without immediate logging (deferred computation).

        Args:
            **kwargs: Metric values to buffer (e.g., loss=0.5, accuracy=0.9)

        Example:
            cache.store(loss=0.5, accuracy=0.9)
            cache.store(loss=0.48)  # Accumulates
        """
        for key, value in kwargs.items():
            # Handle None values gracefully
            if value is None:
                value = float('nan')
            try:
                self._buffer[key].append(float(value))
            except (TypeError, ValueError):
                # Skip non-numeric values silently
                continue

    def set(self, **kwargs) -> None:
        """
        Set metadata values without aggregation (replaces previous values).

        Used for contextual metadata like learning rate, epoch number, etc.
        These values are included in the final data point when summarize() is called.

        Args:
            **kwargs: Metadata to set (e.g., lr=0.001, epoch=5)

        Example:
            cache.set(lr=0.001, epoch=5)
            cache.set(lr=0.0005)  # Replaces lr, keeps epoch
        """
        self._metadata.update(kwargs)

    def _compute_stats(self) -> Dict[str, float]:
        """
        Compute statistics from buffered values (idempotent, read-only).

        Returns:
            Dict with hierarchical metric names (key.mean, key.std, etc.)

        Note: This is idempotent - can be called multiple times without side effects.
        """
        stats_data = {}

        for key, values in self._buffer.items():
            if not values:
                continue

            # Filter out NaN values (ml-logger pattern)
            clean_values = [v for v in values if not (isinstance(v, float) and v != v)]

            if not clean_values:
                continue

            # Compute statistics with hierarchical naming
            stats_data[f"{key}.mean"] = statistics.mean(clean_values)
            stats_data[f"{key}.min"] = min(clean_values)
            stats_data[f"{key}.max"] = max(clean_values)
            stats_data[f"{key}.count"] = len(clean_values)

            # Std dev requires at least 2 values
            if len(clean_values) >= 2:
                stats_data[f"{key}.std"] = statistics.stdev(clean_values)
            else:
                stats_data[f"{key}.std"] = 0.0

        return stats_data

    def summarize(self, clear: bool = True) -> None:
        """
        Compute statistics from buffered values and log them (non-idempotent).

        Args:
            clear: If True (default), clear buffer after computing statistics.
                  This creates a "rolling window" behavior matching ml-logger's "tiled" mode.

        Example:
            # After storing 10 loss values and setting lr=0.001:
            cache.store(loss=0.5)
            cache.set(lr=0.001, epoch=5)
            cache.summarize()
            # Logs: {lr: 0.001, epoch: 5, loss.mean: 0.5, loss.std: 0.0, ...}

        Note: This is non-idempotent - calling it multiple times has side effects.
        """
        if not self._buffer and not self._metadata:
            return

        # Compute statistics (delegated to idempotent method)
        stats_data = self._compute_stats()

        # Merge metadata with statistics
        output_data = {**self._metadata, **stats_data}

        if not output_data:
            return

        # Append combined data as a single metric data point
        self._metric_builder.append(**output_data)

        # Clear buffer if requested (default behavior for "tiled" mode)
        if clear:
            self._buffer.clear()
            self._metadata.clear()  # Also clear metadata

    def peek(self, *keys: str, limit: int = 5) -> Dict[str, List[float]]:
        """
        Non-destructive inspection of buffered values (idempotent, read-only).

        Args:
            *keys: Optional specific keys to peek at. If empty, shows all.
            limit: Number of most recent values to show (default 5)

        Returns:
            Dict of buffered values (truncated to last `limit` items)

        Example:
            cache.peek('loss', limit=3)  # {'loss': [0.5, 0.48, 0.52]}
        """
        keys_to_show = keys if keys else self._buffer.keys()
        return {
            k: self._buffer[k][-limit:] if limit else self._buffer[k]
            for k in keys_to_show
            if k in self._buffer and self._buffer[k]
        }


class MetricsManager:
    """
    Manager for metric operations that supports both named and unnamed usage.

    Supports three usage patterns:
    1. Named via call: experiment.metrics("loss").append(value=0.5, step=1)
    2. Named via argument: experiment.metrics.append(name="loss", value=0.5, step=1)
    3. Unnamed: experiment.metrics.append(value=0.5, step=1)  # name=None

    Usage:
        # With explicit metric name (via call)
        experiment.metrics("train_loss").append(value=0.5, step=100)

        # With explicit metric name (via argument)
        experiment.metrics.append(name="train_loss", value=0.5, step=100)

        # Without name (uses None as metric name)
        experiment.metrics.append(value=0.5, step=100)
    """

    def __init__(self, experiment: 'Experiment'):
        """
        Initialize MetricsManager.

        Args:
            experiment: Parent Experiment instance
        """
        self._experiment = experiment
        self._metric_builders: Dict[str, 'MetricBuilder'] = {}  # Cache for MetricBuilder instances

    def __call__(self, name: str, description: Optional[str] = None,
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> 'MetricBuilder':
        """
        Get a MetricBuilder for a specific metric name (cached for reuse).

        Args:
            name: Metric name (unique within experiment)
            description: Optional metric description
            tags: Optional tags for categorization
            metadata: Optional structured metadata

        Returns:
            MetricBuilder instance for the named metric (same instance on repeated calls)

        Examples:
            experiment.metrics("loss").append(value=0.5, step=1)

        Note:
            MetricBuilder instances are cached by name, so repeated calls with the
            same name return the same instance. This ensures summary_cache works
            correctly when called multiple times within a loop.
        """
        # Cache key includes name only (description/tags/metadata are set once on first call)
        if name not in self._metric_builders:
            self._metric_builders[name] = MetricBuilder(self._experiment, name, description, tags, metadata)
        return self._metric_builders[name]

    def append(self, name: Optional[str] = None, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Append a data point to a metric (name can be optional).

        Args:
            name: Metric name (optional, can be None for unnamed metrics)
            data: Data dict (alternative to kwargs)
            **kwargs: Data as keyword arguments

        Returns:
            Response dict with metric metadata

        Examples:
            experiment.metrics.append(name="loss", value=0.5, step=1)
            experiment.metrics.append(value=0.5, step=1)  # name=None
            experiment.metrics.append(name="loss", data={"value": 0.5, "step": 1})
        """
        if data is None:
            data = kwargs
        return self._experiment._append_to_metric(name, data, None, None, None)

    def append_batch(self, name: Optional[str] = None, data_points: Optional[List[Dict[str, Any]]] = None,
                     description: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Append multiple data points to a metric.

        Args:
            name: Metric name (optional, can be None for unnamed metrics)
            data_points: List of data point dicts
            description: Optional metric description
            tags: Optional tags for categorization
            metadata: Optional structured metadata

        Returns:
            Response dict with metric metadata

        Examples:
            experiment.metrics.append_batch(
                name="loss",
                data_points=[
                    {"value": 0.5, "step": 1},
                    {"value": 0.4, "step": 2}
                ]
            )
            experiment.metrics.append_batch(
                data_points=[
                    {"value": 0.5, "step": 1},
                    {"value": 0.4, "step": 2}
                ]
            )  # name=None
        """
        if data_points is None:
            data_points = []
        return self._experiment._append_batch_to_metric(name, data_points, description, tags, metadata)


class MetricBuilder:
    """
    Builder for metric operations.

    Provides fluent API for appending, reading, and querying metric data.

    Usage:
        # Append single data point
        experiment.metric(name="train_loss").append(value=0.5, step=100)

        # Append batch
        experiment.metric(name="train_loss").append_batch([
            {"value": 0.5, "step": 100},
            {"value": 0.45, "step": 101}
        ])

        # Read data
        data = experiment.metric(name="train_loss").read(start_index=0, limit=100)

        # Get statistics
        stats = experiment.metric(name="train_loss").stats()
    """

    def __init__(self, experiment: 'Experiment', name: str, description: Optional[str] = None,
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize MetricBuilder.

        Args:
            experiment: Parent Experiment instance
            name: Metric name (unique within experiment)
            description: Optional metric description
            tags: Optional tags for categorization
            metadata: Optional structured metadata (units, type, etc.)
        """
        self._experiment = experiment
        self._name = name
        self._description = description
        self._tags = tags
        self._metadata = metadata
        self._summary_cache = None  # Lazy initialization

    def append(self, **kwargs) -> 'MetricBuilder':
        """
        Append a single data point to the metric.

        The data point can have any structure - common patterns:
        - {value: 0.5, step: 100}
        - {loss: 0.3, accuracy: 0.92, epoch: 5}
        - {timestamp: "...", temperature: 25.5, humidity: 60}

        Args:
            **kwargs: Data point fields (flexible schema)

        Returns:
            Dict with metricId, index, bufferedDataPoints, chunkSize

        Example:
            result = experiment.metric(name="train_loss").append(value=0.5, step=100, epoch=1)
            print(f"Appended at index {result['index']}")
        """
        result = self._experiment._append_to_metric(
            name=self._name,
            data=kwargs,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )
        return result

    def append_batch(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Append multiple data points in batch (more efficient than multiple append calls).

        Args:
            data_points: List of data point dicts

        Returns:
            Dict with metricId, startIndex, endIndex, count, bufferedDataPoints, chunkSize

        Example:
            result = experiment.metric(name="metrics").append_batch([
                {"loss": 0.5, "acc": 0.8, "step": 1},
                {"loss": 0.4, "acc": 0.85, "step": 2},
                {"loss": 0.3, "acc": 0.9, "step": 3}
            ])
            print(f"Appended {result['count']} points")
        """
        if not data_points:
            raise ValueError("data_points cannot be empty")

        result = self._experiment._append_batch_to_metric(
            name=self._name,
            data_points=data_points,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )
        return result

    def read(self, start_index: int = 0, limit: int = 1000) -> Dict[str, Any]:
        """
        Read data points from the metric by index range.

        Args:
            start_index: Starting index (inclusive, default 0)
            limit: Maximum number of points to read (default 1000, max 10000)

        Returns:
            Dict with keys:
            - data: List of {index: str, data: dict, createdAt: str}
            - startIndex: Starting index
            - endIndex: Ending index
            - total: Number of points returned
            - hasMore: Whether more data exists beyond this range

        Example:
            result = experiment.metric(name="train_loss").read(start_index=0, limit=100)
            for point in result['data']:
                print(f"Index {point['index']}: {point['data']}")
        """
        return self._experiment._read_metric_data(
            name=self._name,
            start_index=start_index,
            limit=limit
        )

    def stats(self) -> Dict[str, Any]:
        """
        Get metric statistics and metadata.

        Returns:
            Dict with metric info:
            - metricId: Unique metric ID
            - name: Metric name
            - description: Metric description (if set)
            - tags: Tags list
            - metadata: User metadata
            - totalDataPoints: Total points (buffered + chunked)
            - bufferedDataPoints: Points in MongoDB (hot storage)
            - chunkedDataPoints: Points in S3 (cold storage)
            - totalChunks: Number of chunks in S3
            - chunkSize: Chunking threshold
            - firstDataAt: Timestamp of first point (if data has timestamp)
            - lastDataAt: Timestamp of last point (if data has timestamp)
            - createdAt: Metric creation time
            - updatedAt: Last update time

        Example:
            stats = experiment.metric(name="train_loss").stats()
            print(f"Total points: {stats['totalDataPoints']}")
            print(f"Buffered: {stats['bufferedDataPoints']}, Chunked: {stats['chunkedDataPoints']}")
        """
        return self._experiment._get_metric_stats(name=self._name)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all metrics in the experiment.

        Returns:
            List of metric summaries with keys:
            - metricId: Unique metric ID
            - name: Metric name
            - description: Metric description
            - tags: Tags list
            - totalDataPoints: Total data points
            - createdAt: Creation timestamp

        Example:
            metrics = experiment.metric().list_all()
            for metric in metrics:
                print(f"{metric['name']}: {metric['totalDataPoints']} points")
        """
        return self._experiment._list_metrics()

    @property
    def summary_cache(self) -> SummaryCache:
        """
        Get summary cache for this metric (lazy initialization).

        The summary cache allows buffering values and computing statistics
        periodically, which is much more efficient than logging every value.

        Returns:
            SummaryCache instance for this metric

        Example:
            metric = experiment.metrics("train")
            # Store values every batch
            metric.summary_cache.store(loss=0.5)
            metric.summary_cache.store(loss=0.48)
            # Set metadata
            metric.summary_cache.set(lr=0.001, epoch=1)
            # Compute stats and log periodically
            metric.summary_cache.summarize()
        """
        if self._summary_cache is None:
            self._summary_cache = SummaryCache(self)
        return self._summary_cache
