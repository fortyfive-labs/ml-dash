"""
Metric API - Time-series data logging for ML experiments.

Metrics are used for storing continuous data series like training metrics,
validation losses, system measurements, etc.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import defaultdict
import statistics

if TYPE_CHECKING:
    from .experiment import Experiment


class BufferManager:
    """
    Global buffer manager for collecting metric values across prefixes.

    Accumulates values via metrics("prefix").buffer(...) and computes
    statistics when log_summary() is called.

    Usage:
        # Accumulate with prefix
        metrics("train").buffer(loss=0.5, accuracy=0.81)
        metrics("val").buffer(loss=0.6, accuracy=0.78)

        # Log summaries (all buffered prefixes)
        metrics.buffer.log_summary()                    # default: "mean"
        metrics.buffer.log_summary("mean", "std", "p95")

        # Log non-buffered values directly
        metrics.log(epoch=epoch, lr=lr)

        # Final flush to storage
        metrics.flush()
    """

    # Supported aggregation functions
    SUPPORTED_AGGS = {
        "mean", "std", "min", "max", "count",
        "median", "sum",
        "p50", "p90", "p95", "p99",
        "last", "first"
    }

    def __init__(self, metrics_manager: 'MetricsManager'):
        """
        Initialize BufferManager.

        Args:
            metrics_manager: Parent MetricsManager instance
        """
        self._metrics_manager = metrics_manager
        # Buffers per prefix: {prefix: {key: [values]}}
        self._buffers: Dict[Optional[str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def _store(self, prefix: Optional[str], **kwargs) -> None:
        """
        Store values in buffer for a specific prefix.

        Args:
            prefix: Metric prefix (e.g., "train", "val")
            **kwargs: Metric values to buffer (e.g., loss=0.5, accuracy=0.9)
        """
        for key, value in kwargs.items():
            # Handle None values gracefully
            if value is None:
                value = float('nan')
            try:
                self._buffers[prefix][key].append(float(value))
            except (TypeError, ValueError):
                # Skip non-numeric values silently
                continue

    def _compute_stats(self, values: List[float], aggs: tuple) -> Dict[str, float]:
        """
        Compute statistics for a list of values.

        Args:
            values: List of numeric values
            aggs: Tuple of aggregation names

        Returns:
            Dict with computed statistics
        """
        # Filter out NaN values
        clean_values = [v for v in values if not (isinstance(v, float) and v != v)]

        if not clean_values:
            return {}

        stats = {}
        for agg in aggs:
            if agg == "mean":
                stats["mean"] = statistics.mean(clean_values)
            elif agg == "std":
                if len(clean_values) >= 2:
                    stats["std"] = statistics.stdev(clean_values)
                else:
                    stats["std"] = 0.0
            elif agg == "min":
                stats["min"] = min(clean_values)
            elif agg == "max":
                stats["max"] = max(clean_values)
            elif agg == "count":
                stats["count"] = len(clean_values)
            elif agg == "median" or agg == "p50":
                stats[agg] = statistics.median(clean_values)
            elif agg == "sum":
                stats["sum"] = sum(clean_values)
            elif agg == "p90":
                stats["p90"] = self._percentile(clean_values, 90)
            elif agg == "p95":
                stats["p95"] = self._percentile(clean_values, 95)
            elif agg == "p99":
                stats["p99"] = self._percentile(clean_values, 99)
            elif agg == "last":
                stats["last"] = clean_values[-1]
            elif agg == "first":
                stats["first"] = clean_values[0]

        return stats

    def _percentile(self, values: List[float], p: int) -> float:
        """Compute percentile of values."""
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_vals) else f
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def log_summary(self, *aggs: str) -> None:
        """
        Compute statistics from buffered values and log them.

        Args:
            *aggs: Aggregation functions to compute. Defaults to ("mean",).
                   Supported: "mean", "std", "min", "max", "count",
                             "median", "sum", "p50", "p90", "p95", "p99",
                             "last", "first"

        Example:
            metrics.buffer.log_summary()                    # default: mean
            metrics.buffer.log_summary("mean", "std")       # mean and std
            metrics.buffer.log_summary("mean", "p95")       # mean and 95th percentile
        """
        # Default to mean
        if not aggs:
            aggs = ("mean",)

        # Validate aggregations
        for agg in aggs:
            if agg not in self.SUPPORTED_AGGS:
                raise ValueError(f"Unsupported aggregation: {agg}. Supported: {self.SUPPORTED_AGGS}")

        # Process each prefix's buffer
        for prefix, buffer in list(self._buffers.items()):
            if not buffer:
                continue

            output_data = {}

            for key, values in buffer.items():
                if not values:
                    continue

                stats = self._compute_stats(values, aggs)

                # Add stats with hierarchical naming (key.agg)
                for stat_name, stat_value in stats.items():
                    output_data[f"{key}.{stat_name}"] = stat_value

            if output_data:
                # Log to the appropriate metric
                self._metrics_manager(prefix).log(**output_data)

        # Clear all buffers
        self._buffers.clear()

    def peek(self, prefix: Optional[str] = None, *keys: str, limit: int = 5) -> Dict[str, List[float]]:
        """
        Non-destructive inspection of buffered values.

        Args:
            prefix: Specific prefix to peek at (None for all)
            *keys: Optional specific keys to peek at. If empty, shows all.
            limit: Number of most recent values to show (default 5)

        Returns:
            Dict of buffered values (truncated to last `limit` items)
        """
        if prefix is not None:
            buffer = self._buffers.get(prefix, {})
            keys_to_show = keys if keys else buffer.keys()
            return {
                k: buffer[k][-limit:] if limit else buffer[k]
                for k in keys_to_show
                if k in buffer and buffer[k]
            }
        else:
            # Return all buffers
            result = {}
            for p, buffer in self._buffers.items():
                prefix_str = p if p else "(default)"
                keys_to_show = keys if keys else buffer.keys()
                for k in keys_to_show:
                    if k in buffer and buffer[k]:
                        result[f"{prefix_str}/{k}"] = buffer[k][-limit:] if limit else buffer[k]
            return result


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

        # Log combined data as a single metric data point
        self._metric_builder.log(**output_data)

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

    Supports two usage patterns:
    1. Named via call: experiment.metrics("train").log(loss=0.5, accuracy=0.9)
    2. Unnamed: experiment.metrics.log(epoch=1).flush()

    Usage:
        # With explicit metric name (via call)
        experiment.metrics("train").log(loss=0.5, accuracy=0.9)

        # With epoch context (unnamed metric)
        experiment.metrics.log(epoch=epoch).flush()

        # Nested dict pattern (single call for all metrics)
        experiment.metrics.log(
            epoch=100,
            train=dict(loss=0.142, accuracy=0.80),
            eval=dict(loss=0.201, accuracy=0.76)
        )
    """

    def __init__(self, experiment: 'Experiment'):
        """
        Initialize MetricsManager.

        Args:
            experiment: Parent Experiment instance
        """
        self._experiment = experiment
        self._metric_builders: Dict[str, 'MetricBuilder'] = {}  # Cache for MetricBuilder instances
        self._buffer_manager: Optional[BufferManager] = None  # Lazy initialization

    @property
    def buffer(self) -> BufferManager:
        """
        Get the global BufferManager for buffered metric operations.

        The buffer manager collects values across prefixes and computes
        statistics when log_summary() is called.

        Returns:
            BufferManager instance

        Example:
            # Accumulate values
            metrics("train").buffer(loss=0.5, accuracy=0.81)
            metrics("val").buffer(loss=0.6, accuracy=0.78)

            # Log summaries
            metrics.buffer.log_summary()                    # default: mean
            metrics.buffer.log_summary("mean", "std", "p95")
        """
        if self._buffer_manager is None:
            self._buffer_manager = BufferManager(self)
        return self._buffer_manager

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
            experiment.metrics("train").log(loss=0.5, accuracy=0.9)

        Note:
            MetricBuilder instances are cached by name, so repeated calls with the
            same name return the same instance. This ensures summary_cache works
            correctly when called multiple times within a loop.
        """
        # Cache key includes name only (description/tags/metadata are set once on first call)
        if name not in self._metric_builders:
            self._metric_builders[name] = MetricBuilder(
                self._experiment, name, description, tags, metadata,
                metrics_manager=self
            )
        return self._metric_builders[name]

    def log(self, _flush: bool = False, **kwargs) -> 'MetricsManager':
        """
        Log a data point to the unnamed (root) metric.

        Supports two patterns:

        1. Simple key-value pairs:
            experiment.metrics.log(epoch=epoch).flush()

        2. Nested dict pattern (logs to multiple prefixed metrics):
            experiment.metrics.log(
                epoch=100,
                train=dict(loss=0.142, accuracy=0.80),
                eval=dict(loss=0.201, accuracy=0.76)
            )

        Args:
            _flush: If True, flush after logging (equivalent to calling .flush())
            **kwargs: Data point fields. Dict values are expanded to prefixed metrics.

        Returns:
            Self for method chaining

        Examples:
            # Log epoch context and flush
            experiment.metrics.log(epoch=epoch).flush()

            # Log with nested dicts (single call for all metrics)
            experiment.metrics.log(
                epoch=100,
                train=dict(loss=0.142, accuracy=0.80),
                eval=dict(loss=0.201, accuracy=0.76)
            )

            # Equivalent to _flush=True
            experiment.metrics.log(epoch=100, _flush=True)
        """
        # Separate nested dicts from scalar values
        scalar_data = {}
        nested_data = {}

        for key, value in kwargs.items():
            if isinstance(value, dict):
                nested_data[key] = value
            else:
                scalar_data[key] = value

        # Log scalar data to unnamed metric
        if scalar_data:
            self._experiment._append_to_metric(None, scalar_data, None, None, None)

        # Log nested dicts to their respective prefixed metrics
        for prefix, data in nested_data.items():
            # Include scalar data (like epoch) with each nested metric
            combined_data = {**scalar_data, **data}
            self(prefix).log(**combined_data)

        if _flush:
            self.flush()

        return self

    def flush(self) -> 'MetricsManager':
        """
        Flush buffered data (for method chaining).

        Currently a no-op as data is written immediately, but supports
        the fluent API pattern:
            experiment.metrics.log(epoch=epoch).flush()

        Returns:
            Self for method chaining
        """
        # Data is written immediately, so nothing to flush
        # This method exists for API consistency and chaining
        return self


class MetricBuilder:
    """
    Builder for metric operations.

    Provides fluent API for logging, reading, and querying metric data.

    Usage:
        # Log single data point
        experiment.metrics("train").log(loss=0.5, accuracy=0.9)

        # Read data
        data = experiment.metrics("train").read(start_index=0, limit=100)

        # Get statistics
        stats = experiment.metrics("train").stats()
    """

    def __init__(self, experiment: 'Experiment', name: str, description: Optional[str] = None,
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
                 metrics_manager: Optional['MetricsManager'] = None):
        """
        Initialize MetricBuilder.

        Args:
            experiment: Parent Experiment instance
            name: Metric name (unique within experiment)
            description: Optional metric description
            tags: Optional tags for categorization
            metadata: Optional structured metadata (units, type, etc.)
            metrics_manager: Parent MetricsManager (for buffer access)
        """
        self._experiment = experiment
        self._name = name
        self._description = description
        self._tags = tags
        self._metadata = metadata
        self._metrics_manager = metrics_manager
        self._summary_cache = None  # Lazy initialization

    def buffer(self, **kwargs) -> 'MetricBuilder':
        """
        Buffer values for later aggregation via metrics.buffer.log_summary().

        Values are accumulated and statistics are computed when log_summary() is called.

        Args:
            **kwargs: Metric values to buffer (e.g., loss=0.5, accuracy=0.9)

        Returns:
            Self for method chaining

        Example:
            # Accumulate values during training
            for batch in dataloader:
                metrics("train").buffer(loss=loss, acc=acc)

            # Log summary at end of epoch
            metrics.buffer.log_summary()        # logs loss.mean, acc.mean
            metrics.buffer.log_summary("mean", "std")  # logs loss.mean, loss.std, etc.
        """
        if self._metrics_manager is None:
            raise RuntimeError("buffer() requires MetricsManager reference")
        self._metrics_manager.buffer._store(self._name, **kwargs)
        return self

    def log(self, **kwargs) -> 'MetricBuilder':
        """
        Log a single data point to the metric.

        The data point can have any structure - common patterns:
        - {loss: 0.3, accuracy: 0.92}
        - {value: 0.5, step: 100}
        - {timestamp: "...", temperature: 25.5, humidity: 60}

        Supports method chaining for fluent API:
            experiment.metrics("train").log(loss=0.5, accuracy=0.9)

        Args:
            **kwargs: Data point fields (flexible schema)

        Returns:
            Self for method chaining

        Example:
            experiment.metrics("train").log(loss=0.5, accuracy=0.9)
            experiment.metrics.log(epoch=epoch).flush()
        """
        self._experiment._append_to_metric(
            name=self._name,
            data=kwargs,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )
        return self

    def flush(self) -> 'MetricBuilder':
        """
        Flush buffered data (for method chaining).

        Currently a no-op as data is written immediately, but supports
        the fluent API pattern:
            experiment.metrics.log(epoch=epoch).flush()

        Returns:
            Self for method chaining
        """
        # Data is written immediately, so nothing to flush
        # This method exists for API consistency and chaining
        return self

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
