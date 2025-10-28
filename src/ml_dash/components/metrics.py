"""Metrics logging component for ML-Logger."""

import json
import time
from typing import Any, Dict, Optional, List
from collections import defaultdict

from ..backends.base import StorageBackend


class MetricsLogger:
    """Logs metrics with support for namespacing and aggregation.

    Metrics are stored in a single JSONL file (metrics.jsonl).

    Args:
        backend: Storage backend
        prefix: Experiment prefix path
        namespace: Optional namespace for metrics (e.g., "train", "val")
    """

    def __init__(
        self,
        backend: StorageBackend,
        prefix: str,
        namespace: str = ""
    ):
        """Initialize metrics logger.

        Args:
            backend: Storage backend
            prefix: Experiment prefix path
            namespace: Optional namespace prefix
        """
        self.backend = backend
        self.prefix = prefix
        self.namespace = namespace
        self.metrics_file = f"{prefix}/metrics.jsonl"
        self._collect_buffer: Dict[str, List[float]] = defaultdict(list)

    def log(self, step: Optional[int] = None, **metrics) -> None:
        """Log metrics immediately.

        Args:
            step: Step number (epoch, iteration, etc.)
            **metrics: Metric name-value pairs
        """
        # Apply namespace to metric names
        namespaced_metrics = {}
        for key, value in metrics.items():
            if self.namespace:
                key = f"{self.namespace}.{key}"
            namespaced_metrics[key] = value

        entry = {
            "timestamp": time.time(),
            "metrics": namespaced_metrics
        }

        if step is not None:
            entry["step"] = step

        line = json.dumps(entry) + "\n"
        self.backend.append_text(self.metrics_file, line)

    def collect(self, step: Optional[int] = None, **metrics) -> None:
        """Collect metrics for later aggregation.

        Args:
            step: Step number (optional, used by flush)
            **metrics: Metric name-value pairs
        """
        for key, value in metrics.items():
            if self.namespace:
                key = f"{self.namespace}.{key}"
            self._collect_buffer[key].append(float(value))

    def flush(
        self,
        _aggregation: str = "mean",
        step: Optional[int] = None,
        **additional_metrics
    ) -> None:
        """Flush collected metrics with aggregation.

        Args:
            _aggregation: Aggregation method ("mean", "sum", "min", "max", "last")
            step: Step number for logged metrics
            **additional_metrics: Additional metrics to log (not aggregated)
        """
        if not self._collect_buffer and not additional_metrics:
            return

        aggregated = {}

        # Aggregate collected metrics
        for key, values in self._collect_buffer.items():
            if not values:
                continue

            if _aggregation == "mean":
                aggregated[key] = sum(values) / len(values)
            elif _aggregation == "sum":
                aggregated[key] = sum(values)
            elif _aggregation == "min":
                aggregated[key] = min(values)
            elif _aggregation == "max":
                aggregated[key] = max(values)
            elif _aggregation == "last":
                aggregated[key] = values[-1]
            else:
                raise ValueError(f"Unknown aggregation method: {_aggregation}")

        # Add non-aggregated metrics
        for key, value in additional_metrics.items():
            if self.namespace:
                key = f"{self.namespace}.{key}"
            aggregated[key] = value

        # Log aggregated metrics
        if aggregated:
            entry = {
                "timestamp": time.time(),
                "metrics": aggregated
            }

            if step is not None:
                entry["step"] = step

            line = json.dumps(entry) + "\n"
            self.backend.append_text(self.metrics_file, line)

        # Clear buffer
        self._collect_buffer.clear()

    def __call__(self, namespace: str) -> "MetricsLogger":
        """Create a namespaced metrics logger.

        Args:
            namespace: Namespace name (e.g., "train", "val")

        Returns:
            New MetricsLogger with the namespace
        """
        new_namespace = f"{self.namespace}.{namespace}" if self.namespace else namespace
        return MetricsLogger(
            backend=self.backend,
            prefix=self.prefix,
            namespace=new_namespace
        )

    def read(self) -> List[Dict[str, Any]]:
        """Read all metrics from file.

        Returns:
            List of metric entries
        """
        if not self.backend.exists(self.metrics_file):
            return []

        content = self.backend.read_text(self.metrics_file)
        metrics = []

        for line in content.strip().split("\n"):
            if not line:
                continue
            metrics.append(json.loads(line))

        return metrics
