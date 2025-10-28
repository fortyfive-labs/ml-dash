"""Parameter management component for ML-Logger."""

import json
import time
from typing import Any, Dict, Optional

from ..backends.base import StorageBackend


def deep_merge(base: Dict, updates: Dict) -> Dict:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        updates: Updates to merge in

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ParameterManager:
    """Manages experiment parameters.

    Parameters are stored in an append-only JSONL file (parameters.jsonl)
    with operations: set, extend, update.

    Args:
        backend: Storage backend
        prefix: Experiment prefix path
    """

    def __init__(self, backend: StorageBackend, prefix: str):
        """Initialize parameter manager.

        Args:
            backend: Storage backend
            prefix: Experiment prefix path
        """
        self.backend = backend
        self.prefix = prefix
        self.params_file = f"{prefix}/parameters.jsonl"
        self._cache: Optional[Dict[str, Any]] = None

    def set(self, **kwargs) -> None:
        """Set parameters (replaces existing).

        Args:
            **kwargs: Parameter key-value pairs
        """
        self._append_operation("set", data=kwargs)
        self._cache = None  # Invalidate cache

    def extend(self, **kwargs) -> None:
        """Extend parameters (deep merge with existing).

        Args:
            **kwargs: Parameter key-value pairs to merge
        """
        self._append_operation("extend", data=kwargs)
        self._cache = None  # Invalidate cache

    def update(self, key: str, value: Any) -> None:
        """Update a single parameter.

        Args:
            key: Parameter key (can be dot-separated like "model.layers")
            value: New value
        """
        self._append_operation("update", key=key, value=value)
        self._cache = None  # Invalidate cache

    def read(self) -> Dict[str, Any]:
        """Read current parameters by replaying operations.

        Returns:
            Current parameter dictionary
        """
        if self._cache is not None:
            return self._cache.copy()

        params = {}

        if not self.backend.exists(self.params_file):
            return params

        # Read and replay all operations
        content = self.backend.read_text(self.params_file)
        for line in content.strip().split("\n"):
            if not line:
                continue

            operation = json.loads(line)
            op_type = operation.get("operation")

            if op_type == "set":
                params = operation.get("data", {})
            elif op_type == "extend":
                params = deep_merge(params, operation.get("data", {}))
            elif op_type == "update":
                key = operation.get("key")
                value = operation.get("value")
                if key:
                    # Support dot-separated keys
                    keys = key.split(".")
                    current = params
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value

        self._cache = params
        return params.copy()

    def log(self, **kwargs) -> None:
        """Alias for set() to match API documentation.

        Args:
            **kwargs: Parameter key-value pairs
        """
        self.set(**kwargs)

    def _append_operation(self, operation: str, **kwargs) -> None:
        """Append an operation to the parameters file.

        Args:
            operation: Operation type (set, extend, update)
            **kwargs: Operation-specific data
        """
        entry = {
            "timestamp": time.time(),
            "operation": operation,
            **kwargs
        }
        line = json.dumps(entry) + "\n"
        self.backend.append_text(self.params_file, line)
