"""ML_Logger - Legacy logging class for backward compatibility.

This class provides a simpler interface for basic logging with filtering capabilities.
"""

import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern

from .backends.local_backend import LocalBackend
from .backends.base import StorageBackend


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class ML_Logger:
    """Legacy logger class with filtering capabilities.

    This class provides a simpler interface for logging with built-in filtering
    by log level, patterns, and custom filter functions.

    Args:
        prefix: Directory prefix for logging (e.g., "../data")
        backend: Storage backend (optional, defaults to LocalBackend)
    """

    def __init__(
        self,
        prefix: str,
        backend: Optional[StorageBackend] = None,
    ):
        """Initialize ML_Logger.

        Args:
            prefix: Directory prefix for logging
            backend: Storage backend (optional)
        """
        self.prefix = prefix
        self.backend = backend or LocalBackend(prefix)

        # Buffer for in-memory log storage
        self.buffer: List[Dict[str, Any]] = []

        # Filtering configuration
        self._min_level = LogLevel.DEBUG
        self._include_patterns: List[Pattern] = []
        self._exclude_patterns: List[Pattern] = []
        self._custom_filters: List[Callable] = []

    def log(self, message: str, level: str = "INFO", **context) -> None:
        """Log a message with optional context.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            **context: Additional context fields
        """
        entry = {
            "message": message,
            "level": level.upper(),
            **context
        }

        # Apply filters
        if self._should_log(entry):
            self.buffer.append(entry)

    def info(self, message: str, **context) -> None:
        """Log an info message.

        Args:
            message: Log message
            **context: Additional context
        """
        self.log(message, level="INFO", **context)

    def warning(self, message: str, **context) -> None:
        """Log a warning message.

        Args:
            message: Log message
            **context: Additional context
        """
        self.log(message, level="WARNING", **context)

    def error(self, message: str, **context) -> None:
        """Log an error message.

        Args:
            message: Log message
            **context: Additional context
        """
        self.log(message, level="ERROR", **context)

    def debug(self, message: str, **context) -> None:
        """Log a debug message.

        Args:
            message: Log message
            **context: Additional context
        """
        self.log(message, level="DEBUG", **context)

    def set_level(self, level: str) -> None:
        """Set minimum log level.

        Args:
            level: Minimum level (DEBUG, INFO, WARNING, ERROR)
        """
        self._min_level = LogLevel[level.upper()]

    def add_include_pattern(self, pattern: str) -> None:
        """Add a pattern to include in logs.

        Args:
            pattern: Regex pattern to match
        """
        self._include_patterns.append(re.compile(pattern))

    def add_exclude_pattern(self, pattern: str) -> None:
        """Add a pattern to exclude from logs.

        Args:
            pattern: Regex pattern to match
        """
        self._exclude_patterns.append(re.compile(pattern))

    def add_filter(self, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a custom filter function.

        Args:
            filter_func: Function that takes log entry and returns True to keep it
        """
        self._custom_filters.append(filter_func)

    def clear_filters(self) -> None:
        """Clear all filters."""
        self._min_level = LogLevel.DEBUG
        self._include_patterns.clear()
        self._exclude_patterns.clear()
        self._custom_filters.clear()

    def get_filtered_logs(
        self,
        level: Optional[str] = None,
        pattern: Optional[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get filtered logs from buffer.

        Args:
            level: Filter by log level
            pattern: Filter by regex pattern
            start_step: Filter by minimum step
            end_step: Filter by maximum step

        Returns:
            List of filtered log entries
        """
        filtered = self.buffer.copy()

        if level:
            filtered = [entry for entry in filtered if entry.get("level") == level.upper()]

        if pattern:
            regex = re.compile(pattern)
            filtered = [
                entry for entry in filtered
                if regex.search(entry.get("message", ""))
            ]

        if start_step is not None or end_step is not None:
            # Assign default step based on index in buffer if not present
            result = []
            for i, entry in enumerate(filtered):
                step = entry.get("step", i)
                if (start_step is None or step >= start_step) and \
                   (end_step is None or step <= end_step):
                    # Add step to entry if it wasn't there
                    entry_with_step = entry.copy()
                    if "step" not in entry_with_step:
                        entry_with_step["step"] = i
                    result.append(entry_with_step)
            return result

        return filtered

    def _should_log(self, entry: Dict[str, Any]) -> bool:
        """Check if an entry should be logged based on filters.

        Args:
            entry: Log entry to check

        Returns:
            True if entry should be logged
        """
        # Check log level
        entry_level = LogLevel[entry.get("level", "INFO")]
        if entry_level.value < self._min_level.value:
            return False

        message = entry.get("message", "")

        # Check include patterns
        if self._include_patterns:
            if not any(pattern.search(message) for pattern in self._include_patterns):
                return False

        # Check exclude patterns
        if self._exclude_patterns:
            if any(pattern.search(message) for pattern in self._exclude_patterns):
                return False

        # Check custom filters
        for filter_func in self._custom_filters:
            if not filter_func(entry):
                return False

        return True

    def clear_buffer(self) -> None:
        """Clear the log buffer."""
        self.buffer.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"ML_Logger(prefix='{self.prefix}', entries={len(self.buffer)})"
