"""
Log API for ML-Dash SDK.

Provides fluent interface for structured logging with validation.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from .experiment import Experiment


class LogLevel(Enum):
    """
    Standard log levels for ML-Dash.

    Supported levels:
    - INFO: Informational messages
    - WARN: Warning messages
    - ERROR: Error messages
    - DEBUG: Debug messages
    - FATAL: Fatal error messages
    """
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    DEBUG = "debug"
    FATAL = "fatal"

    @classmethod
    def validate(cls, level: str) -> str:
        """
        Validate and normalize log level.

        Args:
            level: Log level string (case-insensitive)

        Returns:
            Normalized log level string (lowercase)

        Raises:
            ValueError: If level is not one of the 5 standard levels

        Example:
            >>> LogLevel.validate("INFO")
            "info"
            >>> LogLevel.validate("invalid")
            ValueError: Invalid log level: 'invalid'. Must be one of: info, warn, error, debug, fatal
        """
        level_lower = level.lower()
        try:
            return cls[level_lower.upper()].value
        except KeyError:
            valid_levels = ", ".join([l.value for l in cls])
            raise ValueError(
                f"Invalid log level: '{level}'. "
                f"Must be one of: {valid_levels}"
            )


class LogBuilder:
    """
    Fluent builder for creating log entries.

    This class is returned by Experiment.log() when no message is provided.
    It allows for a fluent API style where metadata is set first, then
    the log level method is called to write the log.

    Example:
        experiment.log(metadata={"epoch": 1}).info("Training started")
        experiment.log().error("Failed", error_code=500)
    """

    def __init__(self, experiment: 'Experiment', metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize LogBuilder.

        Args:
            experiment: Parent Experiment instance
            metadata: Optional metadata dict from log() call
        """
        self._experiment = experiment
        self._metadata = metadata

    def info(self, message: str, **extra_metadata) -> None:
        """
        Write info level log.

        Args:
            message: Log message
            **extra_metadata: Additional metadata as keyword arguments

        Example:
            experiment.log().info("Training started")
            experiment.log().info("Epoch complete", epoch=1, loss=0.5)
        """
        self._write(LogLevel.INFO.value, message, extra_metadata)

    def warn(self, message: str, **extra_metadata) -> None:
        """
        Write warning level log.

        Args:
            message: Log message
            **extra_metadata: Additional metadata as keyword arguments

        Example:
            experiment.log().warn("High loss detected", loss=1.5)
        """
        self._write(LogLevel.WARN.value, message, extra_metadata)

    def error(self, message: str, **extra_metadata) -> None:
        """
        Write error level log.

        Args:
            message: Log message
            **extra_metadata: Additional metadata as keyword arguments

        Example:
            experiment.log().error("Failed to save", path="/models/checkpoint.pth")
        """
        self._write(LogLevel.ERROR.value, message, extra_metadata)

    def debug(self, message: str, **extra_metadata) -> None:
        """
        Write debug level log.

        Args:
            message: Log message
            **extra_metadata: Additional metadata as keyword arguments

        Example:
            experiment.log().debug("Memory usage", memory_mb=2500)
        """
        self._write(LogLevel.DEBUG.value, message, extra_metadata)

    def fatal(self, message: str, **extra_metadata) -> None:
        """
        Write fatal level log.

        Args:
            message: Log message
            **extra_metadata: Additional metadata as keyword arguments

        Example:
            experiment.log().fatal("Unrecoverable error", exit_code=1)
        """
        self._write(LogLevel.FATAL.value, message, extra_metadata)

    def _write(self, level: str, message: str, extra_metadata: Dict[str, Any]) -> None:
        """
        Internal: Execute the actual log write.

        Merges metadata from log() call with metadata from level method,
        then writes immediately (no buffering).

        Args:
            level: Log level (already validated)
            message: Log message
            extra_metadata: Additional metadata from level method kwargs
        """
        # Merge metadata from log() call with metadata from level method
        if self._metadata and extra_metadata:
            final_metadata = {**self._metadata, **extra_metadata}
        elif self._metadata:
            final_metadata = self._metadata
        elif extra_metadata:
            final_metadata = extra_metadata
        else:
            final_metadata = None

        # Write immediately (no buffering)
        self._experiment._write_log(
            message=message,
            level=level,
            metadata=final_metadata,
            timestamp=None
        )
