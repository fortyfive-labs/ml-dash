"""Text logging component for ML-Logger."""

import json
import time
from typing import Any, Dict, Optional, List

from ..backends.base import StorageBackend


class LogManager:
    """Manages structured text logging.

    Logs are stored in a JSONL file (logs.jsonl).

    Args:
        backend: Storage backend
        prefix: Experiment prefix path
    """

    def __init__(self, backend: StorageBackend, prefix: str):
        """Initialize log manager.

        Args:
            backend: Storage backend
            prefix: Experiment prefix path
        """
        self.backend = backend
        self.prefix = prefix
        self.logs_file = f"{prefix}/logs.jsonl"

    def log(self, message: str, level: str = "INFO", **context) -> None:
        """Log a message with context.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            **context: Additional context fields
        """
        entry = {
            "timestamp": time.time(),
            "level": level.upper(),
            "message": message,
        }

        if context:
            entry["context"] = context

        line = json.dumps(entry) + "\n"
        self.backend.append_text(self.logs_file, line)

    def info(self, message: str, **context) -> None:
        """Log an info message.

        Args:
            message: Log message
            **context: Additional context fields
        """
        self.log(message, level="INFO", **context)

    def warning(self, message: str, **context) -> None:
        """Log a warning message.

        Args:
            message: Log message
            **context: Additional context fields
        """
        self.log(message, level="WARNING", **context)

    def error(self, message: str, **context) -> None:
        """Log an error message.

        Args:
            message: Log message
            **context: Additional context fields
        """
        self.log(message, level="ERROR", **context)

    def debug(self, message: str, **context) -> None:
        """Log a debug message.

        Args:
            message: Log message
            **context: Additional context fields
        """
        self.log(message, level="DEBUG", **context)

    def read(self) -> List[Dict[str, Any]]:
        """Read all logs from file.

        Returns:
            List of log entries
        """
        if not self.backend.exists(self.logs_file):
            return []

        content = self.backend.read_text(self.logs_file)
        logs = []

        for line in content.strip().split("\n"):
            if not line:
                continue
            logs.append(json.loads(line))

        return logs
