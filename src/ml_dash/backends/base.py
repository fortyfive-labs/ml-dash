"""Storage backend abstract base class for ML-Logger.

This module defines the abstract interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage backends (local, S3, GCP, ML-Dash) must implement these methods.
    """

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists.

        Args:
            path: Path to check

        Returns:
            True if path exists, False otherwise
        """
        pass

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write binary data to a file.

        Args:
            path: File path
            data: Binary data to write
        """
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read binary data from a file.

        Args:
            path: File path

        Returns:
            Binary data from file
        """
        pass

    @abstractmethod
    def write_text(self, path: str, text: str) -> None:
        """Write text to a file.

        Args:
            path: File path
            text: Text to write
        """
        pass

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read text from a file.

        Args:
            path: File path

        Returns:
            Text content from file
        """
        pass

    @abstractmethod
    def append_text(self, path: str, text: str) -> None:
        """Append text to a file.

        Args:
            path: File path
            text: Text to append
        """
        pass

    @abstractmethod
    def list_dir(self, path: str = "") -> List[str]:
        """List contents of a directory.

        Args:
            path: Directory path (empty string for root)

        Returns:
            List of file/directory names
        """
        pass

    @abstractmethod
    def get_url(self, path: str) -> Optional[str]:
        """Get a URL for accessing a file (if applicable).

        Args:
            path: File path

        Returns:
            URL string or None if not applicable
        """
        pass

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directories recursively.

        Args:
            path: Directory path to create
            exist_ok: Don't raise error if directory exists
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file.

        Args:
            path: File path to delete
        """
        pass
