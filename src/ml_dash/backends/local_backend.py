"""Local file system storage backend for ML-Logger."""

from pathlib import Path
from typing import Optional, List
import os

from .base import StorageBackend


class LocalBackend(StorageBackend):
    """Local file system storage backend.

    Stores all data in the local file system.

    Args:
        root_dir: Root directory for storage (default: ".ml-logger")
    """

    def __init__(self, root_dir: str = ".ml-logger"):
        """Initialize local backend.

        Args:
            root_dir: Root directory for storage
        """
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to root_dir.

        Args:
            path: Relative path

        Returns:
            Absolute Path object
        """
        return self.root_dir / path

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        return self._resolve_path(path).exists()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write binary data to a file."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    def read_bytes(self, path: str) -> bytes:
        """Read binary data from a file."""
        return self._resolve_path(path).read_bytes()

    def write_text(self, path: str, text: str) -> None:
        """Write text to a file."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(text, encoding="utf-8")

    def read_text(self, path: str) -> str:
        """Read text from a file."""
        return self._resolve_path(path).read_text(encoding="utf-8")

    def append_text(self, path: str, text: str) -> None:
        """Append text to a file."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(text)

    def list_dir(self, path: str = "") -> List[str]:
        """List contents of a directory."""
        full_path = self._resolve_path(path) if path else self.root_dir
        if not full_path.exists():
            return []
        return [item.name for item in full_path.iterdir()]

    def get_url(self, path: str) -> Optional[str]:
        """Get a file:// URL for accessing a file."""
        full_path = self._resolve_path(path)
        return f"file://{full_path}" if full_path.exists() else None

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directories recursively."""
        self._resolve_path(path).mkdir(parents=True, exist_ok=exist_ok)

    def delete(self, path: str) -> None:
        """Delete a file."""
        full_path = self._resolve_path(path)
        if full_path.exists():
            full_path.unlink()
