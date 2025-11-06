"""
Files module for ML-Dash SDK.

Provides fluent API for file upload, download, list, and delete operations.
"""

import hashlib
import mimetypes
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .experiment import Experiment


class FileBuilder:
    """
    Fluent interface for file operations.

    Usage:
        # Upload file
        experiment.file(file_path="./model.pt", prefix="/models").save()

        # List files
        files = experiment.file().list()
        files = experiment.file(prefix="/models").list()

        # Download file
        experiment.file(file_id="123").download()
        experiment.file(file_id="123", dest_path="./model.pt").download()

        # Delete file
        experiment.file(file_id="123").delete()
    """

    def __init__(self, experiment: 'Experiment', **kwargs):
        """
        Initialize file builder.

        Args:
            experiment: Parent experiment instance
            **kwargs: File operation parameters
                - file_path: Path to file to upload
                - prefix: Logical path prefix (default: "/")
                - description: Optional description
                - tags: Optional list of tags
                - bindrs: Optional list of bindrs
                - metadata: Optional metadata dict
                - file_id: File ID for download/delete/update operations
                - dest_path: Destination path for download
        """
        self._experiment = experiment
        self._file_path = kwargs.get('file_path')
        self._prefix = kwargs.get('prefix', '/')
        self._description = kwargs.get('description')
        self._tags = kwargs.get('tags', [])
        self._bindrs = kwargs.get('bindrs', [])
        self._metadata = kwargs.get('metadata')
        self._file_id = kwargs.get('file_id')
        self._dest_path = kwargs.get('dest_path')

    def save(self) -> Dict[str, Any]:
        """
        Upload and save the file.

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If file_path not provided or file doesn't exist
            ValueError: If file size exceeds 5GB limit

        Examples:
            result = experiment.file(file_path="./model.pt", prefix="/models").save()
            # Returns: {"id": "123", "path": "/models", "filename": "model.pt", ...}
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        if not self._file_path:
            raise ValueError("file_path is required for save() operation")

        file_path = Path(self._file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {self._file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {self._file_path}")

        # Check file size (max 5GB)
        file_size = file_path.stat().st_size
        MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB in bytes
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size ({file_size} bytes) exceeds 5GB limit")

        # Compute checksum
        checksum = compute_sha256(str(file_path))

        # Detect MIME type
        content_type = get_mime_type(str(file_path))

        # Get filename
        filename = file_path.name

        # Upload through experiment
        return self._experiment._upload_file(
            file_path=str(file_path),
            prefix=self._prefix,
            filename=filename,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata,
            checksum=checksum,
            content_type=content_type,
            size_bytes=file_size
        )

    def list(self) -> List[Dict[str, Any]]:
        """
        List files with optional filters.

        Returns:
            List of file metadata dicts

        Raises:
            RuntimeError: If experiment is not open

        Examples:
            files = experiment.file().list()  # All files
            files = experiment.file(prefix="/models").list()  # Filter by prefix
            files = experiment.file(tags=["checkpoint"]).list()  # Filter by tags
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        return self._experiment._list_files(
            prefix=self._prefix if self._prefix != '/' else None,
            tags=self._tags if self._tags else None
        )

    def download(self) -> str:
        """
        Download file with automatic checksum verification.

        If dest_path not provided, downloads to current directory with original filename.

        Returns:
            Path to downloaded file

        Raises:
            RuntimeError: If experiment is not open
            ValueError: If file_id not provided
            ValueError: If checksum verification fails

        Examples:
            # Download to current directory with original filename
            path = experiment.file(file_id="123").download()

            # Download to custom path
            path = experiment.file(file_id="123", dest_path="./model.pt").download()
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        if not self._file_id:
            raise ValueError("file_id is required for download() operation")

        return self._experiment._download_file(
            file_id=self._file_id,
            dest_path=self._dest_path
        )

    def delete(self) -> Dict[str, Any]:
        """
        Delete file (soft delete).

        Returns:
            Dict with id and deletedAt timestamp

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If file_id not provided

        Examples:
            result = experiment.file(file_id="123").delete()
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        if not self._file_id:
            raise ValueError("file_id is required for delete() operation")

        return self._experiment._delete_file(file_id=self._file_id)

    def update(self) -> Dict[str, Any]:
        """
        Update file metadata (description, tags, metadata).

        Returns:
            Updated file metadata dict

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If file_id not provided

        Examples:
            result = experiment.file(
                file_id="123",
                description="Updated description",
                tags=["new", "tags"],
                metadata={"updated": True}
            ).update()
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        if not self._file_id:
            raise ValueError("file_id is required for update() operation")

        return self._experiment._update_file(
            file_id=self._file_id,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )

    def save_json(self, content: Any, file_name: str) -> Dict[str, Any]:
        """
        Save JSON content to a file.

        Args:
            content: Content to save as JSON (dict, list, or any JSON-serializable object)
            file_name: Name of the file to create

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If content is not JSON-serializable

        Examples:
            config = {"model": "resnet50", "lr": 0.001}
            result = experiment.file(prefix="/configs").save_json(config, "config.json")
        """
        import json
        import tempfile
        import os

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.json', text=True)
        try:
            # Write JSON content to temp file
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(content, f, indent=2)

            # Save using existing save() method
            original_file_path = self._file_path
            self._file_path = temp_path

            # Upload and get result
            result = self.save()

            # Restore original file_path
            self._file_path = original_file_path

            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def save_torch(self, model: Any, file_name: str) -> Dict[str, Any]:
        """
        Save PyTorch model to a file.

        Args:
            model: PyTorch model or state dict to save
            file_name: Name of the file to create (should end with .pt or .pth)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ImportError: If torch is not installed
            ValueError: If model cannot be saved

        Examples:
            import torch
            model = torch.nn.Linear(10, 5)
            result = experiment.file(prefix="/models").save_torch(model, "model.pt")

            # Or save state dict
            result = experiment.file(prefix="/models").save_torch(model.state_dict(), "model.pth")
        """
        import tempfile
        import os

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pt')
        os.close(temp_fd)  # Close the file descriptor

        try:
            # Save model to temp file
            torch.save(model, temp_path)

            # Save using existing save() method
            original_file_path = self._file_path
            self._file_path = temp_path

            # Upload and get result
            result = self.save()

            # Restore original file_path
            self._file_path = original_file_path

            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def compute_sha256(file_path: str) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA256 checksum

    Examples:
        checksum = compute_sha256("./model.pt")
        # Returns: "abc123def456..."
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def get_mime_type(file_path: str) -> str:
    """
    Detect MIME type of a file.

    Args:
        file_path: Path to file

    Returns:
        MIME type string (default: "application/octet-stream")

    Examples:
        mime_type = get_mime_type("./model.pt")
        # Returns: "application/octet-stream"

        mime_type = get_mime_type("./image.png")
        # Returns: "image/png"
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def verify_checksum(file_path: str, expected_checksum: str) -> bool:
    """
    Verify SHA256 checksum of a file.

    Args:
        file_path: Path to file
        expected_checksum: Expected SHA256 checksum (hex-encoded)

    Returns:
        True if checksum matches, False otherwise

    Examples:
        is_valid = verify_checksum("./model.pt", "abc123...")
    """
    actual_checksum = compute_sha256(file_path)
    return actual_checksum == expected_checksum


def generate_snowflake_id() -> str:
    """
    Generate a simple Snowflake-like ID for local mode.

    Not a true Snowflake ID, but provides unique IDs for local storage.

    Returns:
        String representation of generated ID
    """
    import time
    import random

    timestamp = int(time.time() * 1000)
    random_bits = random.randint(0, 4095)
    return str((timestamp << 12) | random_bits)
