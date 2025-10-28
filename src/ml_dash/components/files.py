"""File management component for ML-Logger."""

import json
import pickle
from typing import Any, Optional
from pathlib import Path

from ..backends.base import StorageBackend


class FileManager:
    """Manages file storage and retrieval.

    Files are stored in the files/ subdirectory.

    Args:
        backend: Storage backend
        prefix: Experiment prefix path
        namespace: Optional namespace for files (e.g., "checkpoints")
    """

    def __init__(
        self,
        backend: StorageBackend,
        prefix: str,
        namespace: str = ""
    ):
        """Initialize file manager.

        Args:
            backend: Storage backend
            prefix: Experiment prefix path
            namespace: Optional namespace subdirectory
        """
        self.backend = backend
        self.prefix = prefix
        self.namespace = namespace

    def _get_file_path(self, filename: str) -> str:
        """Get full file path with namespace.

        Args:
            filename: File name

        Returns:
            Full path including prefix, files/, and namespace
        """
        parts = [self.prefix, "files"]
        if self.namespace:
            parts.append(self.namespace)
        parts.append(filename)
        return "/".join(parts)

    def save(self, data: Any, filename: str) -> None:
        """Save data to a file (auto-detects format).

        Supports: JSON (.json), pickle (.pkl, .pickle), PyTorch (.pt, .pth),
        NumPy (.npy, .npz), and raw bytes.

        Args:
            data: Data to save
            filename: File name
        """
        file_path = self._get_file_path(filename)
        suffix = Path(filename).suffix.lower()

        if suffix == ".json":
            # Save as JSON
            json_str = json.dumps(data, indent=2)
            self.backend.write_text(file_path, json_str)

        elif suffix in [".pkl", ".pickle"]:
            # Save as pickle
            pickled = pickle.dumps(data)
            self.backend.write_bytes(file_path, pickled)

        elif suffix in [".pt", ".pth"]:
            # Save PyTorch tensor/model
            try:
                import torch
                import io
                buffer = io.BytesIO()
                torch.save(data, buffer)
                self.backend.write_bytes(file_path, buffer.getvalue())
            except ImportError:
                raise ImportError("PyTorch is required to save .pt/.pth files")

        elif suffix in [".npy", ".npz"]:
            # Save NumPy array
            try:
                import numpy as np
                import io
                buffer = io.BytesIO()
                if suffix == ".npy":
                    np.save(buffer, data)
                else:
                    np.savez(buffer, data)
                self.backend.write_bytes(file_path, buffer.getvalue())
            except ImportError:
                raise ImportError("NumPy is required to save .npy/.npz files")

        else:
            # Save as raw bytes
            if isinstance(data, bytes):
                self.backend.write_bytes(file_path, data)
            elif isinstance(data, str):
                self.backend.write_text(file_path, data)
            else:
                # Fallback to pickle
                pickled = pickle.dumps(data)
                self.backend.write_bytes(file_path, pickled)

    def save_pkl(self, data: Any, filename: str) -> None:
        """Save data as pickle file.

        Args:
            data: Data to save
            filename: File name (will add .pkl if missing)
        """
        if not filename.endswith((".pkl", ".pickle")):
            filename = f"{filename}.pkl"
        self.save(data, filename)

    def load(self, filename: str) -> Any:
        """Load data from a file (auto-detects format).

        Args:
            filename: File name

        Returns:
            Loaded data
        """
        file_path = self._get_file_path(filename)
        suffix = Path(filename).suffix.lower()

        if suffix == ".json":
            # Load JSON
            json_str = self.backend.read_text(file_path)
            return json.loads(json_str)

        elif suffix in [".pkl", ".pickle"]:
            # Load pickle
            pickled = self.backend.read_bytes(file_path)
            return pickle.loads(pickled)

        elif suffix in [".pt", ".pth"]:
            # Load PyTorch
            try:
                import torch
                import io
                data = self.backend.read_bytes(file_path)
                buffer = io.BytesIO(data)
                return torch.load(buffer)
            except ImportError:
                raise ImportError("PyTorch is required to load .pt/.pth files")

        elif suffix in [".npy", ".npz"]:
            # Load NumPy
            try:
                import numpy as np
                import io
                data = self.backend.read_bytes(file_path)
                buffer = io.BytesIO(data)
                if suffix == ".npy":
                    return np.load(buffer)
                else:
                    return np.load(buffer, allow_pickle=True)
            except ImportError:
                raise ImportError("NumPy is required to load .npy/.npz files")

        else:
            # For unknown extensions, try different strategies
            data = self.backend.read_bytes(file_path)

            # If it looks like a binary extension, return bytes directly
            if suffix in [".bin", ".dat", ".raw"]:
                return data

            # Try to unpickle first (handles custom extensions from save())
            try:
                return pickle.loads(data)
            except (pickle.UnpicklingError, EOFError, AttributeError):
                pass

            # Try to decode as text
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                # Return raw bytes as fallback
                return data

    def load_torch(self, filename: str) -> Any:
        """Load PyTorch checkpoint.

        Args:
            filename: File name

        Returns:
            Loaded PyTorch data
        """
        if not filename.endswith((".pt", ".pth")):
            filename = f"{filename}.pt"
        return self.load(filename)

    def __call__(self, namespace: str) -> "FileManager":
        """Create a namespaced file manager.

        Args:
            namespace: Namespace subdirectory (e.g., "checkpoints")

        Returns:
            New FileManager with the namespace
        """
        new_namespace = f"{self.namespace}/{namespace}" if self.namespace else namespace
        return FileManager(
            backend=self.backend,
            prefix=self.prefix,
            namespace=new_namespace
        )

    def exists(self, filename: str) -> bool:
        """Check if a file exists.

        Args:
            filename: File name

        Returns:
            True if file exists
        """
        file_path = self._get_file_path(filename)
        return self.backend.exists(file_path)

    def list(self) -> list:
        """List files in the current namespace.

        Returns:
            List of file names
        """
        dir_path = f"{self.prefix}/files"
        if self.namespace:
            dir_path = f"{dir_path}/{self.namespace}"

        try:
            return self.backend.list_dir(dir_path)
        except Exception:
            return []
