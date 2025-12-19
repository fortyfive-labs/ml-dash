"""
Files module for ML-Dash SDK.

Provides fluent API for file upload, download, list, and delete operations.
"""

import hashlib
import mimetypes
import fnmatch
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .experiment import Experiment


class FileBuilder:
    """
    Fluent interface for file operations.

    Usage:
        # Upload file
        experiment.files("checkpoints").save(net, to="checkpoint.pt")

        # List files
        files = experiment.files("/some/location").list()
        files = experiment.files("/models").list()

        # Download file
        experiment.files("some.text").download()
        experiment.files("some.text").download(to="./model.pt")

        # Download Files via Glob Pattern
        file_paths = experiment.files("images").list("*.png")
        experiment.files("images").download("*.png")

        # Delete files
        experiment.files("some.text").delete()

    Specific File Types:
        dxp.files.save_text("content", to="view.yaml")
        dxp.files.save_json(dict(hey="yo"), to="config.json")
        dxp.files.save_blob(b"xxx", to="data.bin")
    """

    def __init__(self, experiment: 'Experiment', path: Optional[str] = None, **kwargs):
        """
        Initialize file builder.

        Args:
            experiment: Parent experiment instance
            path: File path or prefix for operations. Can be:
                - A prefix/directory (e.g., "checkpoints", "/models")
                - A file path (e.g., "some.text", "images/photo.png")
            **kwargs: Additional file operation parameters (for backwards compatibility)
                - file_path: Path to file to upload (deprecated, use save(to=))
                - prefix: Logical path prefix (deprecated, use path argument)
                - description: Optional description
                - tags: Optional list of tags
                - bindrs: Optional list of bindrs
                - metadata: Optional metadata dict
                - file_id: File ID for download/delete/update operations
                - dest_path: Destination path for download (deprecated, use download(to=))
        """
        self._experiment = experiment
        self._path = path
        # Backwards compatibility
        self._file_path = kwargs.get('file_path')
        self._prefix = kwargs.get('prefix', '/')
        self._description = kwargs.get('description')
        self._tags = kwargs.get('tags', [])
        self._bindrs = kwargs.get('bindrs', [])
        self._metadata = kwargs.get('metadata')
        self._file_id = kwargs.get('file_id')
        self._dest_path = kwargs.get('dest_path')

        # If path is provided, determine if it's a file or prefix
        if path:
            # Normalize path
            path = path.lstrip('/')
            self._normalized_path = '/' + path if not path.startswith('/') else path

    def save(
        self,
        obj: Optional[Any] = None,
        *,
        to: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload and save a file or object.

        Args:
            obj: Object to save. Can be:
                - None: Uses file_path from constructor (backwards compatibility)
                - str: Path to an existing file
                - bytes: Binary data to save
                - dict/list: JSON-serializable data
                - PyTorch model/state_dict: Saved with torch.save()
                - matplotlib figure: Saved as image
                - Any picklable object: Saved with pickle
            to: Target filename (required when obj is not a file path)
            description: Optional description (overrides constructor)
            tags: Optional list of tags (overrides constructor)
            metadata: Optional metadata dict (overrides constructor)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If file_path not provided or file doesn't exist
            ValueError: If file size exceeds 100GB limit

        Examples:
            # Save existing file
            experiment.files("models").save("./model.pt")
            experiment.files("models").save(to="model.pt")  # copies from self._file_path

            # Save PyTorch model
            experiment.files("checkpoints").save(model, to="checkpoint.pt")
            experiment.files("checkpoints").save(model.state_dict(), to="weights.pt")

            # Save dict as JSON
            experiment.files("configs").save({"lr": 0.001}, to="config.json")

            # Save bytes
            experiment.files("data").save(b"binary data", to="data.bin")
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Use provided values or fall back to constructor values
        desc = description if description is not None else self._description
        file_tags = tags if tags is not None else self._tags
        file_metadata = metadata if metadata is not None else self._metadata

        # Determine prefix from path
        prefix = self._prefix
        if self._path:
            prefix = '/' + self._path.lstrip('/')

        # Handle different object types
        if obj is None:
            # Backwards compatibility: use file_path from constructor
            if not self._file_path:
                raise ValueError("No file or object provided. Pass a file path or object to save().")
            return self._save_file(
                file_path=self._file_path,
                prefix=prefix,
                description=desc,
                tags=file_tags,
                metadata=file_metadata
            )

        if isinstance(obj, str) and Path(obj).exists():
            # obj is a path to an existing file
            return self._save_file(
                file_path=obj,
                prefix=prefix,
                description=desc,
                tags=file_tags,
                metadata=file_metadata
            )

        if isinstance(obj, bytes):
            # Save bytes directly
            if not to:
                raise ValueError("'to' parameter is required when saving bytes")
            return self._save_bytes(
                data=obj,
                filename=to,
                prefix=prefix,
                description=desc,
                tags=file_tags,
                metadata=file_metadata
            )

        if isinstance(obj, (dict, list)):
            # Try JSON first
            if not to:
                raise ValueError("'to' parameter is required when saving dict/list")
            return self._save_json(
                content=obj,
                filename=to,
                prefix=prefix,
                description=desc,
                tags=file_tags,
                metadata=file_metadata
            )

        # Check for PyTorch model
        try:
            import torch
            if isinstance(obj, (torch.nn.Module, dict)) or hasattr(obj, 'state_dict'):
                if not to:
                    raise ValueError("'to' parameter is required when saving PyTorch model")
                return self._save_torch(
                    model=obj,
                    filename=to,
                    prefix=prefix,
                    description=desc,
                    tags=file_tags,
                    metadata=file_metadata
                )
        except ImportError:
            pass

        # Check for matplotlib figure
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            if isinstance(obj, Figure):
                if not to:
                    raise ValueError("'to' parameter is required when saving matplotlib figure")
                return self._save_fig(
                    fig=obj,
                    filename=to,
                    prefix=prefix,
                    description=desc,
                    tags=file_tags,
                    metadata=file_metadata
                )
        except ImportError:
            pass

        # Fall back to pickle
        if not to:
            raise ValueError("'to' parameter is required when saving object")
        return self._save_pickle(
            content=obj,
            filename=to,
            prefix=prefix,
            description=desc,
            tags=file_tags,
            metadata=file_metadata
        )

    def _save_file(
        self,
        file_path: str,
        prefix: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal method to save an existing file."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise ValueError(f"File not found: {file_path}")

        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check file size (max 100GB)
        file_size = file_path_obj.stat().st_size
        MAX_FILE_SIZE = 100 * 1024 * 1024 * 1024  # 100GB in bytes
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size ({file_size} bytes) exceeds 100GB limit")

        # Compute checksum
        checksum = compute_sha256(str(file_path_obj))

        # Detect MIME type
        content_type = get_mime_type(str(file_path_obj))

        # Get filename
        filename = file_path_obj.name

        # Upload through experiment
        return self._experiment._upload_file(
            file_path=str(file_path_obj),
            prefix=prefix,
            filename=filename,
            description=description,
            tags=tags or [],
            metadata=metadata,
            checksum=checksum,
            content_type=content_type,
            size_bytes=file_size
        )

    def _save_bytes(
        self,
        data: bytes,
        filename: str,
        prefix: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save bytes data to a file."""
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        try:
            with open(temp_path, 'wb') as f:
                f.write(data)
            return self._save_file(
                file_path=temp_path,
                prefix=prefix,
                description=description,
                tags=tags,
                metadata=metadata
            )
        finally:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def _save_json(
        self,
        content: Any,
        filename: str,
        prefix: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save JSON content to a file."""
        import json
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        try:
            with open(temp_path, 'w') as f:
                json.dump(content, f, indent=2)
            return self._save_file(
                file_path=temp_path,
                prefix=prefix,
                description=description,
                tags=tags,
                metadata=metadata
            )
        finally:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def _save_torch(
        self,
        model: Any,
        filename: str,
        prefix: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save PyTorch model to a file."""
        import tempfile
        import os
        import torch

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        try:
            torch.save(model, temp_path)
            return self._save_file(
                file_path=temp_path,
                prefix=prefix,
                description=description,
                tags=tags,
                metadata=metadata
            )
        finally:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def _save_fig(
        self,
        fig: Any,
        filename: str,
        prefix: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Save matplotlib figure to a file."""
        import tempfile
        import os
        import matplotlib.pyplot as plt

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        try:
            fig.savefig(temp_path, **kwargs)
            plt.close(fig)
            return self._save_file(
                file_path=temp_path,
                prefix=prefix,
                description=description,
                tags=tags,
                metadata=metadata
            )
        finally:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def _save_pickle(
        self,
        content: Any,
        filename: str,
        prefix: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save Python object to a pickle file."""
        import pickle
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(content, f)
            return self._save_file(
                file_path=temp_path,
                prefix=prefix,
                description=description,
                tags=tags,
                metadata=metadata
            )
        finally:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def list(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files with optional glob pattern filtering.

        Args:
            pattern: Optional glob pattern to filter files (e.g., "*.png", "model_*.pt")

        Returns:
            List of file metadata dicts

        Raises:
            RuntimeError: If experiment is not open

        Examples:
            files = experiment.files().list()  # All files
            files = experiment.files("/models").list()  # Files in /models prefix
            files = experiment.files("images").list("*.png")  # PNG files in images
            files = experiment.files().list("**/*.pt")  # All .pt files
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        # Determine prefix filter - support both new (path) and old (prefix) API
        prefix = None
        if self._path:
            prefix = '/' + self._path.lstrip('/')
        elif self._prefix and self._prefix != '/':
            prefix = self._prefix

        # Get all files matching prefix
        files = self._experiment._list_files(
            prefix=prefix,
            tags=self._tags if self._tags else None
        )

        # Apply glob pattern if provided
        if pattern:
            pattern = pattern.lstrip('/')
            filtered = []
            for f in files:
                filename = f.get('filename', '')
                full_path = f.get('path', '/').rstrip('/') + '/' + filename
                full_path = full_path.lstrip('/')

                # Match against filename or full path
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(full_path, pattern):
                    filtered.append(f)
            return filtered

        return files

    def download(
        self,
        pattern: Optional[str] = None,
        *,
        to: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Download file(s) with automatic checksum verification.

        Args:
            pattern: Optional glob pattern for batch download (e.g., "*.png")
            to: Destination path. For single files, this is the file path.
                For patterns, this is the destination directory.

        Returns:
            For single file: Path to downloaded file
            For pattern: List of paths to downloaded files

        Raises:
            RuntimeError: If experiment is not open
            ValueError: If file not found or checksum verification fails

        Examples:
            # Download single file
            path = experiment.files("model.pt").download()
            path = experiment.files("model.pt").download(to="./local_model.pt")

            # Download by file ID (backwards compatibility)
            path = experiment.files(file_id="123").download()

            # Download multiple files matching pattern
            paths = experiment.files("images").download("*.png")
            paths = experiment.files("images").download("*.png", to="./local_images")
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        # If file_id is set (backwards compatibility)
        if self._file_id:
            return self._experiment._download_file(
                file_id=self._file_id,
                dest_path=to or self._dest_path
            )

        # If pattern is provided, download multiple files
        if pattern:
            files = self.list(pattern)
            if not files:
                raise ValueError(f"No files found matching pattern: {pattern}")

            downloaded = []
            dest_dir = Path(to) if to else Path('.')
            if to and not dest_dir.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                file_id = f.get('id')
                filename = f.get('filename', 'file')
                dest_path = str(dest_dir / filename)
                path = self._experiment._download_file(
                    file_id=file_id,
                    dest_path=dest_path
                )
                downloaded.append(path)

            return downloaded

        # Download single file by path
        if self._path:
            # Find file by path
            files = self._experiment._list_files(prefix=None, tags=None)
            matching = []
            search_path = self._path.lstrip('/')

            for f in files:
                filename = f.get('filename', '')
                prefix = f.get('path', '/').lstrip('/')
                full_path = prefix.rstrip('/') + '/' + filename if prefix else filename
                full_path = full_path.lstrip('/')

                if full_path == search_path or filename == search_path:
                    matching.append(f)

            if not matching:
                raise ValueError(f"File not found: {self._path}")

            if len(matching) > 1:
                # If multiple matches, prefer exact path match
                exact = [f for f in matching if
                         (f.get('path', '/').lstrip('/').rstrip('/') + '/' + f.get('filename', '')).lstrip('/') == search_path]
                if exact:
                    matching = exact[:1]
                else:
                    matching = matching[:1]

            file_info = matching[0]
            return self._experiment._download_file(
                file_id=file_info['id'],
                dest_path=to
            )

        raise ValueError("No file path or pattern specified")

    def delete(self, pattern: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Delete file(s) (soft delete).

        Args:
            pattern: Optional glob pattern for batch delete (e.g., "*.png")

        Returns:
            For single file: Dict with id and deletedAt timestamp
            For pattern: List of deletion results

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If file not found

        Examples:
            # Delete single file
            result = experiment.files("some.text").delete()

            # Delete by file ID (backwards compatibility)
            result = experiment.files(file_id="123").delete()

            # Delete multiple files matching pattern
            results = experiment.files("images").delete("*.png")
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # If file_id is set (backwards compatibility)
        if self._file_id:
            return self._experiment._delete_file(file_id=self._file_id)

        # If pattern is provided, delete multiple files
        if pattern:
            files = self.list(pattern)
            if not files:
                raise ValueError(f"No files found matching pattern: {pattern}")

            results = []
            for f in files:
                file_id = f.get('id')
                result = self._experiment._delete_file(file_id=file_id)
                results.append(result)
            return results

        # Delete single file by path
        if self._path:
            files = self._experiment._list_files(prefix=None, tags=None)
            matching = []
            search_path = self._path.lstrip('/')

            for f in files:
                filename = f.get('filename', '')
                prefix = f.get('path', '/').lstrip('/')
                full_path = prefix.rstrip('/') + '/' + filename if prefix else filename
                full_path = full_path.lstrip('/')

                if full_path == search_path or filename == search_path:
                    matching.append(f)

            if not matching:
                raise ValueError(f"File not found: {self._path}")

            # Delete all matching files
            if len(matching) == 1:
                return self._experiment._delete_file(file_id=matching[0]['id'])

            results = []
            for f in matching:
                result = self._experiment._delete_file(file_id=f['id'])
                results.append(result)
            return results

        raise ValueError("No file path or pattern specified")

    def update(self) -> Dict[str, Any]:
        """
        Update file metadata (description, tags, metadata).

        Returns:
            Updated file metadata dict

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If file_id not provided

        Examples:
            result = experiment.files(
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

    # Convenience methods for specific file types

    def save_json(
        self,
        content: Any,
        file_name: Optional[str] = None,
        *,
        to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save JSON content to a file.

        Args:
            content: Content to save as JSON (dict, list, or any JSON-serializable object)
            file_name: Name of the file to create (deprecated, use 'to')
            to: Target filename (preferred)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            config = {"model": "resnet50", "lr": 0.001}
            result = experiment.files("configs").save_json(config, to="config.json")
        """
        filename = to or file_name
        if not filename:
            raise ValueError("'to' parameter is required")

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        return self._save_json(
            content=content,
            filename=filename,
            prefix=prefix,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )

    def save_text(self, content: str, *, to: str) -> Dict[str, Any]:
        """
        Save text content to a file.

        Args:
            content: Text content to save
            to: Target filename

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            result = experiment.files().save_text("Hello, world!", to="greeting.txt")
            result = experiment.files("configs").save_text(yaml_content, to="view.yaml")
        """
        if not to:
            raise ValueError("'to' parameter is required")

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        return self._save_bytes(
            data=content.encode('utf-8'),
            filename=to,
            prefix=prefix,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )

    def save_blob(self, data: bytes, *, to: str) -> Dict[str, Any]:
        """
        Save binary data to a file.

        Args:
            data: Binary data to save
            to: Target filename

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            result = experiment.files("data").save_blob(binary_data, to="model.bin")
        """
        if not to:
            raise ValueError("'to' parameter is required")

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        return self._save_bytes(
            data=data,
            filename=to,
            prefix=prefix,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )

    def save_torch(
        self,
        model: Any,
        file_name: Optional[str] = None,
        *,
        to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save PyTorch model to a file.

        Args:
            model: PyTorch model or state dict to save
            file_name: Name of the file to create (deprecated, use 'to')
            to: Target filename (preferred)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            result = experiment.files("models").save_torch(model, to="model.pt")
            result = experiment.files("models").save_torch(model.state_dict(), to="weights.pth")
        """
        filename = to or file_name
        if not filename:
            raise ValueError("'to' parameter is required")

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        return self._save_torch(
            model=model,
            filename=filename,
            prefix=prefix,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )

    def save_pkl(
        self,
        content: Any,
        file_name: Optional[str] = None,
        *,
        to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save Python object to a pickle file.

        Args:
            content: Python object to pickle (must be pickle-serializable)
            file_name: Name of the file to create (deprecated, use 'to')
            to: Target filename (preferred)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            data = {"model": "resnet50", "weights": np.array([1, 2, 3])}
            result = experiment.files("data").save_pkl(data, to="data.pkl")
        """
        filename = to or file_name
        if not filename:
            raise ValueError("'to' parameter is required")

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        return self._save_pickle(
            content=content,
            filename=filename,
            prefix=prefix,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata
        )

    def save_fig(
        self,
        fig: Optional[Any] = None,
        file_name: Optional[str] = None,
        *,
        to: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Save matplotlib figure to a file.

        Args:
            fig: Matplotlib figure object. If None, uses plt.gcf() (current figure)
            file_name: Name of file to create (deprecated, use 'to')
            to: Target filename (preferred)
            **kwargs: Additional arguments passed to fig.savefig()

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            plt.plot([1, 2, 3], [1, 4, 9])
            result = experiment.files("plots").save_fig(to="plot.png")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")

        filename = to or file_name
        if not filename:
            raise ValueError("'to' parameter is required")

        if fig is None:
            fig = plt.gcf()

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        return self._save_fig(
            fig=fig,
            filename=filename,
            prefix=prefix,
            description=self._description,
            tags=self._tags,
            metadata=self._metadata,
            **kwargs
        )

    def save_video(
        self,
        frame_stack: Union[List, Any],
        file_name: Optional[str] = None,
        *,
        to: Optional[str] = None,
        fps: int = 20,
        **imageio_kwargs
    ) -> Dict[str, Any]:
        """
        Save video frame stack to a file.

        Args:
            frame_stack: List of numpy arrays or stacked array
            file_name: Name of file to create (deprecated, use 'to')
            to: Target filename (preferred)
            fps: Frames per second (default: 20)
            **imageio_kwargs: Additional arguments passed to imageio

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Examples:
            frames = [np.random.rand(480, 640) for _ in range(30)]
            result = experiment.files("videos").save_video(frames, to="output.mp4")
        """
        import tempfile
        import os

        try:
            import imageio.v3 as iio
        except ImportError:
            raise ImportError("imageio is not installed. Install it with: pip install imageio imageio-ffmpeg")

        try:
            from skimage import img_as_ubyte
        except ImportError:
            raise ImportError("scikit-image is not installed. Install it with: pip install scikit-image")

        filename = to or file_name
        if not filename:
            raise ValueError("'to' parameter is required")

        # Validate frame_stack
        try:
            if len(frame_stack) == 0:
                raise ValueError("frame_stack is empty")
        except TypeError:
            raise ValueError("frame_stack must be a list or numpy array")

        prefix = '/' + self._path.lstrip('/') if self._path else self._prefix

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)

        try:
            frames_ubyte = img_as_ubyte(frame_stack)
            try:
                iio.imwrite(temp_path, frames_ubyte, fps=fps, **imageio_kwargs)
            except iio.core.NeedDownloadError:
                import imageio.plugins.ffmpeg
                imageio.plugins.ffmpeg.download()
                iio.imwrite(temp_path, frames_ubyte, fps=fps, **imageio_kwargs)

            return self._save_file(
                file_path=temp_path,
                prefix=prefix,
                description=self._description,
                tags=self._tags,
                metadata=self._metadata
            )
        finally:
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def duplicate(self, source: Union[str, Dict[str, Any]], to: str) -> Dict[str, Any]:
        """
        Duplicate an existing file to a new path within the same experiment.

        Args:
            source: Source file - either file ID (str) or metadata dict with 'id' key
            to: Target path like "models/latest.pt" or "/checkpoints/best.pt"

        Returns:
            File metadata dict for the duplicated file

        Examples:
            snapshot = dxp.files("models").save_torch(model, to=f"model_{epoch:05d}.pt")
            dxp.files().duplicate(snapshot, to="models/latest.pt")
        """
        import tempfile
        import os

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Extract source file ID
        if isinstance(source, str):
            source_id = source
        elif isinstance(source, dict) and 'id' in source:
            source_id = source['id']
        else:
            raise ValueError("source must be a file ID (str) or metadata dict with 'id' key")

        if not source_id:
            raise ValueError("Invalid source: file ID is empty")

        # Parse target path into prefix and filename
        to = to.lstrip('/')
        if '/' in to:
            target_prefix, target_filename = to.rsplit('/', 1)
            target_prefix = '/' + target_prefix
        else:
            target_prefix = '/'
            target_filename = to

        if not target_filename:
            raise ValueError(f"Invalid target path '{to}': must include filename")

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, target_filename)

        try:
            downloaded_path = self._experiment._download_file(
                file_id=source_id,
                dest_path=temp_path
            )

            return self._save_file(
                file_path=downloaded_path,
                prefix=target_prefix,
                description=self._description,
                tags=self._tags,
                metadata=self._metadata
            )
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass


class FilesAccessor:
    """
    Accessor that enables both callable and attribute-style access to file operations.

    This allows:
        experiment.files("path")          # Returns FileBuilder
        experiment.files.save(...)        # Direct method call
        experiment.files.download(...)    # Direct method call
    """

    def __init__(self, experiment: 'Experiment'):
        self._experiment = experiment
        self._builder = FileBuilder(experiment)

    def __call__(self, path: Optional[str] = None, **kwargs) -> FileBuilder:
        """Create a FileBuilder with the given path."""
        return FileBuilder(self._experiment, path=path, **kwargs)

    # Direct methods that don't require a path first

    def save(
        self,
        obj: Optional[Any] = None,
        *,
        to: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Save a file directly.

        Examples:
            experiment.files.save("./model.pt")
            experiment.files.save(model, to="checkpoints/model.pt")
        """
        # Parse 'to' to extract prefix and filename
        if to:
            to_path = to.lstrip('/')
            if '/' in to_path:
                prefix, filename = to_path.rsplit('/', 1)
                prefix = '/' + prefix
            else:
                prefix = '/'
                filename = to_path
            return FileBuilder(self._experiment, path=prefix, **kwargs).save(obj, to=filename)

        if isinstance(obj, str) and Path(obj).exists():
            # obj is a file path, extract prefix from it
            return FileBuilder(self._experiment, **kwargs).save(obj)

        raise ValueError("'to' parameter is required when not saving an existing file path")

    def download(
        self,
        path: str,
        *,
        to: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Download file(s) directly.

        Examples:
            experiment.files.download("model.pt")
            experiment.files.download("images/*.png", to="local_images")
        """
        path = path.lstrip('/')

        # Check if path contains glob pattern
        if '*' in path or '?' in path:
            # Extract prefix and pattern
            if '/' in path:
                parts = path.split('/')
                # Find where the pattern starts
                prefix_parts = []
                pattern_parts = []
                in_pattern = False
                for part in parts:
                    if '*' in part or '?' in part:
                        in_pattern = True
                    if in_pattern:
                        pattern_parts.append(part)
                    else:
                        prefix_parts.append(part)

                prefix = '/'.join(prefix_parts) if prefix_parts else None
                pattern = '/'.join(pattern_parts)
            else:
                prefix = None
                pattern = path

            return FileBuilder(self._experiment, path=prefix).download(pattern, to=to)

        # Single file download
        return FileBuilder(self._experiment, path=path).download(to=to)

    def delete(self, path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Delete file(s) directly.

        Examples:
            experiment.files.delete("some.text")
            experiment.files.delete("images/*.png")
        """
        path = path.lstrip('/')

        # Check if path contains glob pattern
        if '*' in path or '?' in path:
            # Extract prefix and pattern
            if '/' in path:
                parts = path.split('/')
                prefix_parts = []
                pattern_parts = []
                in_pattern = False
                for part in parts:
                    if '*' in part or '?' in part:
                        in_pattern = True
                    if in_pattern:
                        pattern_parts.append(part)
                    else:
                        prefix_parts.append(part)

                prefix = '/'.join(prefix_parts) if prefix_parts else None
                pattern = '/'.join(pattern_parts)
            else:
                prefix = None
                pattern = path

            return FileBuilder(self._experiment, path=prefix).delete(pattern)

        # Single file delete
        return FileBuilder(self._experiment, path=path).delete()

    def list(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files directly.

        Examples:
            files = experiment.files.list()
            files = experiment.files.list("*.pt")
        """
        return FileBuilder(self._experiment).list(pattern)

    def save_text(self, content: str, *, to: str) -> Dict[str, Any]:
        """
        Save text content to a file.

        Examples:
            experiment.files.save_text("content", to="view.yaml")
        """
        to_path = to.lstrip('/')
        if '/' in to_path:
            prefix, filename = to_path.rsplit('/', 1)
            prefix = '/' + prefix
        else:
            prefix = '/'
            filename = to_path
        return FileBuilder(self._experiment, path=prefix).save_text(content, to=filename)

    def save_json(self, content: Any, *, to: str) -> Dict[str, Any]:
        """
        Save JSON content to a file.

        Examples:
            experiment.files.save_json({"key": "value"}, to="config.json")
        """
        to_path = to.lstrip('/')
        if '/' in to_path:
            prefix, filename = to_path.rsplit('/', 1)
            prefix = '/' + prefix
        else:
            prefix = '/'
            filename = to_path
        return FileBuilder(self._experiment, path=prefix).save_json(content, to=filename)

    def save_blob(self, data: bytes, *, to: str) -> Dict[str, Any]:
        """
        Save binary data to a file.

        Examples:
            experiment.files.save_blob(b"data", to="data.bin")
        """
        to_path = to.lstrip('/')
        if '/' in to_path:
            prefix, filename = to_path.rsplit('/', 1)
            prefix = '/' + prefix
        else:
            prefix = '/'
            filename = to_path
        return FileBuilder(self._experiment, path=prefix).save_blob(data, to=filename)


class BindrsBuilder:
    """
    Fluent interface for bindr (collection) operations.

    Usage:
        file_paths = experiment.bindrs("some-bindr").list()
    """

    def __init__(self, experiment: 'Experiment', bindr_name: str):
        self._experiment = experiment
        self._bindr_name = bindr_name

    def list(self) -> List[Dict[str, Any]]:
        """
        List files in this bindr.

        Returns:
            List of file metadata dicts belonging to this bindr
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        # Get all files and filter by bindr
        all_files = self._experiment._list_files(prefix=None, tags=None)
        return [f for f in all_files if self._bindr_name in f.get('bindrs', [])]


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
