"""
Files module for ML-Dash SDK.

Provides fluent API for file upload, download, list, and delete operations.
"""

import hashlib
import mimetypes
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .experiment import Experiment


class FileBuilder:
    """
    Fluent interface for file operations.

    Usage:
        # Upload file
        experiment.files(file_path="./model.pt", prefix="/models").save()

        # List files
        files = experiment.files().list()
        files = experiment.files(prefix="/models").list()

        # Download file
        experiment.files(file_id="123").download()
        experiment.files(file_id="123", dest_path="./model.pt").download()

        # Delete file
        experiment.files(file_id="123").delete()
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
            ValueError: If file size exceeds 100GB limit

        Examples:
            result = experiment.files(file_path="./model.pt", prefix="/models").save()
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

        # Check file size (max 100GB)
        file_size = file_path.stat().st_size
        MAX_FILE_SIZE = 100 * 1024 * 1024 * 1024  # 100GB in bytes
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size ({file_size} bytes) exceeds 100GB limit")

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
            files = experiment.files().list()  # All files
            files = experiment.files(prefix="/models").list()  # Filter by prefix
            files = experiment.files(tags=["checkpoint"]).list()  # Filter by tags
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
            path = experiment.files(file_id="123").download()

            # Download to custom path
            path = experiment.files(file_id="123", dest_path="./model.pt").download()
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
            result = experiment.files(file_id="123").delete()
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
            result = experiment.files(prefix="/configs").save_json(config, "config.json")
        """
        import json
        import tempfile
        import os

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Create temporary file with desired filename
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)
        try:
            # Write JSON content to temp file
            with open(temp_path, 'w') as f:
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
            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
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
            result = experiment.files(prefix="/models").save_torch(model, "model.pt")

            # Or save state dict
            result = experiment.files(prefix="/models").save_torch(model.state_dict(), "model.pth")
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

        # Create temporary file with desired filename
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)

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
            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def save_pkl(self, content: Any, file_name: str) -> Dict[str, Any]:
        """
        Save Python object to a pickle file.

        Args:
            content: Python object to pickle (must be pickle-serializable)
            file_name: Name of the file to create (should end with .pkl or .pickle)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment is not open or write-protected
            ValueError: If content cannot be pickled

        Examples:
            data = {"model": "resnet50", "weights": np.array([1, 2, 3])}
            result = experiment.files(prefix="/data").save_pkl(data, "data.pkl")

            # Or save any Python object
            result = experiment.files(prefix="/models").save_pkl(trained_model, "model.pickle")
        """
        import pickle
        import tempfile
        import os

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Create temporary file with desired filename
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)
        try:
            # Write pickled content to temp file
            with open(temp_path, 'wb') as f:
                pickle.dump(content, f)

            # Save using existing save() method
            original_file_path = self._file_path
            self._file_path = temp_path

            # Upload and get result
            result = self.save()

            # Restore original file_path
            self._file_path = original_file_path

            return result
        finally:
            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def save_fig(self, fig: Optional[Any] = None, file_name: str = "figure.png", **kwargs) -> Dict[str, Any]:
        """
        Save matplotlib figure to a file.

        Args:
            fig: Matplotlib figure object. If None, uses plt.gcf() (current figure)
            file_name: Name of file to create (extension determines format: .png, .pdf, .svg, .jpg)
            **kwargs: Additional arguments passed to fig.savefig():
                - dpi: Resolution (int or 'figure')
                - transparent: Make background transparent (bool)
                - bbox_inches: 'tight' to auto-crop (str or Bbox)
                - quality: JPEG quality 0-100 (int)
                - format: Override format detection (str)

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment not open or write-protected
            ImportError: If matplotlib not installed

        Examples:
            import matplotlib.pyplot as plt

            # Use current figure
            plt.plot([1, 2, 3], [1, 4, 9])
            result = experiment.files(prefix="/plots").save_fig(file_name="plot.png")

            # Specify figure explicitly
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            result = experiment.files(prefix="/plots").save_fig(fig=fig, file_name="plot.pdf", dpi=150)
        """
        import tempfile
        import os

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Get figure
        if fig is None:
            fig = plt.gcf()

        # Create temporary file with desired filename
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)

        try:
            # Save figure to temp file
            fig.savefig(temp_path, **kwargs)

            # Close figure to prevent memory leaks
            plt.close(fig)

            # Save using existing save() method
            original_file_path = self._file_path
            self._file_path = temp_path

            # Upload and get result
            result = self.save()

            # Restore original file_path
            self._file_path = original_file_path

            return result
        finally:
            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def save_video(
        self,
        frame_stack: Union[List, Any],
        file_name: str,
        fps: int = 20,
        **imageio_kwargs
    ) -> Dict[str, Any]:
        """
        Save video frame stack to a file.

        Args:
            frame_stack: List of numpy arrays or stacked array (shape: [N, H, W] or [N, H, W, C])
            file_name: Name of file to create (extension determines format: .mp4, .gif, .avi, .webm)
            fps: Frames per second (default: 20)
            **imageio_kwargs: Additional arguments passed to imageio.v3.imwrite():
                - codec: Video codec (e.g., 'libx264', 'h264')
                - quality: Quality level (int, higher is better)
                - pixelformat: Pixel format (e.g., 'yuv420p')
                - macro_block_size: Macro block size for encoding

        Returns:
            File metadata dict with id, path, filename, checksum, etc.

        Raises:
            RuntimeError: If experiment not open or write-protected
            ImportError: If imageio or scikit-image not installed
            ValueError: If frame_stack is empty or invalid format

        Examples:
            import numpy as np

            # Grayscale frames (float values 0-1)
            frames = [np.random.rand(480, 640) for _ in range(30)]
            result = experiment.files(prefix="/videos").save_video(frames, "output.mp4")

            # RGB frames with custom FPS
            frames = [np.random.rand(480, 640, 3) for _ in range(60)]
            result = experiment.files(prefix="/videos").save_video(frames, "output.mp4", fps=30)

            # Save as GIF
            frames = [np.random.rand(200, 200) for _ in range(20)]
            result = experiment.files(prefix="/videos").save_video(frames, "animation.gif")

            # With custom codec and quality
            result = experiment.files(prefix="/videos").save_video(
                frames, "output.mp4", fps=30, codec='libx264', quality=8
            )
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

        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Validate frame_stack
        try:
            # Handle both list and numpy array
            if len(frame_stack) == 0:
                raise ValueError("frame_stack is empty")
        except TypeError:
            raise ValueError("frame_stack must be a list or numpy array")

        # Create temporary file with desired filename
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)

        try:
            # Convert frames to uint8 format (handles float32/64, grayscale, RGB, etc.)
            # img_as_ubyte automatically scales [0.0, 1.0] floats to [0, 255] uint8
            frames_ubyte = img_as_ubyte(frame_stack)

            # Encode video to temp file
            try:
                iio.imwrite(temp_path, frames_ubyte, fps=fps, **imageio_kwargs)
            except iio.core.NeedDownloadError:
                # Auto-download FFmpeg if not available
                import imageio.plugins.ffmpeg
                imageio.plugins.ffmpeg.download()
                iio.imwrite(temp_path, frames_ubyte, fps=fps, **imageio_kwargs)

            # Save using existing save() method
            original_file_path = self._file_path
            self._file_path = temp_path

            # Upload and get result
            result = self.save()

            # Restore original file_path
            self._file_path = original_file_path

            return result
        finally:
            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
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
