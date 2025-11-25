"""
Experiment class for ML-Dash SDK.

Supports three usage styles:
1. Decorator: @ml_dash_experiment(...)
2. Context manager: with Experiment(...) as exp:
3. Direct instantiation: exp = Experiment(...)
"""

from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import functools
from pathlib import Path
from datetime import datetime

from .client import RemoteClient
from .storage import LocalStorage
from .log import LogLevel, LogBuilder
from .params import ParametersBuilder
from .files import FileBuilder


class OperationMode(Enum):
    """Operation mode for the experiment."""
    LOCAL = "local"
    REMOTE = "remote"
    HYBRID = "hybrid"  # Future: sync local to remote


class RunManager:
    """
    Lifecycle manager for experiments.

    Supports three usage patterns:
    1. Method calls: experiment.run.start(), experiment.run.complete()
    2. Context manager: with Experiment(...).run as exp:
    3. Decorator: @exp.run or @Experiment(...).run
    """

    def __init__(self, experiment: "Experiment"):
        """
        Initialize RunManager.

        Args:
            experiment: Parent Experiment instance
        """
        self._experiment = experiment

    def start(self) -> "Experiment":
        """
        Start the experiment (sets status to RUNNING).

        Returns:
            The experiment instance for chaining
        """
        return self._experiment._open()

    def complete(self) -> None:
        """Mark experiment as completed (status: COMPLETED)."""
        self._experiment._close(status="COMPLETED")

    def fail(self) -> None:
        """Mark experiment as failed (status: FAILED)."""
        self._experiment._close(status="FAILED")

    def cancel(self) -> None:
        """Mark experiment as cancelled (status: CANCELLED)."""
        self._experiment._close(status="CANCELLED")

    def __enter__(self) -> "Experiment":
        """Context manager entry - starts the experiment."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - completes or fails the experiment."""
        if exc_type is not None:
            self.fail()
        else:
            self.complete()
        return False

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator support for wrapping functions with experiment lifecycle.

        Usage:
            @exp.run
            def train(exp):
                exp.log("Training...")
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self as exp:
                return func(exp, *args, **kwargs)
        return wrapper


class Experiment:
    """
    ML-Dash experiment for metricing experiments.

    Usage examples:

    # Remote mode
    experiment = Experiment(
        name="my-experiment",
        project="my-project",
        remote="http://localhost:3000",
        api_key="your-jwt-token"
    )

    # Local mode
    experiment = Experiment(
        name="my-experiment",
        project="my-project",
        local_path=".ml-dash"
    )

    # Context manager
    with Experiment(...) as exp:
        exp.log(...)

    # Decorator
    @ml_dash_experiment(name="exp", project="ws", remote="...")
    def train():
        ...
    """

    def __init__(
        self,
        name: str,
        project: str,
        *,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bindrs: Optional[List[str]] = None,
        folder: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Mode configuration
        remote: Optional[str] = None,
        api_key: Optional[str] = None,
        user_name: Optional[str] = None,
        local_path: Optional[str] = None,
        # Internal parameters
        _write_protected: bool = False,
    ):
        """
        Initialize an ML-Dash experiment.

        Args:
            name: Experiment name (unique within project)
            project: Project name
            description: Optional experiment description
            tags: Optional list of tags
            bindrs: Optional list of bindrs
            folder: Optional folder path (e.g., "/experiments/baseline")
            metadata: Optional metadata dict
            remote: Remote API URL (e.g., "http://localhost:3000")
            api_key: JWT token for authentication (if not provided, will be generated from user_name)
            user_name: Username for authentication (generates API key if api_key not provided)
            local_path: Local storage root path (for local mode)
            _write_protected: Internal parameter - if True, experiment becomes immutable after creation
        """
        self.name = name
        self.project = project
        self.description = description
        self.tags = tags
        self.bindrs = bindrs
        self.folder = folder
        self._write_protected = _write_protected
        self.metadata = metadata

        # Generate API key from username if not provided
        if remote and not api_key and user_name:
            api_key = self._generate_api_key_from_username(user_name)

        # Determine operation mode
        if remote and local_path:
            self.mode = OperationMode.HYBRID
        elif remote:
            self.mode = OperationMode.REMOTE
        elif local_path:
            self.mode = OperationMode.LOCAL
        else:
            raise ValueError(
                "Must specify either 'remote' (with api_key/user_name) or 'local_path'"
            )

        # Initialize backend
        self._client: Optional[RemoteClient] = None
        self._storage: Optional[LocalStorage] = None
        self._experiment_id: Optional[str] = None
        self._experiment_data: Optional[Dict[str, Any]] = None
        self._is_open = False

        if self.mode in (OperationMode.REMOTE, OperationMode.HYBRID):
            if not api_key:
                raise ValueError("Either api_key or user_name is required for remote mode")
            self._client = RemoteClient(base_url=remote, api_key=api_key)

        if self.mode in (OperationMode.LOCAL, OperationMode.HYBRID):
            if not local_path:
                raise ValueError("local_path is required for local mode")
            self._storage = LocalStorage(root_path=Path(local_path))

    @staticmethod
    def _generate_api_key_from_username(user_name: str) -> str:
        """
        Generate a deterministic API key (JWT) from username.

        This is a temporary solution until proper user authentication is implemented.
        Generates a unique user ID from the username and creates a JWT token.

        Args:
            user_name: Username to generate API key from

        Returns:
            JWT token string
        """
        import hashlib
        import time
        import jwt

        # Generate deterministic user ID from username (first 10 digits of SHA256 hash)
        user_id = str(int(hashlib.sha256(user_name.encode()).hexdigest()[:16], 16))[:10]

        # JWT payload
        payload = {
            "userId": user_id,
            "userName": user_name,
            "iat": int(time.time()),
            "exp": int(time.time()) + (30 * 24 * 60 * 60)  # 30 days expiration
        }

        # Secret key for signing (should match server's JWT_SECRET)
        secret = "your-secret-key-change-this-in-production"

        # Generate JWT
        token = jwt.encode(payload, secret, algorithm="HS256")

        return token

    def _open(self) -> "Experiment":
        """
        Internal method to open the experiment (create or update on server/filesystem).

        Returns:
            self for chaining
        """
        if self._is_open:
            return self

        if self._client:
            # Remote mode: create/update experiment via API
            response = self._client.create_or_update_experiment(
                project=self.project,
                name=self.name,
                description=self.description,
                tags=self.tags,
                bindrs=self.bindrs,
                folder=self.folder,
                write_protected=self._write_protected,
                metadata=self.metadata,
            )
            self._experiment_data = response
            self._experiment_id = response["experiment"]["id"]

        if self._storage:
            # Local mode: create experiment directory structure
            self._storage.create_experiment(
                project=self.project,
                name=self.name,
                description=self.description,
                tags=self.tags,
                bindrs=self.bindrs,
                folder=self.folder,
                metadata=self.metadata,
            )

        self._is_open = True
        return self

    def _close(self, status: str = "COMPLETED"):
        """
        Internal method to close the experiment and update status.

        Args:
            status: Status to set - "COMPLETED" (default), "FAILED", or "CANCELLED"
        """
        if not self._is_open:
            return

        # Flush any pending writes
        if self._storage:
            self._storage.flush()

        # Update experiment status in remote mode
        if self._client and self._experiment_id:
            try:
                self._client.update_experiment_status(
                    experiment_id=self._experiment_id,
                    status=status
                )
            except Exception as e:
                # Log error but don't fail the close operation
                print(f"Warning: Failed to update experiment status: {e}")

        self._is_open = False

    @property
    def run(self) -> RunManager:
        """
        Get the RunManager for lifecycle operations.

        Usage:
            # Method calls
            experiment.run.start()
            experiment.run.complete()

            # Context manager
            with Experiment(...).run as exp:
                exp.log("Training...")

            # Decorator
            @experiment.run
            def train(exp):
                exp.log("Training...")

        Returns:
            RunManager instance
        """
        return RunManager(self)

    @property
    def params(self) -> ParametersBuilder:
        """
        Get a ParametersBuilder for parameter operations.

        Usage:
            # Set parameters
            experiment.params.set(lr=0.001, batch_size=32)

            # Get parameters
            params = experiment.params.get()

        Returns:
            ParametersBuilder instance

        Raises:
            RuntimeError: If experiment is not open
        """
        if not self._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        return ParametersBuilder(self)

    def log(
        self,
        message: Optional[str] = None,
        level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra_metadata
    ) -> Optional[LogBuilder]:
        """
        Create a log entry or return a LogBuilder for fluent API.

        This method supports two styles:

        1. Fluent style (no message provided):
           Returns a LogBuilder that allows chaining with level methods.

           Examples:
               experiment.log(metadata={"epoch": 1}).info("Training started")
               experiment.log().error("Failed", error_code=500)

        2. Traditional style (message provided):
           Writes the log immediately and returns None.

           Examples:
               experiment.log("Training started", level="info", epoch=1)
               experiment.log("Training started")  # Defaults to "info"

        Args:
            message: Optional log message (for traditional style)
            level: Optional log level (for traditional style, defaults to "info")
            metadata: Optional metadata dict
            **extra_metadata: Additional metadata as keyword arguments

        Returns:
            LogBuilder if no message provided (fluent mode)
            None if log was written directly (traditional mode)

        Raises:
            RuntimeError: If experiment is not open
            ValueError: If log level is invalid
        """
        if not self._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        # Fluent mode: return LogBuilder
        if message is None:
            combined_metadata = {**(metadata or {}), **extra_metadata}
            return LogBuilder(self, combined_metadata if combined_metadata else None)

        # Traditional mode: write immediately
        level = level or LogLevel.INFO.value  # Default to "info"
        level = LogLevel.validate(level)  # Validate level

        combined_metadata = {**(metadata or {}), **extra_metadata}
        self._write_log(
            message=message,
            level=level,
            metadata=combined_metadata if combined_metadata else None,
            timestamp=None
        )
        return None

    def _write_log(
        self,
        message: str,
        level: str,
        metadata: Optional[Dict[str, Any]],
        timestamp: Optional[datetime]
    ) -> None:
        """
        Internal method to write a log entry immediately.
        No buffering - writes directly to storage/remote.

        Args:
            message: Log message
            level: Log level (already validated)
            metadata: Optional metadata dict
            timestamp: Optional custom timestamp (defaults to now)
        """
        log_entry = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
            "level": level,
            "message": message,
        }

        if metadata:
            log_entry["metadata"] = metadata

        # Write immediately (no buffering)
        if self._client:
            # Remote mode: send to API (wrapped in array for batch API)
            self._client.create_log_entries(
                experiment_id=self._experiment_id,
                logs=[log_entry]  # Single log in array
            )

        if self._storage:
            # Local mode: write to file immediately
            self._storage.write_log(
                project=self.project,
                experiment=self.name,
                message=log_entry["message"],
                level=log_entry["level"],
                metadata=log_entry.get("metadata"),
                timestamp=log_entry["timestamp"]
            )

    def files(self, **kwargs) -> FileBuilder:
        """
        Get a FileBuilder for fluent file operations.

        Returns:
            FileBuilder instance for chaining

        Raises:
            RuntimeError: If experiment is not open

        Examples:
            # Upload file
            experiment.files(file_path="./model.pt", prefix="/models").save()

            # List files
            files = experiment.files().list()
            files = experiment.files(prefix="/models").list()

            # Download file
            experiment.files(file_id="123").download()

            # Delete file
            experiment.files(file_id="123").delete()
        """
        if not self._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        return FileBuilder(self, **kwargs)

    def _upload_file(
        self,
        file_path: str,
        prefix: str,
        filename: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        checksum: str,
        content_type: str,
        size_bytes: int
    ) -> Dict[str, Any]:
        """
        Internal method to upload a file.

        Args:
            file_path: Local file path
            prefix: Logical path prefix
            filename: Original filename
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            checksum: SHA256 checksum
            content_type: MIME type
            size_bytes: File size in bytes

        Returns:
            File metadata dict
        """
        result = None

        if self._client:
            # Remote mode: upload to API
            result = self._client.upload_file(
                experiment_id=self._experiment_id,
                file_path=file_path,
                prefix=prefix,
                filename=filename,
                description=description,
                tags=tags,
                metadata=metadata,
                checksum=checksum,
                content_type=content_type,
                size_bytes=size_bytes
            )

        if self._storage:
            # Local mode: copy to local storage
            result = self._storage.write_file(
                project=self.project,
                experiment=self.name,
                file_path=file_path,
                prefix=prefix,
                filename=filename,
                description=description,
                tags=tags,
                metadata=metadata,
                checksum=checksum,
                content_type=content_type,
                size_bytes=size_bytes
            )

        return result

    def _list_files(
        self,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Internal method to list files.

        Args:
            prefix: Optional prefix filter
            tags: Optional tags filter

        Returns:
            List of file metadata dicts
        """
        files = []

        if self._client:
            # Remote mode: fetch from API
            files = self._client.list_files(
                experiment_id=self._experiment_id,
                prefix=prefix,
                tags=tags
            )

        if self._storage:
            # Local mode: read from metadata file
            files = self._storage.list_files(
                project=self.project,
                experiment=self.name,
                prefix=prefix,
                tags=tags
            )

        return files

    def _download_file(
        self,
        file_id: str,
        dest_path: Optional[str] = None
    ) -> str:
        """
        Internal method to download a file.

        Args:
            file_id: File ID
            dest_path: Optional destination path (defaults to original filename)

        Returns:
            Path to downloaded file
        """
        if self._client:
            # Remote mode: download from API
            return self._client.download_file(
                experiment_id=self._experiment_id,
                file_id=file_id,
                dest_path=dest_path
            )

        if self._storage:
            # Local mode: copy from local storage
            return self._storage.read_file(
                project=self.project,
                experiment=self.name,
                file_id=file_id,
                dest_path=dest_path
            )

        raise RuntimeError("No client or storage configured")

    def _delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Internal method to delete a file.

        Args:
            file_id: File ID

        Returns:
            Dict with id and deletedAt
        """
        result = None

        if self._client:
            # Remote mode: delete via API
            result = self._client.delete_file(
                experiment_id=self._experiment_id,
                file_id=file_id
            )

        if self._storage:
            # Local mode: soft delete in metadata
            result = self._storage.delete_file(
                project=self.project,
                experiment=self.name,
                file_id=file_id
            )

        return result

    def _update_file(
        self,
        file_id: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Internal method to update file metadata.

        Args:
            file_id: File ID
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Updated file metadata dict
        """
        result = None

        if self._client:
            # Remote mode: update via API
            result = self._client.update_file(
                experiment_id=self._experiment_id,
                file_id=file_id,
                description=description,
                tags=tags,
                metadata=metadata
            )

        if self._storage:
            # Local mode: update in metadata file
            result = self._storage.update_file_metadata(
                project=self.project,
                experiment=self.name,
                file_id=file_id,
                description=description,
                tags=tags,
                metadata=metadata
            )

        return result


    def _write_params(self, flattened_params: Dict[str, Any]) -> None:
        """
        Internal method to write/merge parameters.

        Args:
            flattened_params: Already-flattened parameter dict with dot notation
        """
        if self._client:
            # Remote mode: send to API
            self._client.set_parameters(
                experiment_id=self._experiment_id,
                data=flattened_params
            )

        if self._storage:
            # Local mode: write to file
            self._storage.write_parameters(
                project=self.project,
                experiment=self.name,
                data=flattened_params
            )

    def _read_params(self) -> Optional[Dict[str, Any]]:
        """
        Internal method to read parameters.

        Returns:
            Flattened parameters dict, or None if no parameters exist
        """
        params = None

        if self._client:
            # Remote mode: fetch from API
            try:
                params = self._client.get_parameters(experiment_id=self._experiment_id)
            except Exception:
                # Parameters don't exist yet
                params = None

        if self._storage:
            # Local mode: read from file
            params = self._storage.read_parameters(
                project=self.project,
                experiment=self.name
            )

        return params

    @property
    def metrics(self) -> 'MetricsManager':
        """
        Get a MetricsManager for metric operations.

        Supports two usage patterns:
        1. Named: experiment.metrics("loss").append(value=0.5, step=1)
        2. Unnamed: experiment.metrics.append(name="loss", value=0.5, step=1)

        Returns:
            MetricsManager instance

        Raises:
            RuntimeError: If experiment is not open

        Examples:
            # Named metric
            experiment.metrics("train_loss").append(value=0.5, step=100)

            # Unnamed (name in append call)
            experiment.metrics.append(name="train_loss", value=0.5, step=100)

            # Append batch
            experiment.metrics("metrics").append_batch([
                {"loss": 0.5, "acc": 0.8, "step": 1},
                {"loss": 0.4, "acc": 0.85, "step": 2}
            ])

            # Read data
            data = experiment.metrics("train_loss").read(start_index=0, limit=100)

            # Get statistics
            stats = experiment.metrics("train_loss").stats()
        """
        from .metric import MetricsManager

        if not self._is_open:
            raise RuntimeError(
                "Cannot use metrics on closed experiment. "
                "Use 'with Experiment(...).run as experiment:' or call experiment.run.start() first."
            )

        return MetricsManager(self)

    def _append_to_metric(
        self,
        name: Optional[str],
        data: Dict[str, Any],
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Internal method to append a single data point to a metric.

        Args:
            name: Metric name (can be None for unnamed metrics)
            data: Data point (flexible schema)
            description: Optional metric description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Dict with metricId, index, bufferedDataPoints, chunkSize
        """
        result = None

        if self._client:
            # Remote mode: append via API
            result = self._client.append_to_metric(
                experiment_id=self._experiment_id,
                metric_name=name,
                data=data,
                description=description,
                tags=tags,
                metadata=metadata
            )

        if self._storage:
            # Local mode: append to local storage
            result = self._storage.append_to_metric(
                project=self.project,
                experiment=self.name,
                metric_name=name,
                data=data,
                description=description,
                tags=tags,
                metadata=metadata
            )

        return result

    def _append_batch_to_metric(
        self,
        name: Optional[str],
        data_points: List[Dict[str, Any]],
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Internal method to append multiple data points to a metric.

        Args:
            name: Metric name (can be None for unnamed metrics)
            data_points: List of data points
            description: Optional metric description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Dict with metricId, startIndex, endIndex, count
        """
        result = None

        if self._client:
            # Remote mode: append batch via API
            result = self._client.append_batch_to_metric(
                experiment_id=self._experiment_id,
                metric_name=name,
                data_points=data_points,
                description=description,
                tags=tags,
                metadata=metadata
            )

        if self._storage:
            # Local mode: append batch to local storage
            result = self._storage.append_batch_to_metric(
                project=self.project,
                experiment=self.name,
                metric_name=name,
                data_points=data_points,
                description=description,
                tags=tags,
                metadata=metadata
            )

        return result

    def _read_metric_data(
        self,
        name: str,
        start_index: int,
        limit: int
    ) -> Dict[str, Any]:
        """
        Internal method to read data points from a metric.

        Args:
            name: Metric name
            start_index: Starting index
            limit: Max points to read

        Returns:
            Dict with data, startIndex, endIndex, total, hasMore
        """
        result = None

        if self._client:
            # Remote mode: read via API
            result = self._client.read_metric_data(
                experiment_id=self._experiment_id,
                metric_name=name,
                start_index=start_index,
                limit=limit
            )

        if self._storage:
            # Local mode: read from local storage
            result = self._storage.read_metric_data(
                project=self.project,
                experiment=self.name,
                metric_name=name,
                start_index=start_index,
                limit=limit
            )

        return result

    def _get_metric_stats(self, name: str) -> Dict[str, Any]:
        """
        Internal method to get metric statistics.

        Args:
            name: Metric name

        Returns:
            Dict with metric stats
        """
        result = None

        if self._client:
            # Remote mode: get stats via API
            result = self._client.get_metric_stats(
                experiment_id=self._experiment_id,
                metric_name=name
            )

        if self._storage:
            # Local mode: get stats from local storage
            result = self._storage.get_metric_stats(
                project=self.project,
                experiment=self.name,
                metric_name=name
            )

        return result

    def _list_metrics(self) -> List[Dict[str, Any]]:
        """
        Internal method to list all metrics in experiment.

        Returns:
            List of metric summaries
        """
        result = None

        if self._client:
            # Remote mode: list via API
            result = self._client.list_metrics(experiment_id=self._experiment_id)

        if self._storage:
            # Local mode: list from local storage
            result = self._storage.list_metrics(
                project=self.project,
                experiment=self.name
            )

        return result or []

    @property
    def id(self) -> Optional[str]:
        """Get the experiment ID (only available after open in remote mode)."""
        return self._experiment_id

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """Get the full experiment data (only available after open in remote mode)."""
        return self._experiment_data


def ml_dash_experiment(
    name: str,
    project: str,
    **kwargs
) -> Callable:
    """
    Decorator for wrapping functions with an ML-Dash experiment.

    Usage:
        @ml_dash_experiment(
            name="my-experiment",
            project="my-project",
            remote="http://localhost:3000",
            api_key="your-token"
        )
        def train_model():
            # Function code here
            pass

    The decorated function will receive an 'experiment' keyword argument
    with the active Experiment instance.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            with Experiment(name=name, project=project, **kwargs).run as experiment:
                # Inject experiment into function kwargs
                func_kwargs['experiment'] = experiment
                return func(*args, **func_kwargs)
        return wrapper
    return decorator
