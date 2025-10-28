"""Experiment class - main API for ML-Logger."""

import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .backends.base import StorageBackend
from .backends.dash_backend import DashBackend
from .backends.local_backend import LocalBackend
from .components.files import FileManager
from .components.logs import LogManager
from .components.metrics import MetricsLogger
from .components.parameters import ParameterManager


class Experiment:
  """Main experiment tracking class.

  Represents a single training execution with parameters, metrics, files, and logs.

  Args:
      namespace: User/team namespace (required)
      workspace: Project workspace (required)
      prefix: Experiment path (required)
      remote: Remote server URL (optional)
      local_root: Local storage directory (default: ".ml-logger")
      directory: Directory path for organizing experiments (optional)
      readme: Searchable description (optional)
      experiment_id: Server-side experiment ID (optional)
  """

  def __init__(
    self,
    namespace: str,
    workspace: str,
    prefix: str,
    remote: Optional[str] = None,
    local_root: str = ".ml-logger",
    directory: Optional[str] = None,
    readme: Optional[str] = None,
    experiment_id: Optional[str] = None,
    tags: Optional[list] = None,
  ):
    """Initialize experiment.

    Args:
        namespace: User/team namespace
        workspace: Project workspace
        prefix: Experiment path (used as experiment name)
        remote: Remote server URL (optional)
        local_root: Local storage directory
        directory: Directory path for organizing experiments (e.g., "dir1/dir2")
        readme: Searchable description
        experiment_id: Server-side experiment ID
        tags: Experiment tags
    """
    self.namespace = namespace
    self.workspace = workspace
    self.prefix = prefix
    self.remote = remote
    self.local_root = local_root
    self.directory = directory
    self.readme = readme or ""
    self.experiment_id = experiment_id
    self.run_id: Optional[str] = None
    self.charts: Dict[str, Any] = {}
    self.tags = tags or []

    # Full path: {local_root}/{namespace}/{workspace}/{directory}/{prefix}
    # If directory is provided, insert it before prefix
    if directory:
      self.local_path = f"{namespace}/{workspace}/{directory}/{prefix}"
    else:
      self.local_path = f"{namespace}/{workspace}/{prefix}"

    # Initialize backend
    if remote:
      # Use remote DashBackend
      self.backend: StorageBackend = DashBackend(
        server_url=remote,
        namespace=namespace,
        workspace=workspace,
        experiment_name=prefix,
        experiment_id=experiment_id,
        directory=directory,
      )
      # Initialize experiment on server
      try:
        exp_data = self.backend.initialize_experiment(description=readme, tags=tags)
        self.experiment_id = exp_data.get("id")
        print(f"✓ Initialized experiment on remote server: {self.experiment_id}")
      except Exception as e:
        print(f"Warning: Failed to initialize experiment on remote server: {e}")
        # Fall back to local backend
        self.backend = LocalBackend(local_root)
    else:
      # Use local backend
      self.backend = LocalBackend(local_root)

    # Initialize components
    self.params = ParameterManager(self.backend, self.local_path)
    self.metrics = MetricsLogger(self.backend, self.local_path)
    self.files = FileManager(self.backend, self.local_path)
    self.logs = LogManager(self.backend, self.local_path)

    # Metadata
    self._meta_file = f"{self.local_path}/.ml-logger.meta.json"
    self._status = "created"
    self._started_at: Optional[float] = None
    self._completed_at: Optional[float] = None
    self._hostname = socket.gethostname()

    # Load or create metadata (only for local backend)
    if not remote:
      self._load_metadata()

  def _load_metadata(self) -> None:
    """Load experiment metadata from file."""
    if self.backend.exists(self._meta_file):
      try:
        content = self.backend.read_text(self._meta_file)
        meta = json.loads(content)
        self._status = meta.get("status", "created")
        self._started_at = meta.get("started_at")
        self._completed_at = meta.get("completed_at")
        self.readme = meta.get("readme", self.readme)
        self.charts = meta.get("charts", {})
      except Exception:
        pass

  def _save_metadata(self) -> None:
    """Save experiment metadata to file."""
    meta = {
      "namespace": self.namespace,
      "workspace": self.workspace,
      "prefix": self.prefix,
      "remote": self.remote,
      "experiment_id": self.experiment_id,
      "readme": self.readme,
      "charts": self.charts,
      "status": self._status,
      "started_at": self._started_at,
      "completed_at": self._completed_at,
      "hostname": self._hostname,
      "updated_at": time.time(),
    }
    content = json.dumps(meta, indent=2)
    self.backend.write_text(self._meta_file, content)

  def run(self, func: Optional[Callable] = None):
    """Mark experiment as started (supports 3 patterns).

    Pattern 1 - Direct call:
        experiment.run()
        # ... training code ...
        experiment.complete()

    Pattern 2 - Context manager:
        with experiment.run():
            # ... training code ...

    Pattern 3 - Decorator:
        @experiment.run
        def train():
            # ... training code ...

    Args:
        func: Function to wrap (for decorator pattern)

    Returns:
        Context manager or decorated function
    """
    if func is None:
      # Pattern 1 (direct) or Pattern 2 (context manager)
      self._status = "running"
      self._started_at = time.time()
      # Only save metadata for local backends
      if not isinstance(self.backend, DashBackend):
        self._save_metadata()
      return self._run_context()
    else:
      # Pattern 3 (decorator)
      @wraps(func)
      def wrapper(*args, **kwargs):
        with self.run():
          return func(*args, **kwargs)

      return wrapper

  @contextmanager
  def _run_context(self):
    """Context manager for run lifecycle."""
    try:
      # Create run on remote server if using DashBackend
      if isinstance(self.backend, DashBackend) and not self.run_id:
        try:
          run_data = self.backend.create_run(name=self.prefix, tags=self.tags)
          self.run_id = run_data.get("id")
          print(f"✓ Created run on remote server: {self.run_id}")
        except Exception as e:
          print(f"Warning: Failed to create run on remote server: {e}")

      yield self
      self.complete()
    except Exception as e:
      self.fail(str(e))
      raise

  def complete(self) -> None:
    """Mark experiment as completed."""
    self._status = "completed"
    self._completed_at = time.time()

    # Update run status on remote server
    if isinstance(self.backend, DashBackend) and self.run_id:
      try:
        self.backend.update_run(status="COMPLETED")
        print("✓ Marked run as COMPLETED on remote server")
      except Exception as e:
        print(f"Warning: Failed to update run status: {e}")

    # Save metadata locally
    if not isinstance(self.backend, DashBackend):
      self._save_metadata()

  def fail(self, error: str) -> None:
    """Mark experiment as failed.

    Args:
        error: Error message
    """
    self._status = "failed"
    self._completed_at = time.time()

    # Log error
    self.logs.error("Experiment failed", error=error)

    # Update run status on remote server
    if isinstance(self.backend, DashBackend) and self.run_id:
      try:
        self.backend.update_run(status="FAILED", metadata={"error": error})
        print("✓ Marked run as FAILED on remote server")
      except Exception as e:
        print(f"Warning: Failed to update run status: {e}")

    # Save metadata locally
    if not isinstance(self.backend, DashBackend):
      self._save_metadata()

  # Convenience methods for logging
  def info(self, message: str, **context) -> None:
    """Log info message (convenience method).

    Args:
        message: Log message
        **context: Additional context
    """
    self.logs.info(message, **context)

  def error(self, message: str, **context) -> None:
    """Log error message (convenience method).

    Args:
        message: Log message
        **context: Additional context
    """
    self.logs.error(message, **context)

  def warning(self, message: str, **context) -> None:
    """Log warning message (convenience method).

    Args:
        message: Log message
        **context: Additional context
    """
    self.logs.warning(message, **context)

  def debug(self, message: str, **context) -> None:
    """Log debug message (convenience method).

    Args:
        message: Log message
        **context: Additional context
    """
    self.logs.debug(message, **context)

  @classmethod
  def _auto_configure(cls) -> "Experiment":
    """Create auto-configured experiment from environment.

    Reads configuration from:
    - ML_DASH_NAMESPACE (default: "default")
    - ML_DASH_WORKSPACE (default: "experiments")
    - ML_DASH_PREFIX (default: timestamp+uuid)
    - ML_DASH_REMOTE (optional)

    Returns:
        Auto-configured Experiment instance
    """
    namespace = os.environ.get("ML_DASH_NAMESPACE", "default")
    workspace = os.environ.get("ML_DASH_WORKSPACE", "experiments")

    # Generate default prefix with timestamp + short UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    default_prefix = f"{timestamp}_{short_id}"

    prefix = os.environ.get("ML_DASH_PREFIX", default_prefix)
    remote = os.environ.get("ML_DASH_REMOTE")

    return cls(
      namespace=namespace,
      workspace=workspace,
      prefix=prefix,
      remote=remote,
    )

  def __repr__(self) -> str:
    """String representation."""
    return (
      f"Experiment(namespace='{self.namespace}', "
      f"workspace='{self.workspace}', "
      f"prefix='{self.prefix}', "
      f"status='{self._status}')"
    )
