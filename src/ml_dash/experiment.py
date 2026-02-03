"""
Experiment class for ML-Dash SDK.

Supports three usage styles:
1. Decorator: @ml_dash_experiment(...)
2. Context manager: with Experiment(...).run as exp:
3. Direct instantiation: exp = Experiment(...)
"""

import functools
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .buffer import BackgroundBufferManager, BufferConfig
from .client import RemoteClient
from .files import BindrsBuilder, FilesAccessor
from .log import LogBuilder, LogLevel
from .params import ParametersBuilder
from .run import RUN, requires_open
from .storage import LocalStorage


def _expand_exp_template(template: str) -> str:
  """
  Expand {EXP.attr} placeholders in template string.

  Handles both regular attributes and property descriptors on the EXP class.

  Args:
      template: String containing {EXP.attr} placeholders

  Returns:
      String with placeholders expanded to actual values
  """
  import re

  def replace_match(match):
    attr_name = match.group(1)
    # Get the attribute from the class __dict__, handling properties correctly
    # EXP is a params_proto class where properties are stored in EXP.__dict__
    attr = RUN.__dict__.get(attr_name)
    if isinstance(attr, property):
      # For properties, call the getter with EXP as self
      return str(attr.fget(RUN))
    else:
      # For regular attributes, access via getattr
      return str(getattr(RUN, attr_name))

  # Match {EXP.attr_name} pattern
  pattern = r"\{EXP\.(\w+)\}"
  return re.sub(pattern, replace_match, template)


class OperationMode(Enum):
  """Operation mode for the experiment."""

  LOCAL = "local"
  REMOTE = "remote"
  HYBRID = "hybrid"  # Future: sync local to remote


class Experiment:
  """
  ML-Dash experiment for metricing experiments.

  Prefix format: {owner}/{project}/path.../[name]
    - owner: First segment (e.g., your username)
    - project: Second segment (e.g., project name)
    - path: Remaining segments form the folder structure
    - name: Derived from last segment (may be a seed/id)

  Usage examples:

  # Local mode (default)
  experiment = Experiment(prefix="ge/my-project/experiments/exp1")

  # Custom local storage directory
  experiment = Experiment(
      prefix="ge/my-project/experiments/exp1",
      dash_root=".dash"
  )

  # Remote mode with custom server
  experiment = Experiment(
      prefix="ge/my-project/experiments/exp1",
      dash_url="https://custom-server.com"
  )

  # Context manager
  with Experiment(prefix="ge/my-project/exp1").run as exp:
      exp.logs.info("Training started")

  # Decorator
  @ml_dash_experiment(prefix="ge/ws/experiments/exp", dash_url="https://api.dash.ml")
  def train():
      ...
  """

  run: RUN
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

  def __init__(
    self,
    prefix: Optional[str] = None,
    *,
    readme: Optional[str] = None,
    # Ge: this is an instance only property
    tags: Optional[List[str]] = None,
    # Ge: Bindrs is an instance-only property, it is not set inside the RUN namespace.
    bindrs: Optional[List[str]] = None,
    # Ge: This is also instance-only
    metadata: Optional[Dict[str, Any]] = None,
    # Mode configuration
    dash_url: Optional[Union[str, bool]] = None,
    dash_root: Optional[str] = ".dash",
    # Deprecated parameters (for backward compatibility)
    remote: Optional[Union[str, bool]] = None,
    local_path: Optional[str] = None,
    # Internal parameters
    _write_protected: bool = False,
    # The rest of the params go directly to populate the RUN object.
    **run_params,
  ):
    """
    Initialize an ML-Dash experiment.

    Args:
        prefix: Full experiment path like "owner/project/folder.../name" (defaults to DASH_PREFIX env var).
                Format: {owner}/{project}/path.../[name]
                - owner: First segment (e.g., username)
                - project: Second segment (e.g., project name)
                - path: Remaining segments form the folder path
                - name: Derived from last segment (may be a seed/id, not always meaningful)
        readme: Optional experiment readme/description
        tags: Optional list of tags
        bindrs: Optional list of bindrs
        metadata: Optional metadata dict
        dash_url: Remote API URL. True=use EXP.API_URL, str=custom URL, None=no remote. Token auto-loaded from ~/.dash/token.enc
        dash_root: Local storage root path (defaults to ".dash"). Set to None for remote-only mode.
        remote: (Deprecated) Use dash_url instead
        local_path: (Deprecated) Use dash_root instead
        _write_protected: Internal parameter - if True, experiment becomes immutable after creation

    Mode Selection:
        - Default (no dash_url): Local-only mode (writes to ".dash/")
        - dash_url + dash_root: Hybrid mode (local + remote)
        - dash_url + dash_root=None: Remote-only mode
    """
    import warnings

    # Handle backward compatibility
    if remote is not None:
      warnings.warn(
        "Parameter 'remote' is deprecated. Use 'dash_url' instead.",
        DeprecationWarning,
        stacklevel=2,
      )
      if dash_url is None:
        dash_url = remote

    if local_path is not None:
      warnings.warn(
        "Parameter 'local_path' is deprecated. Use 'dash_root' instead.",
        DeprecationWarning,
        stacklevel=2,
      )
      if dash_root == ".dash":  # Only override if dash_root is default
        dash_root = local_path

    if prefix:
      run_params["prefix"] = prefix

    self.run = RUN(_experiment=self, **run_params)

    self.readme = readme
    self.tags = tags
    self._bindrs_list = bindrs
    self._write_protected = _write_protected
    self.metadata = metadata

    # Determine operation mode
    # dash_root defaults to ".dash", dash_url defaults to None
    if dash_url and dash_root:
      self.mode = OperationMode.HYBRID
    elif dash_url:
      self.mode = OperationMode.REMOTE
    else:
      self.mode = OperationMode.LOCAL

    # Initialize backend
    self._experiment_id: Optional[str] = None
    self._experiment_data: Optional[Dict[str, Any]] = None
    self._is_open = False
    self._metrics_manager: Optional["MetricsManager"] = None  # Cached metrics manager
    self._tracks_manager: Optional["TracksManager"] = None  # Cached tracks manager

    # Initialize buffer manager
    self._buffer_config = BufferConfig.from_env()
    self._buffer_manager: Optional[BackgroundBufferManager] = None

    if self.mode in (OperationMode.REMOTE, OperationMode.HYBRID):
      # RemoteClient will autoload token from ~/.dash/token.enc
      # Use RUN.api_url if dash_url=True (boolean), otherwise use the provided URL
      api_url = RUN.api_url if dash_url is True else dash_url
      self.run._client = RemoteClient(base_url=api_url, namespace=self.run.owner)

    if self.mode in (OperationMode.LOCAL, OperationMode.HYBRID):
      self.run._storage = LocalStorage(root_path=Path(dash_root))

  def _open(self) -> "Experiment":
    """
    Internal method to open the experiment (create or update on server/filesystem).

    Returns:
        self for chaining
    """
    if self._is_open:
      return self

    if self.run._client:
      # Remote mode: create/update experiment via API
      try:
        response = self.run._client.create_or_update_experiment(
          project=self.run.project,
          name=self.run.name,
          description=self.readme,
          tags=self.tags,
          bindrs=self._bindrs_list,
          prefix=self.run._folder_path,
          write_protected=self._write_protected,
          metadata=self.metadata,
        )
        self._experiment_data = response
        self._experiment_id = response["experiment"]["id"]

        # Display message about viewing data online
        try:
          from rich.console import Console

          console = Console()
          experiment_url = f"https://dash.ml/{self.run.prefix}"
          console.print(
            f"[dim]✓ Experiment started: [bold]{self.run.name}[/bold] (project: {self.run.project})[/dim]\n"
            f"[dim]View your data, statistics, and plots online at:[/dim] "
            f"[link={experiment_url}]{experiment_url}[/link]"
          )
        except ImportError:
          # Fallback if rich is not available
          experiment_url = f"https://dash.ml/{self.run.prefix}"
          print(f"✓ Experiment started: {self.run.name} (project: {self.run.project})")
          print(f"View your data at: {experiment_url}")

      except Exception as e:
        # Check if it's an authentication error
        from .auth.exceptions import AuthenticationError

        if isinstance(e, AuthenticationError):
          try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            message = (
              "[bold red]Authentication Required[/bold red]\n\n"
              "You need to authenticate before using remote experiments.\n\n"
              "[bold]To authenticate:[/bold]\n"
              "  [cyan]ml-dash login[/cyan]\n\n"
              "[dim]This will open your browser for secure OAuth2 authentication.\n"
              "Your token will be stored securely in your system keychain.[/dim]\n\n"
              "[bold]Alternative:[/bold]\n"
              "  Use [cyan]local_path[/cyan] instead of [cyan]remote[/cyan] for offline experiments"
            )

            panel = Panel(
              message,
              title="[bold yellow]⚠ Not Authenticated[/bold yellow]",
              border_style="yellow",
              expand=False,
            )
            console.print("\n")
            console.print(panel)
            console.print("\n")
          except ImportError:
            # Fallback if rich is not available
            print("\n" + "=" * 60)
            print("⚠ Authentication Required")
            print("=" * 60)
            print("\nYou need to authenticate before using remote experiments.\n")
            print("To authenticate:")
            print("  ml-dash login\n")
            print("Alternative:")
            print("  Use local_path instead of remote for offline experiments\n")
            print("=" * 60 + "\n")

          import sys

          sys.exit(1)
        else:
          # Re-raise other exceptions
          raise

    if self.run._storage:
      # Local mode: create experiment directory structure
      self.run._storage.create_experiment(
        project=self.run.project,
        prefix=self.run._folder_path,
        description=self.readme,
        tags=self.tags,
        bindrs=self._bindrs_list,
        metadata=self.metadata,
      )

    # Start background buffer
    if self._buffer_config.buffer_enabled:
      self._buffer_manager = BackgroundBufferManager(self, self._buffer_config)
      self._buffer_manager.start()

    self._is_open = True
    return self

  def _close(self, status: str = "COMPLETED"):
    """
    Internal method to close the experiment and update status.

    Args:
        status: Status to set - "COMPLETED" (default), "FAILED", or "CANCELLED"
    """
    # if not self._is_open:
    #   return
    #
    # note-ge: do NOT flush because the upload will be async. we will NEVER reuse
    # experiment objects.
    # # Flush any pending writes
    # if self.run._storage:
    #   self.run._storage.flush()

    # Flush and stop buffer BEFORE status update
    # Waits indefinitely for all data to be flushed (important for large files)
    if self._buffer_manager:
      self._buffer_manager.stop()

    # Update experiment status in remote mode
    if self.run._client and self._experiment_id:
      try:
        self.run._client.update_experiment_status(
          experiment_id=self._experiment_id, status=status
        )

        # Display completion message with link to view results
        status_emoji = {"COMPLETED": "✓", "FAILED": "✗", "CANCELLED": "⊘"}.get(
          status, "•"
        )

        status_color = {
          "COMPLETED": "green",
          "FAILED": "red",
          "CANCELLED": "yellow",
        }.get(status, "white")

        try:
          from rich.console import Console

          console = Console()
          experiment_url = f"https://dash.ml/{self.run.prefix}"
          console.print(
            f"[{status_color}]{status_emoji} Experiment {status.lower()}: "
            f"[bold]{self.run.name}[/bold] (project: {self.run.project})[/{status_color}]\n"
            f"[dim]View results, statistics, and plots online at:[/dim] "
            f"[link={experiment_url}]{experiment_url}[/link]"
          )
        except ImportError:
          # Fallback if rich is not available
          experiment_url = f"https://dash.ml/{self.run.prefix}"
          print(
            f"{status_emoji} Experiment {status.lower()}: {self.run.name} (project: {self.run.project})"
          )
          print(f"View results at: {experiment_url}")

      except Exception as e:
        # Raise on status update failure
        raise RuntimeError(
          f"Failed to update experiment status to COMPLETED: {e}\n"
          f"Experiment may not be marked as completed on the server."
        ) from e

    self._is_open = False

  @property
  @requires_open
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
    return ParametersBuilder(self)

  @property
  @requires_open
  def logs(self) -> LogBuilder:
    """
    Get a LogBuilder for fluent-style logging.

    Returns a LogBuilder that allows chaining with level methods like
    .info(), .warn(), .error(), .debug(), .fatal().

    Returns:
        LogBuilder instance for fluent logging

    Raises:
        RuntimeError: If experiment is not open

    Examples:
        exp.logs.info("Training started", epoch=1)
        exp.logs.error("Failed to load data", error_code=500)
        exp.logs.warn("GPU memory low", memory_available="1GB")
        exp.logs.debug("Debug info", step=100)
    """
    return LogBuilder(self, metadata=None)

  @requires_open
  def log(
    self,
    message: Optional[str] = None,
    level: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **extra_metadata,
  ) -> Optional[LogBuilder]:
    """
    Create a log entry (traditional style).

    .. deprecated::
        The fluent style (calling without message) is deprecated.
        Use the `logs` property instead: `exp.logs.info("message")`

    Recommended usage:
        exp.logs.info("Training started", epoch=1)
        exp.logs.error("Failed", error_code=500)

    Traditional style (still supported):
        experiment.log("Training started", level="info", epoch=1)
        experiment.log("Training started")  # Defaults to "info"

    Args:
        message: Log message (required for recommended usage)
        level: Log level (defaults to "info")
        metadata: Optional metadata dict
        **extra_metadata: Additional metadata as keyword arguments

    Returns:
        None when used in traditional style (message provided)
        LogBuilder when used in deprecated fluent style (message=None)

    Raises:
        RuntimeError: If experiment is not open
        ValueError: If log level is invalid
    """

    # Fluent mode: return LogBuilder (deprecated)
    if message is None:
      import warnings

      warnings.warn(
        "Using exp.log() without a message is deprecated. "
        "Use exp.logs.info('message') instead.",
        DeprecationWarning,
        stacklevel=2,
      )
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
      timestamp=None,
    )
    return None

  def _write_log(
    self,
    message: str,
    level: str,
    metadata: Optional[Dict[str, Any]],
    timestamp: Optional[datetime],
  ) -> None:
    """
    Internal method to write a log entry.
    Uses buffering if enabled, otherwise writes directly.

    Args:
        message: Log message
        level: Log level (already validated)
        metadata: Optional metadata dict
        timestamp: Optional custom timestamp (defaults to now)
    """
    # Print to console immediately (user visibility)
    self._print_log(message, level, metadata)

    # Buffer or write immediately
    if self._buffer_manager and self._buffer_config.buffer_enabled:
      self._buffer_manager.buffer_log(message, level, metadata, timestamp)
    else:
      # Immediate write (backward compatibility)
      log_entry = {
        "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
        "level": level,
        "message": message,
      }

      if metadata:
        log_entry["metadata"] = metadata

      if self.run._client:
        # Remote mode: send to API (wrapped in array for batch API)
        try:
          self.run._client.create_log_entries(
            experiment_id=self._experiment_id,
            logs=[log_entry],  # Single log in array
          )
        except Exception as e:
          raise RuntimeError(
            f"Failed to write log to remote server: {e}\n"
            f"Data loss occurred. Check your network connection and server status."
          ) from e

      if self.run._storage:
        # Local mode: write to file immediately
        try:
          self.run._storage.write_log(
            owner=self.run.owner,
            project=self.run.project,
            prefix=self.run._folder_path,
            message=log_entry["message"],
            level=log_entry["level"],
            metadata=log_entry.get("metadata"),
            timestamp=log_entry["timestamp"],
          )
        except Exception as e:
          raise RuntimeError(
            f"Failed to write log to local storage: {e}\n"
            f"Check disk space and file permissions."
          ) from e

  def _print_log(
    self, message: str, level: str, metadata: Optional[Dict[str, Any]]
  ) -> None:
    """
    Print log to stdout or stderr based on level.

    ERROR and FATAL go to stderr, all others go to stdout.

    Args:
        message: Log message
        level: Log level
        metadata: Optional metadata dict
    """
    import sys

    # Format the log message
    level_upper = level.upper()

    # Build metadata string if present
    metadata_str = ""
    if metadata:
      # Format metadata as key=value pairs
      pairs = [f"{k}={v}" for k, v in metadata.items()]
      metadata_str = f" [{', '.join(pairs)}]"

    # Format: [LEVEL] message [key=value, ...]
    formatted_message = f"[{level_upper}] {message}{metadata_str}"

    # Route to stdout or stderr based on level
    if level in ("error", "fatal"):
      print(formatted_message, file=sys.stderr)
    else:
      print(formatted_message, file=sys.stdout)

  @property
  @requires_open
  def files(self) -> FilesAccessor:
    """
    Get a FilesAccessor for fluent file operations.

    Returns:
        FilesAccessor instance for chaining

    Raises:
        RuntimeError: If experiment is not open

    Examples:
        # Upload file - supports flexible syntax
        experiment.files("checkpoints").upload("./model.pt", to="checkpoint.pt")
        experiment.files(prefix="checkpoints").upload("./model.pt")
        experiment.files().upload("./model.pt", to="models/model.pt")  # root

        # List files
        files = experiment.files("/some/location").list()
        files = experiment.files("/models").list()

        # Download file
        experiment.files("some.text").download()
        experiment.files("some.text").download(to="./model.pt")

        # Download files via glob pattern
        file_paths = experiment.files("images").list("*.png")
        experiment.files("images").download("*.png")

        # This is equivalent to downloading to a directory
        experiment.files.download("images/*.png", to="local_images")

        # Delete files
        experiment.files("some.text").delete()
        experiment.files.delete("some.text")

        # Specific file types
        dxp.files.save_text("content", to="view.yaml")
        dxp.files.save_json(dict(hey="yo"), to="config.json")
        dxp.files.save_blob(b"xxx", to="data.bin")
    """
    return FilesAccessor(self)

  @requires_open
  def bindrs(self, bindr_name: str) -> BindrsBuilder:
    """
    Get a BindrsBuilder for working with file collections (bindrs).

    Bindrs are collections of files that can span multiple prefixes.

    Args:
        bindr_name: Name of the bindr (collection)

    Returns:
        BindrsBuilder instance for chaining

    Raises:
        RuntimeError: If experiment is not open

    Examples:
        # List files in a bindr
        file_paths = experiment.bindrs("some-bindr").list()

    Note:
        This is a placeholder for future bindr functionality.
    """
    return BindrsBuilder(self, bindr_name)

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
    size_bytes: int,
  ) -> Dict[str, Any]:
    """
    Internal method to upload a file.
    Uses buffering if enabled, otherwise uploads directly.

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
        File metadata dict (or pending status if buffering)
    """
    # Buffer or upload immediately
    if self._buffer_manager and self._buffer_config.buffer_enabled:
      self._buffer_manager.buffer_file(
        file_path, prefix, filename, description, tags, metadata,
        checksum, content_type, size_bytes
      )
      return {"id": "pending", "status": "queued"}
    else:
      # Immediate upload (backward compatibility)
      result = None

      if self.run._client:
        # Remote mode: upload to API
        result = self.run._client.upload_file(
          experiment_id=self._experiment_id,
          file_path=file_path,
          prefix=prefix,
          filename=filename,
          description=description,
          tags=tags,
          metadata=metadata,
          checksum=checksum,
          content_type=content_type,
          size_bytes=size_bytes,
        )

      if self.run._storage:
        # Local mode: copy to local storage
        result = self.run._storage.write_file(
          owner=self.run.owner,
          project=self.run.project,
          prefix=self.run._folder_path,
          file_path=file_path,
          path=prefix,
          filename=filename,
          description=description,
          tags=tags,
          metadata=metadata,
          checksum=checksum,
          content_type=content_type,
          size_bytes=size_bytes,
        )

      return result

  def _list_files(
    self, prefix: Optional[str] = None, tags: Optional[List[str]] = None
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

    if self.run._client:
      # Remote mode: fetch from API
      files = self.run._client.list_files(
        experiment_id=self._experiment_id, prefix=prefix, tags=tags
      )

    if self.run._storage:
      # Local mode: read from metadata file
      files = self.run._storage.list_files(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        path_prefix=prefix,
        tags=tags,
      )

    return files

  def _download_file(self, file_id: str, dest_path: Optional[str] = None) -> str:
    """
    Internal method to download a file.

    Args:
        file_id: File ID
        dest_path: Optional destination path (defaults to original filename)

    Returns:
        Path to downloaded file
    """
    if self.run._client:
      # Remote mode: download from API
      return self.run._client.download_file(
        experiment_id=self._experiment_id, file_id=file_id, dest_path=dest_path
      )

    if self.run._storage:
      # Local mode: copy from local storage
      return self.run._storage.read_file(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        file_id=file_id,
        dest_path=dest_path,
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

    if self.run._client:
      # Remote mode: delete via API
      result = self.run._client.delete_file(
        experiment_id=self._experiment_id, file_id=file_id
      )

    if self.run._storage:
      # Local mode: soft delete in metadata
      result = self.run._storage.delete_file(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        file_id=file_id,
      )

    return result

  def _update_file(
    self,
    file_id: str,
    description: Optional[str],
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
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

    if self.run._client:
      # Remote mode: update via API
      result = self.run._client.update_file(
        experiment_id=self._experiment_id,
        file_id=file_id,
        description=description,
        tags=tags,
        metadata=metadata,
      )

    if self.run._storage:
      # Local mode: update in metadata file
      result = self.run._storage.update_file_metadata(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        file_id=file_id,
        description=description,
        tags=tags,
        metadata=metadata,
      )

    return result

  def _write_params(self, flattened_params: Dict[str, Any]) -> None:
    """
    Internal method to write/merge parameters.

    Args:
        flattened_params: Already-flattened parameter dict with dot notation
    """
    if self.run._client:
      # Remote mode: send to API
      self.run._client.set_parameters(
        experiment_id=self._experiment_id, data=flattened_params
      )

    if self.run._storage:
      # Local mode: write to file
      self.run._storage.write_parameters(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        data=flattened_params,
      )

  def _read_params(self) -> Optional[Dict[str, Any]]:
    """
    Internal method to read parameters.

    Returns:
        Flattened parameters dict, or None if no parameters exist
    """
    params = None

    if self.run._client:
      # Remote mode: fetch from API
      try:
        params = self.run._client.get_parameters(experiment_id=self._experiment_id)
      except Exception:
        # Parameters don't exist yet
        params = None

    if self.run._storage:
      # Local mode: read from file
      params = self.run._storage.read_parameters(
        owner=self.run.owner, project=self.run.project, prefix=self.run._folder_path
      )

    return params

  @property
  @requires_open
  def metrics(self) -> "MetricsManager":
    """
    Get a MetricsManager for metric operations.

    Supports two usage patterns:
    1. Named: experiment.metrics("train").log(loss=0.5, accuracy=0.9)
    2. Unnamed: experiment.metrics.log(epoch=epoch).flush()

    Returns:
        MetricsManager instance

    Raises:
        RuntimeError: If experiment is not open

    Examples:
        # Named metric with multi-field logging
        experiment.metrics("train").log(loss=0.5, accuracy=0.9)
        experiment.metrics("eval").log(loss=0.6, accuracy=0.85)
        experiment.metrics.log(epoch=epoch).flush()

        # Nested dict pattern (single call for all metrics)
        experiment.metrics.log(
            epoch=100,
            train=dict(loss=0.142, accuracy=0.80),
            eval=dict(loss=0.201, accuracy=0.76)
        )

        # Read data
        data = experiment.metrics("train").read(start_index=0, limit=100)

        # Get statistics
        stats = experiment.metrics("train").stats()
    """
    from .metric import MetricsManager

    # Cache the MetricsManager instance to preserve MetricBuilder cache across calls
    if self._metrics_manager is None:
      self._metrics_manager = MetricsManager(self)
    return self._metrics_manager

  @property
  @requires_open
  def tracks(self) -> "TracksManager":
    """
    Get a TracksManager for timestamped track operations.

    Supports topic-based logging with automatic timestamp merging:
    - experiment.tracks("robot/position").append(q=[0.1, 0.2], _ts=0.0)
    - experiment.tracks.flush()  # Flush all topics
    - experiment.tracks("robot/position").flush()  # Flush specific topic

    Returns:
        TracksManager instance

    Raises:
        RuntimeError: If experiment is not open

    Examples:
        # Log track data with timestamp
        experiment.tracks("robot/position").append(
            q=[0.1, -0.22, 0.45],
            e=[0.5, 0.0, 0.6],
            _ts=2.0
        )

        # Entries with same timestamp are automatically merged
        experiment.tracks("camera/rgb").append(frame_id=0, _ts=0.0)
        experiment.tracks("camera/rgb").append(path="frame_0.png", _ts=0.0)

        # Read track data
        data = experiment.tracks("robot/position").read(format="json")

        # Download in different formats
        jsonl = experiment.tracks("robot/position").read(format="jsonl")
        parquet = experiment.tracks("robot/position").read(format="parquet")
        mocap = experiment.tracks("robot/position").read(format="mocap")
    """
    from .track import TracksManager

    # Cache the TracksManager instance to preserve TrackBuilder cache across calls
    if self._tracks_manager is None:
      self._tracks_manager = TracksManager(self)
    return self._tracks_manager

  def _append_to_metric(
    self,
    name: Optional[str],
    data: Dict[str, Any],
    description: Optional[str],
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
  ) -> Optional[Dict[str, Any]]:
    """
    Internal method to append a single data point to a metric.
    Uses buffering if enabled, otherwise writes directly.

    Args:
        name: Metric name (can be None for unnamed metrics)
        data: Data point (flexible schema)
        description: Optional metric description
        tags: Optional tags
        metadata: Optional metadata

    Returns:
        Dict with metricId, index, bufferedDataPoints, chunkSize or None if buffering enabled/all backends fail
    """
    # Buffer or write immediately
    if self._buffer_manager and self._buffer_config.buffer_enabled:
      self._buffer_manager.buffer_metric(name, data, description, tags, metadata)
      return None  # No immediate response when buffering
    else:
      # Immediate write (backward compatibility)
      result = None

      if self.run._client:
        # Remote mode: append via API
        try:
          result = self.run._client.append_to_metric(
            experiment_id=self._experiment_id,
            metric_name=name,
            data=data,
            description=description,
            tags=tags,
            metadata=metadata,
          )
        except Exception as e:
          metric_display = f"'{name}'" if name else "unnamed metric"
          raise RuntimeError(
            f"Failed to log {metric_display} to remote server: {e}\n"
            f"Data loss occurred. Check your network connection and server status."
          ) from e

      if self.run._storage:
        # Local mode: append to local storage
        try:
          result = self.run._storage.append_to_metric(
            owner=self.run.owner,
            project=self.run.project,
            prefix=self.run._folder_path,
            metric_name=name,
            data=data,
            description=description,
            tags=tags,
            metadata=metadata,
          )
        except Exception as e:
          metric_display = f"'{name}'" if name else "unnamed metric"
          raise RuntimeError(
            f"Failed to log {metric_display} to local storage: {e}\n"
            f"Check disk space and file permissions."
          ) from e

      return result

  def _write_track(
    self,
    topic: str,
    timestamp: float,
    data: Dict[str, Any],
  ) -> None:
    """
    Internal method to write a track entry with timestamp.
    Uses buffering with timestamp-based merging if enabled.

    Args:
        topic: Track topic (e.g., "robot/position")
        timestamp: Entry timestamp
        data: Data fields

    Note:
        Entries with the same timestamp are automatically merged in the buffer.
    """
    # Buffer or write immediately
    if self._buffer_manager and self._buffer_config.buffer_enabled:
      self._buffer_manager.buffer_track(topic, timestamp, data)
    else:
      # Immediate write (no buffering)
      if self.run._client:
        # Remote mode: append via API
        try:
          self.run._client.append_batch_to_track(
            experiment_id=self._experiment_id,
            topic=topic,
            entries=[{"timestamp": timestamp, **data}],
          )
        except Exception as e:
          raise RuntimeError(
            f"Failed to log track '{topic}' to remote server: {e}\n"
            f"Data loss occurred. Check your network connection and server status."
          ) from e

      if self.run._storage:
        # Local mode: append to local storage
        try:
          self.run._storage.append_batch_to_track(
            owner=self.run.owner,
            project=self.run.project,
            prefix=self.run._folder_path,
            topic=topic,
            entries=[{"timestamp": timestamp, **data}],
          )
        except Exception as e:
          raise RuntimeError(
            f"Failed to log track '{topic}' to local storage: {e}\n"
            f"Check disk space and file permissions."
          ) from e

  def _append_batch_to_metric(
    self,
    name: Optional[str],
    data_points: List[Dict[str, Any]],
    description: Optional[str],
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
  ) -> Optional[Dict[str, Any]]:
    """
    Internal method to append multiple data points to a metric.

    Args:
        name: Metric name (can be None for unnamed metrics)
        data_points: List of data points
        description: Optional metric description
        tags: Optional tags
        metadata: Optional metadata

    Returns:
        Dict with metricId, startIndex, endIndex, count or None if all backends fail
    """
    result = None

    if self.run._client:
      # Remote mode: append batch via API
      try:
        result = self.run._client.append_batch_to_metric(
          experiment_id=self._experiment_id,
          metric_name=name,
          data_points=data_points,
          description=description,
          tags=tags,
          metadata=metadata,
        )
      except Exception as e:
        metric_display = f"'{name}'" if name else "unnamed metric"
        raise RuntimeError(
          f"Failed to log batch to {metric_display} on remote server: {e}\n"
          f"Data loss occurred. Check your network connection and server status."
        ) from e

    if self.run._storage:
      # Local mode: append batch to local storage
      try:
        result = self.run._storage.append_batch_to_metric(
          owner=self.run.owner,
          project=self.run.project,
          prefix=self.run._folder_path,
          metric_name=name,
          data_points=data_points,
          description=description,
          tags=tags,
          metadata=metadata,
        )
      except Exception as e:
        metric_display = f"'{name}'" if name else "unnamed metric"
        raise RuntimeError(
          f"Failed to log batch to {metric_display} in local storage: {e}\n"
          f"Check disk space and file permissions."
        ) from e

    return result

  def _read_metric_data(
    self, name: str, start_index: int, limit: int
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

    if self.run._client:
      # Remote mode: read via API
      result = self.run._client.read_metric_data(
        experiment_id=self._experiment_id,
        metric_name=name,
        start_index=start_index,
        limit=limit,
      )

    if self.run._storage:
      # Local mode: read from local storage
      result = self.run._storage.read_metric_data(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        metric_name=name,
        start_index=start_index,
        limit=limit,
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

    if self.run._client:
      # Remote mode: get stats via API
      result = self.run._client.get_metric_stats(
        experiment_id=self._experiment_id, metric_name=name
      )

    if self.run._storage:
      # Local mode: get stats from local storage
      result = self.run._storage.get_metric_stats(
        owner=self.run.owner,
        project=self.run.project,
        prefix=self.run._folder_path,
        metric_name=name,
      )

    return result

  def _list_metrics(self) -> List[Dict[str, Any]]:
    """
    Internal method to list all metrics in experiment.

    Returns:
        List of metric summaries
    """
    result = None

    if self.run._client:
      # Remote mode: list via API
      result = self.run._client.list_metrics(experiment_id=self._experiment_id)

    if self.run._storage:
      # Local mode: list from local storage
      result = self.run._storage.list_metrics(
        owner=self.run.owner, project=self.run.project, prefix=self.run._folder_path
      )

    return result or []

  @property
  def owner(self) -> Optional[str]:
    """Get the owner (first segment of prefix)."""
    return self.run.owner

  @owner.setter
  def owner(self, value: str) -> None:
    """Set the owner."""
    self.run.owner = value

  @property
  def project(self) -> Optional[str]:
    """Get the project (second segment of prefix or RUN.project)."""
    return self.run.project

  @project.setter
  def project(self, value: str) -> None:
    """Set the project."""
    self.run.project = value

  @property
  def name(self) -> Optional[str]:
    """Get the experiment name (last segment of prefix)."""
    return self.run.name

  @name.setter
  def name(self, value: str) -> None:
    """Set the name."""
    self.run.name = value

  @property
  def _folder_path(self) -> Optional[str]:
    """Get the full folder path (same as prefix)."""
    return self.run._folder_path

  @_folder_path.setter
  def _folder_path(self, value: str) -> None:
    """Set the full folder path and re-parse into components."""
    self.run._folder_path = value
    self.run.prefix = value
    # Re-parse prefix into components
    if value:
      parts = value.strip("/").split("/")
      if len(parts) >= 2:
        self.run.owner = parts[0]
        self.run.project = parts[1]
        self.run.name = parts[-1] if len(parts) > 2 else parts[1]

  @property
  def _client(self):
    """Get the remote client."""
    return self.run._client

  @_client.setter
  def _client(self, value) -> None:
    """Set the remote client."""
    self.run._client = value

  @property
  def _storage(self):
    """Get the local storage."""
    return self.run._storage

  @_storage.setter
  def _storage(self, value) -> None:
    """Set the local storage."""
    self.run._storage = value

  def flush(self) -> None:
    """
    Manually flush all buffered data.

    Forces immediate flush of all queued logs, metrics, and files.
    Waits for all file uploads to complete.

    Examples:
        with Experiment("my-project/exp").run as exp:
            for epoch in range(100):
                exp.metrics("train").log(loss=loss)

            exp.flush()  # Ensure metrics written before checkpoint
            torch.save(model, "model.pt")
    """
    if self._buffer_manager:
      self._buffer_manager.flush_all()

  @property
  def id(self) -> Optional[str]:
    """Get the experiment ID (only available after open in remote mode)."""
    return self._experiment_id

  @property
  def data(self) -> Optional[Dict[str, Any]]:
    """Get the full experiment data (only available after open in remote mode)."""
    return self._experiment_data


def ml_dash_experiment(prefix: str, **kwargs) -> Callable:
  """
  Decorator for wrapping functions with an ML-Dash experiment.

  Args:
      prefix: Full experiment path like "owner/project/folder.../name"
      **kwargs: Additional arguments passed to Experiment constructor

  Usage:
      @ml_dash_experiment(
          prefix="ge/my-project/experiments/my-experiment",
          dash_url="https://api.dash.ml"
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
      with Experiment(prefix=prefix, **kwargs).run as experiment:
        # Inject experiment into function kwargs
        func_kwargs["experiment"] = experiment
        return func(*args, **func_kwargs)

    return wrapper

  return decorator
