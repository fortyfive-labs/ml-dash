"""
Local filesystem storage for ML-Dash.
"""

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class LocalStorage:
  """
  Local filesystem storage backend.

  Directory structure:
  <root>/
    <project>/
      <experiment_name>/
        experiment.json        # Experiment metadata
        logs/
          logs.jsonl        # Log entries
          .log_sequence     # Sequence counter
        metrics/
          <metric_name>.jsonl
        files/
          <uploaded_files>
        parameters.json     # Flattened parameters
  """

  def __init__(self, root_path: Path):
    """
    Initialize local storage.

    Args:
        root_path: Root directory for local storage
    """
    self.root_path = Path(root_path)
    self.root_path.mkdir(parents=True, exist_ok=True)

  @contextmanager
  def _file_lock(self, lock_file: Path):
    """
    Context manager for file-based locking (works across processes and threads).

    Args:
        lock_file: Path to the lock file

    Yields:
        File handle with exclusive lock
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None

    try:
      # Open lock file
      lock_fd = open(lock_file, "a")

      # Acquire exclusive lock (blocking)
      # Use fcntl on Unix-like systems
      if hasattr(fcntl, "flock"):
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
      elif hasattr(fcntl, "lockf"):
        fcntl.lockf(lock_fd.fileno(), fcntl.LOCK_EX)
      else:
        # Fallback for systems without fcntl (like Windows)
        # Use simple file existence as lock (not perfect but better than nothing)
        pass

      yield lock_fd

    finally:
      # Release lock and close file
      if lock_fd:
        try:
          if hasattr(fcntl, "flock"):
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
          elif hasattr(fcntl, "lockf"):
            fcntl.lockf(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
          pass
        lock_fd.close()

  def create_experiment(
    self,
    project: str,
    prefix: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    bindrs: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
  ) -> Path:
    """
    Create an experiment directory structure.

    Structure: root / prefix
    where prefix = owner/project/folder_1/.../exp_name

    Args:
        project: Project name (extracted from prefix, kept for API compatibility)
        prefix: Full experiment path (owner/project/folder_1/.../exp_name)
        description: Optional description
        tags: Optional tags
        bindrs: Optional bindrs
        metadata: Optional metadata

    Returns:
        Path to experiment directory
    """
    # Normalize prefix path
    prefix_clean = prefix.rstrip("/")
    prefix_path = prefix_clean.lstrip("/")

    # Create experiment directory directly from prefix
    experiment_dir = self.root_path / prefix_path
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "metrics").mkdir(exist_ok=True)
    (experiment_dir / "files").mkdir(exist_ok=True)

    # Extract experiment name from last segment of prefix
    name = prefix_clean.split("/")[-1]

    # Write experiment metadata
    experiment_metadata = {
      "name": name,
      "project": project,
      "description": description,
      "tags": tags or [],
      "bindrs": bindrs or [],
      "prefix": prefix,
      "metadata": metadata,
      "created_at": datetime.utcnow().isoformat() + "Z",
      "write_protected": False,
    }

    experiment_file = experiment_dir / "experiment.json"

    # File-based lock for concurrent experiment creation/update
    lock_file = experiment_dir / ".experiment.lock"
    with self._file_lock(lock_file):
      if not experiment_file.exists():
        # Only create if doesn't exist (don't overwrite)
        with open(experiment_file, "w") as f:
          json.dump(experiment_metadata, f, indent=2)
      else:
        # Update existing experiment
        try:
          with open(experiment_file, "r") as f:
            existing = json.load(f)
        except (json.JSONDecodeError, IOError):
          # File might be corrupted or empty, recreate it
          with open(experiment_file, "w") as f:
            json.dump(experiment_metadata, f, indent=2)
          return experiment_dir

        # Merge updates
        if description is not None:
          existing["description"] = description
        if tags is not None:
          existing["tags"] = tags
        if bindrs is not None:
          existing["bindrs"] = bindrs
        if prefix is not None:
          existing["prefix"] = prefix
        if metadata is not None:
          existing["metadata"] = metadata
        existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with open(experiment_file, "w") as f:
          json.dump(existing, f, indent=2)

    return experiment_dir

  def flush(self):
    """Flush any pending writes (no-op for now)."""
    pass

  def write_log(
    self,
    owner: str,
    project: str,
    prefix: str,
    message: str = "",
    level: str = "info",
    timestamp: str = "",
    metadata: Optional[Dict[str, Any]] = None,
  ):
    """
    Write a single log entry immediately to JSONL file.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        message: Log message
        level: Log level
        timestamp: ISO timestamp string
        metadata: Optional metadata
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    logs_dir = experiment_dir / "logs"
    logs_file = logs_dir / "logs.jsonl"
    seq_file = logs_dir / ".log_sequence"

    # File-based lock for concurrent log writes (prevents sequence collision)
    lock_file = logs_dir / ".log_sequence.lock"
    with self._file_lock(lock_file):
      # Read and increment sequence counter
      sequence_number = 0
      if seq_file.exists():
        try:
          sequence_number = int(seq_file.read_text().strip())
        except (ValueError, IOError):
          sequence_number = 0

      log_entry = {
        "sequenceNumber": sequence_number,
        "timestamp": timestamp,
        "level": level,
        "message": message,
      }

      if metadata:
        log_entry["metadata"] = metadata

      # Write log immediately
      with open(logs_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

      # Update sequence counter
      seq_file.write_text(str(sequence_number + 1))

  def write_metric_data(
    self,
    project: str,
    experiment: str,
    metric_name: str,
    data: Any,
  ):
    """
    Write metric data point.

    Args:
        project: Project name
        experiment: Experiment name
        metric_name: Metric name
        data: Data point
    """
    experiment_dir = self._get_experiment_dir(project, experiment)
    metric_file = experiment_dir / "metrics" / f"{metric_name}.jsonl"

    data_point = {
      "timestamp": datetime.utcnow().isoformat() + "Z",
      "data": data,
    }

    with open(metric_file, "a") as f:
      f.write(json.dumps(data_point) + "\n")

  def write_parameters(
    self,
    owner: str,
    project: str,
    prefix: str,
    data: Optional[Dict[str, Any]] = None,
  ):
    """
    Write/merge parameters. Always merges with existing parameters.

    File format:
    {
      "version": 2,
      "data": {"model.lr": 0.001, "model.batch_size": 32},
      "updatedAt": "2024-01-15T10:30:00Z"
    }

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix path
        data: Flattened parameter dict with dot notation (already flattened)
    """
    if data is None:
      data = {}
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    params_file = experiment_dir / "parameters.json"

    # File-based lock for concurrent parameter writes (prevents data loss and version conflicts)
    lock_file = experiment_dir / ".parameters.lock"
    with self._file_lock(lock_file):
      # Read existing if present
      if params_file.exists():
        try:
          with open(params_file, "r") as f:
            existing_doc = json.load(f)
        except (json.JSONDecodeError, IOError):
          # Corrupted file, recreate
          existing_doc = None

        if existing_doc:
          # Merge with existing data
          existing_data = existing_doc.get("data", {})
          existing_data.update(data)

          # Increment version
          version = existing_doc.get("version", 1) + 1

          params_doc = {
            "version": version,
            "data": existing_data,
            "updatedAt": datetime.utcnow().isoformat() + "Z",
          }
        else:
          # Create new if corrupted
          params_doc = {
            "version": 1,
            "data": data,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "updatedAt": datetime.utcnow().isoformat() + "Z",
          }
      else:
        # Create new parameters document
        params_doc = {
          "version": 1,
          "data": data,
          "createdAt": datetime.utcnow().isoformat() + "Z",
          "updatedAt": datetime.utcnow().isoformat() + "Z",
        }

      with open(params_file, "w") as f:
        json.dump(params_doc, f, indent=2)

  def read_parameters(
    self,
    owner: str,
    project: str,
    prefix: str,
  ) -> Optional[Dict[str, Any]]:
    """
    Read parameters from local file.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix path

    Returns:
        Flattened parameter dict, or None if file doesn't exist
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    params_file = experiment_dir / "parameters.json"

    if not params_file.exists():
      return None

    try:
      with open(params_file, "r") as f:
        params_doc = json.load(f)
      return params_doc.get("data", {})
    except (json.JSONDecodeError, IOError):
      return None

  def write_file(
    self,
    owner: str,
    project: str,
    prefix: str,
    file_path: str,
    path: str = "",
    filename: str = "",
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    checksum: str = "",
    content_type: str = "",
    size_bytes: int = 0,
  ) -> Dict[str, Any]:
    """
    Write file to local storage.

    Stores at: root / owner / project / prefix / files / path / file_id / filename

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix (folder_1/folder_2/.../exp_name)
        file_path: Source file path (where to copy from)
        path: Subdirectory within experiment/files for organizing files
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
    import shutil

    from .files import generate_snowflake_id

    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    files_dir = experiment_dir / "files"
    metadata_file = files_dir / ".files_metadata.json"

    # Generate Snowflake ID for file
    file_id = generate_snowflake_id()

    # Normalize path (remove leading slashes to avoid absolute paths)
    normalized_path = path.lstrip("/") if path else ""

    # Create storage subdirectory, then file directory
    storage_dir = files_dir / normalized_path if normalized_path else files_dir
    storage_dir.mkdir(parents=True, exist_ok=True)

    file_dir = storage_dir / file_id
    file_dir.mkdir(parents=True, exist_ok=True)

    # Copy file
    dest_file = file_dir / filename
    shutil.copy2(file_path, dest_file)

    # Create file metadata
    file_metadata = {
      "id": file_id,
      "experimentId": f"{project}/{prefix}",  # Local mode doesn't have real experiment ID
      "path": path,
      "filename": filename,
      "description": description,
      "tags": tags or [],
      "contentType": content_type,
      "sizeBytes": size_bytes,
      "checksum": checksum,
      "metadata": metadata,
      "uploadedAt": datetime.utcnow().isoformat() + "Z",
      "updatedAt": datetime.utcnow().isoformat() + "Z",
      "deletedAt": None,
    }

    # File-based lock for concurrent safety (works across processes/threads/instances)
    lock_file = files_dir / ".files_metadata.lock"
    with self._file_lock(lock_file):
      # Read existing metadata
      files_metadata = {"files": []}
      if metadata_file.exists():
        try:
          with open(metadata_file, "r") as f:
            files_metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
          files_metadata = {"files": []}

      # Check if file with same path+filename exists (overwrite behavior)
      existing_index = None
      for i, existing_file in enumerate(files_metadata["files"]):
        if (
          existing_file["path"] == path
          and existing_file["filename"] == filename
          and existing_file["deletedAt"] is None
        ):
          existing_index = i
          break

      if existing_index is not None:
        # Overwrite: remove old file and update metadata
        old_file = files_metadata["files"][existing_index]
        old_prefix = old_file["path"].lstrip("/") if old_file["path"] else ""
        if old_prefix:
          old_file_dir = files_dir / old_prefix / old_file["id"]
        else:
          old_file_dir = files_dir / old_file["id"]
        if old_file_dir.exists():
          shutil.rmtree(old_file_dir)
        files_metadata["files"][existing_index] = file_metadata
      else:
        # New file: append to list
        files_metadata["files"].append(file_metadata)

      # Write updated metadata
      with open(metadata_file, "w") as f:
        json.dump(files_metadata, f, indent=2)

    return file_metadata

  def list_files(
    self,
    owner: str,
    project: str,
    prefix: str,
    path_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
  ) -> List[Dict[str, Any]]:
    """
    List files from local storage.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix (folder_1/folder_2/.../exp_name)
        path_prefix: Optional file path prefix filter
        tags: Optional tags filter

    Returns:
        List of file metadata dicts (only non-deleted files)
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    metadata_file = experiment_dir / "files/.files_metadata.json"

    if not metadata_file.exists():
      return []

    try:
      with open(metadata_file, "r") as f:
        files_metadata = json.load(f)
    except (json.JSONDecodeError, IOError):
      return []

    files = files_metadata.get("files", [])

    # Filter out deleted files
    files = [f for f in files if f.get("deletedAt") is None]

    # Apply path prefix filter
    if path_prefix:
      files = [f for f in files if f["path"].startswith(path_prefix)]

    # Apply tags filter
    if tags:
      files = [f for f in files if any(tag in f.get("tags", []) for tag in tags)]

    return files

  def read_file(
    self,
    owner: str,
    project: str,
    prefix: str,
    file_id: str,
    dest_path: Optional[str] = None,
  ) -> str:
    """
    Read/copy file from local storage.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        file_id: File ID
        dest_path: Optional destination path (defaults to original filename)

    Returns:
        Path to copied file

    Raises:
        FileNotFoundError: If file not found
        ValueError: If checksum verification fails
    """
    import shutil

    from .files import verify_checksum

    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    files_dir = experiment_dir / "files"
    metadata_file = files_dir / ".files_metadata.json"

    if not metadata_file.exists():
      raise FileNotFoundError(f"File {file_id} not found")

    # Find file metadata
    with open(metadata_file, "r") as f:
      files_metadata = json.load(f)

    file_metadata = None
    for f in files_metadata.get("files", []):
      if f["id"] == file_id and f.get("deletedAt") is None:
        file_metadata = f
        break

    if not file_metadata:
      raise FileNotFoundError(f"File {file_id} not found")

    # Get source file
    file_prefix = file_metadata["path"].lstrip("/") if file_metadata["path"] else ""
    if file_prefix:
      source_file = files_dir / file_prefix / file_id / file_metadata["filename"]
    else:
      source_file = files_dir / file_id / file_metadata["filename"]
    if not source_file.exists():
      raise FileNotFoundError(f"File {file_id} not found on disk")

    # Determine destination
    if dest_path is None:
      dest_path = file_metadata["filename"]

    # Copy file
    shutil.copy2(source_file, dest_path)

    # Verify checksum
    expected_checksum = file_metadata["checksum"]
    if not verify_checksum(dest_path, expected_checksum):
      import os

      os.remove(dest_path)
      raise ValueError(f"Checksum verification failed for file {file_id}")

    return dest_path

  def delete_file(
    self, owner: str, project: str, prefix: str, file_id: str
  ) -> Dict[str, Any]:
    """
    Delete file from local storage (soft delete in metadata).

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        file_id: File ID

    Returns:
        Dict with id and deletedAt

    Raises:
        FileNotFoundError: If file not found
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    files_dir = experiment_dir / "files"
    metadata_file = files_dir / ".files_metadata.json"

    if not metadata_file.exists():
      raise FileNotFoundError(f"File {file_id} not found")

    # File-based lock for concurrent safety (works across processes/threads/instances)
    lock_file = files_dir / ".files_metadata.lock"
    with self._file_lock(lock_file):
      # Read metadata
      with open(metadata_file, "r") as f:
        files_metadata = json.load(f)

      # Find and soft delete file
      file_found = False
      for file_meta in files_metadata.get("files", []):
        if file_meta["id"] == file_id:
          if file_meta.get("deletedAt") is not None:
            raise FileNotFoundError(f"File {file_id} already deleted")
          file_meta["deletedAt"] = datetime.utcnow().isoformat() + "Z"
          file_meta["updatedAt"] = file_meta["deletedAt"]
          file_found = True
          break

      if not file_found:
        raise FileNotFoundError(f"File {file_id} not found")

      # Write updated metadata
      with open(metadata_file, "w") as f:
        json.dump(files_metadata, f, indent=2)

    return {"id": file_id, "deletedAt": file_meta["deletedAt"]}

  def update_file_metadata(
    self,
    owner: str,
    project: str,
    prefix: str,
    file_id: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """
    Update file metadata in local storage.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        file_id: File ID
        description: Optional description
        tags: Optional tags
        metadata: Optional metadata

    Returns:
        Updated file metadata dict

    Raises:
        FileNotFoundError: If file not found
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    files_dir = experiment_dir / "files"
    metadata_file = files_dir / ".files_metadata.json"

    if not metadata_file.exists():
      raise FileNotFoundError(f"File {file_id} not found")

    # File-based lock for concurrent safety (works across processes/threads/instances)
    lock_file = files_dir / ".files_metadata.lock"
    with self._file_lock(lock_file):
      # Read metadata
      with open(metadata_file, "r") as f:
        files_metadata = json.load(f)

      # Find and update file
      file_found = False
      updated_file = None
      for file_meta in files_metadata.get("files", []):
        if file_meta["id"] == file_id:
          if file_meta.get("deletedAt") is not None:
            raise FileNotFoundError(f"File {file_id} has been deleted")

          # Update fields
          if description is not None:
            file_meta["description"] = description
          if tags is not None:
            file_meta["tags"] = tags
          if metadata is not None:
            file_meta["metadata"] = metadata

          file_meta["updatedAt"] = datetime.utcnow().isoformat() + "Z"
          file_found = True
          updated_file = file_meta
          break

      if not file_found:
        raise FileNotFoundError(f"File {file_id} not found")

      # Write updated metadata
      with open(metadata_file, "w") as f:
        json.dump(files_metadata, f, indent=2)

    return updated_file

  def _get_experiment_dir(self, owner: str, project: str, prefix: str) -> Path:
    """
    Get experiment directory path.

    Structure: root / prefix
    where prefix = owner/project/folder_1/.../exp_name
    """
    prefix_path = prefix.lstrip("/")
    return self.root_path / prefix_path

  def append_to_metric(
    self,
    owner: str,
    project: str,
    prefix: str,
    metric_name: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """
    Append a single data point to a metric in local storage.

    Storage format:
    .dash/{owner}/{project}/{prefix}/metrics/{metric_name}/
        data.jsonl  # Data points (one JSON object per line)
        metadata.json  # Metric metadata (name, description, tags, stats)

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        metric_name: Metric name (None for unnamed metrics)
        data: Data point (flexible schema)
        description: Optional metric description
        tags: Optional tags
        metadata: Optional metric metadata

    Returns:
        Dict with metricId, index, bufferedDataPoints, chunkSize
    """
    if data is None:
      data = {}
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Convert None to string for directory name
    dir_name = str(metric_name) if metric_name is not None else "None"
    metric_dir = metrics_dir / dir_name
    metric_dir.mkdir(exist_ok=True)

    data_file = metric_dir / "data.jsonl"
    metadata_file = metric_dir / "metadata.json"

    # File-based lock for concurrent metric appends (prevents index collision and count errors)
    lock_file = metric_dir / ".metadata.lock"
    with self._file_lock(lock_file):
      # Load or initialize metadata
      if metadata_file.exists():
        try:
          with open(metadata_file, "r") as f:
            metric_meta = json.load(f)
        except (json.JSONDecodeError, IOError):
          # Corrupted metadata, reinitialize
          metric_meta = {
            "metricId": f"local-metric-{metric_name}",
            "name": metric_name,
            "description": description,
            "tags": tags or [],
            "metadata": metadata,
            "totalDataPoints": 0,
            "nextIndex": 0,
            "createdAt": datetime.utcnow().isoformat() + "Z",
          }
      else:
        metric_meta = {
          "metricId": f"local-metric-{metric_name}",
          "name": metric_name,
          "description": description,
          "tags": tags or [],
          "metadata": metadata,
          "totalDataPoints": 0,
          "nextIndex": 0,
          "createdAt": datetime.utcnow().isoformat() + "Z",
        }

      # Get next index
      index = metric_meta["nextIndex"]

      # Append data point to JSONL file
      data_entry = {
        "index": index,
        "data": data,
        "createdAt": datetime.utcnow().isoformat() + "Z",
      }

      with open(data_file, "a") as f:
        f.write(json.dumps(data_entry) + "\n")

      # Update metadata
      metric_meta["nextIndex"] = index + 1
      metric_meta["totalDataPoints"] = metric_meta["totalDataPoints"] + 1
      metric_meta["updatedAt"] = datetime.utcnow().isoformat() + "Z"

      with open(metadata_file, "w") as f:
        json.dump(metric_meta, f, indent=2)

    return {
      "metricId": metric_meta["metricId"],
      "index": str(index),
      "bufferedDataPoints": str(metric_meta["totalDataPoints"]),
      "chunkSize": 10000,  # Default chunk size for local mode
    }

  def append_batch_to_metric(
    self,
    owner: str,
    project: str,
    prefix: str,
    metric_name: Optional[str],
    data_points: List[Dict[str, Any]],
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """
    Append multiple data points to a metric in local storage (batch).

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        metric_name: Metric name (None for unnamed metrics)
        data_points: List of data points
        description: Optional metric description
        tags: Optional tags
        metadata: Optional metric metadata

    Returns:
        Dict with metricId, startIndex, endIndex, count
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Convert None to string for directory name
    dir_name = str(metric_name) if metric_name is not None else "None"
    metric_dir = metrics_dir / dir_name
    metric_dir.mkdir(exist_ok=True)

    data_file = metric_dir / "data.jsonl"
    metadata_file = metric_dir / "metadata.json"

    # File-based lock for concurrent batch appends (prevents index collision and count errors)
    lock_file = metric_dir / ".metadata.lock"
    with self._file_lock(lock_file):
      # Load or initialize metadata
      if metadata_file.exists():
        try:
          with open(metadata_file, "r") as f:
            metric_meta = json.load(f)
        except (json.JSONDecodeError, IOError):
          # Corrupted metadata, reinitialize
          metric_meta = {
            "metricId": f"local-metric-{metric_name}",
            "name": metric_name,
            "description": description,
            "tags": tags or [],
            "metadata": metadata,
            "totalDataPoints": 0,
            "nextIndex": 0,
            "createdAt": datetime.utcnow().isoformat() + "Z",
          }
      else:
        metric_meta = {
          "metricId": f"local-metric-{metric_name}",
          "name": metric_name,
          "description": description,
          "tags": tags or [],
          "metadata": metadata,
          "totalDataPoints": 0,
          "nextIndex": 0,
          "createdAt": datetime.utcnow().isoformat() + "Z",
        }

      start_index = metric_meta["nextIndex"]
      end_index = start_index + len(data_points) - 1

      # Append data points to JSONL file
      with open(data_file, "a") as f:
        for i, data in enumerate(data_points):
          data_entry = {
            "index": start_index + i,
            "data": data,
            "createdAt": datetime.utcnow().isoformat() + "Z",
          }
          f.write(json.dumps(data_entry) + "\n")

      # Update metadata
      metric_meta["nextIndex"] = end_index + 1
      metric_meta["totalDataPoints"] = metric_meta["totalDataPoints"] + len(data_points)
      metric_meta["updatedAt"] = datetime.utcnow().isoformat() + "Z"

      with open(metadata_file, "w") as f:
        json.dump(metric_meta, f, indent=2)

    return {
      "metricId": metric_meta["metricId"],
      "startIndex": str(start_index),
      "endIndex": str(end_index),
      "count": len(data_points),
      "bufferedDataPoints": str(metric_meta["totalDataPoints"]),
      "chunkSize": 10000,
    }

  def read_metric_data(
    self,
    owner: str,
    project: str,
    prefix: str,
    metric_name: str,
    start_index: int = 0,
    limit: int = 1000,
  ) -> Dict[str, Any]:
    """
    Read data points from a metric in local storage.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        metric_name: Metric name
        start_index: Starting index
        limit: Max points to read

    Returns:
        Dict with data, startIndex, endIndex, total, hasMore
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    metric_dir = experiment_dir / "metrics" / metric_name
    data_file = metric_dir / "data.jsonl"

    if not data_file.exists():
      return {
        "data": [],
        "startIndex": start_index,
        "endIndex": start_index - 1,
        "total": 0,
        "hasMore": False,
      }

    # Read all data points from JSONL file
    data_points = []
    with open(data_file, "r") as f:
      for line in f:
        if line.strip():
          entry = json.loads(line)
          # Filter by index range
          if start_index <= entry["index"] < start_index + limit:
            data_points.append(entry)

    # Get total count
    metadata_file = metric_dir / "metadata.json"
    total_count = 0
    if metadata_file.exists():
      with open(metadata_file, "r") as f:
        metric_meta = json.load(f)
        total_count = metric_meta["totalDataPoints"]

    return {
      "data": data_points,
      "startIndex": start_index,
      "endIndex": start_index + len(data_points) - 1
      if data_points
      else start_index - 1,
      "total": len(data_points),
      "hasMore": start_index + len(data_points) < total_count,
    }

  def get_metric_stats(
    self, owner: str, project: str, prefix: str, metric_name: str
  ) -> Dict[str, Any]:
    """
    Get metric statistics from local storage.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        metric_name: Metric name

    Returns:
        Dict with metric stats
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    metric_dir = experiment_dir / "metrics" / metric_name
    metadata_file = metric_dir / "metadata.json"

    if not metadata_file.exists():
      raise FileNotFoundError(f"Metric {metric_name} not found")

    with open(metadata_file, "r") as f:
      metric_meta = json.load(f)

    return {
      "metricId": metric_meta["metricId"],
      "name": metric_meta["name"],
      "description": metric_meta.get("description"),
      "tags": metric_meta.get("tags", []),
      "metadata": metric_meta.get("metadata"),
      "totalDataPoints": str(metric_meta["totalDataPoints"]),
      "bufferedDataPoints": str(
        metric_meta["totalDataPoints"]
      ),  # All buffered in local mode
      "chunkedDataPoints": "0",  # No chunking in local mode
      "totalChunks": 0,
      "chunkSize": 10000,
      "firstDataAt": metric_meta.get("createdAt"),
      "lastDataAt": metric_meta.get("updatedAt"),
      "createdAt": metric_meta.get("createdAt"),
      "updatedAt": metric_meta.get("updatedAt", metric_meta.get("createdAt")),
    }

  def list_metrics(self, owner: str, project: str, prefix: str) -> List[Dict[str, Any]]:
    """
    List all metrics in an experiment from local storage.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix

    Returns:
        List of metric summaries
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    metrics_dir = experiment_dir / "metrics"

    if not metrics_dir.exists():
      return []

    metrics = []
    for metric_dir in metrics_dir.iterdir():
      if metric_dir.is_dir():
        metadata_file = metric_dir / "metadata.json"
        if metadata_file.exists():
          with open(metadata_file, "r") as f:
            metric_meta = json.load(f)
            metrics.append(
              {
                "metricId": metric_meta["metricId"],
                "name": metric_meta["name"],
                "description": metric_meta.get("description"),
                "tags": metric_meta.get("tags", []),
                "totalDataPoints": str(metric_meta["totalDataPoints"]),
                "createdAt": metric_meta.get("createdAt"),
              }
            )

    return metrics

  # ============================================================================
  # Track Storage Methods
  # ============================================================================

  def _serialize_value(self, value: Any) -> Any:
    """
    Convert value to JSON-serializable format.

    Handles numpy arrays, nested dicts/lists, etc.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable value
    """
    # Check for numpy array
    if hasattr(value, '__array__') or (hasattr(value, 'tolist') and hasattr(value, 'dtype')):
      # It's a numpy array
      try:
        return value.tolist()
      except AttributeError:
        pass

    # Check for numpy scalar types
    if hasattr(value, 'item'):
      try:
        return value.item()
      except (AttributeError, ValueError):
        pass

    # Recursively handle dicts
    if isinstance(value, dict):
      return {k: self._serialize_value(v) for k, v in value.items()}

    # Recursively handle lists
    if isinstance(value, (list, tuple)):
      return [self._serialize_value(v) for v in value]

    # Return as-is for other types (int, float, str, bool, None)
    return value

  def _flatten_dict(self, obj: Any, prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested dict with dot notation (e.g., camera.pos).

    Args:
        obj: Object to flatten
        prefix: Current key prefix

    Returns:
        Flattened dict
    """
    result = {}

    if not isinstance(obj, dict):
      # Serialize the value before returning
      serialized = self._serialize_value(obj)
      return {prefix: serialized} if prefix else serialized

    for key, value in obj.items():
      new_key = f"{prefix}.{key}" if prefix else key

      if isinstance(value, dict):
        result.update(self._flatten_dict(value, new_key))
      else:
        # Serialize the value
        result[new_key] = self._serialize_value(value)

    return result

  def append_batch_to_track(
    self,
    owner: str,
    project: str,
    prefix: str,
    topic: str,
    entries: List[Dict[str, Any]],
  ) -> Dict[str, Any]:
    """
    Append batch of timestamped entries to a track in local storage.

    Storage format:
    .dash/{owner}/{project}/{prefix}/tracks/{topic_safe}/
        data.jsonl  # Timestamped entries (one JSON object per line)
        metadata.json  # Track metadata (topic, columns, stats)

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        topic: Track topic (e.g., "robot/position")
        entries: List of entries with timestamp and data fields

    Returns:
        Dict with trackId, count
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    tracks_dir = experiment_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize topic for directory name (replace / with _)
    topic_safe = topic.replace("/", "_")
    track_dir = tracks_dir / topic_safe
    track_dir.mkdir(exist_ok=True)

    data_file = track_dir / "data.jsonl"
    metadata_file = track_dir / "metadata.json"

    # File-based lock for concurrent writes
    lock_file = track_dir / ".metadata.lock"
    with self._file_lock(lock_file):
      # Load or initialize metadata
      if metadata_file.exists():
        try:
          with open(metadata_file, "r") as f:
            track_meta = json.load(f)
        except (json.JSONDecodeError, IOError):
          # Corrupted metadata, reinitialize
          track_meta = {
            "trackId": f"local-track-{topic_safe}",
            "topic": topic,
            "columns": [],
            "totalEntries": 0,
            "firstTimestamp": None,
            "lastTimestamp": None,
            "createdAt": datetime.utcnow().isoformat() + "Z",
          }
      else:
        track_meta = {
          "trackId": f"local-track-{topic_safe}",
          "topic": topic,
          "columns": [],
          "totalEntries": 0,
          "firstTimestamp": None,
          "lastTimestamp": None,
          "createdAt": datetime.utcnow().isoformat() + "Z",
        }

      # Process entries and update metadata
      all_columns = set(track_meta["columns"])
      min_ts = track_meta["firstTimestamp"]
      max_ts = track_meta["lastTimestamp"]

      processed_entries = []
      for entry in entries:
        timestamp = entry.get("timestamp")
        if timestamp is None:
          continue

        # Extract data fields (everything except timestamp)
        data_fields = {k: v for k, v in entry.items() if k != "timestamp"}

        # Flatten nested structures
        flattened = self._flatten_dict(data_fields)

        # Update column set
        all_columns.update(flattened.keys())

        # Update timestamp range
        if min_ts is None or timestamp < min_ts:
          min_ts = timestamp
        if max_ts is None or timestamp > max_ts:
          max_ts = timestamp

        processed_entries.append({
          "timestamp": timestamp,
          **flattened
        })

      # Append entries to JSONL file (sorted by timestamp for consistency)
      processed_entries.sort(key=lambda x: x["timestamp"])
      with open(data_file, "a") as f:
        for entry in processed_entries:
          f.write(json.dumps(entry) + "\n")

      # Update metadata
      track_meta["columns"] = sorted(list(all_columns))
      track_meta["totalEntries"] += len(processed_entries)
      track_meta["firstTimestamp"] = min_ts
      track_meta["lastTimestamp"] = max_ts

      # Write metadata
      with open(metadata_file, "w") as f:
        json.dump(track_meta, f, indent=2)

      return {
        "trackId": track_meta["trackId"],
        "count": len(processed_entries),
      }

  def read_track_data(
    self,
    owner: str,
    project: str,
    prefix: str,
    topic: str,
    start_timestamp: Optional[float] = None,
    end_timestamp: Optional[float] = None,
    columns: Optional[List[str]] = None,
    format: str = "json",
  ) -> Any:
    """
    Read track data from local storage with optional filtering.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        topic: Track topic
        start_timestamp: Optional start timestamp filter
        end_timestamp: Optional end timestamp filter
        columns: Optional list of columns to retrieve
        format: Export format ('json', 'jsonl', 'parquet', 'mocap')

    Returns:
        Track data in requested format
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    topic_safe = topic.replace("/", "_")
    track_dir = experiment_dir / "tracks" / topic_safe
    data_file = track_dir / "data.jsonl"

    if not data_file.exists():
      if format == "json":
        return {"entries": [], "count": 0}
      elif format == "jsonl":
        return b""
      elif format == "parquet":
        # Return empty parquet file
        import pyarrow as pa
        import pyarrow.parquet as pq
        import io
        table = pa.table({"timestamp": []})
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue()
      elif format == "mocap":
        return {
          "version": "1.0",
          "metadata": {"topic": topic, "frameCount": 0, "duration": 0},
          "channels": [],
          "frames": []
        }

    # Read all entries from JSONL file
    entries = []
    with open(data_file, "r") as f:
      for line in f:
        if line.strip():
          entry = json.loads(line)

          # Filter by timestamp range
          timestamp = entry.get("timestamp")
          if start_timestamp is not None and timestamp < start_timestamp:
            continue
          if end_timestamp is not None and timestamp > end_timestamp:
            continue

          # Filter by columns
          if columns:
            filtered_entry = {"timestamp": timestamp}
            for col in columns:
              if col in entry:
                filtered_entry[col] = entry[col]
            entries.append(filtered_entry)
          else:
            entries.append(entry)

    # Return in requested format
    if format == "json":
      return {"entries": entries, "count": len(entries)}

    elif format == "jsonl":
      lines = [json.dumps(entry) for entry in entries]
      return "\n".join(lines).encode('utf-8')

    elif format == "parquet":
      # Convert to Apache Parquet
      import pyarrow as pa
      import pyarrow.parquet as pq
      import io

      if not entries:
        table = pa.table({"timestamp": []})
      else:
        # Build schema from entries
        table = pa.Table.from_pylist(entries)

      buf = io.BytesIO()
      pq.write_table(table, buf, compression='zstd')
      return buf.getvalue()

    elif format == "mocap":
      # Read metadata
      metadata_file = track_dir / "metadata.json"
      track_meta = {}
      if metadata_file.exists():
        with open(metadata_file, "r") as f:
          track_meta = json.load(f)

      # Build mocap format
      if not entries:
        return {
          "version": "1.0",
          "metadata": {
            "topic": topic,
            "frameCount": 0,
            "duration": 0,
            "startTime": 0,
            "endTime": 0,
          },
          "channels": [],
          "frames": []
        }

      first_ts = entries[0]["timestamp"]
      last_ts = entries[-1]["timestamp"]
      duration = last_ts - first_ts
      fps = track_meta.get("metadata", {}).get("fps", 30) if isinstance(track_meta.get("metadata"), dict) else 30

      # Get all channels (columns)
      all_channels = set()
      for entry in entries:
        all_channels.update(k for k in entry.keys() if k != "timestamp")

      return {
        "version": "1.0",
        "metadata": {
          "topic": topic,
          "description": track_meta.get("description"),
          "tags": track_meta.get("tags", []),
          "fps": fps,
          "duration": duration,
          "frameCount": len(entries),
          "startTime": first_ts,
          "endTime": last_ts,
        },
        "channels": sorted(list(all_channels)),
        "frames": [{"time": e["timestamp"], **{k: v for k, v in e.items() if k != "timestamp"}} for e in entries]
      }

    else:
      raise ValueError(f"Unsupported format: {format}")

  def list_tracks(
    self,
    owner: str,
    project: str,
    prefix: str,
    topic_filter: Optional[str] = None,
  ) -> List[Dict[str, Any]]:
    """
    List all tracks in an experiment.

    Args:
        owner: Owner/user
        project: Project name
        prefix: Experiment prefix
        topic_filter: Optional topic filter (e.g., "robot/*")

    Returns:
        List of track summaries
    """
    experiment_dir = self._get_experiment_dir(owner, project, prefix)
    tracks_dir = experiment_dir / "tracks"

    if not tracks_dir.exists():
      return []

    tracks = []
    for track_dir in tracks_dir.iterdir():
      if track_dir.is_dir():
        metadata_file = track_dir / "metadata.json"
        if metadata_file.exists():
          with open(metadata_file, "r") as f:
            track_meta = json.load(f)

            topic = track_meta["topic"]

            # Apply topic filter
            if topic_filter:
              if topic_filter.endswith("/*"):
                # Prefix match
                prefix_match = topic_filter[:-2]
                if not topic.startswith(prefix_match):
                  continue
              elif topic != topic_filter:
                # Exact match
                continue

            tracks.append({
              "id": track_meta["trackId"],
              "topic": topic,
              "totalEntries": track_meta["totalEntries"],
              "firstTimestamp": track_meta.get("firstTimestamp"),
              "lastTimestamp": track_meta.get("lastTimestamp"),
              "columns": track_meta.get("columns", []),
              "createdAt": track_meta.get("createdAt"),
            })

    return tracks
