"""
Local filesystem storage for ML-Dash.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from datetime import datetime


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

    def create_experiment(
        self,
        project: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bindrs: Optional[List[str]] = None,
        folder: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Create a experiment directory structure.

        Args:
            project: Project name
            name: Experiment name
            description: Optional description
            tags: Optional tags
            bindrs: Optional bindrs
            folder: Optional folder path (used for organization)
            metadata: Optional metadata

        Returns:
            Path to experiment directory
        """
        # Determine base path - include folder in hierarchy if specified
        if folder is not None:
            # Strip leading / to make it relative, then use as base path
            folder_path = folder.lstrip('/')
            base_path = self.root_path / folder_path
        else:
            base_path = self.root_path

        # Create project directory
        project_dir = base_path / project
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment directory
        experiment_dir = project_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (experiment_dir / "logs").mkdir(exist_ok=True)
        (experiment_dir / "metrics").mkdir(exist_ok=True)
        (experiment_dir / "files").mkdir(exist_ok=True)

        # Write experiment metadata
        experiment_metadata = {
            "name": name,
            "project": project,
            "description": description,
            "tags": tags or [],
            "bindrs": bindrs or [],
            "folder": folder,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "write_protected": False,
        }

        experiment_file = experiment_dir / "experiment.json"
        if not experiment_file.exists():
            # Only create if doesn't exist (don't overwrite)
            with open(experiment_file, "w") as f:
                json.dump(experiment_metadata, f, indent=2)
        else:
            # Update existing experiment
            with open(experiment_file, "r") as f:
                existing = json.load(f)
            # Merge updates
            if description is not None:
                existing["description"] = description
            if tags is not None:
                existing["tags"] = tags
            if bindrs is not None:
                existing["bindrs"] = bindrs
            if folder is not None:
                existing["folder"] = folder
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
        project: str,
        experiment: str,
        message: str,
        level: str,
        timestamp: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Write a single log entry immediately to JSONL file.

        Args:
            project: Project name
            experiment: Experiment name
            message: Log message
            level: Log level
            timestamp: ISO timestamp string
            metadata: Optional metadata
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        logs_dir = experiment_dir / "logs"
        logs_file = logs_dir / "logs.jsonl"
        seq_file = logs_dir / ".log_sequence"

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
        project: str,
        experiment: str,
        data: Dict[str, Any],
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
            project: Project name
            experiment: Experiment name
            data: Flattened parameter dict with dot notation (already flattened)
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        params_file = experiment_dir / "parameters.json"

        # Read existing if present
        if params_file.exists():
            with open(params_file, "r") as f:
                existing_doc = json.load(f)

            # Merge with existing data
            existing_data = existing_doc.get("data", {})
            existing_data.update(data)

            # Increment version
            version = existing_doc.get("version", 1) + 1

            params_doc = {
                "version": version,
                "data": existing_data,
                "updatedAt": datetime.utcnow().isoformat() + "Z"
            }
        else:
            # Create new parameters document
            params_doc = {
                "version": 1,
                "data": data,
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "updatedAt": datetime.utcnow().isoformat() + "Z"
            }

        with open(params_file, "w") as f:
            json.dump(params_doc, f, indent=2)

    def read_parameters(
        self,
        project: str,
        experiment: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read parameters from local file.

        Args:
            project: Project name
            experiment: Experiment name

        Returns:
            Flattened parameter dict, or None if file doesn't exist
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
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
        project: str,
        experiment: str,
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
        Write file to local storage.

        Copies file to: files/<prefix>/<file_id>/<filename>
        Updates .files_metadata.json with file metadata

        Args:
            project: Project name
            experiment: Experiment name
            file_path: Source file path
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
        import shutil
        from .files import generate_snowflake_id

        experiment_dir = self._get_experiment_dir(project, experiment)
        files_dir = experiment_dir / "files"
        metadata_file = files_dir / ".files_metadata.json"

        # Generate Snowflake ID for file
        file_id = generate_snowflake_id()

        # Normalize prefix (remove leading slashes to avoid absolute paths)
        normalized_prefix = prefix.lstrip("/") if prefix else ""

        # Create prefix directory, then file directory
        prefix_dir = files_dir / normalized_prefix if normalized_prefix else files_dir
        prefix_dir.mkdir(parents=True, exist_ok=True)

        file_dir = prefix_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        # Copy file
        dest_file = file_dir / filename
        shutil.copy2(file_path, dest_file)

        # Create file metadata
        file_metadata = {
            "id": file_id,
            "experimentId": f"{project}/{experiment}",  # Local mode doesn't have real experiment ID
            "path": prefix,
            "filename": filename,
            "description": description,
            "tags": tags or [],
            "contentType": content_type,
            "sizeBytes": size_bytes,
            "checksum": checksum,
            "metadata": metadata,
            "uploadedAt": datetime.utcnow().isoformat() + "Z",
            "updatedAt": datetime.utcnow().isoformat() + "Z",
            "deletedAt": None
        }

        # Read existing metadata
        files_metadata = {"files": []}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    files_metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                files_metadata = {"files": []}

        # Check if file with same prefix+filename exists (overwrite behavior)
        existing_index = None
        for i, existing_file in enumerate(files_metadata["files"]):
            if (existing_file["path"] == prefix and
                existing_file["filename"] == filename and
                existing_file["deletedAt"] is None):
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
        project: str,
        experiment: str,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List files from local storage.

        Args:
            project: Project name
            experiment: Experiment name
            prefix: Optional prefix filter
            tags: Optional tags filter

        Returns:
            List of file metadata dicts (only non-deleted files)
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        metadata_file = experiment_dir / "files" / ".files_metadata.json"

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

        # Apply prefix filter
        if prefix:
            files = [f for f in files if f["path"].startswith(prefix)]

        # Apply tags filter
        if tags:
            files = [f for f in files if any(tag in f.get("tags", []) for tag in tags)]

        return files

    def read_file(
        self,
        project: str,
        experiment: str,
        file_id: str,
        dest_path: Optional[str] = None
    ) -> str:
        """
        Read/copy file from local storage.

        Args:
            project: Project name
            experiment: Experiment name
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

        experiment_dir = self._get_experiment_dir(project, experiment)
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
        self,
        project: str,
        experiment: str,
        file_id: str
    ) -> Dict[str, Any]:
        """
        Delete file from local storage (soft delete in metadata).

        Args:
            project: Project name
            experiment: Experiment name
            file_id: File ID

        Returns:
            Dict with id and deletedAt

        Raises:
            FileNotFoundError: If file not found
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        metadata_file = experiment_dir / "files" / ".files_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"File {file_id} not found")

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

        return {
            "id": file_id,
            "deletedAt": file_meta["deletedAt"]
        }

    def update_file_metadata(
        self,
        project: str,
        experiment: str,
        file_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update file metadata in local storage.

        Args:
            project: Project name
            experiment: Experiment name
            file_id: File ID
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Updated file metadata dict

        Raises:
            FileNotFoundError: If file not found
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        metadata_file = experiment_dir / "files" / ".files_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"File {file_id} not found")

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

    def _get_experiment_dir(self, project: str, experiment: str, folder: Optional[str] = None) -> Path:
        """
        Get experiment directory path.

        If folder is not provided, tries to read it from experiment.json metadata.
        Falls back to root_path/project/experiment if not found.
        """
        # If folder explicitly provided, use it
        if folder is not None:
            folder_path = folder.lstrip('/')
            return self.root_path / folder_path / project / experiment

        # Try to read folder from experiment metadata
        # Check common locations where experiment might exist
        possible_paths = []

        # First, try without folder (most common case)
        default_path = self.root_path / project / experiment
        possible_paths.append(default_path)

        # Then scan for experiment.json in subdirectories (for folder-based experiments)
        try:
            for item in self.root_path.rglob(f"*/{project}/{experiment}/experiment.json"):
                exp_dir = item.parent
                if exp_dir not in [p for p in possible_paths]:
                    possible_paths.insert(0, exp_dir)  # Prioritize found paths
        except:
            pass

        # Check each possible path for experiment.json with folder metadata
        for path in possible_paths:
            exp_json = path / "experiment.json"
            if exp_json.exists():
                try:
                    with open(exp_json, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get('folder'):
                            folder_path = metadata['folder'].lstrip('/')
                            return self.root_path / folder_path / project / experiment
                except:
                    pass
                # Found experiment.json, use this path even if no folder metadata
                return path

        # Fallback to default path
        return default_path

    def append_to_metric(
        self,
        project: str,
        experiment: str,
        metric_name: Optional[str],
        data: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Append a single data point to a metric in local storage.

        Storage format:
        .ml-dash/{project}/{experiment}/metrics/{metric_name}/
            data.jsonl  # Data points (one JSON object per line)
            metadata.json  # Metric metadata (name, description, tags, stats)

        Args:
            project: Project name
            experiment: Experiment name
            metric_name: Metric name (None for unnamed metrics)
            data: Data point (flexible schema)
            description: Optional metric description
            tags: Optional tags
            metadata: Optional metric metadata

        Returns:
            Dict with metricId, index, bufferedDataPoints, chunkSize
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        metrics_dir = experiment_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Convert None to string for directory name
        dir_name = str(metric_name) if metric_name is not None else "None"
        metric_dir = metrics_dir / dir_name
        metric_dir.mkdir(exist_ok=True)

        data_file = metric_dir / "data.jsonl"
        metadata_file = metric_dir / "metadata.json"

        # Load or initialize metadata
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metric_meta = json.load(f)
        else:
            metric_meta = {
                "metricId": f"local-metric-{metric_name}",
                "name": metric_name,
                "description": description,
                "tags": tags or [],
                "metadata": metadata,
                "totalDataPoints": 0,
                "nextIndex": 0,
                "createdAt": datetime.utcnow().isoformat() + "Z"
            }

        # Get next index
        index = metric_meta["nextIndex"]

        # Append data point to JSONL file
        data_entry = {
            "index": index,
            "data": data,
            "createdAt": datetime.utcnow().isoformat() + "Z"
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
            "chunkSize": 10000  # Default chunk size for local mode
        }

    def append_batch_to_metric(
        self,
        project: str,
        experiment: str,
        metric_name: Optional[str],
        data_points: List[Dict[str, Any]],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Append multiple data points to a metric in local storage (batch).

        Args:
            project: Project name
            experiment: Experiment name
            metric_name: Metric name (None for unnamed metrics)
            data_points: List of data points
            description: Optional metric description
            tags: Optional tags
            metadata: Optional metric metadata

        Returns:
            Dict with metricId, startIndex, endIndex, count
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        metrics_dir = experiment_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Convert None to string for directory name
        dir_name = str(metric_name) if metric_name is not None else "None"
        metric_dir = metrics_dir / dir_name
        metric_dir.mkdir(exist_ok=True)

        data_file = metric_dir / "data.jsonl"
        metadata_file = metric_dir / "metadata.json"

        # Load or initialize metadata
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metric_meta = json.load(f)
        else:
            metric_meta = {
                "metricId": f"local-metric-{metric_name}",
                "name": metric_name,
                "description": description,
                "tags": tags or [],
                "metadata": metadata,
                "totalDataPoints": 0,
                "nextIndex": 0,
                "createdAt": datetime.utcnow().isoformat() + "Z"
            }

        start_index = metric_meta["nextIndex"]
        end_index = start_index + len(data_points) - 1

        # Append data points to JSONL file
        with open(data_file, "a") as f:
            for i, data in enumerate(data_points):
                data_entry = {
                    "index": start_index + i,
                    "data": data,
                    "createdAt": datetime.utcnow().isoformat() + "Z"
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
            "chunkSize": 10000
        }

    def read_metric_data(
        self,
        project: str,
        experiment: str,
        metric_name: str,
        start_index: int = 0,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Read data points from a metric in local storage.

        Args:
            project: Project name
            experiment: Experiment name
            metric_name: Metric name
            start_index: Starting index
            limit: Max points to read

        Returns:
            Dict with data, startIndex, endIndex, total, hasMore
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
        metric_dir = experiment_dir / "metrics" / metric_name
        data_file = metric_dir / "data.jsonl"

        if not data_file.exists():
            return {
                "data": [],
                "startIndex": start_index,
                "endIndex": start_index - 1,
                "total": 0,
                "hasMore": False
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
            "endIndex": start_index + len(data_points) - 1 if data_points else start_index - 1,
            "total": len(data_points),
            "hasMore": start_index + len(data_points) < total_count
        }

    def get_metric_stats(
        self,
        project: str,
        experiment: str,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Get metric statistics from local storage.

        Args:
            project: Project name
            experiment: Experiment name
            metric_name: Metric name

        Returns:
            Dict with metric stats
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
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
            "bufferedDataPoints": str(metric_meta["totalDataPoints"]),  # All buffered in local mode
            "chunkedDataPoints": "0",  # No chunking in local mode
            "totalChunks": 0,
            "chunkSize": 10000,
            "firstDataAt": metric_meta.get("createdAt"),
            "lastDataAt": metric_meta.get("updatedAt"),
            "createdAt": metric_meta.get("createdAt"),
            "updatedAt": metric_meta.get("updatedAt", metric_meta.get("createdAt"))
        }

    def list_metrics(
        self,
        project: str,
        experiment: str
    ) -> List[Dict[str, Any]]:
        """
        List all metrics in an experiment from local storage.

        Args:
            project: Project name
            experiment: Experiment name

        Returns:
            List of metric summaries
        """
        experiment_dir = self._get_experiment_dir(project, experiment)
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
                        metrics.append({
                            "metricId": metric_meta["metricId"],
                            "name": metric_meta["name"],
                            "description": metric_meta.get("description"),
                            "tags": metric_meta.get("tags", []),
                            "totalDataPoints": str(metric_meta["totalDataPoints"]),
                            "createdAt": metric_meta.get("createdAt")
                        })

        return metrics
