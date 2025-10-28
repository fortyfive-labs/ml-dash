"""ML-Dash server storage backend for ML-Logger.

This backend syncs data to an ML-Dash server via HTTP API.
"""

import json
import time
from typing import Optional, List, Dict, Any
import requests

from .base import StorageBackend


class DashBackend(StorageBackend):
    """ML-Dash server storage backend.

    Syncs data to a remote ML-Dash server via HTTP API.

    Args:
        server_url: URL of the ML-Dash server (e.g., "http://localhost:4000")
        namespace: User/team namespace
        workspace: Project workspace
        experiment_name: Name of the experiment
        experiment_id: Server-side experiment ID (optional, will be created if not provided)
        run_id: Server-side run ID (optional, will be created when needed)
        directory: Directory path for organizing experiments (optional)
    """

    def __init__(
        self,
        server_url: str,
        namespace: str,
        workspace: str,
        experiment_name: str,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        directory: Optional[str] = None,
    ):
        """Initialize ML-Dash backend.

        Args:
            server_url: URL of the ML-Dash server
            namespace: User/team namespace
            workspace: Project workspace
            experiment_name: Name of the experiment
            experiment_id: Server-side experiment ID (optional)
            run_id: Server-side run ID (optional)
            directory: Directory path for organizing experiments (e.g., "dir1/dir2")
        """
        self.server_url = server_url.rstrip("/")
        self.namespace = namespace
        self.workspace = workspace
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.namespace_id: Optional[str] = None
        self.run_id = run_id
        self.directory = directory
        self._session = requests.Session()

    def initialize_experiment(self, description: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create or get the experiment on the server.

        Args:
            description: Experiment description
            tags: Experiment tags

        Returns:
            Experiment data from server
        """
        url = f"{self.server_url}/api/v1/experiments"
        data = {
            "namespace": self.namespace,
            "workspace": self.workspace,
            "experimentName": self.experiment_name,
        }

        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags
        if self.directory:
            data["directory"] = self.directory

        response = self._session.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            experiment = result.get("experiment", {})
            self.experiment_id = experiment.get("id")
            self.namespace_id = experiment.get("namespaceId")
            return experiment
        else:
            raise Exception(f"Failed to create experiment: {result}")

    def create_run(self, name: Optional[str] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new run on the server.

        Args:
            name: Run name
            tags: Run tags
            metadata: Run metadata

        Returns:
            Run data from server
        """
        if not self.experiment_id or not self.namespace_id:
            raise Exception("Must call initialize_experiment() before create_run()")

        url = f"{self.server_url}/api/v1/runs"
        data = {
            "experimentId": self.experiment_id,
            "namespaceId": self.namespace_id,
        }

        if name:
            data["name"] = name
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata

        response = self._session.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            run = result.get("run", {})
            self.run_id = run.get("id")
            return run
        else:
            raise Exception(f"Failed to create run: {result}")

    def update_run(self, status: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Update the run status.

        Args:
            status: Run status (RUNNING, COMPLETED, FAILED, STOPPED)
            metadata: Additional metadata

        Returns:
            Updated run data from server
        """
        if not self.run_id:
            raise Exception("No run_id available. Call create_run() first")

        url = f"{self.server_url}/api/v1/runs/{self.run_id}"
        data = {}

        if status:
            data["status"] = status
        if metadata:
            data["metadata"] = metadata
        if status in ["COMPLETED", "FAILED", "STOPPED"]:
            data["endedAt"] = time.time() * 1000  # Convert to milliseconds

        response = self._session.put(url, json=data)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            return result.get("run", {})
        else:
            raise Exception(f"Failed to update run: {result}")

    def log_metrics(self, metrics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Log metrics to the server.

        Args:
            metrics: Dictionary mapping metric names to lists of {step, timestamp, value} dicts

        Returns:
            Server response
        """
        if not self.experiment_id or not self.run_id:
            raise Exception("Must initialize experiment and create run before logging metrics")

        url = f"{self.server_url}/api/v1/metrics"
        data = {
            "experimentId": self.experiment_id,
            "runId": self.run_id,
            "metrics": metrics,
        }

        response = self._session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def log_parameters(self, parameters: Dict[str, Any], operation: str = "set") -> Dict[str, Any]:
        """Log parameters to the server.

        Args:
            parameters: Parameter dictionary
            operation: Operation type (set, extend, update)

        Returns:
            Server response
        """
        if not self.namespace_id:
            raise Exception("Must initialize experiment before logging parameters")

        url = f"{self.server_url}/api/v1/parameters"
        data = {
            "namespaceId": self.namespace_id,
            "parameters": parameters,
            "operation": operation,
        }

        if self.experiment_id:
            data["experimentId"] = self.experiment_id
        if self.run_id:
            data["runId"] = self.run_id

        response = self._session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def log_entry(self, level: str, message: str, metadata: Optional[Dict] = None, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Log a text entry to the server.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            metadata: Additional metadata
            timestamp: Timestamp (defaults to current time)

        Returns:
            Server response
        """
        if not self.run_id:
            raise Exception("Must create run before logging entries")

        url = f"{self.server_url}/api/v1/logs"
        data = {
            "runId": self.run_id,
            "level": level.upper(),
            "message": message,
        }

        if timestamp:
            data["timestamp"] = timestamp * 1000  # Convert to milliseconds
        if metadata:
            data["metadata"] = metadata

        response = self._session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def upload_file(self, name: str, file_data: bytes, artifact_type: str = "OTHER", mime_type: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Upload a file to the server.

        Args:
            name: File name
            file_data: File content as bytes
            artifact_type: Type of artifact (IMAGE, VIDEO, MODEL, etc.)
            mime_type: MIME type
            metadata: Additional metadata

        Returns:
            Server response with artifact info
        """
        if not self.run_id:
            raise Exception("Must create run before uploading files")

        url = f"{self.server_url}/api/v1/files"

        # Prepare form data
        files = {"file": (name, file_data, mime_type or "application/octet-stream")}
        data = {
            "runId": self.run_id,
            "name": name,
            "type": artifact_type,
        }

        if metadata:
            data["metadata"] = json.dumps(metadata)

        response = self._session.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()

    # StorageBackend interface methods

    def exists(self, path: str) -> bool:
        """Check if a file exists (not fully supported in remote backend)."""
        # For logs.jsonl, check if we have a run_id (meaning logs can be fetched)
        if "logs.jsonl" in path:
            return self.run_id is not None

        # For metrics.jsonl and parameters.jsonl, assume they can be read
        if "metrics.jsonl" in path or "parameters.jsonl" in path:
            return self.run_id is not None or self.experiment_id is not None

        # For other files, assume they don't exist locally
        return False

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write binary data to the server as a file."""
        # Extract filename from path
        filename = path.split("/")[-1]

        # Determine artifact type from extension
        artifact_type = "OTHER"
        if any(path.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]):
            artifact_type = "IMAGE"
        elif any(path.endswith(ext) for ext in [".mp4", ".avi", ".mov"]):
            artifact_type = "VIDEO"
        elif any(path.endswith(ext) for ext in [".pt", ".pth", ".h5", ".pkl"]):
            artifact_type = "MODEL"
        elif any(path.endswith(ext) for ext in [".json"]):
            artifact_type = "JSON"
        elif any(path.endswith(ext) for ext in [".yaml", ".yml"]):
            artifact_type = "YAML"
        elif any(path.endswith(ext) for ext in [".md"]):
            artifact_type = "MARKDOWN"
        elif any(path.endswith(ext) for ext in [".csv"]):
            artifact_type = "CSV"

        self.upload_file(filename, data, artifact_type=artifact_type)

    def read_bytes(self, path: str) -> bytes:
        """Read binary data from the server."""
        # Not implemented - files are stored in S3, would need presigned URLs
        raise NotImplementedError("DashBackend.read_bytes() not implemented - files are stored in S3")

    def write_text(self, path: str, text: str) -> None:
        """Write text to the server."""
        self.write_bytes(path, text.encode("utf-8"))

    def read_text(self, path: str) -> str:
        """Read text from the server.

        For logs.jsonl, metrics.jsonl, and parameters.jsonl, fetches data from server
        and formats as JSONL.
        """
        # Handle logs.jsonl
        if "logs.jsonl" in path:
            if not self.run_id:
                return ""

            url = f"{self.server_url}/api/v1/runs/{self.run_id}/logs"
            response = self._session.get(url)
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise Exception(f"Failed to fetch logs: {result}")

            # Convert logs to JSONL format
            logs = result.get("logs", [])
            lines = []
            for log in logs:
                # Convert to the format used by local backend
                entry = {
                    "timestamp": log.get("timestamp"),
                    "level": log.get("level"),
                    "message": log.get("message"),
                }
                if log.get("metadata"):
                    entry["context"] = log["metadata"]

                lines.append(json.dumps(entry))

            return "\n".join(lines)

        # Handle parameters.jsonl
        if "parameters.jsonl" in path:
            if not self.run_id and not self.experiment_id:
                return ""

            url = f"{self.server_url}/api/v1/parameters"
            params = {}
            if self.run_id:
                params["runId"] = self.run_id
            elif self.experiment_id:
                params["experimentId"] = self.experiment_id

            response = self._session.get(url, params=params)
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise Exception(f"Failed to fetch parameters: {result}")

            # The server stores only the final state, not operation history
            # We return a single 'set' operation with the merged data
            parameters_list = result.get("parameters", [])

            if not parameters_list:
                return ""

            # Merge all parameter data (in case there are multiple records)
            merged_data = {}
            for param in parameters_list:
                param_data = param.get("data", {})
                merged_data.update(param_data)

            # Return as a single set operation
            entry = {
                "timestamp": time.time(),
                "operation": "set",
                "data": merged_data,
            }

            return json.dumps(entry)

        # Handle metrics.jsonl
        if "metrics.jsonl" in path:
            if not self.experiment_id:
                return ""

            url = f"{self.server_url}/api/v1/experiments/{self.experiment_id}/metrics"
            response = self._session.get(url)
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise Exception(f"Failed to fetch metrics: {result}")

            # Convert metrics to JSONL format, grouped by step/timestamp
            metrics_list = result.get("metrics", [])

            # Group metrics by (step, timestamp)
            grouped = {}

            for metric in metrics_list:
                metric_name = metric.get("name")
                data = metric.get("data", {})

                # If we have a specific run_id, filter to that run
                if self.run_id and self.run_id in data:
                    run_data = data[self.run_id]
                    for point in run_data:
                        step = point.get("step")
                        timestamp = point.get("timestamp", time.time()) / 1000
                        value = point.get("value")

                        # Create key for grouping
                        key = (step, int(timestamp * 1000))  # Group by step and timestamp (ms precision)

                        if key not in grouped:
                            grouped[key] = {
                                "timestamp": timestamp,
                                "metrics": {},
                            }
                            if step is not None:
                                grouped[key]["step"] = step

                        grouped[key]["metrics"][metric_name] = value

                elif not self.run_id:
                    # If no specific run, include all runs
                    for run_id, run_data in data.items():
                        for point in run_data:
                            step = point.get("step")
                            timestamp = point.get("timestamp", time.time()) / 1000
                            value = point.get("value")

                            key = (step, int(timestamp * 1000))

                            if key not in grouped:
                                grouped[key] = {
                                    "timestamp": timestamp,
                                    "metrics": {},
                                }
                                if step is not None:
                                    grouped[key]["step"] = step

                            grouped[key]["metrics"][metric_name] = value

            # Convert to JSONL, sorted by step
            lines = []
            for key in sorted(grouped.keys()):
                lines.append(json.dumps(grouped[key]))

            return "\n".join(lines) if lines else ""

        # For other files, use read_bytes
        return self.read_bytes(path).decode("utf-8")

    def append_text(self, path: str, text: str) -> None:
        """Append text to a file on the server.

        For JSONL files (metrics, logs, parameters), we parse and send to appropriate endpoints.
        """
        # Determine what type of data this is based on the path
        if "metrics.jsonl" in path:
            self._append_metrics(text)
        elif "logs.jsonl" in path:
            self._append_log(text)
        elif "parameters.jsonl" in path:
            self._append_parameters(text)
        else:
            # For other files, we can't really append in S3, so we skip
            pass

    def _append_metrics(self, line: str) -> None:
        """Parse and send metrics from a JSONL line."""
        try:
            entry = json.loads(line.strip())
            metrics_data = entry.get("metrics", {})
            step = entry.get("step")
            timestamp = entry.get("timestamp", time.time())

            # Convert to the format expected by the API
            formatted_metrics = {}
            for name, value in metrics_data.items():
                formatted_metrics[name] = [{
                    "step": step,
                    "timestamp": timestamp * 1000,  # Convert to milliseconds
                    "value": float(value),
                }]

            if formatted_metrics:
                self.log_metrics(formatted_metrics)
        except Exception as e:
            print(f"Warning: Failed to send metrics: {e}")

    def _append_log(self, line: str) -> None:
        """Parse and send log entry from a JSONL line."""
        try:
            entry = json.loads(line.strip())
            level = entry.get("level", "INFO")
            message = entry.get("message", "")
            timestamp = entry.get("timestamp", time.time())
            context = entry.get("context")

            self.log_entry(level, message, metadata=context, timestamp=timestamp)
        except Exception as e:
            print(f"Warning: Failed to send log entry: {e}")

    def _append_parameters(self, line: str) -> None:
        """Parse and send parameters from a JSONL line."""
        try:
            entry = json.loads(line.strip())
            operation = entry.get("operation", "set")

            if operation == "set":
                data = entry.get("data", {})
                self.log_parameters(data, operation="set")
            elif operation == "extend":
                data = entry.get("data", {})
                self.log_parameters(data, operation="extend")
            elif operation == "update":
                # For single key updates, we can send as a set operation
                key = entry.get("key")
                value = entry.get("value")
                if key:
                    self.log_parameters({key: value}, operation="update")
        except Exception as e:
            print(f"Warning: Failed to send parameters: {e}")

    def list_dir(self, path: str = "") -> List[str]:
        """List contents of a directory on the server."""
        # Not implemented - would need a separate API endpoint
        raise NotImplementedError("DashBackend.list_dir() not implemented")

    def get_url(self, path: str) -> Optional[str]:
        """Get a URL for accessing a file on the server."""
        # Files are stored in S3, URL would come from the artifact record
        return f"{self.server_url}/files/{self.namespace}/{self.workspace}/{path}"

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directories on the server."""
        # Server-side directories are created automatically
        pass

    def delete(self, path: str) -> None:
        """Delete a file on the server."""
        # Not implemented - would need a separate API endpoint
        raise NotImplementedError("DashBackend.delete() not implemented")
