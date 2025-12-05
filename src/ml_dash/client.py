"""
Remote API client for ML-Dash server.
"""

from typing import Optional, Dict, Any, List
import httpx


class RemoteClient:
    """Client for communicating with ML-Dash server."""

    def __init__(self, base_url: str, api_key: str):
        """
        Initialize remote client.

        Args:
            base_url: Base URL of ML-Dash server (e.g., "http://localhost:3000")
            api_key: JWT token for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                # Note: Don't set Content-Type here as default
                # It will be set per-request (json or multipart)
            },
            timeout=30.0,
        )

    def create_or_update_experiment(
        self,
        project: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bindrs: Optional[List[str]] = None,
        folder: Optional[str] = None,
        write_protected: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create or update an experiment.

        Args:
            project: Project name
            name: Experiment name
            description: Optional description
            tags: Optional list of tags
            bindrs: Optional list of bindrs
            folder: Optional folder path
            write_protected: If True, experiment becomes immutable
            metadata: Optional metadata dict

        Returns:
            Response dict with experiment, project, folder, and namespace data

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = {
            "name": name,
        }

        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
        if bindrs is not None:
            payload["bindrs"] = bindrs
        if folder is not None:
            payload["folder"] = folder
        if write_protected:
            payload["writeProtected"] = write_protected
        if metadata is not None:
            payload["metadata"] = metadata

        response = self._client.post(
            f"/projects/{project}/experiments",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
    ) -> Dict[str, Any]:
        """
        Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: Status value - "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"

        Returns:
            Response dict with updated experiment data

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = {
            "status": status,
        }

        response = self._client.patch(
            f"/experiments/{experiment_id}/status",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def create_log_entries(
        self,
        experiment_id: str,
        logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create log entries in batch.

        Supports both single log and multiple logs via array.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            logs: List of log entries, each with fields:
                - timestamp: ISO 8601 string
                - level: "info"|"warn"|"error"|"debug"|"fatal"
                - message: Log message string
                - metadata: Optional dict

        Returns:
            Response dict:
            {
                "created": 1,
                "startSequence": 42,
                "endSequence": 42,
                "experimentId": "123456789"
            }

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.post(
            f"/experiments/{experiment_id}/logs",
            json={"logs": logs}
        )
        response.raise_for_status()
        return response.json()

    def set_parameters(
        self,
        experiment_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set/merge parameters for an experiment.

        Always merges with existing parameters (upsert behavior).

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            data: Flattened parameter dict with dot notation
                Example: {"model.lr": 0.001, "model.batch_size": 32}

        Returns:
            Response dict:
            {
                "id": "snowflake_id",
                "experimentId": "experiment_id",
                "data": {...},
                "version": 2,
                "createdAt": "...",
                "updatedAt": "..."
            }

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.post(
            f"/experiments/{experiment_id}/parameters",
            json={"data": data}
        )
        response.raise_for_status()
        return response.json()

    def get_parameters(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get parameters for an experiment.

        Args:
            experiment_id: Experiment ID (Snowflake ID)

        Returns:
            Flattened parameter dict with dot notation
            Example: {"model.lr": 0.001, "model.batch_size": 32}

        Raises:
            httpx.HTTPStatusError: If request fails or parameters don't exist
        """
        response = self._client.get(f"/experiments/{experiment_id}/parameters")
        response.raise_for_status()
        result = response.json()
        return result.get("data", {})

    def upload_file(
        self,
        experiment_id: str,
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
        Upload a file to an experiment.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
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

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        # Prepare multipart form data
        # Read file content first (httpx needs content, not file handle)
        with open(file_path, "rb") as f:
            file_content = f.read()

        files = {"file": (filename, file_content, content_type)}
        data = {
            "prefix": prefix,
            "checksum": checksum,
            "sizeBytes": str(size_bytes),
        }
        if description:
            data["description"] = description
        if tags:
            data["tags"] = ",".join(tags)
        if metadata:
            import json
            data["metadata"] = json.dumps(metadata)

        # httpx will automatically set multipart/form-data content-type
        response = self._client.post(
            f"/experiments/{experiment_id}/files",
            files=files,
            data=data
        )

        response.raise_for_status()
        return response.json()

    def list_files(
        self,
        experiment_id: str,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List files in an experiment.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            prefix: Optional prefix filter
            tags: Optional tags filter

        Returns:
            List of file metadata dicts

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        params = {}
        if prefix:
            params["prefix"] = prefix
        if tags:
            params["tags"] = ",".join(tags)

        response = self._client.get(
            f"/experiments/{experiment_id}/files",
            params=params
        )
        response.raise_for_status()
        result = response.json()
        return result.get("files", [])

    def get_file(self, experiment_id: str, file_id: str) -> Dict[str, Any]:
        """
        Get file metadata.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            file_id: File ID (Snowflake ID)

        Returns:
            File metadata dict

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.get(f"/experiments/{experiment_id}/files/{file_id}")
        response.raise_for_status()
        return response.json()

    def download_file(
        self,
        experiment_id: str,
        file_id: str,
        dest_path: Optional[str] = None
    ) -> str:
        """
        Download a file from a experiment.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            file_id: File ID (Snowflake ID)
            dest_path: Optional destination path (defaults to original filename)

        Returns:
            Path to downloaded file

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If checksum verification fails
        """
        # Get file metadata first to get filename and checksum
        file_metadata = self.get_file(experiment_id, file_id)
        filename = file_metadata["filename"]
        expected_checksum = file_metadata["checksum"]

        # Determine destination path
        if dest_path is None:
            dest_path = filename

        # Download file
        response = self._client.get(
            f"/experiments/{experiment_id}/files/{file_id}/download"
        )
        response.raise_for_status()

        # Write to file
        with open(dest_path, "wb") as f:
            f.write(response.content)

        # Verify checksum
        from .files import verify_checksum
        if not verify_checksum(dest_path, expected_checksum):
            # Delete corrupted file
            import os
            os.remove(dest_path)
            raise ValueError(f"Checksum verification failed for file {file_id}")

        return dest_path

    def delete_file(self, experiment_id: str, file_id: str) -> Dict[str, Any]:
        """
        Delete a file (soft delete).

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            file_id: File ID (Snowflake ID)

        Returns:
            Dict with id and deletedAt

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.delete(f"/experiments/{experiment_id}/files/{file_id}")
        response.raise_for_status()
        return response.json()

    def update_file(
        self,
        experiment_id: str,
        file_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update file metadata.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            file_id: File ID (Snowflake ID)
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Updated file metadata dict

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = {}
        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata

        response = self._client.patch(
            f"/experiments/{experiment_id}/files/{file_id}",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def append_to_metric(
        self,
        experiment_id: str,
        metric_name: str,
        data: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Append a single data point to a metric.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            metric_name: Metric name (unique within experiment)
            data: Data point (flexible schema)
            description: Optional metric description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Dict with metricId, index, bufferedDataPoints, chunkSize

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = {"data": data}
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        response = self._client.post(
            f"/experiments/{experiment_id}/metrics/{metric_name}/append",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def append_batch_to_metric(
        self,
        experiment_id: str,
        metric_name: str,
        data_points: List[Dict[str, Any]],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Append multiple data points to a metric in batch.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            metric_name: Metric name (unique within experiment)
            data_points: List of data points
            description: Optional metric description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Dict with metricId, startIndex, endIndex, count, bufferedDataPoints, chunkSize

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = {"dataPoints": data_points}
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        response = self._client.post(
            f"/experiments/{experiment_id}/metrics/{metric_name}/append-batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def read_metric_data(
        self,
        experiment_id: str,
        metric_name: str,
        start_index: int = 0,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Read data points from a metric.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            metric_name: Metric name
            start_index: Starting index (default 0)
            limit: Max points to read (default 1000, max 10000)

        Returns:
            Dict with data, startIndex, endIndex, total, hasMore

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.get(
            f"/experiments/{experiment_id}/metrics/{metric_name}/data",
            params={"startIndex": start_index, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_metric_stats(
        self,
        experiment_id: str,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Get metric statistics and metadata.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            metric_name: Metric name

        Returns:
            Dict with metric stats (totalDataPoints, bufferedDataPoints, etc.)

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.get(
            f"/experiments/{experiment_id}/metrics/{metric_name}/stats"
        )
        response.raise_for_status()
        return response.json()

    def list_metrics(
        self,
        experiment_id: str
    ) -> List[Dict[str, Any]]:
        """
        List all metrics in an experiment.

        Args:
            experiment_id: Experiment ID (Snowflake ID)

        Returns:
            List of metric summaries

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.get(f"/experiments/{experiment_id}/metrics")
        response.raise_for_status()
        return response.json()["metrics"]

    def graphql_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Optional variables for the query

        Returns:
            Query result data

        Raises:
            httpx.HTTPStatusError: If request fails
            Exception: If GraphQL returns errors
        """
        response = self._client.post(
            "/graphql",
            json={"query": query, "variables": variables or {}}
        )
        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")

        return result.get("data", {})

    def list_projects_graphql(self, namespace_slug: str) -> List[Dict[str, Any]]:
        """
        List all projects in a namespace via GraphQL.

        Args:
            namespace_slug: Namespace slug

        Returns:
            List of project dicts with experimentCount

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query Projects($namespaceSlug: String!) {
          projects(namespaceSlug: $namespaceSlug) {
            id
            name
            slug
            description
            tags
          }
        }
        """
        result = self.graphql_query(query, {"namespaceSlug": namespace_slug})
        projects = result.get("projects", [])

        # For each project, count experiments
        for project in projects:
            exp_query = """
            query ExperimentsCount($namespaceSlug: String!, $projectSlug: String!) {
              experiments(namespaceSlug: $namespaceSlug, projectSlug: $projectSlug) {
                id
              }
            }
            """
            exp_result = self.graphql_query(exp_query, {
                "namespaceSlug": namespace_slug,
                "projectSlug": project['slug']
            })
            experiments = exp_result.get("experiments", [])
            project['experimentCount'] = len(experiments)

        return projects

    def list_experiments_graphql(
        self, namespace_slug: str, project_slug: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments in a project via GraphQL.

        Args:
            namespace_slug: Namespace slug
            project_slug: Project slug
            status: Optional experiment status filter (RUNNING, COMPLETED, FAILED, CANCELLED)

        Returns:
            List of experiment dicts with metadata

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query Experiments($namespaceSlug: String!, $projectSlug: String!, $status: ExperimentStatus) {
          experiments(namespaceSlug: $namespaceSlug, projectSlug: $projectSlug, status: $status) {
            id
            name
            description
            tags
            status
            startedAt
            endedAt
            metadata
            project {
              slug
            }
            logMetadata {
              totalLogs
            }
            metrics {
              name
              metricMetadata {
                totalDataPoints
              }
            }
            files {
              id
              filename
              path
              contentType
              sizeBytes
              checksum
              description
              tags
              metadata
            }
            parameters {
              id
              data
            }
          }
        }
        """
        result = self.graphql_query(query, {
            "namespaceSlug": namespace_slug,
            "projectSlug": project_slug,
            "status": status
        })
        return result.get("experiments", [])

    def get_experiment_graphql(
        self, namespace_slug: str, project_slug: str, experiment_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single experiment via GraphQL.

        Args:
            namespace_slug: Namespace slug
            project_slug: Project slug
            experiment_name: Experiment name

        Returns:
            Experiment dict with metadata, or None if not found

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query Experiment($namespaceSlug: String!, $projectSlug: String!, $experimentName: String!) {
          experiment(namespaceSlug: $namespaceSlug, projectSlug: $projectSlug, experimentName: $experimentName) {
            id
            name
            description
            tags
            status
            metadata
            project {
              slug
            }
            logMetadata {
              totalLogs
            }
            metrics {
              name
              metricMetadata {
                totalDataPoints
              }
            }
            files {
              id
              filename
              path
              contentType
              sizeBytes
              checksum
              description
              tags
              metadata
            }
            parameters {
              id
              data
            }
          }
        }
        """
        result = self.graphql_query(query, {
            "namespaceSlug": namespace_slug,
            "projectSlug": project_slug,
            "experimentName": experiment_name
        })
        return result.get("experiment")

    def download_file_streaming(
        self, experiment_id: str, file_id: str, dest_path: str
    ) -> str:
        """
        Download a file with streaming for large files.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            file_id: File ID (Snowflake ID)
            dest_path: Destination path to save file

        Returns:
            Path to downloaded file

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If checksum verification fails
        """
        # Get metadata first for checksum
        file_metadata = self.get_file(experiment_id, file_id)
        expected_checksum = file_metadata["checksum"]

        # Stream download
        with self._client.stream("GET", f"/experiments/{experiment_id}/files/{file_id}/download") as response:
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        # Verify checksum
        from .files import verify_checksum
        if not verify_checksum(dest_path, expected_checksum):
            import os
            os.remove(dest_path)
            raise ValueError(f"Checksum verification failed for file {file_id}")

        return dest_path

    def query_logs(
        self,
        experiment_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        level: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query logs for an experiment.

        Args:
            experiment_id: Experiment ID
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            order_by: Field to order by (timestamp or sequenceNumber)
            order: Sort order (asc or desc)
            level: List of log levels to filter by
            start_time: Filter logs after this timestamp
            end_time: Filter logs before this timestamp
            search: Search query for log messages

        Returns:
            Dict with logs array and pagination info

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        params: Dict[str, str] = {}

        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        if order_by is not None:
            params["orderBy"] = order_by
        if order is not None:
            params["order"] = order
        if level is not None:
            params["level"] = ",".join(level)
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if search is not None:
            params["search"] = search

        response = self._client.get(f"/experiments/{experiment_id}/logs", params=params)
        response.raise_for_status()
        return response.json()

    def get_metric_data(
        self,
        experiment_id: str,
        metric_name: str,
        start_index: Optional[int] = None,
        limit: Optional[int] = None,
        buffer_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Get data points for a metric.

        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            start_index: Starting index for pagination
            limit: Maximum number of data points to return
            buffer_only: If True, only fetch buffer data (skip chunks)

        Returns:
            Dict with dataPoints array and pagination info

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        params: Dict[str, str] = {}

        if start_index is not None:
            params["startIndex"] = str(start_index)
        if limit is not None:
            params["limit"] = str(limit)
        if buffer_only:
            params["bufferOnly"] = "true"

        response = self._client.get(
            f"/experiments/{experiment_id}/metrics/{metric_name}/data",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def download_metric_chunk(
        self,
        experiment_id: str,
        metric_name: str,
        chunk_number: int,
    ) -> Dict[str, Any]:
        """
        Download a specific chunk by chunk number.

        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            chunk_number: Chunk number to download

        Returns:
            Dict with chunk data including chunkNumber, startIndex, endIndex, dataCount, and data array

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.get(
            f"/experiments/{experiment_id}/metrics/{metric_name}/chunks/{chunk_number}"
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
