"""
Remote API client for ML-Dash server.
"""

from typing import Optional, Dict, Any, List
import httpx


class UserInfo:
    """
    Singleton user info object that fetches current user from API server.

    Fetches user info from API server on first access (lazy loading).
    This queries the API for fresh user data, ensuring up-to-date information.

    Usage:
        >>> from ml_dash import userinfo
        >>> if userinfo.username:
        ...     print(f"Namespace: {userinfo.username}")
        ...     print(f"Email: {userinfo.email}")
        ...     print(f"Project: {userinfo.username}/my-project")
    """

    def __init__(self):
        self._data = None
        self._fetched = False

    def _fetch(self):
        """Fetch user info from API server (lazy loading)."""
        if self._fetched:
            return

        self._fetched = True
        try:
            client = RemoteClient("https://api.dash.ml")
            self._data = client.get_current_user()
        except Exception:
            self._data = None

    @property
    def username(self) -> Optional[str]:
        """Username (namespace) - e.g., 'tom_tao_e4c2c9'"""
        self._fetch()
        return self._data.get("username") if self._data else None

    @property
    def email(self) -> Optional[str]:
        """User email"""
        self._fetch()
        return self._data.get("email") if self._data else None

    @property
    def name(self) -> Optional[str]:
        """Full name"""
        self._fetch()
        return self._data.get("name") if self._data else None

    @property
    def given_name(self) -> Optional[str]:
        """First/given name"""
        self._fetch()
        return self._data.get("given_name") if self._data else None

    @property
    def family_name(self) -> Optional[str]:
        """Last/family name"""
        self._fetch()
        return self._data.get("family_name") if self._data else None

    @property
    def picture(self) -> Optional[str]:
        """Profile picture URL"""
        self._fetch()
        return self._data.get("picture") if self._data else None

    @property
    def id(self) -> Optional[str]:
        """User ID"""
        self._fetch()
        return self._data.get("id") if self._data else None

    def __bool__(self) -> bool:
        """Return True if user is authenticated and data was fetched successfully."""
        self._fetch()
        return self._data is not None

    def __repr__(self) -> str:
        self._fetch()
        if self._data:
            return f"UserInfo(username='{self.username}', email='{self.email}')"
        return "UserInfo(not authenticated)"


# Create singleton instance
userinfo = UserInfo()


def _serialize_value(value: Any) -> Any:
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
        return {k: _serialize_value(v) for k, v in value.items()}

    # Recursively handle lists
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    # Return as-is for other types (int, float, str, bool, None)
    return value


class RemoteClient:
    """Client for communicating with ML-Dash server."""

    def __init__(self, base_url: str, namespace: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize remote client.

        Args:
            base_url: Base URL of ML-Dash server (e.g., "http://localhost:3000")
            namespace: Namespace slug (e.g., "my-namespace"). If not provided, will be queried from server.
            api_key: JWT token for authentication (optional - auto-loads from storage if not provided)

        Note:
            If no api_key is provided, token will be loaded from storage on first API call.
            If still not found, AuthenticationError will be raised at that time.
            If no namespace is provided, it will be fetched from the server on first API call.
        """
        # Store original base URL for GraphQL (no /api prefix)
        self.graphql_base_url = base_url.rstrip("/")

        # Add /api prefix to base URL for REST API calls
        self.base_url = base_url.rstrip("/") + "/api"

        # If no api_key provided, try to load from storage
        if not api_key:
            from .auth.token_storage import get_token_storage

            storage = get_token_storage()
            api_key = storage.load("ml-dash-token")

        self.api_key = api_key

        # Store namespace (can be None, will be fetched on first API call if needed)
        self._namespace = namespace
        self._namespace_fetched = False

        self._rest_client = None
        self._gql_client = None
        self._id_cache: Dict[str, str] = {}  # Cache for slug -> ID mappings

    @property
    def namespace(self) -> str:
        """
        Get namespace, fetching from server if not already set.

        Returns:
            Namespace slug

        Raises:
            AuthenticationError: If not authenticated
            ValueError: If namespace cannot be determined
        """
        if self._namespace:
            return self._namespace

        if not self._namespace_fetched:
            # Fetch namespace from server
            self._namespace = self._fetch_namespace_from_server()
            self._namespace_fetched = True

        if not self._namespace:
            raise ValueError("Could not determine namespace. Please provide --namespace explicitly.")

        return self._namespace

    @namespace.setter
    def namespace(self, value: str):
        """Set namespace."""
        self._namespace = value
        self._namespace_fetched = True

    def _fetch_namespace_from_server(self) -> Optional[str]:
        """
        Fetch current user's namespace from server.

        Returns:
            Namespace slug or None if cannot be determined
        """
        try:
            self._ensure_authenticated()

            # Query server for current user's namespace
            query = """
            query GetMyNamespace {
              me {
                username
              }
            }
            """
            result = self.graphql_query(query)
            username = result.get("me", {}).get("username")
            return username
        except Exception:
            return None

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user's info from server.

        This queries the API server for fresh user data, ensuring up-to-date information.

        Returns:
            User info dict with keys: username, email, name, given_name, family_name, picture
            Returns None if not authenticated or if query fails

        Example:
            >>> client = RemoteClient("https://api.dash.ml")
            >>> user = client.get_current_user()
            >>> print(user["username"])  # e.g., "tom_tao_e4c2c9"
            >>> print(user["email"])     # e.g., "user@example.com"
        """
        try:
            self._ensure_authenticated()

            # Query server for current user's complete profile
            query = """
            query GetCurrentUser {
              me {
                id
                username
                email
                name
                given_name
                family_name
                picture
              }
            }
            """
            result = self.graphql_query(query)
            return result.get("me")
        except Exception:
            return None

    def _ensure_authenticated(self):
        """Check if authenticated, raise error if not."""
        if not self.api_key:
            from .auth.exceptions import AuthenticationError
            raise AuthenticationError(
                "Not authenticated. Run 'ml-dash login' to authenticate, "
                "or provide an explicit api_key parameter."
            )

    @property
    def _client(self):
        """Lazy REST API client (with /api prefix)."""
        if self._rest_client is None:
            self._ensure_authenticated()
            self._rest_client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    # Note: Don't set Content-Type here as default
                    # It will be set per-request (json or multipart)
                },
                timeout=30.0,
            )
        return self._rest_client

    @property
    def _graphql_client(self):
        """Lazy GraphQL client (without /api prefix)."""
        if self._gql_client is None:
            self._ensure_authenticated()
            self._gql_client = httpx.Client(
                base_url=self.graphql_base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=30.0,
            )
        return self._gql_client

    def _get_project_id(self, project_slug: str) -> Optional[str]:
        """
        Resolve project ID from slug using GraphQL.

        Args:
            project_slug: Project slug

        Returns:
            Project ID (Snowflake ID) if found, None if not found
            When None is returned, the server will auto-create the project
        """
        cache_key = f"project:{self.namespace}:{project_slug}"
        if cache_key in self._id_cache:
            return self._id_cache[cache_key]

        query = """
        query GetProject($namespace: String!) {
          namespace(slug: $namespace) {
            projects {
              id
              slug
            }
          }
        }
        """
        result = self.graphql_query(query, {
            "namespace": self.namespace
        })

        namespace_data = result.get("namespace")
        if namespace_data is None:
            raise ValueError(f"Namespace '{self.namespace}' not found. Please check the namespace exists on the server.")

        projects = namespace_data.get("projects", [])
        for project in projects:
            if project["slug"] == project_slug:
                project_id = project["id"]
                self._id_cache[cache_key] = project_id
                return project_id

        # Project not found - return None to let server auto-create it
        return None

    def delete_project(self, project_slug: str) -> Dict[str, Any]:
        """
        Delete a project and all its experiments, metrics, files, and logs.

        Args:
            project_slug: Project slug

        Returns:
            Dict with projectId, deleted count, experiments count, and message

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If project not found
        """
        # Get project ID first
        project_id = self._get_project_id(project_slug)
        if not project_id:
            raise ValueError(f"Project '{project_slug}' not found in namespace '{self.namespace}'")

        # Delete using project-specific endpoint
        response = self._client.delete(f"projects/{project_id}")
        response.raise_for_status()
        return response.json()

    def _get_experiment_node_id(self, experiment_id: str) -> str:
        """
        Resolve node ID from experiment ID using GraphQL.

        Args:
            experiment_id: Experiment ID

        Returns:
            Node ID

        Raises:
            ValueError: If experiment node not found
        """
        cache_key = f"exp_node:{experiment_id}"
        if cache_key in self._id_cache:
            return self._id_cache[cache_key]

        query = """
        query GetExperimentNode($experimentId: ID!) {
          experimentNode(experimentId: $experimentId) {
            id
          }
        }
        """
        result = self.graphql_query(query, {"experimentId": experiment_id})

        node = result.get("experimentNode")
        if not node:
            raise ValueError(f"No node found for experiment ID '{experiment_id}'")

        node_id = node["id"]
        self._id_cache[cache_key] = node_id
        return node_id

    def create_or_update_experiment(
        self,
        project: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bindrs: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        write_protected: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create or update an experiment using unified node API.

        Args:
            project: Project slug
            name: Experiment name
            description: Optional description
            tags: Optional list of tags
            bindrs: Optional list of bindrs
            prefix: Full prefix path (ignored in new API - use folders instead)
            write_protected: If True, experiment becomes immutable
            metadata: Optional metadata dict

        Returns:
            Response dict with experiment, node, and project data
            Note: Project will be auto-created if it doesn't exist

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        # Resolve project ID from slug (returns None if not found)
        project_id = self._get_project_id(project)

        # Parse prefix to create folder hierarchy for experiment
        # prefix format: "namespace/project/folder1/folder2/experiment_name"
        # We need to create folders: folder1 -> folder2 and place experiment under folder2
        parent_id = "ROOT"

        if prefix:
            # Parse prefix to extract folder path
            parts = prefix.strip('/').split('/')
            # parts: [namespace, project, folder1, folder2, ..., experiment_name]

            if len(parts) >= 3:
                # We have at least namespace/project/something
                # Extract folder parts (everything between project and experiment name)
                # Skip namespace (parts[0]) and project (parts[1])
                # Skip experiment name (parts[-1])
                folder_parts = parts[2:-1] if len(parts) > 3 else []

                if folder_parts:
                    # Ensure we have a project_id for folder creation
                    if not project_id:
                        # Create the project first since we need its ID for folders
                        project_response = self._client.post(
                            f"namespaces/{self.namespace}/nodes",
                            json={
                                "type": "PROJECT",
                                "name": project,
                                "slug": project,
                            }
                        )
                        project_response.raise_for_status()
                        project_data = project_response.json()
                        project_id = project_data.get("project", {}).get("id")

                    if project_id:
                        # Create folder hierarchy
                        current_parent_id = "ROOT"
                        for folder_name in folder_parts:
                            if not folder_name:
                                continue
                            # Create folder (server handles upsert)
                            # NOTE: Do NOT pass experimentId for project-level folders
                            folder_response = self._client.post(
                                f"namespaces/{self.namespace}/nodes",
                                json={
                                    "type": "FOLDER",
                                    "projectId": project_id,
                                    "parentId": current_parent_id,
                                    "name": folder_name
                                    # experimentId intentionally omitted - these are project-level folders
                                }
                            )
                            folder_response.raise_for_status()
                            folder_data = folder_response.json()
                            current_parent_id = folder_data.get("node", {}).get("id")

                        # Update parent_id for experiment
                        parent_id = current_parent_id

        # Build payload for unified node API
        payload = {
            "type": "EXPERIMENT",
            "name": name,
            "parentId": parent_id,
        }

        # Send projectId if available, otherwise projectSlug (server will auto-create)
        if project_id:
            payload["projectId"] = project_id
        else:
            payload["projectSlug"] = project

        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
        if bindrs is not None:
            payload["bindrs"] = bindrs
        if write_protected:
            payload["writeProtected"] = write_protected
        if metadata is not None:
            payload["metadata"] = metadata

        # Call unified node creation API
        response = self._client.post(
            f"namespaces/{self.namespace}/nodes",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        # Cache the experiment node ID mapping
        if "experiment" in result and "node" in result:
            exp_id = result["experiment"]["id"]
            node_id = result["node"]["id"]
            self._id_cache[f"exp_node:{exp_id}"] = node_id

        return result

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
    ) -> Dict[str, Any]:
        """
        Update experiment status using unified node API.

        Args:
            experiment_id: Experiment ID
            status: Status value - "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"

        Returns:
            Response dict with updated node data

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If experiment node not found
        """
        # Resolve node ID from experiment ID
        node_id = self._get_experiment_node_id(experiment_id)

        # Update node with new status
        payload = {"status": status}

        response = self._client.patch(
            f"nodes/{node_id}",
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
            f"experiments/{experiment_id}/logs",
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
            f"experiments/{experiment_id}/parameters",
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
        response = self._client.get(f"experiments/{experiment_id}/parameters")
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
        size_bytes: int,
        project_id: Optional[str] = None,
        parent_id: str = "ROOT"
    ) -> Dict[str, Any]:
        """
        Upload a file to an experiment using unified node API.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            file_path: Local file path
            prefix: Logical path prefix for folder structure (e.g., "models/checkpoints")
                   Will create nested folders automatically. May include namespace/project
                   parts which will be stripped automatically (e.g., "ns/proj/folder1/folder2"
                   will create folders: folder1 -> folder2)
            filename: Original filename
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            checksum: SHA256 checksum
            content_type: MIME type
            size_bytes: File size in bytes
            project_id: Project ID (optional - will be resolved from experiment if not provided)
            parent_id: Parent node ID (folder) or "ROOT" for root level.
                      If prefix is provided, folders will be created under this parent.

        Returns:
            Response dict with node and physicalFile data

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        # If project_id not provided, need to resolve it from experiment
        # For now, assuming we have it or it will be queried separately
        if project_id is None:
            # Query experiment to get project ID
            query = """
            query GetExperimentProject($experimentId: ID!) {
              experimentById(id: $experimentId) {
                projectId
              }
            }
            """
            result = self.graphql_query(query, {"experimentId": experiment_id})
            project_id = result.get("experimentById", {}).get("projectId")
            if not project_id:
                raise ValueError(f"Could not resolve project ID for experiment {experiment_id}")

        # Resolve experiment node ID (files should be children of the experiment node, not ROOT)
        # Check cache first, otherwise query
        experiment_node_id = self._id_cache.get(f"exp_node:{experiment_id}")
        if not experiment_node_id:
            # Query to get the experiment node ID
            query = """
            query GetExperimentNode($experimentId: ID!) {
              experimentById(id: $experimentId) {
                id
              }
            }
            """
            # Note: experimentById returns the Experiment record, not the Node
            # We need to find the Node with type=EXPERIMENT and experimentId=experiment_id
            # Use the project nodes query instead
            query = """
            query GetExperimentNode($projectId: ID!, $experimentId: ID!) {
              project(id: $projectId) {
                nodes(parentId: null, maxDepth: 10) {
                  id
                  type
                  experimentId
                  children {
                    id
                    type
                    experimentId
                    children {
                      id
                      type
                      experimentId
                    }
                  }
                }
              }
            }
            """
            result = self.graphql_query(query, {"projectId": project_id, "experimentId": experiment_id})

            # Find the experiment node
            def find_experiment_node(nodes, exp_id):
                for node in nodes:
                    if node.get("type") == "EXPERIMENT" and node.get("experimentId") == exp_id:
                        return node.get("id")
                    if node.get("children"):
                        found = find_experiment_node(node["children"], exp_id)
                        if found:
                            return found
                return None

            project_nodes = result.get("project", {}).get("nodes", [])
            experiment_node_id = find_experiment_node(project_nodes, experiment_id)

            if experiment_node_id:
                # Cache it for future uploads
                self._id_cache[f"exp_node:{experiment_id}"] = experiment_node_id
            else:
                # Fallback to ROOT if we can't find the experiment node
                # This might happen for old experiments or legacy data
                experiment_node_id = "ROOT"

        # Get experiment node path to strip from prefix
        # When we use experiment_node_id as parent, we need to strip the experiment's
        # folder path from the prefix to avoid creating duplicate folders
        # We'll cache this in the id_cache to avoid repeated queries
        cache_key = f"exp_folder_path:{experiment_id}"
        experiment_folder_path = self._id_cache.get(cache_key)

        if experiment_folder_path is None and experiment_node_id != "ROOT":
            # Query experiment to get its project info for the GraphQL query
            exp_query = """
            query GetExpInfo($experimentId: ID!) {
              experimentById(id: $experimentId) {
                project {
                  slug
                  namespace {
                    slug
                  }
                }
              }
            }
            """
            exp_result = self.graphql_query(exp_query, {"experimentId": experiment_id})
            project_slug = exp_result.get("experimentById", {}).get("project", {}).get("slug")
            namespace_slug = exp_result.get("experimentById", {}).get("project", {}).get("namespace", {}).get("slug")

            if project_slug and namespace_slug:
                # Query to get the experiment node's path
                # This includes all ancestor folders up to the experiment
                query = """
                query GetExperimentPath($namespaceSlug: String!, $projectSlug: String!) {
                  project(namespaceSlug: $namespaceSlug, projectSlug: $projectSlug) {
                    nodes(parentId: null, maxDepth: 10) {
                      id
                      name
                      type
                      experimentId
                      parentId
                      children {
                        id
                        name
                        type
                        experimentId
                        parentId
                        children {
                          id
                          name
                          type
                          experimentId
                          parentId
                        }
                      }
                    }
                  }
                }
                """
                result = self.graphql_query(query, {"namespaceSlug": namespace_slug, "projectSlug": project_slug})

                # Build path to experiment node
                def find_node_path(nodes, target_id, current_path=None):
                    if current_path is None:
                        current_path = []
                    for node in nodes:
                        new_path = current_path + [node.get("name")]
                        if node.get("id") == target_id:
                            return new_path
                        if node.get("children"):
                            found = find_node_path(node["children"], target_id, new_path)
                            if found:
                                return found
                    return None

                project_nodes = result.get("project", {}).get("nodes", [])
                path_parts = find_node_path(project_nodes, experiment_node_id)
                if path_parts:
                    # IMPORTANT: Don't include the experiment node's name itself
                    # We want the path TO the experiment's parent folder, not the experiment
                    # E.g., if path is ["examples", "exp-name"], we want "examples"
                    if len(path_parts) > 1:
                        experiment_folder_path = "/".join(path_parts[:-1])
                    else:
                        # Experiment is at root level, no parent folders
                        experiment_folder_path = ""
                    # Cache it
                    self._id_cache[cache_key] = experiment_folder_path
                else:
                    # Couldn't find path, set empty string to avoid re-querying
                    experiment_folder_path = ""
                    self._id_cache[cache_key] = experiment_folder_path

        # Use experiment node ID as the parent for file uploads
        # Files and folders should be children of the experiment node
        if parent_id == "ROOT" and experiment_node_id != "ROOT":
            parent_id = experiment_node_id

        # Parse prefix to create folder hierarchy
        # prefix like "models/checkpoints" should create folders: models -> checkpoints
        # NOTE: The prefix may contain namespace/project parts (e.g., "ns/proj/folder1/folder2")
        # We need to strip the namespace and project parts since we're already in an experiment context
        if prefix and prefix != '/' and prefix.strip():
            # Clean and normalize prefix
            prefix = prefix.strip('/')

            # Try to detect and strip namespace/project from prefix
            # Common patterns: "namespace/project/folders..." or just "folders..."
            # Since we're in experiment context, we already know the namespace and project
            # Check if prefix starts with namespace
            if prefix.startswith(self.namespace + '/'):
                # Strip namespace
                prefix = prefix[len(self.namespace) + 1:]

                # Now check if it starts with project slug/name
                # We need to query the experiment to get the project info
                query = """
                query GetExperimentProject($experimentId: ID!) {
                  experimentById(id: $experimentId) {
                    project {
                      slug
                      name
                    }
                  }
                }
                """
                exp_result = self.graphql_query(query, {"experimentId": experiment_id})
                project_info = exp_result.get("experimentById", {}).get("project", {})
                project_slug = project_info.get("slug", "")
                project_name = project_info.get("name", "")

                # Try to strip project slug or name
                if project_slug and prefix.startswith(project_slug + '/'):
                    prefix = prefix[len(project_slug) + 1:]
                elif project_name and prefix.startswith(project_name + '/'):
                    prefix = prefix[len(project_name) + 1:]

            # Strip experiment folder path from prefix since we're using experiment node as parent
            # For example: if prefix is "examples/exp1/models" and experiment is at "examples/exp1",
            # strip "examples/exp1/" to get "models"
            if experiment_folder_path and prefix.startswith(experiment_folder_path + '/'):
                prefix = prefix[len(experiment_folder_path) + 1:]
            elif experiment_folder_path and prefix == experiment_folder_path:
                # Prefix is exactly the experiment path, no subfolders
                prefix = ""

            if prefix:
                folder_parts = prefix.split('/')
                current_parent_id = parent_id

                # Create or find each folder in the hierarchy
                # Server handles upsert - will return existing folder if it exists
                for folder_name in folder_parts:
                    if not folder_name:  # Skip empty parts
                        continue

                    # Create folder (server will return existing if duplicate)
                    folder_response = self._client.post(
                        f"namespaces/{self.namespace}/nodes",
                        json={
                            "type": "FOLDER",
                            "projectId": project_id,
                            "experimentId": experiment_id,
                            "parentId": current_parent_id,
                            "name": folder_name
                        }
                    )
                    folder_response.raise_for_status()
                    folder_data = folder_response.json()
                    current_parent_id = folder_data.get("node", {}).get("id")

                # Update parent_id to the final folder in the hierarchy
                parent_id = current_parent_id

        # Prepare multipart form data
        with open(file_path, "rb") as f:
            file_content = f.read()

        files = {"file": (filename, file_content, content_type)}
        data = {
            "type": "FILE",
            "projectId": project_id,
            "experimentId": experiment_id,
            "parentId": parent_id,
            "name": filename,
            "checksum": checksum,
        }
        if description:
            data["description"] = description
        if tags:
            data["tags"] = ",".join(tags)
        if metadata:
            import json
            data["metadata"] = json.dumps(metadata)

        # Call unified node creation API
        response = self._client.post(
            f"namespaces/{self.namespace}/nodes",
            files=files,
            data=data
        )

        response.raise_for_status()
        result = response.json()

        # Transform unified node response to expected file metadata format
        # The server returns {node: {...}, physicalFile: {...}}
        # We need to flatten it to match the expected format
        node = result.get("node", {})
        physical_file = result.get("physicalFile", {})

        # Convert BigInt IDs and sizeBytes from string back to appropriate types
        # Node ID should remain as string for consistency
        node_id = node.get("id")
        if isinstance(node_id, (int, float)):
            # If it was deserialized as a number, convert to string to preserve full precision
            node_id = str(int(node_id))

        size_bytes = physical_file.get("sizeBytes")
        if isinstance(size_bytes, str):
            size_bytes = int(size_bytes)

        # Use experimentId from node, not the parameter (which might be a path string)
        experiment_id_from_node = node.get("experimentId")
        if isinstance(experiment_id_from_node, (int, float)):
            experiment_id_from_node = str(int(experiment_id_from_node))

        return {
            "id": node_id,
            "experimentId": experiment_id_from_node or experiment_id,
            "path": prefix,  # Use prefix as path for backward compatibility
            "filename": filename,
            "description": node.get("description"),
            "tags": node.get("tags", []),
            "contentType": physical_file.get("contentType"),
            "sizeBytes": size_bytes,
            "checksum": physical_file.get("checksum"),
            "metadata": node.get("metadata"),
            "uploadedAt": node.get("createdAt"),
            "updatedAt": node.get("updatedAt"),
            "deletedAt": node.get("deletedAt"),
        }

    def list_files(
        self,
        experiment_id: str,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List files in an experiment using GraphQL.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            prefix: Optional prefix filter (DEPRECATED - filtering not supported in new API)
            tags: Optional tags filter

        Returns:
            List of file node dicts

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query ListExperimentFiles($experimentId: ID!) {
          experimentById(id: $experimentId) {
            files {
              id
              name
              description
              tags
              metadata
              createdAt
              pPath
              physicalFile {
                id
                filename
                contentType
                sizeBytes
                checksum
                s3Url
              }
            }
          }
        }
        """
        result = self.graphql_query(query, {"experimentId": experiment_id})
        files = result.get("experimentById", {}).get("files", [])

        # Apply client-side filtering if tags specified
        if tags:
            filtered_files = []
            for file in files:
                file_tags = file.get("tags", [])
                if any(tag in file_tags for tag in tags):
                    filtered_files.append(file)
            return filtered_files

        return files

    def get_file(self, experiment_id: str, file_id: str) -> Dict[str, Any]:
        """
        Get file metadata using unified node API.

        Args:
            experiment_id: Experiment ID (DEPRECATED - not used in new API)
            file_id: File node ID (Snowflake ID)

        Returns:
            Node metadata dict

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        # file_id is actually the node ID in the new system
        response = self._client.get(f"nodes/{file_id}")
        response.raise_for_status()
        return response.json()

    def download_file(
        self,
        experiment_id: str,
        file_id: str,
        dest_path: Optional[str] = None
    ) -> str:
        """
        Download a file using unified node API.

        Args:
            experiment_id: Experiment ID (DEPRECATED - not used in new API)
            file_id: File node ID (Snowflake ID)
            dest_path: Optional destination path (defaults to original filename)

        Returns:
            Path to downloaded file

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If checksum verification fails
        """
        # Get file metadata first to get filename and checksum
        file_metadata = self.get_file(experiment_id, file_id)
        filename = file_metadata.get("name") or file_metadata.get("physicalFile", {}).get("filename")
        expected_checksum = file_metadata.get("physicalFile", {}).get("checksum")

        # Determine destination path
        if dest_path is None:
            dest_path = filename

        # Download file using node API
        response = self._client.get(f"nodes/{file_id}/download")
        response.raise_for_status()

        # Write to file
        with open(dest_path, "wb") as f:
            f.write(response.content)

        # Verify checksum if available
        if expected_checksum:
            from .files import verify_checksum
            if not verify_checksum(dest_path, expected_checksum):
                # Delete corrupted file
                import os
                os.remove(dest_path)
                raise ValueError(f"Checksum verification failed for file {file_id}")

        return dest_path

    def delete_file(self, experiment_id: str, file_id: str) -> Dict[str, Any]:
        """
        Delete a file using unified node API (soft delete).

        Args:
            experiment_id: Experiment ID (DEPRECATED - not used in new API)
            file_id: File node ID (Snowflake ID)

        Returns:
            Dict with id and deletedAt

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = self._client.delete(f"nodes/{file_id}")
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
        Update file metadata using unified node API.

        Args:
            experiment_id: Experiment ID (DEPRECATED - not used in new API)
            file_id: File node ID (Snowflake ID)
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Updated node metadata dict

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
            f"nodes/{file_id}",
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
            f"experiments/{experiment_id}/metrics/{metric_name}/append",
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
            f"experiments/{experiment_id}/metrics/{metric_name}/append-batch",
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
            f"experiments/{experiment_id}/metrics/{metric_name}/data",
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
            f"experiments/{experiment_id}/metrics/{metric_name}/stats"
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
        response = self._client.get(f"experiments/{experiment_id}/metrics")
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
        response = self._graphql_client.post(
            "/graphql",
            json={"query": query, "variables": variables or {}}
        )
        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")

        # Handle case where data is explicitly null in response
        return result.get("data") or {}

    def list_projects_graphql(self) -> List[Dict[str, Any]]:
        """
        List all projects via GraphQL.

        Namespace is automatically inferred from JWT token on the server.

        Returns:
            List of project dicts with experimentCount

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query Projects {
          projects {
            id
            name
            slug
            description
            tags
          }
        }
        """
        result = self.graphql_query(query, {})
        projects = result.get("projects", [])

        # For each project, count experiments
        for project in projects:
            exp_query = """
            query ExperimentsCount($projectSlug: String!) {
              experiments(projectSlug: $projectSlug) {
                id
              }
            }
            """
            exp_result = self.graphql_query(exp_query, {"projectSlug": project['slug']})
            experiments = exp_result.get("experiments", [])
            project['experimentCount'] = len(experiments)

        return projects

    def list_experiments_graphql(
        self, project_slug: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments in a project via GraphQL.

        Namespace is automatically inferred from JWT token on the server.

        Args:
            project_slug: Project slug
            status: Optional experiment status filter (RUNNING, COMPLETED, FAILED, CANCELLED)

        Returns:
            List of experiment dicts with metadata

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query Experiments($projectSlug: String!, $status: ExperimentStatus) {
          experiments(projectSlug: $projectSlug, status: $status) {
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
              namespace {
                slug
              }
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
              name
              pPath
              description
              tags
              metadata
              physicalFile {
                filename
                contentType
                sizeBytes
                checksum
                s3Url
              }
            }
            parameters {
              id
              data
            }
          }
        }
        """
        variables = {"projectSlug": project_slug}
        if status is not None:
            variables["status"] = status

        result = self.graphql_query(query, variables)
        return result.get("experiments", [])

    def get_experiment_graphql(
        self, project_slug: str, experiment_name: str, namespace_slug: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single experiment via GraphQL.

        Args:
            project_slug: Project slug
            experiment_name: Experiment name
            namespace_slug: Namespace slug (optional - defaults to client's namespace)

        Returns:
            Experiment dict with metadata, or None if not found

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        query = """
        query Experiment($namespaceSlug: String, $projectSlug: String!, $experimentName: String!) {
          experiment(namespaceSlug: $namespaceSlug, projectSlug: $projectSlug, experimentName: $experimentName) {
            id
            name
            description
            tags
            status
            metadata
            project {
              slug
              namespace {
                slug
              }
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
              name
              pPath
              description
              tags
              metadata
              physicalFile {
                filename
                contentType
                sizeBytes
                checksum
                s3Url
              }
            }
            parameters {
              id
              data
            }
          }
        }
        """
        # Use provided namespace or fall back to client's namespace
        ns = namespace_slug or self.namespace

        variables = {
            "namespaceSlug": ns,
            "projectSlug": project_slug,
            "experimentName": experiment_name
        }

        result = self.graphql_query(query, variables)
        return result.get("experiment")

    def search_experiments_graphql(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search experiments using glob pattern via GraphQL.

        Pattern format: namespace/project/experiment
        Supports wildcards: *, ?, [0-9], [a-z], etc.

        Args:
            pattern: Glob pattern (e.g., "tom*/tutorials/*", "*/project-?/exp*")

        Returns:
            List of experiment dicts matching the pattern

        Raises:
            httpx.HTTPStatusError: If request fails

        Examples:
            >>> client.search_experiments_graphql("tom*/tutorials/*")
            >>> client.search_experiments_graphql("*/my-project/baseline*")
        """
        query = """
        query SearchExperiments($pattern: String!) {
          searchExperiments(pattern: $pattern) {
            id
            name
            description
            tags
            status
            startedAt
            endedAt
            metadata
            project {
              id
              slug
              name
              namespace {
                id
                slug
              }
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
              name
              pPath
              description
              tags
              metadata
              physicalFile {
                filename
                contentType
                sizeBytes
                checksum
                s3Url
              }
            }
          }
        }
        """
        variables = {"pattern": pattern}
        result = self.graphql_query(query, variables)
        return result.get("searchExperiments", [])

    def download_file_streaming(
        self, experiment_id: str, file_id: str, dest_path: str
    ) -> str:
        """
        Download a file with streaming for large files using unified node API.

        Args:
            experiment_id: Experiment ID (DEPRECATED - not used in new API)
            file_id: File node ID (Snowflake ID)
            dest_path: Destination path to save file

        Returns:
            Path to downloaded file

        Raises:
            httpx.HTTPStatusError: If request fails
            ValueError: If checksum verification fails
        """
        # Get metadata first for checksum
        file_metadata = self.get_file(experiment_id, file_id)
        expected_checksum = file_metadata.get("physicalFile", {}).get("checksum")

        # Stream download using node API
        with self._client.stream("GET", f"nodes/{file_id}/download") as response:
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        # Verify checksum if available
        if expected_checksum:
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

        response = self._client.get(f"experiments/{experiment_id}/logs", params=params)
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
            f"experiments/{experiment_id}/metrics/{metric_name}/data",
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
            f"experiments/{experiment_id}/metrics/{metric_name}/chunks/{chunk_number}"
        )
        response.raise_for_status()
        return response.json()

    # =============================================================================
    # Track Methods
    # =============================================================================

    def create_track(
        self,
        experiment_id: str,
        topic: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new track for timestamped multi-modal data.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            topic: Track topic (e.g., "robot/position", "camera/rgb")
            description: Optional track description
            tags: Optional tags
            metadata: Optional metadata (e.g., fps, units)

        Returns:
            Dict with track ID and metadata

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = {
            "topic": topic,
        }
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        response = self._client.post(
            f"experiments/{experiment_id}/tracks",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def append_to_track(
        self,
        experiment_id: str,
        topic: str,
        timestamp: float,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Append a single entry to a track.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            topic: Track topic (e.g., "robot/position")
            timestamp: Numeric timestamp
            data: Data fields as dict (will be flattened with dot-notation)

        Returns:
            Dict with timestamp and flattened data

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        import urllib.parse

        topic_encoded = urllib.parse.quote(topic, safe='')

        response = self._client.post(
            f"experiments/{experiment_id}/tracks/{topic_encoded}/append",
            json={"timestamp": timestamp, "data": data},
        )
        response.raise_for_status()
        return response.json()

    def append_batch_to_track(
        self,
        experiment_id: str,
        topic: str,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Append multiple entries to a track in batch.

        Args:
            experiment_id: Experiment ID (Snowflake ID)
            topic: Track topic (e.g., "robot/position")
            entries: List of entries, each with 'timestamp' and other data fields

        Returns:
            Dict with count of entries added

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        import urllib.parse

        topic_encoded = urllib.parse.quote(topic, safe='')

        # Serialize entries to handle numpy arrays
        serialized_entries = [_serialize_value(entry) for entry in entries]

        response = self._client.post(
            f"experiments/{experiment_id}/tracks/{topic_encoded}/append_batch",
            json={"entries": serialized_entries},
        )
        response.raise_for_status()
        return response.json()

    def get_track_data(
        self,
        experiment_id: str,
        topic: str,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        columns: Optional[List[str]] = None,
        format: str = "json",
    ) -> Any:
        """
        Get track data with optional filtering.

        Args:
            experiment_id: Experiment ID
            topic: Track topic
            start_timestamp: Optional start timestamp filter
            end_timestamp: Optional end timestamp filter
            columns: Optional list of columns to retrieve
            format: Export format ('json', 'jsonl', 'parquet', 'mcap')

        Returns:
            Track data in requested format (dict for json, bytes for jsonl/parquet/mcap)

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        import urllib.parse

        topic_encoded = urllib.parse.quote(topic, safe='')
        params: Dict[str, str] = {"format": format}

        if start_timestamp is not None:
            params["start_timestamp"] = str(start_timestamp)
        if end_timestamp is not None:
            params["end_timestamp"] = str(end_timestamp)
        if columns:
            params["columns"] = ",".join(columns)

        response = self._client.get(
            f"experiments/{experiment_id}/tracks/{topic_encoded}/data",
            params=params,
        )
        response.raise_for_status()

        # Return bytes for binary formats, dict for JSON
        if format in ("jsonl", "parquet", "mcap"):
            return response.content
        return response.json()

    def list_tracks(
        self,
        experiment_id: str,
        topic_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all tracks in an experiment.

        Args:
            experiment_id: Experiment ID
            topic_filter: Optional topic filter (e.g., "robot/*" for prefix match)

        Returns:
            List of track metadata dicts

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        params: Dict[str, str] = {}
        if topic_filter:
            params["topic"] = topic_filter

        response = self._client.get(
            f"experiments/{experiment_id}/tracks",
            params=params,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("tracks", [])

    def close(self):
        """Close the HTTP clients."""
        self._client.close()
        self._graphql_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
