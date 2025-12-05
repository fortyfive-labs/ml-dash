"""Upload command implementation for ML-Dash CLI."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from ..storage import LocalStorage
from ..client import RemoteClient
from ..config import Config

# Initialize rich console
console = Console()


@dataclass
class ExperimentInfo:
    """Information about an experiment to upload."""
    project: str
    experiment: str
    path: Path
    folder: Optional[str] = None
    has_logs: bool = False
    has_params: bool = False
    metric_names: List[str] = field(default_factory=list)
    file_count: int = 0
    estimated_size: int = 0  # in bytes


@dataclass
class ValidationResult:
    """Result of experiment validation."""
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    valid_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadResult:
    """Result of uploading an experiment."""
    experiment: str
    success: bool = False
    uploaded: Dict[str, int] = field(default_factory=dict)  # {"logs": 100, "metrics": 3}
    failed: Dict[str, List[str]] = field(default_factory=dict)  # {"files": ["error msg"]}
    errors: List[str] = field(default_factory=list)
    bytes_uploaded: int = 0  # Total bytes uploaded


@dataclass
class UploadState:
    """Tracks upload state for resume functionality."""
    local_path: str
    remote_url: str
    completed_experiments: List[str] = field(default_factory=list)  # ["project/experiment"]
    failed_experiments: List[str] = field(default_factory=list)
    in_progress_experiment: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "local_path": self.local_path,
            "remote_url": self.remote_url,
            "completed_experiments": self.completed_experiments,
            "failed_experiments": self.failed_experiments,
            "in_progress_experiment": self.in_progress_experiment,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UploadState":
        """Create from dictionary."""
        return cls(
            local_path=data["local_path"],
            remote_url=data["remote_url"],
            completed_experiments=data.get("completed_experiments", []),
            failed_experiments=data.get("failed_experiments", []),
            in_progress_experiment=data.get("in_progress_experiment"),
            timestamp=data.get("timestamp"),
        )

    def save(self, path: Path):
        """Save state to file."""
        import datetime
        self.timestamp = datetime.datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["UploadState"]:
        """Load state from file."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError):
            return None


def generate_api_key_from_username(user_name: str) -> str:
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


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Add upload command parser."""
    parser = subparsers.add_parser(
        "upload",
        help="Upload local experiments to remote server",
        description="Upload locally-stored ML-Dash experiment data to a remote server.",
    )

    # Positional argument
    parser.add_argument(
        "path",
        nargs="?",
        default="./.ml-dash",
        help="Local storage directory to upload from (default: ./.ml-dash)",
    )

    # Remote configuration
    parser.add_argument(
        "--remote",
        type=str,
        help="Remote server URL (required unless set in config)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="JWT token for authentication (required unless --username or config is set)",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Username for authentication (generates API key automatically)",
    )

    # Scope control
    parser.add_argument(
        "--project",
        type=str,
        help="Upload only experiments from this project",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Upload only this specific experiment (requires --project)",
    )

    # Data filtering
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Don't upload logs",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Don't upload metrics",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Don't upload files",
    )
    parser.add_argument(
        "--skip-params",
        action="store_true",
        help="Don't upload parameters",
    )

    # Behavior control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any validation error (default: skip invalid data)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for logs/metrics (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous interrupted upload",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=".ml-dash-upload-state.json",
        help="Path to state file for resume (default: .ml-dash-upload-state.json)",
    )

    return parser


def discover_experiments(
    local_path: Path,
    project_filter: Optional[str] = None,
    experiment_filter: Optional[str] = None,
) -> List[ExperimentInfo]:
    """
    Discover experiments in local storage directory.

    Supports both flat (local_path/project/experiment) and folder-based
    (local_path/folder/project/experiment) hierarchies.

    Args:
        local_path: Root path of local storage
        project_filter: Only discover experiments in this project
        experiment_filter: Only discover this experiment (requires project_filter)

    Returns:
        List of ExperimentInfo objects
    """
    local_path = Path(local_path)

    if not local_path.exists():
        return []

    experiments = []

    # Find all experiment.json files recursively
    for exp_json in local_path.rglob("*/experiment.json"):
        exp_dir = exp_json.parent

        # Extract project and experiment names from path
        # Path structure: local_path / [folder] / project / experiment
        try:
            relative_path = exp_dir.relative_to(local_path)
            parts = relative_path.parts

            if len(parts) < 2:
                continue  # Need at least project/experiment

            # Last two parts are project/experiment
            exp_name = parts[-1]
            project_name = parts[-2]

            # Apply filters
            if project_filter and project_name != project_filter:
                continue
            if experiment_filter and exp_name != experiment_filter:
                continue

            # Read folder from experiment.json
            folder = None
            try:
                with open(exp_json, 'r') as f:
                    metadata = json.load(f)
                    folder = metadata.get('folder')
            except:
                pass

            # Create experiment info
            exp_info = ExperimentInfo(
                project=project_name,
                experiment=exp_name,
                path=exp_dir,
                folder=folder,
            )
        except (ValueError, IndexError):
            continue

        # Check for parameters
        params_file = exp_dir / "parameters.json"
        exp_info.has_params = params_file.exists()

        # Check for logs
        logs_file = exp_dir / "logs" / "logs.jsonl"
        exp_info.has_logs = logs_file.exists()

        # Check for metrics
        metrics_dir = exp_dir / "metrics"
        if metrics_dir.exists():
            for metric_dir in metrics_dir.iterdir():
                if metric_dir.is_dir():
                    data_file = metric_dir / "data.jsonl"
                    if data_file.exists():
                        exp_info.metric_names.append(metric_dir.name)

        # Check for files
        files_dir = exp_dir / "files"
        if files_dir.exists():
            try:
                # Count files recursively
                exp_info.file_count = sum(1 for _ in files_dir.rglob("*") if _.is_file())

                # Estimate size
                exp_info.estimated_size = sum(
                    f.stat().st_size for f in files_dir.rglob("*") if f.is_file()
                )
            except (OSError, PermissionError):
                pass

        experiments.append(exp_info)

    return experiments


class ExperimentValidator:
    """Validates local experiment data before upload."""

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, fail on any validation error
        """
        self.strict = strict

    def validate_experiment(self, exp_info: ExperimentInfo) -> ValidationResult:
        """
        Validate experiment directory structure and data.

        Args:
            exp_info: Experiment information

        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult()
        result.valid_data = {}

        # 1. Validate experiment metadata (required)
        if not self._validate_experiment_metadata(exp_info, result):
            result.is_valid = False
            return result

        # 2. Validate parameters (optional)
        self._validate_parameters(exp_info, result)

        # 3. Validate logs (optional)
        self._validate_logs(exp_info, result)

        # 4. Validate metrics (optional)
        self._validate_metrics(exp_info, result)

        # 5. Validate files (optional)
        self._validate_files(exp_info, result)

        # In strict mode, any warning becomes an error
        if self.strict and result.warnings:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.is_valid = False

        return result

    def _validate_experiment_metadata(self, exp_info: ExperimentInfo, result: ValidationResult) -> bool:
        """Validate experiment.json exists and is valid."""
        exp_json = exp_info.path / "experiment.json"

        if not exp_json.exists():
            result.errors.append("Missing experiment.json")
            return False

        try:
            with open(exp_json, "r") as f:
                metadata = json.load(f)

            # Check required fields
            if "name" not in metadata or "project" not in metadata:
                result.errors.append("experiment.json missing required fields (name, project)")
                return False

            result.valid_data["metadata"] = metadata
            return True

        except json.JSONDecodeError as e:
            result.errors.append(f"Invalid JSON in experiment.json: {e}")
            return False
        except IOError as e:
            result.errors.append(f"Cannot read experiment.json: {e}")
            return False

    def _validate_parameters(self, exp_info: ExperimentInfo, result: ValidationResult):
        """Validate parameters.json format."""
        if not exp_info.has_params:
            return

        params_file = exp_info.path / "parameters.json"
        try:
            with open(params_file, "r") as f:
                params = json.load(f)

            # Check if it's a dict
            if not isinstance(params, dict):
                result.warnings.append("parameters.json is not a dict (will skip)")
                return

            # Check for valid data key if using versioned format
            if "data" in params:
                if not isinstance(params["data"], dict):
                    result.warnings.append("parameters.json data is not a dict (will skip)")
                    return
                result.valid_data["parameters"] = params["data"]
            else:
                result.valid_data["parameters"] = params

        except json.JSONDecodeError as e:
            result.warnings.append(f"Invalid JSON in parameters.json: {e} (will skip)")
        except IOError as e:
            result.warnings.append(f"Cannot read parameters.json: {e} (will skip)")

    def _validate_logs(self, exp_info: ExperimentInfo, result: ValidationResult):
        """Validate logs.jsonl format."""
        if not exp_info.has_logs:
            return

        logs_file = exp_info.path / "logs" / "logs.jsonl"
        invalid_lines = []

        try:
            with open(logs_file, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    try:
                        log_entry = json.loads(line)
                        # Check required fields
                        if "message" not in log_entry:
                            invalid_lines.append(line_num)
                    except json.JSONDecodeError:
                        invalid_lines.append(line_num)

            if invalid_lines:
                count = len(invalid_lines)
                preview = invalid_lines[:5]
                result.warnings.append(
                    f"logs.jsonl has {count} invalid lines (e.g., {preview}...) - will skip these"
                )

        except IOError as e:
            result.warnings.append(f"Cannot read logs.jsonl: {e} (will skip logs)")

    def _validate_metrics(self, exp_info: ExperimentInfo, result: ValidationResult):
        """Validate metrics data."""
        if not exp_info.metric_names:
            return

        for metric_name in exp_info.metric_names:
            metric_dir = exp_info.path / "metrics" / metric_name
            data_file = metric_dir / "data.jsonl"

            invalid_lines = []
            try:
                with open(data_file, "r") as f:
                    for line_num, line in enumerate(f, start=1):
                        try:
                            data_point = json.loads(line)
                            # Check for data field
                            if "data" not in data_point:
                                invalid_lines.append(line_num)
                        except json.JSONDecodeError:
                            invalid_lines.append(line_num)

                if invalid_lines:
                    count = len(invalid_lines)
                    preview = invalid_lines[:5]
                    result.warnings.append(
                        f"metric '{metric_name}' has {count} invalid lines (e.g., {preview}...) - will skip these"
                    )

            except IOError as e:
                result.warnings.append(f"Cannot read metric '{metric_name}': {e} (will skip)")

    def _validate_files(self, exp_info: ExperimentInfo, result: ValidationResult):
        """Validate files existence."""
        files_dir = exp_info.path / "files"
        if not files_dir.exists():
            return

        metadata_file = files_dir / ".files_metadata.json"
        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r") as f:
                files_metadata = json.load(f)

            missing_files = []
            for file_id, file_info in files_metadata.items():
                if isinstance(file_info, dict) and file_info.get("deletedAt") is None:
                    # Check if file exists
                    file_path = files_dir / file_info.get("prefix", "") / file_id / file_info.get("filename", "")
                    if not file_path.exists():
                        missing_files.append(file_info.get("filename", file_id))

            if missing_files:
                count = len(missing_files)
                preview = missing_files[:3]
                result.warnings.append(
                    f"{count} files referenced in metadata but missing on disk (e.g., {preview}...) - will skip these"
                )

        except (json.JSONDecodeError, IOError):
            pass  # If we can't read metadata, just skip file validation


class ExperimentUploader:
    """Handles uploading a single experiment."""

    def __init__(
        self,
        local_storage: LocalStorage,
        remote_client: RemoteClient,
        batch_size: int = 100,
        skip_logs: bool = False,
        skip_metrics: bool = False,
        skip_files: bool = False,
        skip_params: bool = False,
        verbose: bool = False,
        progress: Optional[Progress] = None,
        max_concurrent_metrics: int = 5,
    ):
        """
        Initialize uploader.

        Args:
            local_storage: Local storage instance
            remote_client: Remote client instance
            batch_size: Batch size for logs/metrics
            skip_logs: Skip uploading logs
            skip_metrics: Skip uploading metrics
            skip_files: Skip uploading files
            skip_params: Skip uploading parameters
            verbose: Show verbose output
            progress: Optional rich Progress instance for tracking
            max_concurrent_metrics: Maximum concurrent metric uploads (default: 5)
        """
        self.local = local_storage
        self.remote = remote_client
        self.batch_size = batch_size
        self.skip_logs = skip_logs
        self.skip_metrics = skip_metrics
        self.skip_files = skip_files
        self.skip_params = skip_params
        self.verbose = verbose
        self.progress = progress
        self.max_concurrent_metrics = max_concurrent_metrics
        # Thread-safe lock for shared state updates
        self._lock = threading.Lock()
        # Thread-local storage for remote clients (for thread-safe HTTP requests)
        self._thread_local = threading.local()

    def _get_remote_client(self) -> RemoteClient:
        """Get thread-local remote client for safe concurrent access."""
        if not hasattr(self._thread_local, 'client'):
            # Create a new client for this thread
            self._thread_local.client = RemoteClient(
                base_url=self.remote.base_url,
                api_key=self.remote.api_key
            )
        return self._thread_local.client

    def upload_experiment(
        self, exp_info: ExperimentInfo, validation_result: ValidationResult, task_id=None
    ) -> UploadResult:
        """
        Upload a single experiment with all its data.

        Args:
            exp_info: Experiment information
            validation_result: Validation results
            task_id: Optional progress task ID

        Returns:
            UploadResult with upload status
        """
        result = UploadResult(experiment=f"{exp_info.project}/{exp_info.experiment}")

        # Calculate total steps for progress tracking
        total_steps = 1  # metadata
        if not self.skip_params and "parameters" in validation_result.valid_data:
            total_steps += 1
        if not self.skip_logs and exp_info.has_logs:
            total_steps += 1
        if not self.skip_metrics and exp_info.metric_names:
            total_steps += len(exp_info.metric_names)
        if not self.skip_files and exp_info.file_count > 0:
            total_steps += exp_info.file_count

        current_step = 0

        def update_progress(description: str):
            nonlocal current_step
            current_step += 1
            if self.progress and task_id is not None:
                self.progress.update(task_id, completed=current_step, total=total_steps, description=description)

        try:
            # 1. Create/update experiment metadata
            update_progress("Creating experiment...")
            if self.verbose:
                console.print(f"  [dim]Creating experiment...[/dim]")

            exp_data = validation_result.valid_data

            # Store folder path in metadata (not as folderId which expects Snowflake ID)
            custom_metadata = exp_data.get("metadata") or {}
            if exp_data.get("folder"):
                custom_metadata["folder"] = exp_data["folder"]

            response = self.remote.create_or_update_experiment(
                project=exp_info.project,
                name=exp_info.experiment,
                description=exp_data.get("description"),
                tags=exp_data.get("tags"),
                bindrs=exp_data.get("bindrs"),
                folder=None,  # Don't send folder path as folderId (expects Snowflake ID)
                write_protected=exp_data.get("write_protected", False),
                metadata=custom_metadata if custom_metadata else None,
            )

            # Extract experiment ID from nested response
            experiment_id = response.get("experiment", {}).get("id") or response.get("id")
            if self.verbose:
                console.print(f"  [green]✓[/green] Created experiment (id: {experiment_id})")

            # 2. Upload parameters
            if not self.skip_params and "parameters" in validation_result.valid_data:
                update_progress("Uploading parameters...")
                if self.verbose:
                    console.print(f"  [dim]Uploading parameters...[/dim]")

                params = validation_result.valid_data["parameters"]
                self.remote.set_parameters(experiment_id, params)
                result.uploaded["params"] = len(params)
                # Track bytes (approximate JSON size)
                result.bytes_uploaded += len(json.dumps(params).encode('utf-8'))

                if self.verbose:
                    console.print(f"  [green]✓[/green] Uploaded {len(params)} parameters")

            # 3. Upload logs
            if not self.skip_logs and exp_info.has_logs:
                count = self._upload_logs(experiment_id, exp_info, result, task_id, update_progress)
                result.uploaded["logs"] = count

            # 4. Upload metrics
            if not self.skip_metrics and exp_info.metric_names:
                count = self._upload_metrics(experiment_id, exp_info, result, task_id, update_progress)
                result.uploaded["metrics"] = count

            # 5. Upload files
            if not self.skip_files and exp_info.file_count > 0:
                count = self._upload_files(experiment_id, exp_info, result, task_id, update_progress)
                result.uploaded["files"] = count

            result.success = True

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            if self.verbose:
                console.print(f"  [red]✗ Error: {e}[/red]")

        return result

    def _upload_logs(self, experiment_id: str, exp_info: ExperimentInfo, result: UploadResult,
                     task_id=None, update_progress=None) -> int:
        """Upload logs in batches."""
        if update_progress:
            update_progress("Uploading logs...")
        if self.verbose:
            console.print(f"  [dim]Uploading logs...[/dim]")

        logs_file = exp_info.path / "logs" / "logs.jsonl"
        logs_batch = []
        total_uploaded = 0
        skipped = 0

        try:
            with open(logs_file, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)

                        # Validate required fields
                        if "message" not in log_entry:
                            skipped += 1
                            continue

                        # Prepare log entry for API
                        api_log = {
                            "timestamp": log_entry.get("timestamp"),
                            "level": log_entry.get("level", "info"),
                            "message": log_entry["message"],
                        }
                        if "metadata" in log_entry:
                            api_log["metadata"] = log_entry["metadata"]

                        logs_batch.append(api_log)
                        # Track bytes
                        result.bytes_uploaded += len(line.encode('utf-8'))

                        # Upload batch
                        if len(logs_batch) >= self.batch_size:
                            self.remote.create_log_entries(experiment_id, logs_batch)
                            total_uploaded += len(logs_batch)
                            logs_batch = []

                    except json.JSONDecodeError:
                        skipped += 1
                        continue

            # Upload remaining logs
            if logs_batch:
                self.remote.create_log_entries(experiment_id, logs_batch)
                total_uploaded += len(logs_batch)

            if self.verbose:
                msg = f"  [green]✓[/green] Uploaded {total_uploaded} log entries"
                if skipped > 0:
                    msg += f" (skipped {skipped} invalid)"
                console.print(msg)

        except IOError as e:
            result.failed.setdefault("logs", []).append(str(e))

        return total_uploaded

    def _upload_single_metric(
        self,
        experiment_id: str,
        metric_name: str,
        metric_dir: Path,
        result: UploadResult
    ) -> Dict[str, Any]:
        """
        Upload a single metric (thread-safe helper).

        Returns:
            Dict with 'success', 'uploaded', 'skipped', 'bytes', and 'error' keys
        """
        data_file = metric_dir / "data.jsonl"
        data_batch = []
        total_uploaded = 0
        skipped = 0
        bytes_uploaded = 0

        # Get thread-local client for safe concurrent HTTP requests
        remote_client = self._get_remote_client()

        try:
            with open(data_file, "r") as f:
                for line in f:
                    try:
                        data_point = json.loads(line)

                        # Validate required fields
                        if "data" not in data_point:
                            skipped += 1
                            continue

                        data_batch.append(data_point["data"])
                        bytes_uploaded += len(line.encode('utf-8'))

                        # Upload batch using thread-local client
                        if len(data_batch) >= self.batch_size:
                            remote_client.append_batch_to_metric(
                                experiment_id, metric_name, data_batch
                            )
                            total_uploaded += len(data_batch)
                            data_batch = []

                    except json.JSONDecodeError:
                        skipped += 1
                        continue

            # Upload remaining data points using thread-local client
            if data_batch:
                remote_client.append_batch_to_metric(experiment_id, metric_name, data_batch)
                total_uploaded += len(data_batch)

            return {
                'success': True,
                'uploaded': total_uploaded,
                'skipped': skipped,
                'bytes': bytes_uploaded,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'uploaded': 0,
                'skipped': 0,
                'bytes': 0,
                'error': str(e)
            }

    def _upload_metrics(self, experiment_id: str, exp_info: ExperimentInfo, result: UploadResult,
                        task_id=None, update_progress=None) -> int:
        """Upload metrics in parallel with concurrency limit."""
        if not exp_info.metric_names:
            return 0

        total_metrics = 0

        # Use ThreadPoolExecutor for parallel uploads
        with ThreadPoolExecutor(max_workers=self.max_concurrent_metrics) as executor:
            # Submit all metric upload tasks
            future_to_metric = {}
            for metric_name in exp_info.metric_names:
                metric_dir = exp_info.path / "metrics" / metric_name
                future = executor.submit(
                    self._upload_single_metric,
                    experiment_id,
                    metric_name,
                    metric_dir,
                    result
                )
                future_to_metric[future] = metric_name

            # Process completed uploads as they finish
            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]

                # Update progress
                if update_progress:
                    update_progress(f"Uploading metric '{metric_name}'...")

                try:
                    upload_result = future.result()

                    # Thread-safe update of shared state
                    with self._lock:
                        result.bytes_uploaded += upload_result['bytes']

                    if upload_result['success']:
                        total_metrics += 1

                        # Thread-safe console output
                        if self.verbose:
                            msg = f"  [green]✓[/green] Uploaded {upload_result['uploaded']} data points for '{metric_name}'"
                            if upload_result['skipped'] > 0:
                                msg += f" (skipped {upload_result['skipped']} invalid)"
                            with self._lock:
                                console.print(msg)
                    else:
                        # Record failure
                        error_msg = f"{metric_name}: {upload_result['error']}"
                        with self._lock:
                            result.failed.setdefault("metrics", []).append(error_msg)
                            if self.verbose:
                                console.print(f"  [red]✗[/red] Failed to upload '{metric_name}': {upload_result['error']}")

                except Exception as e:
                    # Handle unexpected errors
                    error_msg = f"{metric_name}: {str(e)}"
                    with self._lock:
                        result.failed.setdefault("metrics", []).append(error_msg)
                        if self.verbose:
                            console.print(f"  [red]✗[/red] Failed to upload '{metric_name}': {e}")

        return total_metrics

    def _upload_files(self, experiment_id: str, exp_info: ExperimentInfo, result: UploadResult,
                      task_id=None, update_progress=None) -> int:
        """Upload files one by one."""
        files_dir = exp_info.path / "files"
        total_uploaded = 0

        # Use LocalStorage to list files
        try:
            files_list = self.local.list_files(exp_info.project, exp_info.experiment)

            for file_info in files_list:
                # Skip deleted files
                if file_info.get("deletedAt") is not None:
                    continue

                try:
                    if update_progress:
                        update_progress(f"Uploading {file_info['filename']}...")

                    # Get file path directly from storage without copying
                    file_id = file_info["id"]
                    experiment_dir = self.local._get_experiment_dir(exp_info.project, exp_info.experiment)
                    files_dir = experiment_dir / "files"

                    # Construct file path
                    file_prefix = file_info["path"].lstrip("/") if file_info["path"] else ""
                    if file_prefix:
                        file_path = files_dir / file_prefix / file_id / file_info["filename"]
                    else:
                        file_path = files_dir / file_id / file_info["filename"]

                    # Upload to remote with correct parameters
                    self.remote.upload_file(
                        experiment_id=experiment_id,
                        file_path=str(file_path),
                        prefix=file_info.get("path", ""),
                        filename=file_info["filename"],
                        description=file_info.get("description"),
                        tags=file_info.get("tags", []),
                        metadata=file_info.get("metadata"),
                        checksum=file_info["checksum"],
                        content_type=file_info["contentType"],
                        size_bytes=file_info["sizeBytes"],
                    )

                    total_uploaded += 1
                    # Track bytes
                    result.bytes_uploaded += file_info.get("sizeBytes", 0)

                    if self.verbose:
                        size_mb = file_info.get("sizeBytes", 0) / (1024 * 1024)
                        console.print(f"    [green]✓[/green] {file_info['filename']} ({size_mb:.1f}MB)")

                except Exception as e:
                    result.failed.setdefault("files", []).append(f"{file_info['filename']}: {e}")

        except Exception as e:
            result.failed.setdefault("files", []).append(str(e))

        if self.verbose and not result.failed.get("files"):
            console.print(f"  [green]✓[/green] Uploaded {total_uploaded} files")

        return total_uploaded


def cmd_upload(args: argparse.Namespace) -> int:
    """
    Execute upload command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Load config
    config = Config()

    # Get remote URL (command line > config)
    remote_url = args.remote or config.remote_url
    if not remote_url:
        console.print("[red]Error:[/red] --remote URL is required (or set in config)")
        return 1

    # Get API key (command line > config > generate from username)
    api_key = args.api_key or config.api_key

    # If no API key, try to generate from username
    if not api_key:
        if args.username:
            console.print(f"[dim]Generating API key from username: {args.username}[/dim]")
            api_key = generate_api_key_from_username(args.username)
        else:
            console.print("[red]Error:[/red] --api-key or --username is required (or set in config)")
            return 1

    # Validate experiment filter requires project
    if args.experiment and not args.project:
        console.print("[red]Error:[/red] --experiment requires --project")
        return 1

    # Discover experiments
    local_path = Path(args.path)
    if not local_path.exists():
        console.print(f"[red]Error:[/red] Local storage path does not exist: {local_path}")
        return 1

    # Handle state file for resume functionality
    state_file = Path(args.state_file)
    upload_state = None

    if args.resume:
        upload_state = UploadState.load(state_file)
        if upload_state:
            # Validate state matches current upload
            if upload_state.local_path != str(local_path.absolute()):
                console.print("[yellow]Warning:[/yellow] State file local path doesn't match. Starting fresh upload.")
                upload_state = None
            elif upload_state.remote_url != remote_url:
                console.print("[yellow]Warning:[/yellow] State file remote URL doesn't match. Starting fresh upload.")
                upload_state = None
            else:
                console.print(f"[green]Resuming previous upload from {upload_state.timestamp}[/green]")
                console.print(f"  Already completed: {len(upload_state.completed_experiments)} experiments")
                console.print(f"  Failed: {len(upload_state.failed_experiments)} experiments")
        else:
            console.print("[yellow]No previous upload state found. Starting fresh upload.[/yellow]")

    # Create new state if not resuming
    if not upload_state:
        upload_state = UploadState(
            local_path=str(local_path.absolute()),
            remote_url=remote_url,
        )

    console.print(f"[bold]Scanning local storage:[/bold] {local_path.absolute()}")
    experiments = discover_experiments(
        local_path,
        project_filter=args.project,
        experiment_filter=args.experiment,
    )

    if not experiments:
        if args.project and args.experiment:
            console.print(f"[yellow]No experiment found:[/yellow] {args.project}/{args.experiment}")
        elif args.project:
            console.print(f"[yellow]No experiments found in project:[/yellow] {args.project}")
        else:
            console.print("[yellow]No experiments found in local storage[/yellow]")
        return 1

    # Filter out already completed experiments when resuming
    if args.resume and upload_state.completed_experiments:
        original_count = len(experiments)
        experiments = [
            exp for exp in experiments
            if f"{exp.project}/{exp.experiment}" not in upload_state.completed_experiments
        ]
        skipped_count = original_count - len(experiments)
        if skipped_count > 0:
            console.print(f"[dim]Skipping {skipped_count} already completed experiment(s)[/dim]")

    console.print(f"[green]Found {len(experiments)} experiment(s) to upload[/green]")

    # Display discovered experiments
    if args.verbose or args.dry_run:
        console.print("\n[bold]Discovered experiments:[/bold]")
        for exp in experiments:
            parts = []
            if exp.has_logs:
                parts.append("logs")
            if exp.has_params:
                parts.append("params")
            if exp.metric_names:
                parts.append(f"{len(exp.metric_names)} metrics")
            if exp.file_count:
                size_mb = exp.estimated_size / (1024 * 1024)
                parts.append(f"{exp.file_count} files ({size_mb:.1f}MB)")

            details = ", ".join(parts) if parts else "metadata only"
            console.print(f"  [cyan]•[/cyan] {exp.project}/{exp.experiment} [dim]({details})[/dim]")

    # Dry-run mode: stop here
    if args.dry_run:
        console.print("\n[yellow bold]DRY RUN[/yellow bold] - No data will be uploaded")
        console.print("Run without --dry-run to proceed with upload.")
        return 0

    # Validate experiments
    console.print("\n[bold]Validating experiments...[/bold]")
    validator = ExperimentValidator(strict=args.strict)
    validation_results = {}
    valid_experiments = []
    invalid_experiments = []

    for exp in experiments:
        validation = validator.validate_experiment(exp)
        validation_results[f"{exp.project}/{exp.experiment}"] = validation

        if validation.is_valid:
            valid_experiments.append(exp)
        else:
            invalid_experiments.append(exp)

        # Show warnings and errors
        if args.verbose or validation.errors:
            exp_key = f"{exp.project}/{exp.experiment}"
            if validation.errors:
                console.print(f"  [red]✗[/red] {exp_key}:")
                for error in validation.errors:
                    console.print(f"      [red]{error}[/red]")
            elif validation.warnings:
                console.print(f"  [yellow]⚠[/yellow] {exp_key}:")
                for warning in validation.warnings:
                    console.print(f"      [yellow]{warning}[/yellow]")

    if invalid_experiments:
        console.print(f"\n[yellow]{len(invalid_experiments)} experiment(s) failed validation and will be skipped[/yellow]")
        if args.strict:
            console.print("[red]Error: Validation failed in --strict mode[/red]")
            return 1

    if not valid_experiments:
        console.print("[red]Error: No valid experiments to upload[/red]")
        return 1

    console.print(f"[green]{len(valid_experiments)} experiment(s) ready to upload[/green]")

    # Initialize remote client and local storage
    remote_client = RemoteClient(base_url=remote_url, api_key=api_key)
    local_storage = LocalStorage(root_path=local_path)

    # Upload experiments with progress tracking
    console.print(f"\n[bold]Uploading to:[/bold] {remote_url}")
    results = []

    # Track upload timing
    import time
    start_time = time.time()

    # Create progress bar for overall upload
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=not args.verbose,  # Keep progress visible in verbose mode
    ) as progress:
        # Create uploader with progress tracking
        uploader = ExperimentUploader(
            local_storage=local_storage,
            remote_client=remote_client,
            batch_size=args.batch_size,
            skip_logs=args.skip_logs,
            skip_metrics=args.skip_metrics,
            skip_files=args.skip_files,
            skip_params=args.skip_params,
            verbose=args.verbose,
            progress=progress,
        )

        for i, exp in enumerate(valid_experiments, start=1):
            exp_key = f"{exp.project}/{exp.experiment}"

            # Create task for this experiment
            task_id = progress.add_task(
                f"[{i}/{len(valid_experiments)}] {exp_key}",
                total=100,  # Will be updated with actual steps
            )

            # Update state - mark as in progress
            upload_state.in_progress_experiment = exp_key
            if not args.dry_run:
                upload_state.save(state_file)

            validation = validation_results[exp_key]
            result = uploader.upload_experiment(exp, validation, task_id=task_id)
            results.append(result)

            # Update state - mark as completed or failed
            upload_state.in_progress_experiment = None
            if result.success:
                upload_state.completed_experiments.append(exp_key)
            else:
                upload_state.failed_experiments.append(exp_key)

            if not args.dry_run:
                upload_state.save(state_file)

            # Update task to completed
            progress.update(task_id, completed=100, total=100)

            if not args.verbose:
                # Show brief status
                if result.success:
                    parts = []
                    if result.uploaded.get("params"):
                        parts.append(f"{result.uploaded['params']} params")
                    if result.uploaded.get("logs"):
                        parts.append(f"{result.uploaded['logs']} logs")
                    if result.uploaded.get("metrics"):
                        parts.append(f"{result.uploaded['metrics']} metrics")
                    if result.uploaded.get("files"):
                        parts.append(f"{result.uploaded['files']} files")
                    status = ", ".join(parts) if parts else "metadata only"
                    console.print(f"  [green]✓[/green] Uploaded ({status})")
                else:
                    console.print(f"  [red]✗[/red] Failed")
                    if result.errors:
                        for error in result.errors[:3]:  # Show first 3 errors
                            console.print(f"      [red]{error}[/red]")

    # Calculate timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_bytes = sum(r.bytes_uploaded for r in results)

    # Print summary with rich Table
    console.print()

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Create summary table
    summary_table = Table(title="Upload Summary", show_header=True, header_style="bold")
    summary_table.add_column("Status", style="cyan")
    summary_table.add_column("Count", justify="right")

    summary_table.add_row("Successful", f"[green]{len(successful)}/{len(results)}[/green]")
    if failed:
        summary_table.add_row("Failed", f"[red]{len(failed)}/{len(results)}[/red]")

    # Add timing information
    summary_table.add_row("Total Time", f"{elapsed_time:.2f}s")

    # Calculate and display upload speed
    if total_bytes > 0 and elapsed_time > 0:
        # Convert to appropriate unit
        if total_bytes < 1024 * 1024:  # Less than 1 MB
            speed_kb = (total_bytes / 1024) / elapsed_time
            summary_table.add_row("Avg Speed", f"{speed_kb:.2f} KB/s")
        else:  # 1 MB or more
            speed_mb = (total_bytes / (1024 * 1024)) / elapsed_time
            summary_table.add_row("Avg Speed", f"{speed_mb:.2f} MB/s")

    console.print(summary_table)

    # Show failed experiments
    if failed:
        console.print("\n[bold red]Failed Experiments:[/bold red]")
        for result in failed:
            console.print(f"  [red]✗[/red] {result.experiment}")
            for error in result.errors:
                console.print(f"      [dim]{error}[/dim]")

    # Data statistics
    total_logs = sum(r.uploaded.get("logs", 0) for r in results)
    total_metrics = sum(r.uploaded.get("metrics", 0) for r in results)
    total_files = sum(r.uploaded.get("files", 0) for r in results)

    if total_logs or total_metrics or total_files:
        data_table = Table(title="Data Uploaded", show_header=True, header_style="bold")
        data_table.add_column("Type", style="cyan")
        data_table.add_column("Count", justify="right", style="green")

        if total_logs:
            data_table.add_row("Logs", f"{total_logs} entries")
        if total_metrics:
            data_table.add_row("Metrics", f"{total_metrics} metrics")
        if total_files:
            data_table.add_row("Files", f"{total_files} files")

        console.print()
        console.print(data_table)

    # Clean up state file if all uploads succeeded
    if not args.dry_run and len(failed) == 0 and state_file.exists():
        state_file.unlink()
        console.print("\n[dim]Upload complete. State file removed.[/dim]")
    elif not args.dry_run and failed:
        console.print(f"\n[yellow]State saved to {state_file}. Use --resume to retry failed uploads.[/yellow]")

    # Return exit code
    return 0 if len(failed) == 0 else 1
