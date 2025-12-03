"""Upload command implementation for ML-Dash CLI."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..storage import LocalStorage
from ..client import RemoteClient
from ..config import Config


@dataclass
class ExperimentInfo:
    """Information about an experiment to upload."""
    project: str
    experiment: str
    path: Path
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
        help="JWT token for authentication (required unless set in config)",
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

    return parser


def discover_experiments(
    local_path: Path,
    project_filter: Optional[str] = None,
    experiment_filter: Optional[str] = None,
) -> List[ExperimentInfo]:
    """
    Discover experiments in local storage directory.

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

    # Iterate through projects
    for project_dir in local_path.iterdir():
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name

        # Apply project filter
        if project_filter and project_name != project_filter:
            continue

        # Iterate through experiments in project
        for exp_dir in project_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name

            # Apply experiment filter
            if experiment_filter and exp_name != experiment_filter:
                continue

            # Check if experiment.json exists (required)
            exp_json = exp_dir / "experiment.json"
            if not exp_json.exists():
                continue

            # Create experiment info
            exp_info = ExperimentInfo(
                project=project_name,
                experiment=exp_name,
                path=exp_dir,
            )

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
        """
        self.local = local_storage
        self.remote = remote_client
        self.batch_size = batch_size
        self.skip_logs = skip_logs
        self.skip_metrics = skip_metrics
        self.skip_files = skip_files
        self.skip_params = skip_params
        self.verbose = verbose

    def upload_experiment(
        self, exp_info: ExperimentInfo, validation_result: ValidationResult
    ) -> UploadResult:
        """
        Upload a single experiment with all its data.

        Args:
            exp_info: Experiment information
            validation_result: Validation results

        Returns:
            UploadResult with upload status
        """
        result = UploadResult(experiment=f"{exp_info.project}/{exp_info.experiment}")

        try:
            # 1. Create/update experiment metadata
            if self.verbose:
                print(f"  Creating experiment...")

            metadata = validation_result.valid_data.get("metadata", {})
            response = self.remote.create_or_update_experiment(
                project=exp_info.project,
                name=exp_info.experiment,
                description=metadata.get("description"),
                tags=metadata.get("tags"),
                bindrs=metadata.get("bindrs"),
                folder=metadata.get("folder"),
                write_protected=metadata.get("write_protected", False),
                metadata=metadata.get("metadata"),
            )

            experiment_id = response["id"]
            if self.verbose:
                print(f"  ✓ Created experiment (id: {experiment_id})")

            # 2. Upload parameters
            if not self.skip_params and "parameters" in validation_result.valid_data:
                if self.verbose:
                    print(f"  Uploading parameters...")

                params = validation_result.valid_data["parameters"]
                self.remote.set_parameters(experiment_id, params)
                result.uploaded["params"] = len(params)

                if self.verbose:
                    print(f"  ✓ Uploaded {len(params)} parameters")

            # 3. Upload logs
            if not self.skip_logs and exp_info.has_logs:
                count = self._upload_logs(experiment_id, exp_info, result)
                result.uploaded["logs"] = count

            # 4. Upload metrics
            if not self.skip_metrics and exp_info.metric_names:
                count = self._upload_metrics(experiment_id, exp_info, result)
                result.uploaded["metrics"] = count

            # 5. Upload files
            if not self.skip_files and exp_info.file_count > 0:
                count = self._upload_files(experiment_id, exp_info, result)
                result.uploaded["files"] = count

            result.success = True

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            if self.verbose:
                print(f"  ✗ Error: {e}")

        return result

    def _upload_logs(self, experiment_id: str, exp_info: ExperimentInfo, result: UploadResult) -> int:
        """Upload logs in batches."""
        if self.verbose:
            print(f"  Uploading logs...")

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
                msg = f"  ✓ Uploaded {total_uploaded} log entries"
                if skipped > 0:
                    msg += f" (skipped {skipped} invalid)"
                print(msg)

        except IOError as e:
            result.failed.setdefault("logs", []).append(str(e))

        return total_uploaded

    def _upload_metrics(self, experiment_id: str, exp_info: ExperimentInfo, result: UploadResult) -> int:
        """Upload metrics in batches."""
        total_metrics = 0

        for metric_name in exp_info.metric_names:
            if self.verbose:
                print(f"  Uploading metric '{metric_name}'...")

            metric_dir = exp_info.path / "metrics" / metric_name
            data_file = metric_dir / "data.jsonl"

            data_batch = []
            total_uploaded = 0
            skipped = 0

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

                            # Upload batch
                            if len(data_batch) >= self.batch_size:
                                self.remote.append_batch_to_metric(
                                    experiment_id, metric_name, data_batch
                                )
                                total_uploaded += len(data_batch)
                                data_batch = []

                        except json.JSONDecodeError:
                            skipped += 1
                            continue

                # Upload remaining data points
                if data_batch:
                    self.remote.append_batch_to_metric(experiment_id, metric_name, data_batch)
                    total_uploaded += len(data_batch)

                if self.verbose:
                    msg = f"  ✓ Uploaded {total_uploaded} data points for '{metric_name}'"
                    if skipped > 0:
                        msg += f" (skipped {skipped} invalid)"
                    print(msg)

                total_metrics += 1

            except IOError as e:
                result.failed.setdefault("metrics", []).append(f"{metric_name}: {e}")

        return total_metrics

    def _upload_files(self, experiment_id: str, exp_info: ExperimentInfo, result: UploadResult) -> int:
        """Upload files one by one."""
        if self.verbose:
            print(f"  Uploading files...")

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
                    # Download file to temp location
                    file_id = file_info["id"]
                    file_path = self.local.read_file(
                        exp_info.project, exp_info.experiment, file_id
                    )

                    # Upload to remote
                    with open(file_path, "rb") as f:
                        self.remote.upload_file(
                            experiment_id=experiment_id,
                            file=f,
                            filename=file_info["filename"],
                            prefix=file_info.get("prefix", ""),
                            description=file_info.get("description"),
                            tags=file_info.get("tags", []),
                            metadata=file_info.get("metadata"),
                        )

                    total_uploaded += 1

                    if self.verbose:
                        size_mb = file_info.get("sizeBytes", 0) / (1024 * 1024)
                        print(f"    ✓ {file_info['filename']} ({size_mb:.1f}MB)")

                except Exception as e:
                    result.failed.setdefault("files", []).append(f"{file_info['filename']}: {e}")

        except Exception as e:
            result.failed.setdefault("files", []).append(str(e))

        if self.verbose and not result.failed.get("files"):
            print(f"  ✓ Uploaded {total_uploaded} files")

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
        print("Error: --remote URL is required (or set in config with 'ml-dash setup')")
        return 1

    # Get API key (command line > config)
    api_key = args.api_key or config.api_key
    if not api_key:
        print("Error: --api-key is required (or set in config with 'ml-dash setup')")
        return 1

    # Validate experiment filter requires project
    if args.experiment and not args.project:
        print("Error: --experiment requires --project")
        return 1

    # Discover experiments
    local_path = Path(args.path)
    if not local_path.exists():
        print(f"Error: Local storage path does not exist: {local_path}")
        return 1

    print(f"Scanning local storage: {local_path.absolute()}")
    experiments = discover_experiments(
        local_path,
        project_filter=args.project,
        experiment_filter=args.experiment,
    )

    if not experiments:
        if args.project and args.experiment:
            print(f"No experiment found: {args.project}/{args.experiment}")
        elif args.project:
            print(f"No experiments found in project: {args.project}")
        else:
            print("No experiments found in local storage")
        return 1

    print(f"Found {len(experiments)} experiment(s)")

    # Display discovered experiments
    if args.verbose or args.dry_run:
        print("\nDiscovered experiments:")
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
            print(f"  - {exp.project}/{exp.experiment} ({details})")

    # Dry-run mode: stop here
    if args.dry_run:
        print("\nDRY RUN - No data will be uploaded")
        print("Run without --dry-run to proceed with upload.")
        return 0

    # Validate experiments
    print("\nValidating experiments...")
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
                print(f"  ✗ {exp_key}:")
                for error in validation.errors:
                    print(f"      {error}")
            elif validation.warnings:
                print(f"  ⚠ {exp_key}:")
                for warning in validation.warnings:
                    print(f"      {warning}")

    if invalid_experiments:
        print(f"\n{len(invalid_experiments)} experiment(s) failed validation and will be skipped")
        if args.strict:
            print("Error: Validation failed in --strict mode")
            return 1

    if not valid_experiments:
        print("Error: No valid experiments to upload")
        return 1

    print(f"{len(valid_experiments)} experiment(s) ready to upload")

    # Initialize remote client and local storage
    remote_client = RemoteClient(base_url=remote_url, api_key=api_key)
    local_storage = LocalStorage(root_path=local_path)

    # Create uploader
    uploader = ExperimentUploader(
        local_storage=local_storage,
        remote_client=remote_client,
        batch_size=args.batch_size,
        skip_logs=args.skip_logs,
        skip_metrics=args.skip_metrics,
        skip_files=args.skip_files,
        skip_params=args.skip_params,
        verbose=args.verbose,
    )

    # Upload experiments
    print(f"\nUploading to: {remote_url}")
    results = []

    for i, exp in enumerate(valid_experiments, start=1):
        exp_key = f"{exp.project}/{exp.experiment}"
        print(f"\n[{i}/{len(valid_experiments)}] Uploading {exp_key}")

        validation = validation_results[exp_key]
        result = uploader.upload_experiment(exp, validation)
        results.append(result)

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
                print(f"  ✓ Uploaded ({status})")
            else:
                print(f"  ✗ Failed")
                if result.errors:
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"      {error}")

    # Print summary
    print("\n" + "=" * 60)
    print("Upload Summary")
    print("=" * 60)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"Successful: {len(successful)}/{len(results)} experiments")
    if failed:
        print(f"Failed: {len(failed)}/{len(results)} experiments")
        for result in failed:
            print(f"  ✗ {result.experiment}")
            for error in result.errors:
                print(f"      {error}")

    # Data statistics
    total_logs = sum(r.uploaded.get("logs", 0) for r in results)
    total_metrics = sum(r.uploaded.get("metrics", 0) for r in results)
    total_files = sum(r.uploaded.get("files", 0) for r in results)

    if total_logs or total_metrics or total_files:
        print("\nData Uploaded:")
        if total_logs:
            print(f"  Logs: {total_logs} entries")
        if total_metrics:
            print(f"  Metrics: {total_metrics} metrics")
        if total_files:
            print(f"  Files: {total_files} files")

    # Return exit code
    return 0 if len(failed) == 0 else 1
