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

    # TODO: Implement actual upload logic in next phase
    print("\nUpload functionality coming soon...")
    print("For now, this command only discovers experiments.")

    return 0
