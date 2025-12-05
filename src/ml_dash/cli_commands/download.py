"""
CLI command for downloading experiments from remote server to local storage.
"""

import argparse
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TransferSpeedColumn
from rich.table import Table
from rich.panel import Panel

from ..client import RemoteClient
from ..storage import LocalStorage
from ..config import Config

console = Console()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExperimentInfo:
    """Information about an experiment to download."""
    project: str
    experiment: str
    experiment_id: str
    has_logs: bool = False
    has_params: bool = False
    metric_names: List[str] = field(default_factory=list)
    file_count: int = 0
    estimated_size: int = 0
    log_count: int = 0
    status: str = "RUNNING"
    folder: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class DownloadState:
    """State for resuming interrupted downloads."""
    remote_url: str
    local_path: str
    namespace: str
    completed_experiments: List[str] = field(default_factory=list)
    failed_experiments: List[str] = field(default_factory=list)
    in_progress_experiment: Optional[str] = None
    in_progress_items: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadState":
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: Path):
        """Save state to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional["DownloadState"]:
        """Load state from JSON file."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load state file: {e}[/yellow]")
            return None


@dataclass
class DownloadResult:
    """Result of downloading an experiment."""
    experiment: str
    success: bool = False
    downloaded: Dict[str, int] = field(default_factory=dict)
    failed: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    bytes_downloaded: int = 0
    skipped: bool = False


# ============================================================================
# Helper Functions
# ============================================================================

def _format_bytes(bytes_count: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_count < 1024:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.2f} TB"


def _format_bytes_per_sec(bytes_per_sec: float) -> str:
    """Format bytes per second as human-readable string."""
    return f"{_format_bytes(bytes_per_sec)}/s"


def _experiment_from_graphql(graphql_data: Dict[str, Any]) -> ExperimentInfo:
    """Convert GraphQL experiment data to ExperimentInfo."""
    log_metadata = graphql_data.get('logMetadata') or {}

    # Extract folder from metadata if it exists
    metadata = graphql_data.get('metadata') or {}
    folder = metadata.get('folder') if isinstance(metadata, dict) else None

    return ExperimentInfo(
        project=graphql_data['project']['slug'],
        experiment=graphql_data['name'],
        experiment_id=graphql_data['id'],
        has_logs=log_metadata.get('totalLogs', 0) > 0,
        has_params=graphql_data.get('parameters') is not None,
        metric_names=[m['name'] for m in graphql_data.get('metrics', []) or []],
        file_count=len(graphql_data.get('files', []) or []),
        log_count=int(log_metadata.get('totalLogs', 0)),
        status=graphql_data.get('status', 'RUNNING'),
        folder=folder,
        description=graphql_data.get('description'),
        tags=graphql_data.get('tags', []) or [],
    )


def discover_experiments(
    remote_client: RemoteClient,
    namespace: str,
    project_filter: Optional[str] = None,
    experiment_filter: Optional[str] = None,
) -> List[ExperimentInfo]:
    """
    Discover experiments on remote server using GraphQL.

    Args:
        remote_client: Remote API client
        namespace: Namespace slug
        project_filter: Optional project slug filter
        experiment_filter: Optional experiment name filter

    Returns:
        List of ExperimentInfo objects
    """
    # Specific experiment requested
    if project_filter and experiment_filter:
        exp_data = remote_client.get_experiment_graphql(namespace, project_filter, experiment_filter)
        if exp_data:
            return [_experiment_from_graphql(exp_data)]
        return []

    # Project filter - get all experiments in project
    if project_filter:
        experiments_data = remote_client.list_experiments_graphql(namespace, project_filter)
        return [_experiment_from_graphql(exp) for exp in experiments_data]

    # No filter - get all projects and their experiments
    projects = remote_client.list_projects_graphql(namespace)
    all_experiments = []
    for project in projects:
        experiments_data = remote_client.list_experiments_graphql(namespace, project['slug'])
        all_experiments.extend([_experiment_from_graphql(exp) for exp in experiments_data])

    return all_experiments


def _get_or_generate_api_key(args: argparse.Namespace, config: Config) -> str:
    """Get API key from args, config, or generate from username."""
    if args.api_key:
        return args.api_key
    if config.api_key:
        return config.api_key
    if args.username:
        from ..cli_commands.upload import generate_api_key_from_username
        return generate_api_key_from_username(args.username)
    return ""


# ============================================================================
# Experiment Downloader
# ============================================================================

class ExperimentDownloader:
    """Downloads a single experiment from remote server."""

    def __init__(
        self,
        local_storage: LocalStorage,
        remote_client: RemoteClient,
        batch_size: int = 1000,
        skip_logs: bool = False,
        skip_metrics: bool = False,
        skip_files: bool = False,
        skip_params: bool = False,
        verbose: bool = False,
        max_concurrent_metrics: int = 5,
        max_concurrent_files: int = 3,
    ):
        self.local = local_storage
        self.remote = remote_client
        self.batch_size = batch_size
        self.skip_logs = skip_logs
        self.skip_metrics = skip_metrics
        self.skip_files = skip_files
        self.skip_params = skip_params
        self.verbose = verbose
        self.max_concurrent_metrics = max_concurrent_metrics
        self.max_concurrent_files = max_concurrent_files
        self._lock = threading.Lock()
        self._thread_local = threading.local()

    def _get_remote_client(self) -> RemoteClient:
        """Get thread-local remote client for safe concurrent access."""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = RemoteClient(
                base_url=self.remote.base_url,
                api_key=self.remote.api_key
            )
        return self._thread_local.client

    def download_experiment(self, exp_info: ExperimentInfo) -> DownloadResult:
        """Download a complete experiment."""
        result = DownloadResult(experiment=f"{exp_info.project}/{exp_info.experiment}")

        try:
            if self.verbose:
                console.print(f"  [dim]Downloading {exp_info.project}/{exp_info.experiment}[/dim]")

            # Step 1: Download metadata and create experiment
            self._download_metadata(exp_info, result)

            # Step 2: Download parameters
            if not self.skip_params and exp_info.has_params:
                self._download_parameters(exp_info, result)

            # Step 3: Download logs
            if not self.skip_logs and exp_info.has_logs:
                self._download_logs(exp_info, result)

            # Step 4: Download metrics (parallel)
            if not self.skip_metrics and exp_info.metric_names:
                self._download_metrics(exp_info, result)

            # Step 5: Download files (parallel)
            if not self.skip_files and exp_info.file_count > 0:
                self._download_files(exp_info, result)

            result.success = True

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            if self.verbose:
                console.print(f"  [red]Error: {e}[/red]")

        return result

    def _download_metadata(self, exp_info: ExperimentInfo, result: DownloadResult):
        """Download and create experiment metadata."""
        # Create experiment directory structure with folder path
        self.local.create_experiment(
            project=exp_info.project,
            name=exp_info.experiment,
            description=exp_info.description,
            tags=exp_info.tags,
            bindrs=[],
            folder=exp_info.folder,
            metadata=None,
        )

    def _download_parameters(self, exp_info: ExperimentInfo, result: DownloadResult):
        """Download parameters."""
        try:
            params_data = self.remote.get_parameters(exp_info.experiment_id)
            if params_data:
                self.local.write_parameters(
                    project=exp_info.project,
                    experiment=exp_info.experiment,
                    data=params_data
                )
                result.downloaded["parameters"] = 1
                result.bytes_downloaded += len(json.dumps(params_data))
        except Exception as e:
            result.failed.setdefault("parameters", []).append(str(e))

    def _download_logs(self, exp_info: ExperimentInfo, result: DownloadResult):
        """Download logs with pagination."""
        try:
            offset = 0
            total_downloaded = 0

            while True:
                logs_data = self.remote.query_logs(
                    experiment_id=exp_info.experiment_id,
                    limit=self.batch_size,
                    offset=offset,
                    order_by="sequenceNumber",
                    order="asc"
                )

                logs = logs_data.get("logs", [])
                if not logs:
                    break

                # Write logs
                for log in logs:
                    self.local.write_log(
                        project=exp_info.project,
                        experiment=exp_info.experiment,
                        message=log['message'],
                        level=log['level'],
                        timestamp=log['timestamp'],
                        metadata=log.get('metadata')
                    )

                total_downloaded += len(logs)
                result.bytes_downloaded += sum(len(json.dumps(log)) for log in logs)

                if not logs_data.get("hasMore", False):
                    break

                offset += len(logs)

            result.downloaded["logs"] = total_downloaded

        except Exception as e:
            result.failed.setdefault("logs", []).append(str(e))

    def _download_metrics(self, exp_info: ExperimentInfo, result: DownloadResult):
        """Download all metrics in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_metrics) as executor:
            future_to_metric = {}

            for metric_name in exp_info.metric_names:
                future = executor.submit(
                    self._download_single_metric,
                    exp_info.experiment_id,
                    exp_info.project,
                    exp_info.experiment,
                    metric_name
                )
                future_to_metric[future] = metric_name

            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                metric_result = future.result()

                with self._lock:
                    if metric_result['success']:
                        result.downloaded["metrics"] = result.downloaded.get("metrics", 0) + 1
                        result.bytes_downloaded += metric_result['bytes']
                    else:
                        result.failed.setdefault("metrics", []).append(
                            f"{metric_name}: {metric_result['error']}"
                        )

    def _download_single_chunk(self, experiment_id: str, metric_name: str, chunk_number: int):
        """Download a single chunk (for parallel downloading)."""
        remote = self._get_remote_client()
        try:
            chunk_data = remote.download_metric_chunk(experiment_id, metric_name, chunk_number)
            return {
                'success': True,
                'chunk_number': chunk_number,
                'data': chunk_data.get('data', []),
                'start_index': int(chunk_data.get('startIndex', 0)),
                'end_index': int(chunk_data.get('endIndex', 0)),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'chunk_number': chunk_number,
                'error': str(e),
                'data': []
            }

    def _download_single_metric(self, experiment_id: str, project: str, experiment: str, metric_name: str):
        """Download a single metric using chunk-aware approach (thread-safe)."""
        remote = self._get_remote_client()

        total_downloaded = 0
        bytes_downloaded = 0

        try:
            # Get metric metadata to determine download strategy
            metadata = remote.get_metric_stats(experiment_id, metric_name)
            total_chunks = metadata.get('totalChunks', 0)
            buffered_points = int(metadata.get('bufferedDataPoints', 0))

            all_data = []

            # Download chunks in parallel if they exist
            if total_chunks > 0:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                # Download all chunks in parallel (max 10 workers)
                with ThreadPoolExecutor(max_workers=min(10, total_chunks)) as executor:
                    chunk_futures = {
                        executor.submit(self._download_single_chunk, experiment_id, metric_name, i): i
                        for i in range(total_chunks)
                    }

                    for future in as_completed(chunk_futures):
                        result = future.result()
                        if result['success']:
                            all_data.extend(result['data'])
                        else:
                            # If a chunk fails, fall back to pagination
                            raise Exception(f"Chunk {result['chunk_number']} download failed: {result['error']}")

            # Download buffer data if exists
            if buffered_points > 0:
                response = remote.get_metric_data(
                    experiment_id, metric_name,
                    buffer_only=True
                )
                buffer_data = response.get('data', [])
                all_data.extend(buffer_data)

            # Sort all data by index
            all_data.sort(key=lambda x: int(x.get('index', 0)))

            # Write to local storage in batches
            batch_size = 10000
            for i in range(0, len(all_data), batch_size):
                batch = all_data[i:i + batch_size]
                self.local.append_batch_to_metric(
                    project, experiment, metric_name,
                    data_points=[d['data'] for d in batch]
                )
                total_downloaded += len(batch)
                bytes_downloaded += sum(len(json.dumps(d)) for d in batch)

            return {'success': True, 'downloaded': total_downloaded, 'bytes': bytes_downloaded, 'error': None}

        except Exception as e:
            # Fall back to pagination if chunk download fails
            console.print(f"[yellow]Chunk download failed for {metric_name}, falling back to pagination: {e}[/yellow]")
            return self._download_metric_with_pagination(experiment_id, project, experiment, metric_name)

    def _download_metric_with_pagination(self, experiment_id: str, project: str, experiment: str, metric_name: str):
        """Original pagination-based download (fallback method)."""
        remote = self._get_remote_client()

        total_downloaded = 0
        bytes_downloaded = 0
        start_index = 0

        try:
            while True:
                response = remote.get_metric_data(
                    experiment_id, metric_name,
                    start_index=start_index,
                    limit=self.batch_size
                )

                data_points = response.get('data', [])
                if not data_points:
                    break

                # Write to local storage
                self.local.append_batch_to_metric(
                    project, experiment, metric_name,
                    data_points=[d['data'] for d in data_points]
                )

                total_downloaded += len(data_points)
                bytes_downloaded += sum(len(json.dumps(d)) for d in data_points)

                if not response.get('hasMore', False):
                    break

                start_index += len(data_points)

            return {'success': True, 'downloaded': total_downloaded, 'bytes': bytes_downloaded, 'error': None}

        except Exception as e:
            return {'success': False, 'error': str(e), 'downloaded': 0, 'bytes': 0}

    def _download_files(self, exp_info: ExperimentInfo, result: DownloadResult):
        """Download files in parallel."""
        # Get file list
        try:
            files_data = self.remote.list_files(exp_info.experiment_id)
        except Exception as e:
            result.failed.setdefault("files", []).append(f"List files failed: {e}")
            return

        if not files_data:
            return

        with ThreadPoolExecutor(max_workers=self.max_concurrent_files) as executor:
            future_to_file = {}

            for file_info in files_data:
                future = executor.submit(
                    self._download_single_file,
                    exp_info.experiment_id,
                    exp_info.project,
                    exp_info.experiment,
                    file_info
                )
                future_to_file[future] = file_info['filename']

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                file_result = future.result()

                with self._lock:
                    if file_result['success']:
                        result.downloaded["files"] = result.downloaded.get("files", 0) + 1
                        result.bytes_downloaded += file_result['bytes']
                    else:
                        result.failed.setdefault("files", []).append(
                            f"{filename}: {file_result['error']}"
                        )

    def _download_single_file(self, experiment_id: str, project: str, experiment: str, file_info: Dict[str, Any]):
        """Download a single file with streaming (thread-safe)."""
        remote = self._get_remote_client()

        try:
            # Stream download to temp file
            temp_fd, temp_path = tempfile.mkstemp(prefix="ml_dash_download_")
            os.close(temp_fd)

            remote.download_file_streaming(
                experiment_id, file_info['id'], dest_path=temp_path
            )

            # Write to local storage
            self.local.write_file(
                project=project,
                experiment=experiment,
                file_path=temp_path,
                prefix=file_info['path'],
                filename=file_info['filename'],
                description=file_info.get('description'),
                tags=file_info.get('tags', []),
                metadata=file_info.get('metadata'),
                checksum=file_info['checksum'],
                content_type=file_info['contentType'],
                size_bytes=file_info['sizeBytes']
            )

            # Clean up temp file
            os.remove(temp_path)

            return {'success': True, 'bytes': file_info['sizeBytes'], 'error': None}

        except Exception as e:
            return {'success': False, 'error': str(e), 'bytes': 0}


# ============================================================================
# Main Command
# ============================================================================

def cmd_download(args: argparse.Namespace) -> int:
    """Execute download command."""
    # Load configuration
    config = Config()
    remote_url = args.remote or config.remote_url
    api_key = _get_or_generate_api_key(args, config)
    namespace = args.namespace or args.username

    # Validate inputs
    if not remote_url:
        console.print("[red]Error:[/red] --remote is required (or set in config)")
        return 1

    if not api_key:
        console.print("[red]Error:[/red] --api-key or --username is required")
        return 1

    if not namespace:
        console.print("[red]Error:[/red] --namespace or --username is required")
        return 1

    # Initialize clients
    remote_client = RemoteClient(base_url=remote_url, api_key=api_key)
    local_storage = LocalStorage(root_path=Path(args.path))

    # Load or create state
    state_file = Path(args.state_file)
    if args.resume:
        state = DownloadState.load(state_file)
        if state:
            console.print(f"[cyan]Resuming from previous download ({len(state.completed_experiments)} completed)[/cyan]")
        else:
            console.print("[yellow]No previous state found, starting fresh[/yellow]")
            state = DownloadState(
                remote_url=remote_url,
                local_path=str(args.path),
                namespace=namespace
            )
    else:
        state = DownloadState(
            remote_url=remote_url,
            local_path=str(args.path),
            namespace=namespace
        )

    # Discover experiments
    console.print("[bold]Discovering experiments on remote server...[/bold]")
    try:
        experiments = discover_experiments(
            remote_client, namespace, args.project, args.experiment
        )
    except Exception as e:
        console.print(f"[red]Failed to discover experiments: {e}[/red]")
        return 1

    if not experiments:
        console.print("[yellow]No experiments found[/yellow]")
        return 0

    console.print(f"Found {len(experiments)} experiment(s)")

    # Filter out completed experiments
    experiments_to_download = []
    for exp in experiments:
        exp_key = f"{exp.project}/{exp.experiment}"

        # Skip if already completed
        if exp_key in state.completed_experiments and not args.overwrite:
            console.print(f"  [dim]Skipping {exp_key} (already completed)[/dim]")
            continue

        # Check if exists locally
        exp_json = local_storage.root_path / exp.project / exp.experiment / "experiment.json"
        if exp_json.exists() and not args.overwrite:
            console.print(f"  [yellow]Skipping {exp_key} (already exists locally)[/yellow]")
            continue

        experiments_to_download.append(exp)

    if not experiments_to_download:
        console.print("[green]All experiments already downloaded[/green]")
        return 0

    # Dry run mode
    if args.dry_run:
        console.print("\n[bold]Dry run - would download:[/bold]")
        for exp in experiments_to_download:
            console.print(f"  • {exp.project}/{exp.experiment}")
            console.print(f"    Logs: {exp.log_count}, Metrics: {len(exp.metric_names)}, Files: {exp.file_count}")
        return 0

    # Download experiments
    console.print(f"\n[bold]Downloading {len(experiments_to_download)} experiment(s)...[/bold]")
    results = []
    start_time = time.time()

    for i, exp in enumerate(experiments_to_download, 1):
        exp_key = f"{exp.project}/{exp.experiment}"
        console.print(f"\n[cyan][{i}/{len(experiments_to_download)}] {exp_key}[/cyan]")

        # Mark as in-progress
        state.in_progress_experiment = exp_key
        state.save(state_file)

        # Download
        downloader = ExperimentDownloader(
            local_storage=local_storage,
            remote_client=remote_client,
            batch_size=args.batch_size,
            skip_logs=args.skip_logs,
            skip_metrics=args.skip_metrics,
            skip_files=args.skip_files,
            skip_params=args.skip_params,
            verbose=args.verbose,
            max_concurrent_metrics=args.max_concurrent_metrics,
            max_concurrent_files=args.max_concurrent_files,
        )

        result = downloader.download_experiment(exp)
        results.append(result)

        # Update state
        if result.success:
            state.completed_experiments.append(exp_key)
            console.print(f"  [green]✓ Downloaded successfully[/green]")
        else:
            state.failed_experiments.append(exp_key)
            console.print(f"  [red]✗ Failed: {', '.join(result.errors)}[/red]")

        state.in_progress_experiment = None
        state.save(state_file)

    # Show summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_bytes = sum(r.bytes_downloaded for r in results)
    successful = sum(1 for r in results if r.success)

    console.print("\n[bold]Download Summary[/bold]")
    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Experiments", str(len(results)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", str(len(results) - successful))
    summary_table.add_row("Total Data", _format_bytes(total_bytes))
    summary_table.add_row("Total Time", f"{elapsed_time:.2f}s")

    if elapsed_time > 0:
        speed = total_bytes / elapsed_time
        summary_table.add_row("Avg Speed", _format_bytes_per_sec(speed))

    console.print(summary_table)

    # Clean up state file if all successful
    if all(r.success for r in results):
        state_file.unlink(missing_ok=True)

    return 0 if all(r.success for r in results) else 1


def add_parser(subparsers):
    """Add download command parser."""
    parser = subparsers.add_parser(
        "download",
        help="Download experiments from remote server to local storage"
    )

    # Positional arguments
    parser.add_argument(
        "path",
        nargs="?",
        default="./.ml-dash",
        help="Local storage directory (default: ./.ml-dash)"
    )

    # Remote configuration
    parser.add_argument("--remote", help="Remote server URL")
    parser.add_argument("--api-key", help="JWT authentication token")
    parser.add_argument("--username", help="Username for auto-generating API key")

    # Scope control
    parser.add_argument("--project", help="Download only this project")
    parser.add_argument("--experiment", help="Download specific experiment (requires --project)")
    parser.add_argument("--namespace", help="Namespace slug (defaults to username)")

    # Data filtering
    parser.add_argument("--skip-logs", action="store_true", help="Don't download logs")
    parser.add_argument("--skip-metrics", action="store_true", help="Don't download metrics")
    parser.add_argument("--skip-files", action="store_true", help="Don't download files")
    parser.add_argument("--skip-params", action="store_true", help="Don't download parameters")

    # Behavior control
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiments")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted download")
    parser.add_argument(
        "--state-file",
        default=".ml-dash-download-state.json",
        help="State file path for resume (default: .ml-dash-download-state.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for logs/metrics (default: 1000, max: 10000)"
    )
    parser.add_argument(
        "--max-concurrent-metrics",
        type=int,
        default=5,
        help="Parallel metric downloads (default: 5)"
    )
    parser.add_argument(
        "--max-concurrent-files",
        type=int,
        default=3,
        help="Parallel file downloads (default: 3)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed progress output")

    parser.set_defaults(func=cmd_download)
