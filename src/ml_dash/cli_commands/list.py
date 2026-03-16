"""
List command for ML-Dash CLI.

Allows users to discover projects and experiments on the remote server.
"""

import argparse
import sys
import tty
import termios
from typing import Optional, List, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box

from ..client import RemoteClient
from ..config import Config

console = Console()

PAGE_SIZE = 50


def _read_key() -> str:
    """Read a single keypress without requiring Enter. Returns 'next', 'prev', or 'quit'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'C':   # right arrow
                    return 'next'
                if ch3 == 'D':   # left arrow
                    return 'prev'
        if ch in ('n', '\r', '\n', ' '):
            return 'next'
        if ch in ('p', 'b'):
            return 'prev'
        return 'quit'  # q, ctrl-c, esc, anything else
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def list_tracks(
    remote_client: RemoteClient,
    experiment_path: str,
    topic_filter: Optional[str] = None,
    verbose: bool = False
) -> int:
    """
    List tracks in an experiment.

    Args:
        remote_client: Remote API client
        experiment_path: Experiment path (namespace/project/experiment)
        topic_filter: Optional topic filter (e.g., "robot/*")
        verbose: Show verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse experiment path
    parts = experiment_path.strip("/").split("/")
    if len(parts) < 3:
        console.print("[red]Error:[/red] Experiment path must be 'namespace/project/experiment'")
        return 1

    namespace = parts[0]
    project = parts[1]
    experiment = parts[2]

    try:
        # Get experiment ID
        exp_data = remote_client.get_experiment_graphql(project, experiment)
        if not exp_data:
            console.print(f"[red]Error:[/red] Experiment '{experiment}' not found in project '{project}'")
            return 1

        experiment_id = exp_data["id"]

        # List tracks
        tracks = remote_client.list_tracks(experiment_id, topic_filter)

        if not tracks:
            console.print(f"[yellow]No tracks found[/yellow]")
            return 0

        console.print(f"\n[bold]Tracks in {experiment_path}[/bold]\n")

        # Create table
        table = Table(box=box.ROUNDED)
        table.add_column("Topic", style="cyan", no_wrap=True)
        table.add_column("Entries", justify="right")
        table.add_column("Columns", style="dim")
        table.add_column("Time Range", style="dim")

        for track in tracks:
            topic = track["topic"]
            entries = str(track.get("totalEntries", 0))
            columns = ", ".join(track.get("columns", [])[:5])
            if len(track.get("columns", [])) > 5:
                columns += f", ... (+{len(track['columns']) - 5})"

            first_ts = track.get("firstTimestamp")
            last_ts = track.get("lastTimestamp")
            if first_ts is not None and last_ts is not None:
                time_range = f"{first_ts:.3f} - {last_ts:.3f}"
            else:
                time_range = "N/A"

            table.add_row(topic, entries, columns, time_range)

        with console.pager(styles=True):
            console.print(table)

        return 0

    except Exception as e:
        console.print(f"[red]Error listing tracks:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def _format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp as human-readable relative time."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo)
        diff = now - dt

        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    except:
        return iso_timestamp


def _get_status_style(status: str) -> str:
    """Get rich style for status."""
    status_styles = {
        "COMPLETED": "green",
        "RUNNING": "yellow",
        "FAILED": "red",
        "ARCHIVED": "dim",
    }
    return status_styles.get(status, "white")


def list_projects(
    remote_client: RemoteClient,
    namespace_slug: Optional[str] = None,
    verbose: bool = False
) -> int:
    try:
        offset = 0
        while True:
            result = remote_client.list_projects_graphql(
                namespace_slug=namespace_slug,
                limit=PAGE_SIZE,
                offset=offset,
            )
            projects = result["projects"]
            total_count = result["totalCount"]

            if not projects and offset == 0:
                console.print("[yellow]No projects found[/yellow]")
                return 0

            total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
            current_page = offset // PAGE_SIZE + 1

            table = Table(
                title="\nProjects",
                box=box.ROUNDED,
                caption=f"Page {current_page}/{total_pages}  ·  {total_count} project{'s' if total_count != 1 else ''} total",
            )
            table.add_column("Project", style="cyan", no_wrap=True)
            table.add_column("Experiments", justify="right")
            table.add_column("Description", style="dim")

            for project in projects:
                exp_count = project.get('experimentCount', 0)
                description = project.get('description', '') or ''
                if len(description) > 50:
                    description = description[:47] + "..."
                table.add_row(project['slug'], str(exp_count), description)

            console.print(table)

            has_next = offset + PAGE_SIZE < total_count
            has_prev = offset > 0

            if not has_next and not has_prev:
                break

            nav = []
            if has_prev:
                nav.append("[b/←] prev")
            if has_next:
                nav.append("[n/→] next")
            nav.append("[q] quit")
            console.print(f"[dim]{'  '.join(nav)}[/dim]", end="  ")

            key = _read_key()
            console.print()
            if key == 'next' and has_next:
                offset += PAGE_SIZE
                console.clear()
            elif key == 'prev' and has_prev:
                offset -= PAGE_SIZE
                console.clear()
            else:
                break

        return 0

    except Exception as e:
        console.print(f"[red]Error listing projects:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def list_experiments(
    remote_client: RemoteClient,
    project: str,
    namespace_slug: Optional[str] = None,
    status_filter: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    detailed: bool = False,
    verbose: bool = False,
) -> int:
    try:
        offset = 0
        while True:
            result = remote_client.list_experiments_graphql(
                project,
                status=status_filter,
                namespace_slug=namespace_slug,
                limit=PAGE_SIZE,
                offset=offset,
            )
            experiments = result["experiments"]
            total_count = result["totalCount"]

            # Client-side tags filter (applies within fetched page)
            if tags_filter:
                experiments = [
                    exp for exp in experiments
                    if any(tag in exp.get('tags', []) for tag in tags_filter)
                ]

            if not experiments and offset == 0:
                console.print(f"[yellow]No experiments found in project: {project}[/yellow]")
                return 0

            total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
            current_page = offset // PAGE_SIZE + 1

            table = Table(
                title=f"\nExperiments in project: {project}",
                box=box.ROUNDED,
                caption=f"Page {current_page}/{total_pages}  ·  {total_count} experiment{'s' if total_count != 1 else ''} total",
            )
            table.add_column("Experiment", style="cyan", no_wrap=True)
            table.add_column("Status", justify="center")
            table.add_column("Metrics", justify="right")
            table.add_column("Logs", justify="right")
            table.add_column("Tracks", justify="right")
            table.add_column("Files", justify="right")

            if detailed:
                table.add_column("Tags", style="dim")
                table.add_column("Created", style="dim")

            for exp in experiments:
                status = exp.get('status', 'UNKNOWN')
                status_style = _get_status_style(status)
                metrics_count = len(exp.get('metrics', []))
                log_metadata = exp.get('logMetadata') or {}
                logs_count = log_metadata.get('totalLogs', 0)
                tracks_count = exp.get('trackCount', 0)
                files_count = len(exp.get('files', []))
                exp_display_name = exp.get('displayPath') or exp['name']

                row = [
                    exp_display_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    str(metrics_count),
                    str(logs_count),
                    str(tracks_count),
                    str(files_count),
                ]

                if detailed:
                    tags = exp.get('tags', [])
                    tags_str = ', '.join(tags[:3])
                    if len(tags) > 3:
                        tags_str += f" +{len(tags) - 3}"
                    row.append(tags_str or '-')
                    created_at = exp.get('createdAt', '')
                    row.append(_format_timestamp(created_at) if created_at else '-')

                table.add_row(*row)

            console.print(table)

            has_next = offset + PAGE_SIZE < total_count
            has_prev = offset > 0

            if not has_next and not has_prev:
                break

            nav = []
            if has_prev:
                nav.append("[b/←] prev")
            if has_next:
                nav.append("[n/→] next")
            nav.append("[q] quit")
            console.print(f"[dim]{'  '.join(nav)}[/dim]", end="  ")

            key = _read_key()
            console.print()
            if key == 'next' and has_next:
                offset += PAGE_SIZE
                console.clear()
            elif key == 'prev' and has_prev:
                offset -= PAGE_SIZE
                console.clear()
            else:
                break

        return 0

    except Exception as e:
        console.print(f"[red]Error listing experiments:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """
    Execute list command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Handle track listing if --tracks is specified
    if args.tracks:
        # Load config
        config = Config()
        remote_url = args.dash_url or config.remote_url
        api_key = args.api_key or config.api_key

        if not remote_url:
            console.print("[red]Error:[/red] --dash-url is required (or set in config)")
            return 1

        if not args.project:
            console.print("[red]Error:[/red] --project is required for listing tracks")
            console.print("Example: ml-dash list --tracks --project namespace/project/experiment")
            return 1

        # Extract namespace from project path
        parts = args.project.strip("/").split("/")
        if len(parts) < 3:
            console.print("[red]Error:[/red] For tracks, --project must be 'namespace/project/experiment'")
            return 1

        namespace = parts[0]

        # Create remote client
        try:
            remote_client = RemoteClient(base_url=remote_url, namespace=namespace, api_key=api_key)
        except Exception as e:
            console.print(f"[red]Error connecting to remote:[/red] {e}")
            return 1

        return list_tracks(
            remote_client=remote_client,
            experiment_path=args.project,
            topic_filter=args.topic_filter,
            verbose=args.verbose
        )

    # Load config
    config = Config()

    # Get remote URL (command line > config)
    remote_url = args.dash_url or config.remote_url
    if not remote_url:
        console.print("[red]Error:[/red] --dash-url is required (or set in config)")
        return 1

    # Get API key (command line > config > auto-loaded from storage)
    api_key = args.api_key or config.api_key

    # Extract namespace and project slug from project argument
    namespace = None
    project_slug = None
    if args.project:
        # Parse namespace from project filter (format: "namespace/project")
        project_parts = args.project.strip("/").split("/")
        # For simple patterns without '/', treat as project-only pattern
        if '/' in args.project and len(project_parts) >= 2:
            namespace = project_parts[0]
            project_slug = project_parts[1]

    has_wildcards = args.project and any(c in args.project for c in ['*', '?', '['])

    # If --project has no '/' and no wildcards, treat as plain project slug
    # and auto-resolve namespace from the stored token
    if args.project and not namespace and not has_wildcards:
        project_slug = args.project

    # Create remote client (namespace=None: RemoteClient auto-detects from token)
    try:
        remote_client = RemoteClient(base_url=remote_url, namespace=namespace, api_key=api_key)
    except Exception as e:
        console.print(f"[red]Error connecting to remote:[/red] {e}")
        return 1

    # List projects or experiments
    if args.project:
        # Parse tags if provided
        tags_filter = None
        if args.tags:
            tags_filter = [tag.strip() for tag in args.tags.split(',')]

        if has_wildcards:
            # Use searchExperimentsPaginated GraphQL query for glob patterns
            try:
                # Expand simple project patterns to full namespace/project/experiment format
                if '/' not in args.project:
                    search_pattern = f"*/{args.project}/*"
                else:
                    search_pattern = args.project

                search_offset = 0
                while True:
                    result = remote_client.search_experiments_graphql(
                        search_pattern,
                        limit=PAGE_SIZE,
                        offset=search_offset,
                    )
                    experiments = result["experiments"]
                    total_count = result["totalCount"]

                    # Client-side filters (server doesn't support these in search yet)
                    if args.status:
                        experiments = [e for e in experiments if e.get('status') == args.status]
                    if tags_filter:
                        experiments = [
                            e for e in experiments
                            if any(tag in e.get('tags', []) for tag in tags_filter)
                        ]

                    if not experiments and search_offset == 0:
                        console.print(f"[yellow]No experiments match pattern: {search_pattern}[/yellow]")
                        return 0

                    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
                    current_page = search_offset // PAGE_SIZE + 1

                    table = Table(
                        title=f"\nSearch results for: {search_pattern}",
                        box=box.ROUNDED,
                        caption=f"Page {current_page}/{total_pages}  ·  {total_count} experiment{'s' if total_count != 1 else ''} total",
                    )
                    table.add_column("Project", style="dim", no_wrap=True)
                    table.add_column("Experiment", style="cyan", no_wrap=True)
                    table.add_column("Status", justify="center")
                    table.add_column("Metrics", justify="right")
                    table.add_column("Logs", justify="right")
                    table.add_column("Tracks", justify="right")
                    table.add_column("Files", justify="right")

                    if args.detailed:
                        table.add_column("Tags", style="dim")
                        table.add_column("Started", style="dim")

                    for exp in experiments:
                        status = exp.get('status', 'UNKNOWN')
                        status_style = _get_status_style(status)
                        metrics_count = len(exp.get('metrics', []))
                        log_metadata = exp.get('logMetadata') or {}
                        logs_count = log_metadata.get('totalLogs', 0)
                        tracks_count = exp.get('trackCount', 0)
                        files_count = len(exp.get('files', []))
                        exp_display_name = exp.get('displayPath') or exp['name']
                        proj_slug = exp.get('project', {}).get('slug', '')

                        row = [
                            proj_slug,
                            exp_display_name,
                            f"[{status_style}]{status}[/{status_style}]",
                            str(metrics_count),
                            str(logs_count),
                            str(tracks_count),
                            str(files_count),
                        ]

                        if args.detailed:
                            tags = exp.get('tags', [])
                            tags_str = ', '.join(tags[:3])
                            if len(tags) > 3:
                                tags_str += f" +{len(tags) - 3}"
                            row.append(tags_str or '-')
                            started_at = exp.get('startedAt', '')
                            row.append(_format_timestamp(started_at) if started_at else '-')

                        table.add_row(*row)

                    console.print(table)

                    has_next = search_offset + PAGE_SIZE < total_count
                    has_prev = search_offset > 0

                    if not has_next and not has_prev:
                        break

                    nav = []
                    if has_prev:
                        nav.append("[b/←] prev")
                    if has_next:
                        nav.append("[n/→] next")
                    nav.append("[q] quit")
                    console.print(f"[dim]{'  '.join(nav)}[/dim]", end="  ")

                    key = _read_key()
                    console.print()
                    if key == 'next' and has_next:
                        search_offset += PAGE_SIZE
                        console.clear()
                    elif key == 'prev' and has_prev:
                        search_offset -= PAGE_SIZE
                        console.clear()
                    else:
                        break

                return 0

            except Exception as e:
                console.print(f"[red]Error searching experiments:[/red] {e}")
                if args.verbose:
                    import traceback
                    console.print(traceback.format_exc())
                return 1
        else:
            # No wildcards, use existing list method for exact project match
            # Show the effective namespace for clarity
            try:
                console.print(f"[dim]Using namespace: {remote_client.namespace}[/dim]")
            except Exception:
                pass
            return list_experiments(
                remote_client=remote_client,
                project=project_slug,
                namespace_slug=namespace,
                status_filter=args.status,
                tags_filter=tags_filter,
                detailed=args.detailed,
                verbose=args.verbose
            )
    else:
        return list_projects(
            remote_client=remote_client,
            namespace_slug=args.namespace,
            verbose=args.verbose
        )


def add_parser(subparsers) -> None:
    """Add list command parser to subparsers."""
    parser = subparsers.add_parser(
        "list",
        help="List projects and experiments on remote server",
        description="Discover projects and experiments available on the remote ML-Dash server."
    )

    # Remote configuration
    parser.add_argument(
        "--dash-url", "--api-url",
        dest="dash_url",
        type=str,
        help="ML-Dash server URL (defaults to config or https://api.dash.ml)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="JWT authentication token (auto-loaded from storage if not provided)"
    )
    # Namespace / filtering options
    parser.add_argument(
        "-n", "--namespace",
        type=str,
        dest="namespace",
        help="Namespace slug to list projects for (defaults to authenticated user's namespace)",
    )
    parser.add_argument(
        "-p",
        "--pref",
        "--prefix",
        "--proj",
        "--project",
        dest="project",
        type=str,
        help="List experiments in this project. Supports glob patterns — always quote them to prevent shell expansion: -p 'tom/tut*', -p 'alice/test-?'"
    )
    parser.add_argument("--status", type=str,
                       choices=["COMPLETED", "RUNNING", "FAILED", "ARCHIVED"],
                       help="Filter experiments by status")
    parser.add_argument("--tags", type=str, help="Filter experiments by tags (comma-separated)")

    # Track listing mode
    parser.add_argument(
        "--tracks",
        action="store_true",
        help="List tracks in experiment (requires --project as 'namespace/project/experiment')"
    )
    parser.add_argument(
        "--topic-filter",
        type=str,
        help="Filter tracks by topic (e.g., 'robot/*')"
    )

    # Output options
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
