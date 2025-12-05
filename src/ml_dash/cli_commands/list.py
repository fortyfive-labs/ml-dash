"""
List command for ML-Dash CLI.

Allows users to discover projects and experiments on the remote server.
"""

import argparse
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box

from ..client import RemoteClient
from ..config import Config

console = Console()


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
    namespace: str,
    output_json: bool = False,
    verbose: bool = False
) -> int:
    """
    List all projects for the user.

    Args:
        remote_client: Remote API client
        namespace: Namespace slug
        output_json: Output as JSON
        verbose: Show verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get projects via GraphQL
        projects = remote_client.list_projects_graphql(namespace)

        if output_json:
            # JSON output
            output = {
                "namespace": namespace,
                "projects": projects,
                "count": len(projects)
            }
            console.print(json.dumps(output, indent=2))
            return 0

        # Human-readable output
        if not projects:
            console.print(f"[yellow]No projects found for namespace: {namespace}[/yellow]")
            return 0

        console.print(f"\n[bold]Projects for {namespace}[/bold]\n")

        # Create table
        table = Table(box=box.ROUNDED)
        table.add_column("Project", style="cyan", no_wrap=True)
        table.add_column("Experiments", justify="right")
        table.add_column("Description", style="dim")

        for project in projects:
            exp_count = project.get('experimentCount', 0)
            description = project.get('description', '') or ''
            if len(description) > 50:
                description = description[:47] + "..."

            table.add_row(
                project['slug'],
                str(exp_count),
                description
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(projects)} project(s)[/dim]\n")

        return 0

    except Exception as e:
        console.print(f"[red]Error listing projects:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def list_experiments(
    remote_client: RemoteClient,
    namespace: str,
    project: str,
    status_filter: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    output_json: bool = False,
    detailed: bool = False,
    verbose: bool = False
) -> int:
    """
    List experiments in a project.

    Args:
        remote_client: Remote API client
        namespace: Namespace slug
        project: Project slug
        status_filter: Filter by status (COMPLETED, RUNNING, FAILED, ARCHIVED)
        tags_filter: Filter by tags
        output_json: Output as JSON
        detailed: Show detailed information
        verbose: Show verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get experiments via GraphQL
        experiments = remote_client.list_experiments_graphql(
            namespace, project, status=status_filter
        )

        # Filter by tags if specified
        if tags_filter:
            experiments = [
                exp for exp in experiments
                if any(tag in exp.get('tags', []) for tag in tags_filter)
            ]

        if output_json:
            # JSON output
            output = {
                "namespace": namespace,
                "project": project,
                "experiments": experiments,
                "count": len(experiments)
            }
            console.print(json.dumps(output, indent=2))
            return 0

        # Human-readable output
        if not experiments:
            console.print(f"[yellow]No experiments found in project: {project}[/yellow]")
            return 0

        console.print(f"\n[bold]Experiments in project: {project}[/bold]\n")

        # Create table
        table = Table(box=box.ROUNDED)
        table.add_column("Experiment", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Metrics", justify="right")
        table.add_column("Logs", justify="right")
        table.add_column("Files", justify="right")

        if detailed:
            table.add_column("Tags", style="dim")
            table.add_column("Created", style="dim")

        for exp in experiments:
            status = exp.get('status', 'UNKNOWN')
            status_style = _get_status_style(status)

            # Count metrics
            metrics_count = len(exp.get('metrics', []))

            # Count logs
            log_metadata = exp.get('logMetadata') or {}
            logs_count = log_metadata.get('totalLogs', 0)

            # Count files
            files_count = len(exp.get('files', []))

            row = [
                exp['name'],
                f"[{status_style}]{status}[/{status_style}]",
                str(metrics_count),
                str(logs_count),
                str(files_count),
            ]

            if detailed:
                # Add tags
                tags = exp.get('tags', [])
                tags_str = ', '.join(tags[:3])
                if len(tags) > 3:
                    tags_str += f" +{len(tags) - 3}"
                row.append(tags_str or '-')

                # Add created time
                created_at = exp.get('createdAt', '')
                row.append(_format_timestamp(created_at) if created_at else '-')

            table.add_row(*row)

        console.print(table)
        console.print(f"\n[dim]Total: {len(experiments)} experiment(s)[/dim]\n")

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
            from .upload import generate_api_key_from_username
            api_key = generate_api_key_from_username(args.username)
            if args.verbose:
                console.print(f"[dim]Generated API key from username: {args.username}[/dim]")
        else:
            console.print("[red]Error:[/red] --api-key or --username is required")
            return 1

    # Get namespace (defaults to username or config)
    namespace = args.namespace or args.username or config.namespace
    if not namespace:
        console.print("[red]Error:[/red] --namespace or --username is required")
        return 1

    # Create remote client
    try:
        remote_client = RemoteClient(base_url=remote_url, api_key=api_key)
    except Exception as e:
        console.print(f"[red]Error connecting to remote:[/red] {e}")
        return 1

    # List projects or experiments
    if args.project:
        # Parse tags if provided
        tags_filter = None
        if args.tags:
            tags_filter = [tag.strip() for tag in args.tags.split(',')]

        return list_experiments(
            remote_client=remote_client,
            namespace=namespace,
            project=args.project,
            status_filter=args.status,
            tags_filter=tags_filter,
            output_json=args.json,
            detailed=args.detailed,
            verbose=args.verbose
        )
    else:
        return list_projects(
            remote_client=remote_client,
            namespace=namespace,
            output_json=args.json,
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
    parser.add_argument("--remote", type=str, help="Remote server URL")
    parser.add_argument("--api-key", type=str, help="JWT authentication token")
    parser.add_argument("--username", type=str, help="Username for auto-generating API key")
    parser.add_argument("--namespace", type=str, help="Namespace slug (defaults to username)")

    # Filtering options
    parser.add_argument("--project", type=str, help="List experiments in this project")
    parser.add_argument("--status", type=str,
                       choices=["COMPLETED", "RUNNING", "FAILED", "ARCHIVED"],
                       help="Filter experiments by status")
    parser.add_argument("--tags", type=str, help="Filter experiments by tags (comma-separated)")

    # Output options
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
