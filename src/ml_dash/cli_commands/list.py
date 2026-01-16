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
    output_json: bool = False,
    verbose: bool = False
) -> int:
    """
    List all projects for the user.

    Args:
        remote_client: Remote API client
        output_json: Output as JSON
        verbose: Show verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get projects via GraphQL
        projects = remote_client.list_projects_graphql()

        if output_json:
            # JSON output
            output = {
                "projects": projects,
                "count": len(projects)
            }
            console.print(json.dumps(output, indent=2))
            return 0

        # Human-readable output
        if not projects:
            console.print(f"[yellow]No projects found[/yellow]")
            return 0

        console.print(f"\n[bold]Projects[/bold]\n")

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
            project, status=status_filter
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
    remote_url = args.dash_url or config.remote_url
    if not remote_url:
        console.print("[red]Error:[/red] --dash-url is required (or set in config)")
        return 1

    # Get API key (command line > config > auto-loaded from storage)
    api_key = args.api_key or config.api_key

    # Extract namespace from project argument
    namespace = None
    if args.project:
        # Parse namespace from project filter (format: "namespace/project")
        project_parts = args.project.strip("/").split("/")
        # For simple patterns without '/', treat as project-only pattern
        if '/' in args.project and len(project_parts) >= 2:
            namespace = project_parts[0]

    if not namespace:
        console.print(
            "[red]Error:[/red] --project must be in format 'namespace/project'"
        )
        console.print("Example: ml-dash list --project alice/my-project")
        console.print("Or use glob patterns: ml-dash list --project alice/proj-*")
        return 1

    # Create remote client
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

        # Check if pattern contains wildcards
        has_wildcards = any(c in args.project for c in ['*', '?', '['])

        if has_wildcards:
            # Use searchExperiments GraphQL query for glob patterns
            try:
                # Expand simple project patterns to full namespace/project/experiment format
                # If pattern has no slashes, assume it's just a project pattern
                if '/' not in args.project:
                    # Simple project pattern: "tut*" -> "*/tut*/*"
                    search_pattern = f"*/{args.project}/*"
                else:
                    # Full or partial pattern: use as-is
                    search_pattern = args.project

                experiments = remote_client.search_experiments_graphql(search_pattern)

                # Apply status filter if specified (server doesn't support it in searchExperiments yet)
                if args.status:
                    experiments = [
                        exp for exp in experiments
                        if exp.get('status') == args.status
                    ]

                # Apply tags filter if specified
                if tags_filter:
                    experiments = [
                        exp for exp in experiments
                        if any(tag in exp.get('tags', []) for tag in tags_filter)
                    ]

                if args.json:
                    # JSON output
                    output = {
                        "pattern": search_pattern,
                        "experiments": experiments,
                        "count": len(experiments)
                    }
                    console.print(json.dumps(output, indent=2))
                    return 0

                # Human-readable output
                if not experiments:
                    console.print(f"[yellow]No experiments match pattern: {search_pattern}[/yellow]")
                    return 0

                # Group experiments by project for better display
                from collections import defaultdict
                projects_map = defaultdict(list)
                for exp in experiments:
                    project_slug = exp.get('project', {}).get('slug', 'unknown')
                    projects_map[project_slug].append(exp)

                # Display each project's experiments
                for project_slug in sorted(projects_map.keys()):
                    project_experiments = projects_map[project_slug]
                    console.print(f"\n[bold]Experiments in project: {project_slug}[/bold]\n")

                    # Create table
                    table = Table(box=box.ROUNDED)
                    table.add_column("Experiment", style="cyan", no_wrap=True)
                    table.add_column("Status", justify="center")
                    table.add_column("Metrics", justify="right")
                    table.add_column("Logs", justify="right")
                    table.add_column("Files", justify="right")

                    if args.detailed:
                        table.add_column("Tags", style="dim")
                        table.add_column("Started", style="dim")

                    for exp in project_experiments:
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

                        if args.detailed:
                            # Add tags
                            tags = exp.get('tags', [])
                            tags_str = ', '.join(tags[:3])
                            if len(tags) > 3:
                                tags_str += f" +{len(tags) - 3}"
                            row.append(tags_str or '-')

                            # Add started time (createdAt doesn't exist, use startedAt)
                            started_at = exp.get('startedAt', '')
                            row.append(_format_timestamp(started_at) if started_at else '-')

                        table.add_row(*row)

                    console.print(table)
                    console.print(f"[dim]Total: {len(project_experiments)} experiment(s)[/dim]")

                console.print(f"\n[dim]Grand total: {len(experiments)} experiment(s) across {len(projects_map)} project(s)[/dim]\n")
                return 0

            except Exception as e:
                console.print(f"[red]Error searching experiments:[/red] {e}")
                if args.verbose:
                    import traceback
                    console.print(traceback.format_exc())
                return 1
        else:
            # No wildcards, use existing list method for exact project match
            return list_experiments(
                remote_client=remote_client,
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
    parser.add_argument(
        "--dash-url",
        type=str,
        help="ML-Dash server URL (defaults to config or https://api.dash.ml)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="JWT authentication token (auto-loaded from storage if not provided)"
    )
    # Filtering options
    parser.add_argument(
        "-p",
        "--pref",
        "--prefix",
        "--proj",
        "--project",
        dest="project",
        type=str,
        help="List experiments in this project (supports glob: 'tutorial*', 'test-?', 'proj-[0-9]*')"
    )
    parser.add_argument("--status", type=str,
                       choices=["COMPLETED", "RUNNING", "FAILED", "ARCHIVED"],
                       help="Filter experiments by status")
    parser.add_argument("--tags", type=str, help="Filter experiments by tags (comma-separated)")

    # Output options
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
