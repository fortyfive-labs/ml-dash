"""Remove command for ml-dash CLI - delete projects."""

import argparse
from typing import Optional

from rich.console import Console

from ml_dash.client import RemoteClient
from ml_dash.config import config


def add_parser(subparsers):
    """Add remove command parser."""
    parser = subparsers.add_parser(
        "remove",
        help="Delete a project",
        description="""Delete a project from ml-dash.

WARNING: This will delete the project and all its experiments, metrics, files, and logs.
This action cannot be undone.

Examples:
  # Delete a project in current user's namespace
  ml-dash remove -p my-project

  # Delete a project in a specific namespace
  ml-dash remove -p geyang/old-project

  # Skip confirmation prompt (use with caution!)
  ml-dash remove -p my-project -y
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        required=True,
        help="Project name or namespace/project",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--dash-url",
        type=str,
        help="ML-Dash server URL (default: https://api.dash.ml)",
    )


def cmd_remove(args) -> int:
    """Execute remove command."""
    console = Console()

    # Get remote URL
    remote_url = args.dash_url or config.remote_url or "https://api.dash.ml"

    # Parse the prefix
    prefix = args.prefix.strip("/")
    parts = prefix.split("/")

    if len(parts) > 2:
        console.print(
            f"[red]Error:[/red] Prefix can have at most 2 parts (namespace/project).\n"
            f"Got: {args.prefix}\n\n"
            f"Examples:\n"
            f"  ml-dash remove -p my-project\n"
            f"  ml-dash remove -p geyang/old-project"
        )
        return 1

    if len(parts) == 1:
        # Format: project (use current user's namespace)
        namespace = None
        project_name = parts[0]
    else:
        # Format: namespace/project
        namespace = parts[0]
        project_name = parts[1]

    return _remove_project(
        namespace=namespace,
        project_name=project_name,
        dash_url=remote_url,
        skip_confirm=args.yes,
        console=console,
    )


def _remove_project(
    namespace: Optional[str],
    project_name: str,
    dash_url: str,
    skip_confirm: bool,
    console: Console,
) -> int:
    """Remove a project."""
    try:
        # Initialize client (namespace will be auto-fetched from server if not provided)
        client = RemoteClient(base_url=dash_url, namespace=namespace)

        # Get namespace (triggers server query if not set)
        namespace = client.namespace

        if not namespace:
            console.print("[red]Error:[/red] Could not determine namespace. Please login first.")
            return 1

        full_path = f"{namespace}/{project_name}"

        # Get project ID to verify it exists
        project_id = client._get_project_id(project_name)
        if not project_id:
            console.print(f"[yellow]⚠[/yellow] Project '[bold]{full_path}[/bold]' not found.")
            return 1

        # Confirmation prompt (unless -y flag is used)
        if not skip_confirm:
            console.print(
                f"\n[red bold]⚠ WARNING ⚠[/red bold]\n\n"
                f"You are about to delete project: [bold]{full_path}[/bold]\n"
                f"This will permanently delete:\n"
                f"  • All experiments in this project\n"
                f"  • All metrics and logs\n"
                f"  • All uploaded files\n\n"
                f"[red]This action CANNOT be undone.[/red]\n"
            )
            confirm = console.input("Type the project name to confirm deletion: ")
            if confirm.strip() != project_name:
                console.print("\n[yellow]Deletion cancelled.[/yellow]")
                return 0

        console.print(f"\n[dim]Deleting project '{full_path}'...[/dim]")

        # Delete project using client method
        result = client.delete_project(project_name)

        # Success message
        console.print(f"[green]✓[/green] {result.get('message', 'Project deleted successfully!')}")
        console.print(f"  Name: [bold]{project_name}[/bold]")
        console.print(f"  Namespace: [bold]{namespace}[/bold]")
        console.print(f"  Project ID: {project_id}")
        console.print(f"  Deleted nodes: {result.get('deleted', 0)}")
        console.print(f"  Deleted experiments: {result.get('experiments', 0)}")

        return 0

    except Exception as e:
        # Check if it's a 404 not found
        if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 404:
            console.print(f"[yellow]⚠[/yellow] Project '[bold]{project_name}[/bold]' not found in namespace '[bold]{namespace}[/bold]'")
            return 1

        # Check if it's a 403 forbidden
        if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 403:
            console.print(f"[red]Error:[/red] Permission denied. You don't have permission to delete this project.")
            return 1

        console.print(f"[red]Error deleting project:[/red] {e}")
        return 1
