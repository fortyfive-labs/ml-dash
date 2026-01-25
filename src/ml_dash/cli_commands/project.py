"""Project commands for ml-dash CLI - create and remove projects."""

import argparse
from typing import Optional

from rich.console import Console

from ml_dash.client import RemoteClient
from ml_dash.config import config


def add_parser(subparsers):
    """Add project command parser with subcommands."""
    project_parser = subparsers.add_parser(
        "project",
        help="Project management commands",
        description="Create and manage projects in ml-dash.",
    )

    project_subparsers = project_parser.add_subparsers(
        dest="project_command",
        help="Project commands",
        metavar="COMMAND",
    )

    # Create subcommand
    create_parser = project_subparsers.add_parser(
        "create",
        help="Create a new project",
        description="""Create a new project in ml-dash.

Examples:
  # Create a project in current user's namespace
  ml-dash project create -p new-project

  # Create a project in a specific namespace
  ml-dash project create -p geyang/new-project

  # Create with description
  ml-dash project create -p geyang/tutorials -d "ML tutorials and examples"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    create_parser.add_argument(
        "-p", "--prefix",
        type=str,
        required=True,
        help="Project name or namespace/project",
    )
    create_parser.add_argument(
        "-d", "--description",
        type=str,
        help="Project description (optional)",
    )
    create_parser.add_argument(
        "--dash-url",
        type=str,
        help="ML-Dash server URL (default: https://api.dash.ml)",
    )

    # Remove subcommand
    remove_parser = project_subparsers.add_parser(
        "remove",
        help="Remove (soft delete) a project",
        description="""Remove a project from ml-dash (soft delete).

The project and its contents will be marked as deleted but can be recovered.

Examples:
  # Remove a project in current user's namespace
  ml-dash project remove -p my-project

  # Remove a project in a specific namespace
  ml-dash project remove -p geyang/old-project

  # Skip confirmation prompt
  ml-dash project remove -p my-project --yes
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    remove_parser.add_argument(
        "-p", "--prefix",
        type=str,
        required=True,
        help="Project name or namespace/project to remove",
    )
    remove_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    remove_parser.add_argument(
        "--dash-url",
        type=str,
        help="ML-Dash server URL (default: https://api.dash.ml)",
    )


def cmd_project(args) -> int:
    """Execute project command."""
    console = Console()

    if args.project_command is None:
        console.print("[yellow]Usage:[/yellow] ml-dash project <create|remove> [options]")
        console.print("\nAvailable commands:")
        console.print("  create  Create a new project")
        console.print("  remove  Remove (soft delete) a project")
        return 1

    if args.project_command == "create":
        return _cmd_create(args, console)
    elif args.project_command == "remove":
        return _cmd_remove(args, console)

    return 1


def _cmd_create(args, console: Console) -> int:
    """Execute project create command."""
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
            f"  ml-dash project create -p new-project\n"
            f"  ml-dash project create -p geyang/new-project"
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

    return _create_project(
        namespace=namespace,
        project_name=project_name,
        description=args.description,
        dash_url=remote_url,
        console=console,
    )


def _cmd_remove(args, console: Console) -> int:
    """Execute project remove command."""
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
            f"  ml-dash project remove -p my-project\n"
            f"  ml-dash project remove -p geyang/old-project"
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
        skip_confirm=args.yes,
        dash_url=remote_url,
        console=console,
    )


def _create_project(
    namespace: Optional[str],
    project_name: str,
    description: Optional[str],
    dash_url: str,
    console: Console,
) -> int:
    """Create a new project."""
    try:
        # Initialize client (namespace will be auto-fetched from server if not provided)
        client = RemoteClient(base_url=dash_url, namespace=namespace)

        # Get namespace (triggers server query if not set)
        namespace = client.namespace

        if not namespace:
            console.print("[red]Error:[/red] Could not determine namespace. Please login first.")
            return 1

        console.print(f"[dim]Creating project '{project_name}' in namespace '{namespace}'[/dim]")

        # Create project using unified node API
        response = client._client.post(
            f"/namespaces/{namespace}/nodes",
            json={
                "type": "PROJECT",
                "name": project_name,
                "slug": project_name,
                "description": description or "",
            }
        )
        response.raise_for_status()
        result = response.json()

        # Extract project info
        project = result.get("project", {})
        project_id = project.get("id")
        project_slug = project.get("slug")

        # Success message
        console.print(f"[green]✓[/green] Project created successfully!")
        console.print(f"  Name: [bold]{project_slug}[/bold]")
        console.print(f"  Namespace: [bold]{namespace}[/bold]")
        console.print(f"  ID: {project_id}")
        if description:
            console.print(f"  Description: {description}")
        console.print(f"\n  View at: https://dash.ml/@{namespace}/{project_slug}")

        return 0

    except Exception as e:
        # Check if it's a 409 conflict (project already exists)
        if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 409:
            console.print(f"[yellow]⚠[/yellow] Project '[bold]{project_name}[/bold]' already exists in namespace '[bold]{namespace}[/bold]'")
            return 0

        console.print(f"[red]Error creating project:[/red] {e}")
        return 1


def _remove_project(
    namespace: Optional[str],
    project_name: str,
    skip_confirm: bool,
    dash_url: str,
    console: Console,
) -> int:
    """Remove (soft delete) a project."""
    try:
        # Initialize client (namespace will be auto-fetched from server if not provided)
        client = RemoteClient(base_url=dash_url, namespace=namespace)

        # Get namespace (triggers server query if not set)
        namespace = client.namespace

        if not namespace:
            console.print("[red]Error:[/red] Could not determine namespace. Please login first.")
            return 1

        # Confirmation prompt
        if not skip_confirm:
            console.print(f"[yellow]Warning:[/yellow] This will delete project '[bold]{project_name}[/bold]' in namespace '[bold]{namespace}[/bold]'")
            console.print("The project and its contents will be marked as deleted.")
            confirm = console.input("\nAre you sure? [y/N] ")
            if confirm.lower() not in ("y", "yes"):
                console.print("[dim]Cancelled.[/dim]")
                return 0

        console.print(f"[dim]Removing project '{project_name}' from namespace '{namespace}'[/dim]")

        # Delete project using REST API
        response = client._client.delete(
            f"/namespaces/{namespace}/projects/{project_name}"
        )
        response.raise_for_status()
        result = response.json()

        # Success message
        console.print(f"[green]✓[/green] Project removed successfully!")
        console.print(f"  Project: [bold]{project_name}[/bold]")
        console.print(f"  Namespace: [bold]{namespace}[/bold]")
        console.print(f"  Deleted at: {result.get('deletedAt', 'unknown')}")

        return 0

    except Exception as e:
        # Check for specific error codes
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            status = e.response.status_code
            if status == 404:
                console.print(f"[red]Error:[/red] Project '[bold]{project_name}[/bold]' not found in namespace '[bold]{namespace}[/bold]'")
                return 1
            if status == 410:
                console.print(f"[yellow]⚠[/yellow] Project '[bold]{project_name}[/bold]' has already been deleted")
                return 0

        console.print(f"[red]Error removing project:[/red] {e}")
        return 1
