"""Create command for ml-dash CLI - create projects."""

import argparse

from rich.console import Console

from ml_dash.client import RemoteClient
from ml_dash.config import config


def add_parser(subparsers):
    """Add create command parser."""
    parser = subparsers.add_parser(
        "create",
        help="Create a new project",
        description="""Create a new project in ml-dash.

Examples:
  # Create a project (uses current user's namespace)
  ml-dash create --project tutorials

  # Create in a specific namespace
  ml-dash create --project tutorials --namespace geyang

  # Create with description
  ml-dash create --project tutorials --description "ML tutorials and examples"

  # Short options
  ml-dash create -p tutorials -d "My project"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        required=True,
        help="Project name/slug",
    )
    parser.add_argument(
        "--namespace", "-n",
        type=str,
        help="Namespace to create the project in (default: current user's namespace)",
    )
    parser.add_argument(
        "--description", "-d",
        type=str,
        help="Project description (optional)",
    )
    parser.add_argument(
        "--dash-url",
        type=str,
        help="ML-Dash server URL (default: https://api.dash.ml)",
    )


def cmd_create(args) -> int:
    """Execute create command."""
    console = Console()

    # Get remote URL
    remote_url = args.dash_url or config.remote_url or "https://api.dash.ml"

    # Get parameters
    namespace = args.namespace  # Can be None, client will fetch from server
    project_name = args.project
    description = args.description

    try:
        # Initialize client (namespace will be auto-fetched from server if not provided)
        client = RemoteClient(base_url=remote_url, namespace=namespace)

        # Get namespace (triggers server query if not set)
        namespace = client.namespace

        if not args.namespace:
            console.print(f"[dim]Using namespace: {namespace}[/dim]")

        # Create project using unified node API
        response = client._client.post(
            f"/namespaces/{namespace}/nodes",
            json={
                "type": "PROJECT",
                "name": project_name,
                "slug": project_name,
                "description": description,
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
        console.print(f"\n  View at: [link=https://dash.ml]https://dash.ml[/link]")

        return 0

    except Exception as e:
        # Check if it's a 409 conflict (project already exists)
        if hasattr(e, 'response') and e.response.status_code == 409:
            console.print(f"[yellow]⚠[/yellow] Project '[bold]{project_name}[/bold]' already exists in namespace '[bold]{namespace}[/bold]'")
            return 0

        console.print(f"[red]Error creating project: {e}[/red]")
        return 1
