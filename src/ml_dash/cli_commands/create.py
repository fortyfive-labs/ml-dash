"""Create command for ml-dash CLI - create projects."""

import argparse
from typing import Optional

from rich.console import Console

from ml_dash.client import RemoteClient
from ml_dash.config import DEFAULT_API_URL, config


def add_parser(subparsers):
  """Add create command parser."""
  parser = subparsers.add_parser(
    "create",
    help="Create a new project",
    description="""Create a new project in ml-dash.

Examples:
  # Create a project in current user's namespace
  ml-dash create -p new-project

  # Create a project in a specific namespace
  ml-dash create -p geyang/new-project

  # Create with description
  ml-dash create -p geyang/tutorials -d "ML tutorials and examples"
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "-p", "--project",
    type=str,
    required=True,
    help="Project name or namespace/project (e.g. 'my-project' or 'tom/my-project')",
  )
  parser.add_argument(
    "-d", "--description",
    type=str,
    help="Project description (optional)",
  )
  parser.add_argument(
    "--dash-url", "--api-url",
    dest="dash_url",
    type=str,
    help="ML-Dash server URL (default: https://api.dash.ml)",
  )


def cmd_create(args) -> int:
  """Execute create command."""
  console = Console()

  # Get remote URL
  remote_url = args.dash_url or config.remote_url or DEFAULT_API_URL

  # Parse the project argument
  prefix = args.project.strip("/")
  parts = prefix.split("/")

  if len(parts) > 2:
    console.print(
      "[red]Error:[/red] Project can have at most 2 parts (namespace/project).\n"
      f"Got: {args.project}\n\n"
      "Examples:\n"
      "  ml-dash create -p new-project\n"
      "  ml-dash create -p geyang/new-project"
    )
    return 1

  if len(parts) == 1:
    namespace = None
    project_name = parts[0]
  else:
    namespace = parts[0]
    project_name = parts[1]

  return _create_project(
    namespace=namespace,
    project_name=project_name,
    description=args.description,
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
    client = RemoteClient(base_url=dash_url, namespace=namespace)

    namespace = client.namespace

    if not namespace:
      console.print("[red]Error:[/red] Could not determine namespace. Please login first.")
      return 1

    console.print(f"[dim]Creating project '{project_name}' in namespace '{namespace}'[/dim]")

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

    project = result.get("project") or result
    project_slug = project.get("slug") or project_name
    project_id = project.get("id")

    console.print("[green]✓[/green] Project created successfully!")
    console.print(f"  Name: [bold]{project_slug}[/bold]")
    console.print(f"  Namespace: [bold]{namespace}[/bold]")
    if project_id:
      console.print(f"  ID: {project_id}")
    if description:
      console.print(f"  Description: {description}")
    console.print(f"\n  View at: https://dash.ml/@{namespace}/{project_slug}")

    return 0

  except Exception as e:
    if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 409:
      console.print(f"[yellow]⚠[/yellow] Project '[bold]{project_name}[/bold]' already exists in namespace '[bold]{namespace}[/bold]'")
      return 0

    console.print(f"[red]Error creating project:[/red] {e}")
    return 1
