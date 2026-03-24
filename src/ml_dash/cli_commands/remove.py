"""Remove command for ml-dash CLI - delete projects."""

import argparse
from typing import Optional

from rich.console import Console

from ml_dash.client import RemoteClient
from ml_dash.config import DEFAULT_API_URL, config


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
    "-p", "--project",
    type=str,
    required=True,
    help="Project name or namespace/project (e.g. 'my-project' or 'tom/my-project')",
  )
  parser.add_argument(
    "-y", "--yes",
    action="store_true",
    help="Skip confirmation prompt",
  )
  parser.add_argument(
    "--dash-url", "--api-url",
    dest="dash_url",
    type=str,
    help="ML-Dash server URL (default: https://api.dash.ml)",
  )


def cmd_remove(args) -> int:
  """Execute remove command."""
  console = Console()

  remote_url = args.dash_url or config.remote_url or DEFAULT_API_URL

  prefix = args.project.strip("/")
  parts = prefix.split("/")

  if len(parts) > 2:
    console.print(
      "[red]Error:[/red] Project can have at most 2 parts (namespace/project).\n"
      f"Got: {args.project}\n\n"
      "Examples:\n"
      "  ml-dash remove -p my-project\n"
      "  ml-dash remove -p geyang/old-project"
    )
    return 1

  if len(parts) == 1:
    namespace = None
    project_name = parts[0]
  else:
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
    client = RemoteClient(base_url=dash_url, namespace=namespace)

    namespace = client.namespace

    if not namespace:
      console.print("[red]Error:[/red] Could not determine namespace. Please login first.")
      return 1

    full_path = f"{namespace}/{project_name}"

    project_id = client._get_project_id(project_name)
    if not project_id:
      console.print(f"[yellow]⚠[/yellow] Project '[bold]{full_path}[/bold]' not found.")
      return 0

    if not skip_confirm:
      console.print(
        "\n[red bold]⚠ WARNING ⚠[/red bold]\n\n"
        f"You are about to delete project: [bold]{full_path}[/bold]\n"
        "This will permanently delete:\n"
        "  • All experiments in this project\n"
        "  • All metrics and logs\n"
        "  • All uploaded files\n\n"
        "[red]This action CANNOT be undone.[/red]\n"
      )
      confirm = console.input("Type the project name to confirm deletion: ")
      if confirm.strip() != project_name:
        console.print("\n[yellow]Deletion cancelled.[/yellow]")
        return 0

    console.print(f"\n[dim]Deleting project '{full_path}'...[/dim]")

    result = client.delete_project(project_name)

    console.print(f"[green]✓[/green] Project '[bold]{project_name}[/bold]' deleted from namespace '[bold]{namespace}[/bold]'")
    deleted = result.get('deleted')
    experiments = result.get('experiments')
    if deleted is not None:
      console.print(f"  Deleted nodes: {deleted}")
    if experiments is not None:
      console.print(f"  Deleted experiments: {experiments}")

    return 0

  except Exception as e:
    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
      if e.response.status_code == 404:
        console.print(f"[yellow]⚠[/yellow] Project '[bold]{project_name}[/bold]' not found.")
        return 0
      if e.response.status_code == 403:
        console.print("[red]Error:[/red] Permission denied.")
        return 1
    console.print(f"[red]Error deleting project:[/red] {e}")
    return 1
