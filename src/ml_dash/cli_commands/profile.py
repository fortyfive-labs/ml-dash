"""Profile command for ml-dash CLI - shows current user and configuration."""

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_dash.auth.token_storage import decode_jwt_payload, get_token_storage
from ml_dash.config import config


def add_parser(subparsers):
  """Add profile command parser."""
  parser = subparsers.add_parser(
    "profile",
    help="Show current user profile",
    description="Display the current authenticated user profile and configuration.",
  )
  parser.add_argument(
    "--json",
    action="store_true",
    help="Output as JSON",
  )


def cmd_profile(args) -> int:
  """Execute info command."""
  console = Console()

  # Load token
  storage = get_token_storage()
  token = storage.load("ml-dash-token")

  import getpass

  info = {
    "authenticated": False,
    "remote_url": config.remote_url,
    "local_user": getpass.getuser(),
  }

  if token:
    info["authenticated"] = True
    info["user"] = decode_jwt_payload(token)

  if args.json:
    console.print_json(json.dumps(info))
    return 0

  # Rich display
  if not info["authenticated"]:
    console.print(
      Panel(
        f"[bold cyan]OS Username:[/bold cyan]  {info.get('local_user')}\n\n"
        "[yellow]Not authenticated[/yellow]\n\n"
        "Run [cyan]ml-dash login[/cyan] to authenticate.",
        title="[bold]ML-Dash Info[/bold]",
        border_style="yellow",
      )
    )
    return 0

  # Build info table
  table = Table(show_header=False, box=None, padding=(0, 2))
  table.add_column("Key", style="bold cyan")
  table.add_column("Value")

  # table.add_row("OS Username", info.get("local_user"))
  user = info.get("user", {})
  if user.get("username"):
    table.add_row("Username", user["username"])
  else:
    table.add_row("Username", "[red]Unavailable[/red]")
  if user.get("sub"):
    table.add_row("User ID", user["sub"])
  table.add_row("Name", user.get("name") or "Unknown")
  if user.get("email"):
    table.add_row("Email", user["email"])
  table.add_row("Remote", info.get("remote_url") or "https://api.dash.ml")
  if info.get("token_expires"):
    table.add_row("Token Expires", info["token_expires"])

  console.print(
    Panel(
      table,
      title="[bold green]✓ Authenticated[/bold green]",
      border_style="green",
    )
  )

  return 0
