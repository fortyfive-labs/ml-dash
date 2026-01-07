"""Profile command for ml-dash CLI - shows current user and configuration."""

import json
from base64 import urlsafe_b64decode

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_dash.auth.token_storage import get_token_storage
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


def decode_jwt_payload(token: str) -> dict:
  """Decode JWT payload without verification (for display only).

  Args:
      token: JWT token string

  Returns:
      Decoded payload dict
  """
  try:
    # JWT format: header.payload.signature
    parts = token.split(".")
    if len(parts) != 3:
      return {}

    # Decode payload (second part)
    payload = parts[1]
    # Add padding if needed
    padding = 4 - len(payload) % 4
    if padding != 4:
      payload += "=" * padding

    decoded = urlsafe_b64decode(payload)
    return json.loads(decoded)
  except Exception:
    return {}


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

    # Get user profile from server
    try:
      from ml_dash.client import RemoteClient

      client = RemoteClient(base_url=config.remote_url or "https://api.dash.ml")
      user_profile = client.get_current_user()
      info["user"] = user_profile
    except Exception as e:
      # Fallback to JWT payload if server call fails
      payload = decode_jwt_payload(token)
      info["user"] = payload
      info["server_error"] = str(e)

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
