"""Info command for ml-dash CLI - shows current user and configuration."""

import json
from base64 import urlsafe_b64decode

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_dash.auth.token_storage import get_token_storage
from ml_dash.config import config


def add_parser(subparsers):
  """Add info command parser."""
  parser = subparsers.add_parser(
    "info",
    help="Show current user and configuration",
    description="Display information about the current authenticated user and configuration.",
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


def cmd_info(args) -> int:
  """Execute info command."""
  console = Console()

  # Load token
  storage = get_token_storage()
  token = storage.load("ml-dash-token")

  info = {
    "authenticated": False,
    "remote_url": config.remote_url,
  }

  if token:
    payload = decode_jwt_payload(token)
    info["authenticated"] = True
    info["user"] = {
      "id": payload.get("sub"),
      "email": payload.get("email"),
      "name": payload.get("name"),
    }
    # Token expiry
    if "exp" in payload:
      from datetime import datetime
      exp = datetime.fromtimestamp(payload["exp"])
      info["token_expires"] = exp.isoformat()

  if args.json:
    console.print_json(json.dumps(info))
    return 0

  # Rich display
  if not info["authenticated"]:
    console.print(
      Panel(
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

  user = info.get("user", {})
  table.add_row("User", user.get("name") or user.get("email") or "Unknown")
  if user.get("email"):
    table.add_row("Email", user["email"])
  if user.get("id"):
    table.add_row("User ID", user["id"])
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
