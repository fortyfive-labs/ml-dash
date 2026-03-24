"""Profile command for ml-dash CLI - shows current user and configuration."""

import getpass
import json
import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from ml_dash.auth.token_storage import decode_jwt_payload, get_token_storage
from ml_dash.client import RemoteClient
from ml_dash.config import DEFAULT_API_URL, config


def add_parser(subparsers):
  """Add profile command parser."""
  parser = subparsers.add_parser(
    "profile",
    help="Show current user profile",
    description="Display the current authenticated user profile and configuration.",
  )
  parser.add_argument(
    "--dash-url", "--api-url",
    dest="dash_url",
    type=str,
    help="ML-Dash server URL (default: from config)",
  )
  parser.add_argument(
    "--json",
    action="store_true",
    help="Output as JSON",
  )
  parser.add_argument(
    "--cached",
    action="store_true",
    help="Use cached token data (default: fetch fresh from server)",
  )


def _fetch_fresh_profile(remote_url: str, token: str) -> dict:
  """Fetch user profile from the API server, or None on failure."""
  try:
    client = RemoteClient(remote_url, api_key=token)
    user_data = client.get_current_user()

    if user_data:
      return {
        "sub": user_data.get("id"),
        "username": user_data.get("username"),
        "name": user_data.get("name"),
        "email": user_data.get("email"),
        "given_name": user_data.get("given_name"),
        "family_name": user_data.get("family_name"),
      }
  except Exception:
    return None

  return None


def _check_token_expiration(token_payload: dict) -> tuple[bool, str]:
  """Return (is_expired, status_message) for the given JWT payload."""
  exp = token_payload.get("exp")
  if not exp:
    return False, None

  current_time = int(time.time())
  time_left = exp - current_time

  if time_left < 0:
    return True, "[red]Token expired[/red]"
  elif time_left < 86400:  # Less than 1 day
    hours_left = time_left // 3600
    return False, f"[yellow]Token expires in {hours_left} hours[/yellow]"
  else:
    days_left = time_left // 86400
    return False, f"Expires in {days_left} days"


def cmd_profile(args) -> int:
  """Execute profile command."""
  console = Console()

  storage = get_token_storage()
  token = storage.load("ml-dash-token")

  remote_url = args.dash_url or config.remote_url

  info = {
    "authenticated": False,
    "remote_url": remote_url,
    "local_user": getpass.getuser(),
  }

  if token:
    info["authenticated"] = True

    token_payload = decode_jwt_payload(token)
    is_expired, expiry_message = _check_token_expiration(token_payload)

    if is_expired:
      info["authenticated"] = False
      info["error"] = "Token expired. Please run 'ml-dash login' to re-authenticate."
    else:
      if args.cached:
        info["user"] = token_payload
        info["source"] = "token"
      else:
        fresh_profile = _fetch_fresh_profile(remote_url, token)
        if fresh_profile:
          info["user"] = fresh_profile
          info["source"] = "server"
        else:
          info["user"] = token_payload
          info["source"] = "token"
          info["warning"] = "Could not fetch fresh profile from server, using cached token data"

      if expiry_message:
        info["token_status"] = expiry_message

  if args.json:
    console.print_json(json.dumps(info))
    return 0

  if not info["authenticated"]:
    error_msg = info.get("error", "Not authenticated")
    console.print(
      Panel(
        f"[bold cyan]OS Username:[/bold cyan]  {info['local_user']}\n\n"
        f"[yellow]{error_msg}[/yellow]\n\n"
        "Run [cyan]ml-dash login[/cyan] to authenticate.",
        title="[bold]ML-Dash Info[/bold]",
        border_style="yellow",
      )
    )
    return 0

  table = Table(show_header=False, box=None, padding=(0, 2))
  table.add_column("Key", style="bold cyan")
  table.add_column("Value")

  user = info["user"]
  if user.get("username"):
    table.add_row("Username", user["username"])
  else:
    table.add_row("Username", "[red]Unavailable[/red]")
  if user.get("sub"):
    table.add_row("User ID", user["sub"])
  table.add_row("Name", user.get("name") or "Unknown")
  if user.get("email"):
    table.add_row("Email", user["email"])
  table.add_row("Remote", info["remote_url"] or DEFAULT_API_URL)

  if info.get("token_status"):
    table.add_row("Token Status", info["token_status"])

  source = info["source"]
  if source == "server":
    table.add_row("Data Source", "[green]Server (Fresh)[/green]")
  else:
    table.add_row("Data Source", "[yellow]Token (Cached)[/yellow]")

  warning_text = f"\n[yellow]⚠ {info['warning']}[/yellow]" if info.get("warning") else None
  tip_text = "\n[dim]Tip: Use --cached to use cached token data (faster but may be outdated)[/dim]" if source == "server" else None

  panel_content = table
  if warning_text or tip_text:
    items = [table]
    if warning_text:
      items.append(warning_text)
    if tip_text:
      items.append(tip_text)
    panel_content = Group(*items)

  console.print(
    Panel(
      panel_content,
      title="[bold green]✓ Authenticated[/bold green]",
      border_style="green",
    )
  )

  return 0
