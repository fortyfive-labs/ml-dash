"""Profile command for ml-dash CLI - shows current user and configuration."""

import json
import time

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
  parser.add_argument(
    "--refresh",
    action="store_true",
    help="Fetch fresh profile from server (not from cached token)",
  )


def _fetch_fresh_profile(remote_url: str, token: str) -> dict:
  """Fetch fresh user profile from the API server.

  Args:
      remote_url: API server URL
      token: JWT authentication token

  Returns:
      User profile dict with username, email, name, etc.
  """
  try:
    from ml_dash.client import RemoteClient

    client = RemoteClient(remote_url, api_key=token)

    # Query for full user profile
    query = """
    query GetUserProfile {
      me {
        id
        username
        name
        email
      }
    }
    """

    result = client.graphql_query(query)
    me = result.get("me", {})

    if me:
      return {
        "sub": me.get("id"),
        "username": me.get("username"),
        "name": me.get("name"),
        "email": me.get("email"),
      }
  except Exception as e:
    # If API call fails, return None to fall back to token decoding
    return None

  return None


def _check_token_expiration(token_payload: dict) -> tuple[bool, str]:
  """Check if token is expired or close to expiring.

  Args:
      token_payload: Decoded JWT payload

  Returns:
      Tuple of (is_expired, message)
  """
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

  return False, None


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

    # Decode token payload for initial data and expiration check
    token_payload = decode_jwt_payload(token)

    # Check token expiration
    is_expired, expiry_message = _check_token_expiration(token_payload)

    if is_expired:
      info["authenticated"] = False
      info["error"] = "Token expired. Please run 'ml-dash login' to re-authenticate."
    else:
      # Fetch fresh profile from server if requested, or fall back to token
      if args.refresh:
        fresh_profile = _fetch_fresh_profile(config.remote_url, token)
        if fresh_profile:
          info["user"] = fresh_profile
          info["source"] = "server"
        else:
          info["user"] = token_payload
          info["source"] = "token"
          info["warning"] = "Could not fetch fresh profile from server, using cached token data"
      else:
        info["user"] = token_payload
        info["source"] = "token"

      if expiry_message:
        info["token_status"] = expiry_message

  if args.json:
    console.print_json(json.dumps(info))
    return 0

  # Rich display
  if not info["authenticated"]:
    error_msg = info.get("error", "Not authenticated")
    console.print(
      Panel(
        f"[bold cyan]OS Username:[/bold cyan]  {info.get('local_user')}\n\n"
        f"[yellow]{error_msg}[/yellow]\n\n"
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

  # Show token status (expiration)
  if info.get("token_status"):
    table.add_row("Token Status", info["token_status"])

  # Show data source
  source = info.get("source", "token")
  if source == "server":
    table.add_row("Data Source", "[green]Server (Fresh)[/green]")
  else:
    table.add_row("Data Source", "[yellow]Token (Cached)[/yellow]")

  # Show warning if any
  warning_text = None
  if info.get("warning"):
    warning_text = f"\n[yellow]⚠ {info['warning']}[/yellow]"

  # Show tip for refreshing
  if source == "token":
    tip_text = "\n[dim]Tip: Use --refresh to fetch fresh data from server[/dim]"
  else:
    tip_text = None

  # Build panel content
  panel_content = table
  if warning_text or tip_text:
    from rich.console import Group
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
