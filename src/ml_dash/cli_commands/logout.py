"""Logout command for ml-dash CLI."""

from rich.console import Console

from ml_dash.auth.token_storage import get_token_storage
from ml_dash.auth.exceptions import StorageError


def add_parser(subparsers):
    """Add logout command parser.

    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser(
        "logout",
        help="Clear stored authentication token",
        description="Logout from ml-dash by clearing stored authentication token",
    )


def cmd_logout(args) -> int:
    """Execute logout command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    console = Console()

    try:
        # Get storage backend
        storage = get_token_storage()

        # Delete stored token
        storage.delete("ml-dash-token")

        console.print(
            "[bold green]✓ Logged out successfully![/bold green]\n\n"
            "Your authentication token has been cleared.\n\n"
            "To log in again:\n"
            "  ml-dash login"
        )

        return 0

    except StorageError as e:
        console.print(f"[red]✗ Storage error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]✗ Unexpected error:[/red] {e}")
        return 1
