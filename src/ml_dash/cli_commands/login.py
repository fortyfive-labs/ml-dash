"""Login command for ml-dash CLI."""

import sys
import webbrowser
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ml_dash.auth.device_flow import DeviceFlowClient
from ml_dash.auth.device_secret import get_or_create_device_secret
from ml_dash.auth.token_storage import get_token_storage
from ml_dash.auth.exceptions import (
    DeviceCodeExpiredError,
    AuthorizationDeniedError,
    TokenExchangeError,
)
from ml_dash.config import config


def add_parser(subparsers):
    """Add login command parser.

    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser(
        "login",
        help="Authenticate with ml-dash using device authorization flow",
        description="Login to ml-dash server using OAuth2 device authorization flow",
    )

    parser.add_argument(
        "--remote",
        type=str,
        help="ML-Dash server URL (e.g., https://api.ml-dash.com)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser for authorization",
    )


def generate_qr_code_ascii(url: str) -> str:
    """Generate ASCII QR code for the given URL.

    Args:
        url: URL to encode in QR code

    Returns:
        ASCII art QR code string
    """
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)

        # Generate ASCII art
        output = []
        for row in qr.get_matrix():
            line = ""
            for cell in row:
                line += "██" if cell else "  "
            output.append(line)

        return "\n".join(output)
    except ImportError:
        return "[QR code unavailable - install qrcode: pip install qrcode]"
    except Exception:
        return "[QR code generation failed]"


def cmd_login(args) -> int:
    """Execute login command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    console = Console()

    # Get remote URL
    remote_url = args.remote or config.remote_url
    if not remote_url:
        console.print(
            "[red]Error: No remote URL configured.[/red]\n\n"
            "Please specify with --remote or set default:\n"
            "  ml-dash login --remote https://api.ml-dash.com"
        )
        return 1

    try:
        # Initialize device flow
        console.print("[bold]Initializing device authorization...[/bold]\n")

        device_secret = get_or_create_device_secret(config)
        device_client = DeviceFlowClient(
            device_secret=device_secret, ml_dash_server_url=remote_url
        )

        # Start device flow with vuer-auth
        flow = device_client.start_device_flow()

        # Generate QR code
        qr_code = generate_qr_code_ascii(flow.verification_uri_complete)

        # Display rich UI with QR code
        panel_content = (
            f"[bold cyan]1. Visit this URL:[/bold cyan]\n\n"
            f"   {flow.verification_uri}\n\n"
            f"[bold cyan]2. Enter this code:[/bold cyan]\n\n"
            f"   [bold green]{flow.user_code}[/bold green]\n\n"
        )

        # Add QR code if available
        if "unavailable" not in qr_code and "failed" not in qr_code:
            panel_content += f"[bold cyan]Or scan QR code:[/bold cyan]\n\n{qr_code}\n\n"

        panel_content += f"[dim]Code expires in {flow.expires_in // 60} minutes[/dim]"

        panel = Panel(
            panel_content,
            title="[bold blue]DEVICE AUTHORIZATION REQUIRED[/bold blue]",
            border_style="blue",
            expand=False,
        )
        console.print(panel)
        console.print()

        # Auto-open browser unless disabled
        if not args.no_browser:
            try:
                webbrowser.open(flow.verification_uri_complete)
                console.print("[dim]✓ Opened browser automatically[/dim]\n")
            except Exception:
                # Silent failure - user can manually open URL
                pass

        # Poll for authorization with progress indicator
        console.print("[bold]Waiting for authorization...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Polling", total=None)

            def update_progress(elapsed: int):
                progress.update(task, description=f"Waiting ({elapsed}s)")

            try:
                vuer_auth_token = device_client.poll_for_token(
                    max_attempts=120, progress_callback=update_progress
                )
            except DeviceCodeExpiredError:
                console.print(
                    "\n[red]✗ Device code expired[/red]\n\n"
                    "The authorization code expired after 10 minutes.\n"
                    "Please run 'ml-dash login' again."
                )
                return 1
            except AuthorizationDeniedError:
                console.print(
                    "\n[red]✗ Authorization denied[/red]\n\n"
                    "You declined the authorization request in your browser.\n\n"
                    "To try again:\n"
                    "  ml-dash login"
                )
                return 1
            except TimeoutError:
                console.print(
                    "\n[red]✗ Authorization timed out[/red]\n\n"
                    "No response after 10 minutes.\n\n"
                    "Please run 'ml-dash login' again."
                )
                return 1

        console.print("[green]✓ Authorization successful![/green]\n")

        # Exchange vuer-auth token for ml-dash token
        console.print("[bold]Exchanging token with ml-dash server...[/bold]")

        try:
            ml_dash_token = device_client.exchange_token(vuer_auth_token)
        except TokenExchangeError as e:
            console.print(f"\n[red]✗ Token exchange failed:[/red] {e}\n")
            return 1

        # Store ml-dash permanent token
        storage = get_token_storage()
        storage.store("ml-dash-token", ml_dash_token)

        console.print("[green]✓ Token exchanged successfully![/green]\n")

        # Success message
        console.print(
            "[bold green]✓ Logged in successfully![/bold green]\n\n"
            "Your authentication token has been securely stored.\n"
            "You can now use ml-dash commands without --api-key.\n\n"
            "Examples:\n"
            "  ml-dash upload ./experiments\n"
            "  ml-dash download ./output\n"
            "  ml-dash list"
        )

        return 0

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Login cancelled by user.[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback

        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        return 1
