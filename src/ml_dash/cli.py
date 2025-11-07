"""
CLI interface for ML-Dash.

Provides command-line commands for authentication and configuration.
"""

import argparse
import sys
from typing import Optional, List
from .auth import OAuth2AuthFlow


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for ML-Dash CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="ml-dash",
        description="ML-Dash CLI for authentication and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Authenticate with default auth server
  ml-dash setup

  # Authenticate with custom auth server
  ml-dash setup --auth-server https://auth.example.com

  # Check authentication status
  ml-dash status

  # Logout (clear saved token)
  ml-dash logout

For more information, visit: https://docs.ml-dash.com
        """
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Authenticate with ML-Dash",
        description="Start the OAuth2 authentication flow to authenticate with ML-Dash"
    )
    setup_parser.add_argument(
        "--auth-server",
        help="Auth server URL (default: https://staging-auth.ml-dash.com)",
        default=None
    )
    setup_parser.add_argument(
        "--port",
        type=int,
        default=52845,
        help="Local callback server port (default: 52845)"
    )
    setup_parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds (default: 180)"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check authentication status",
        description="Display current authentication status and saved token info"
    )

    # Logout command
    logout_parser = subparsers.add_parser(
        "logout",
        help="Clear saved authentication token",
        description="Remove saved authentication token from config file"
    )

    return parser


def cmd_setup(args: argparse.Namespace) -> int:
    """
    Handle 'setup' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    flow = OAuth2AuthFlow(
        auth_server=args.auth_server,
        callback_port=args.port,
        timeout=args.timeout
    )

    success = flow.authenticate()
    return 0 if success else 1


def cmd_status(args: argparse.Namespace) -> int:
    """
    Handle 'status' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for authenticated, 1 for not authenticated)
    """
    flow = OAuth2AuthFlow()
    authenticated = flow.check_status()
    return 0 if authenticated else 1


def cmd_logout(args: argparse.Namespace) -> int:
    """
    Handle 'logout' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    flow = OAuth2AuthFlow()
    success = flow.logout()
    return 0 if success else 1


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for ML-Dash CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Route to command handler
    if args.command == "setup":
        return cmd_setup(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "logout":
        return cmd_logout(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
