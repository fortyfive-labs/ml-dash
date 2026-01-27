"""ML-Dash command-line interface."""

import argparse
import sys
from typing import Optional, List


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ml-dash",
        description=(
            "ML-Dash: ML experiment tracking and data storage CLI\n\n"
            "View your experiments, statistics, and plots online at:\n"
            "  https://dash.ml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Import and add command parsers
    from .cli_commands import upload, download, list as list_cmd, login, logout, profile, api, create, remove

    # Authentication commands
    login.add_parser(subparsers)
    logout.add_parser(subparsers)
    profile.add_parser(subparsers)

    # API commands
    api.add_parser(subparsers)

    # Project commands
    create.add_parser(subparsers)
    remove.add_parser(subparsers)

    # Data commands
    upload.add_parser(subparsers)
    download.add_parser(subparsers)
    list_cmd.add_parser(subparsers)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        return 0

    # Route to command handlers
    if args.command == "login":
        from .cli_commands import login
        return login.cmd_login(args)
    elif args.command == "logout":
        from .cli_commands import logout
        return logout.cmd_logout(args)
    elif args.command == "profile":
        from .cli_commands import profile
        return profile.cmd_profile(args)
    elif args.command == "create":
        from .cli_commands import create
        return create.cmd_create(args)
    elif args.command == "remove":
        from .cli_commands import remove
        return remove.cmd_remove(args)
    elif args.command == "upload":
        from .cli_commands import upload
        return upload.cmd_upload(args)
    elif args.command == "download":
        from .cli_commands import download
        return download.cmd_download(args)
    elif args.command == "list":
        from .cli_commands import list as list_cmd
        return list_cmd.cmd_list(args)
    elif args.command == "api":
        from .cli_commands import api
        return api.cmd_api(args)

    # Unknown command (shouldn't happen due to subparsers)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
