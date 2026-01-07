"""API command for ml-dash CLI - send GraphQL queries to the server."""

import argparse
import json

from rich.console import Console

from ml_dash.client import RemoteClient
from ml_dash.config import config


def add_parser(subparsers):
  """Add api command parser."""
  parser = subparsers.add_parser(
    "api",
    help="Send GraphQL queries to ml-dash server",
    description="""Send GraphQL queries to the ml-dash server.

Examples:
  # Query current user
  ml-dash api --query "me { username name email }"

  # Query with arguments (single quotes auto-converted to double)
  ml-dash api --query "user(title: 'hello') { id title }"

  # Extract specific field with jq-like syntax
  ml-dash api --query "me { username }" --jq ".data.me.username"

  # Mutation to update username
  ml-dash api --mutation "updateUser(username: 'newname') { username }"

Notes:
  - Single quotes are auto-converted to double quotes for GraphQL
  - Use --jq for dot-path extraction (built-in, no deps)
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
    "--query", "-q",
    metavar="QUERY",
    help="GraphQL query string",
  )
  group.add_argument(
    "--mutation", "-m",
    metavar="MUTATION",
    help="GraphQL mutation string",
  )
  parser.add_argument(
    "--jq",
    metavar="PATH",
    help="Extract value using dot-path (e.g., .data.me.username)",
  )
  parser.add_argument(
    "--remote",
    type=str,
    help="ML-Dash server URL (default: https://api.dash.ml)",
  )


def extract_path(data, path: str):
  """Extract value from nested dict using dot-path notation.

  Args:
      data: Nested dict/list structure
      path: Dot-separated path (e.g., ".data.me.username")

  Returns:
      Extracted value

  Examples:
      >>> extract_path({"data": {"me": {"username": "ge"}}}, ".data.me.username")
      "ge"
  """
  for key in path.lstrip(".").split("."):
    if key:
      if isinstance(data, dict):
        data = data[key]
      elif isinstance(data, list):
        data = data[int(key)]
      else:
        raise KeyError(f"Cannot access '{key}' on {type(data).__name__}")
  return data


def fix_quotes(query: str) -> str:
  """Convert single quotes to double quotes for GraphQL.

  GraphQL requires double quotes for strings. This allows users to write
  queries with single quotes for shell convenience.

  Args:
      query: GraphQL query string with possible single quotes

  Returns:
      Query with single quotes converted to double quotes
  """
  # Simple conversion - assumes single quotes are for strings
  # This handles: user(title: 'hello') -> user(title: "hello")
  return query.replace("'", '"')


def build_query(query: str, is_mutation: bool) -> str:
  """Build complete GraphQL query string.

  Args:
      query: Query or mutation body
      is_mutation: Whether this is a mutation

  Returns:
      Complete GraphQL query string
  """
  query = query.strip()
  query = fix_quotes(query)

  # If already properly formatted, return as-is
  if query.startswith("{") or query.startswith("mutation") or query.startswith("query"):
    return query

  # Wrap appropriately
  if is_mutation:
    return "mutation { " + query + " }"
  else:
    return "{ " + query + " }"


def cmd_api(args) -> int:
  """Execute api command."""
  console = Console()

  # Get remote URL
  remote_url = args.remote or config.remote_url or "https://api.dash.ml"

  try:
    # Initialize client
    client = RemoteClient(base_url=remote_url)

    # Determine query type and build query
    if args.mutation:
      query = build_query(args.mutation, is_mutation=True)
    else:
      query = build_query(args.query, is_mutation=False)

    # Execute GraphQL query
    result = client.graphql_query(query)

    # Apply jq path extraction if specified
    if args.jq:
      try:
        result = extract_path(result, args.jq)
      except (KeyError, IndexError, TypeError) as e:
        console.print(f"[red]Error extracting path '{args.jq}': {e}[/red]")
        return 1

    # Output result
    if isinstance(result, (dict, list)):
      console.print_json(json.dumps(result))
    else:
      console.print(json.dumps(result))

    return 0

  except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    return 1
