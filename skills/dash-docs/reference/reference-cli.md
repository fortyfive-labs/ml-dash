# Command-Line Interface (CLI)

ML-Dash provides a command-line interface for authenticating, exploring, and managing experiments on the remote server.

## Commands

| Command | Description |
|---|---|
| `version` | Show ml-dash version |
| `login` | Authenticate using OAuth2 device authorization flow |
| `logout` | Clear stored authentication token |
| `profile` | Show current user profile |
| `api` | Send GraphQL queries to the server |
| `list` | List projects and experiments on the remote server |
| `create` | Create a new project |
| `remove` | Delete a project |
| `upload` | Upload local experiments to the remote server |
| `download` | Download experiments from the remote server |

## Installation

The CLI is included with the ML-Dash Python package:

```bash
pip install ml-dash
# or
uv pip install ml-dash
```

Verify installation:

```bash
ml-dash --help
ml-dash version
```

---

## `ml-dash version`

Print the installed ml-dash version.

```bash
ml-dash version
# ml-dash 0.6.21
```

---

## `ml-dash login`

Authenticate with the ML-Dash server using OAuth2 device authorization flow.

```bash
ml-dash login [--dash-url URL] [--auth-url URL] [--no-browser]
```

**Options:**

| Flag | Description |
|---|---|
| `--dash-url`, `--api-url` | ML-Dash server URL (e.g. `https://api.dash.ml`) |
| `--auth-url` | OAuth authorization server URL (e.g. `https://auth.vuer.ai`) |
| `--no-browser` | Don't automatically open the browser |

**What it does:**

1. Starts an OAuth2 device flow — displays a URL, user code, and QR code
2. Opens your browser automatically (unless `--no-browser`)
3. Polls for authorization (10-minute timeout)
4. Exchanges the OAuth token with the ml-dash server for a permanent token
5. Stores the token securely in your system keychain

```bash
# Login to default server
ml-dash login

# Login to a custom server
ml-dash login --dash-url https://your-server.com

# Display code only, don't open browser
ml-dash login --no-browser
```

After logging in, all other commands pick up the stored token automatically — no `--api-key` needed.

---

## `ml-dash logout`

Clear the stored authentication token.

```bash
ml-dash logout
```

Removes the token from your system keychain. Run `ml-dash login` to re-authenticate.

---

## `ml-dash profile`

Show the current authenticated user and configuration.

```bash
ml-dash profile [--dash-url URL] [--json] [--cached]
```

**Options:**

| Flag | Description |
|---|---|
| `--dash-url`, `--api-url` | ML-Dash server URL |
| `--json` | Output as JSON |
| `--cached` | Use token data instead of fetching fresh from server |

By default, profile fetches live data from the server (`me { username email ... }`). Use `--cached` for an offline/faster check using the stored token payload.

**Displays:** username, user ID, name, email, remote URL, token expiration status, data source.

```bash
# Show profile (fetches from server)
ml-dash profile

# Faster offline check using cached token
ml-dash profile --cached

# JSON output
ml-dash profile --json
```

---

## `ml-dash api`

Send raw GraphQL queries or mutations to the server.

```bash
ml-dash api (--query QUERY | --mutation MUTATION) [--jq PATH] [--dash-url URL]
```

**Options:**

| Flag | Description |
|---|---|
| `--query`, `-q` | GraphQL query string (mutually exclusive with `--mutation`) |
| `--mutation`, `-m` | GraphQL mutation string |
| `--jq PATH` | Extract a value using dot-path notation (e.g. `.me.username`) |
| `--dash-url`, `--api-url` | ML-Dash server URL |

**Notes:**
- Single quotes in queries are auto-converted to double quotes for GraphQL compatibility
- `--jq` paths start from the unwrapped response — there is no top-level `data` key (e.g. use `.me.username`, not `.data.me.username`)
- Bare query bodies are automatically wrapped: `me { username }` → `{ me { username } }`

```bash
# Query current user
ml-dash api --query "me { username email }" --api-url http://localhost:3000

# Extract a specific field
ml-dash api --query "me { username }" --jq ".me.username" --api-url http://localhost:3000

# Using single quotes (auto-converted)
ml-dash api --query "user(title: 'hello') { id title }" --api-url http://localhost:3000

# Send a mutation
ml-dash api --mutation "updateUser(username: 'newname') { username }" --api-url http://localhost:3000
```

---

## `ml-dash create`

Create a new project on the remote server.

```bash
ml-dash create -p PROJECT [-d DESCRIPTION] [--dash-url URL]
```

**Options:**

| Flag | Description |
|---|---|
| `-p`, `--project` | Project name — `project` or `namespace/project` (required) |
| `-d`, `--description` | Project description (optional) |
| `--dash-url`, `--api-url` | Server URL |

If no namespace is provided, it is auto-resolved from the authenticated user's account.

If the project already exists, a warning is shown and the command exits successfully.

```bash
# Create in your own namespace (auto-resolved)
ml-dash create -p my-project --api-url https://api.dash.ml

# Create with explicit namespace
ml-dash create -p tom/my-project --api-url https://api.dash.ml

# Create with description
ml-dash create -p tom/my-project -d "Baseline experiments" --api-url https://api.dash.ml
```

---

## `ml-dash list`

List projects and experiments on the remote server with real server-side pagination.

```bash
ml-dash list [-p PROJECT] [-n NAMESPACE] [--status STATUS] [--tags TAGS]
             [--detailed] [--tracks] [--topic-filter TOPIC]
             [--dash-url URL] [--api-key TOKEN] [-v]
```

**Options:**

| Flag | Description |
|---|---|
| `-p`, `--project` | Project filter. Without this, lists all projects. Supports glob patterns — **always quote them** to prevent shell expansion |
| `-n`, `--namespace` | Namespace for all queries (defaults to the authenticated user's namespace) |
| `--status` | Filter experiments by status: `COMPLETED`, `RUNNING`, `FAILED`, `ARCHIVED` |
| `--tags` | Filter experiments by tags (comma-separated) |
| `--detailed` | Show additional columns: tags and created time |
| `--tracks` | List tracks inside an experiment (requires full `namespace/project/experiment` path) |
| `--topic-filter` | Filter tracks by topic pattern (e.g. `robot/*`) |
| `--dash-url`, `--api-url` | ML-Dash server URL |
| `-v`, `--verbose` | Show full error tracebacks |

**Pagination:**

Results are fetched and displayed one page at a time (50 per page). Navigate with:

| Key | Action |
|---|---|
| `n`, `→`, `Space`, `Enter` | Next page |
| `p`, `b`, `←` | Previous page |
| Any other key | Quit |

Each page shows a `Page X/Y · N total` caption.

**Pattern expansion for glob searches:**

| Input | Expanded to |
|---|---|
| `tes*` | `{current_namespace}/tes*/*` |
| `tom/tes*` | `tom/tes*/*` |
| `tom/test/exp*` | `tom/test/exp*` (unchanged) |

> Always quote glob patterns to prevent the shell from expanding `*` before Python sees it.

```bash
# List all your projects (paginated)
ml-dash list --api-url http://localhost:3000

# List projects in another namespace
ml-dash list -n alice --api-url http://localhost:3000

# List experiments in a specific project
ml-dash list -p test --api-url http://localhost:3000

# List experiments with namespace/project
ml-dash list -p tom/test --api-url http://localhost:3000

# Wildcard search across projects (quote the pattern!)
ml-dash list -p 'tom/tes*' --api-url http://localhost:3000

# Filter by status
ml-dash list -p test --status RUNNING --api-url http://localhost:3000

# Show detailed view
ml-dash list -p test --detailed --api-url http://localhost:3000

# List tracks in a specific experiment
ml-dash list --tracks -p tom/test/my-experiment --api-url http://localhost:3000

# Filter tracks by topic
ml-dash list --tracks -p tom/test/my-experiment --topic-filter "robot/*" --api-url http://localhost:3000
```

---

## Global Options

All commands support:

| Flag | Description |
|---|---|
| `--dash-url`, `--api-url` | Remote server URL (defaults to `https://api.dash.ml` or stored config) |
| `--help` | Show command-specific help |

---

## Authentication Flow

```bash
# 1. Authenticate once — token stored in system keychain
ml-dash login

# 2. Check who you're logged in as
ml-dash profile

# 3. Use any command — no --api-key needed
ml-dash list

# 4. Logout when done
ml-dash logout
```

For CI/CD or scripting, set the token in `~/.dash/config.json`:

```json
{
  "remote_url": "https://api.dash.ml",
  "api_key": "your-jwt-token"
}
```

---

## Troubleshooting

### Command not found

```bash
python -m ml_dash.cli --help
# or
uv run ml-dash --help
```

### Authentication errors

```bash
# Re-authenticate
ml-dash logout && ml-dash login

# Check current auth state
ml-dash profile
```

### Token expired

```bash
ml-dash profile  # shows "Token expired" if so
ml-dash login    # re-authenticate
```
