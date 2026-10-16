# CLI Reference

The `ml-dash` command logs you in, inspects projects and experiments, and
moves experiment data to and from an ML-Dash server. This page covers CLI
0.1.1.

The CLI is a separate program from the Python SDK. Install it with the
standalone installer or `npm install -g @dreamlake/ml-dash` (see
[Install](reference/overview.md#cli)). `pip install ml-dash` no longer installs it (since SDK
0.7.0).

| Command | Description |
|---|---|
| [`login`](#ml-dash-login) | Authenticate with the OAuth device flow |
| [`logout`](#ml-dash-logout) | Clear the stored token |
| [`profile`](#ml-dash-profile) | Show the current user and configuration |
| [`list`](#ml-dash-list) | List projects, experiments, and tracks on the server |
| [`create`](#ml-dash-create) | Create a project |
| [`remove`](#ml-dash-remove) | Delete a project and everything in it |
| [`upload`](#ml-dash-upload) | Upload locally stored experiments to the server |
| [`download`](#ml-dash-download) | Download experiments from the server to local storage |
| [`api`](#ml-dash-api) | Send a raw GraphQL query or mutation |
| [`update`](#ml-dash-update) | Update the CLI itself |
| [`version`](#ml-dash-version) | Print the CLI version |

Run `ml-dash <command> --help` for the exact flags of your installed version.

## Common options

| Flag | Description |
|---|---|
| `--dash-url`, `--api-url` URL | Server to talk to. Defaults to `remote_url` in `~/.dash/config.json`, then `https://api.dash.ml`. Accepted by every command that contacts the server. |
| `-h`, `--help` | Show help for the command |

---

## `ml-dash login`

```bash
ml-dash login [--dash-url URL] [--auth-url URL] [--no-browser]
```

| Flag | Description |
|---|---|
| `--dash-url`, `--api-url` | ML-Dash server to log in to |
| `--auth-url` | OAuth authorization server (default `https://auth.vuer.ai`) |
| `--no-browser` | Print the code and URL, but don't open a browser |

Starts the device authorization flow. It prints a code, a URL, and a QR code,
waits up to 10 minutes for you to approve in a browser, exchanges the result
for an ML-Dash token, and stores it. See [Authentication](reference/get-started-authentication.md)
for where the token goes and how the SDK reads it.

## `ml-dash logout`

```bash
ml-dash logout
```

Clears the stored token from the keychain and from the `~/.dash/` token files.

## `ml-dash profile`

```bash
ml-dash profile [--dash-url URL] [--json] [--cached]
```

| Flag | Description |
|---|---|
| `--json` | Output as JSON |
| `--cached` | Read the stored token only, without fetching from the server |

Shows your username, name, email, server URL, token expiry, and whether the
data came from the server or the cached token.

---

## `ml-dash list`

```bash
ml-dash list [-p PROJECT] [-n NAMESPACE] [--status STATUS] [--tags TAGS]
             [--detailed] [--tracks] [--topic-filter TOPIC] [--dash-url URL] [-v]
```

| Flag | Description |
|---|---|
| `-p`, `--project` (aliases `--prefix`, `--pref`, `--proj`) | List experiments in this project. Without it, lists projects. Accepts glob patterns: always quote them, e.g. `-p 'tom/tut*'` |
| `-n`, `--namespace` | Namespace for all queries. Defaults to your own |
| `--status` | Filter experiments: `COMPLETED`, `RUNNING`, `FAILED`, or `ARCHIVED` |
| `--tags` | Filter experiments by tags (comma-separated) |
| `--detailed` | Show more columns |
| `--tracks` | List the tracks in one experiment. Needs `-p namespace/project/experiment` |
| `--topic-filter` | Filter tracks by topic, e.g. `'robot/*'` |
| `-v`, `--verbose` | Verbose output |

Results come 50 per page. In a terminal, move between pages with `n` / `→` /
`Space` / `Enter` (next) and `p` / `b` / `←` (previous). Any other key quits.
When output is piped, `list` prints the first page and notes that more results
are available.

A glob without a namespace is expanded against your own namespace:

| Input | Searches |
|---|---|
| `tes*` | `<your-namespace>/tes*/*` |
| `tom/tes*` | `tom/tes*/*` |
| `tom/test/exp*` | `tom/test/exp*` |

```bash
ml-dash list                                   # your projects
ml-dash list -n alice                          # alice's projects
ml-dash list -p my-project                     # experiments in a project
ml-dash list -p 'tom/tes*' --status RUNNING    # glob across projects
ml-dash list --tracks -p tom/test/run-1 --topic-filter 'robot/*'
```

## `ml-dash create`

```bash
ml-dash create -p PROJECT [-d DESCRIPTION] [--dash-url URL]
```

| Flag | Description |
|---|---|
| `-p`, `--project` | `project` or `namespace/project`. Without a namespace, your own is used |
| `-d`, `--description` | Optional description |

If the project already exists, `create` prints a warning and exits
successfully.

## `ml-dash remove`

```bash
ml-dash remove -p PROJECT [-y] [--dash-url URL]
```

| Flag | Description |
|---|---|
| `-p`, `--project` | `project` or `namespace/project` |
| `-y`, `--yes` | Skip the confirmation prompt |

> **Warning:** `remove` deletes the project and all of its experiments, metrics, files, and
> logs. It cannot be undone.

---

## `ml-dash upload`

```bash
ml-dash upload [PATH] [options]
```

Uploads experiments that the SDK wrote in local mode. `PATH` is the local
storage directory and defaults to `./.dash`.

| Flag | Description |
|---|---|
| `-p`, `--project` (aliases `--prefix`, `--pref`, `--proj`) | Only upload experiments matching this prefix or glob, e.g. `'tom/*/exp*'` |
| `-t`, `--target` | Upload under this prefix on the server instead, e.g. `alice/shared-project` |
| `--skip-logs`, `--skip-metrics`, `--skip-files`, `--skip-params` | Leave that kind of data out |
| `--dry-run` | Show what would be uploaded without uploading |
| `--strict` | Fail on any validation error. By default, invalid data is skipped |
| `--batch-size N` | Batch size for logs and metrics (default 100) |
| `--resume` | Resume an interrupted upload |
| `--state-file FILE` | State file for `--resume` (default `.dash-upload-state.json`) |
| `--tracks FILE` | Upload a single track data file (e.g. `robot_position.jsonl`). Needs `--remote-path` |
| `--remote-path PATH` | Where the track goes, e.g. `namespace/project/exp/robot/position` |
| `-v`, `--verbose` | Show detailed progress |

```bash
ml-dash upload                                  # everything in ./.dash
ml-dash upload ./.dash -p 'tom/*/exp*'          # a subset
ml-dash upload --dry-run -v                     # preview
ml-dash upload -t alice/shared-project          # into another project
ml-dash upload --tracks robot_position.jsonl --remote-path tom/proj/exp/robot/position
```

## `ml-dash download`

```bash
ml-dash download [PATH] [options]
```

Downloads experiments into a local storage directory (`PATH`, default
`./.dash`) in the same layout local mode writes.

| Flag | Description |
|---|---|
| `-p`, `--project` (aliases `--prefix`, `--pref`, `--proj`) | Project or glob to download, e.g. `alice/my-project`, `'tut*'` |
| `--experiment NAME` | Only this experiment. Needs `--project` |
| `--skip-logs`, `--skip-metrics`, `--skip-files`, `--skip-params` | Leave that kind of data out |
| `--dry-run` | Preview without downloading |
| `--overwrite` | Overwrite experiments that already exist locally |
| `--resume` | Resume an interrupted download |
| `--state-file FILE` | State file for `--resume` (default `.dash-download-state.json`) |
| `--batch-size N` | Batch size for logs and metrics (default 1000, max 10000) |
| `--max-concurrent-metrics N` | Parallel metric downloads (default 5) |
| `--max-concurrent-files N` | Parallel file downloads (default 3) |
| `--tracks PATH` | Download one track instead, e.g. `namespace/project/exp/robot/position` |
| `-f`, `--format` | Track export format (default `jsonl`) |
| `-o`, `--output FILE` | Track output file (default: named after the topic) |
| `-v`, `--verbose` | Detailed progress |

```bash
ml-dash download -p alice/my-project
ml-dash download ./backup -p alice/my-project --experiment exp-one
ml-dash download -p 'alice/tut*' --dry-run
ml-dash download --tracks alice/proj/exp/robot/position -f jsonl -o joints.jsonl
```

---

## `ml-dash api`

```bash
ml-dash api (--query QUERY | --mutation MUTATION) [--jq PATH] [--dash-url URL]
```

| Flag | Description |
|---|---|
| `-q`, `--query` | GraphQL query |
| `-m`, `--mutation` | GraphQL mutation |
| `--jq PATH` | Pull out one value with a dot path, e.g. `.me.username` |

- A bare body is wrapped for you: `me { username }` is sent as `{ me { username } }`.
- Single quotes are converted to double quotes, so you can write
  `user(title: 'hello')` inside a double-quoted shell string.
- `--jq` paths start from the response data. There is no top-level `data` key,
  so write `.me.username`, not `.data.me.username`.

```bash
ml-dash api --query "me { username name email }"
ml-dash api --query "me { username }" --jq ".me.username"
```

## `ml-dash update`

```bash
ml-dash update [--check] [--version VERSION] [--json]
```

| Flag | Description |
|---|---|
| `--check` | Report whether an update exists. Change nothing |
| `--version` | Install this exact release instead of the latest |
| `--json` | Output as JSON |

Updates through the channel you installed from. An npm install is updated with
`npm install -g @dreamlake/ml-dash@<version>`. A standalone binary downloads the
new build, checks its sha256 and size against the release manifest, runs it
once, and only then replaces itself. `update` refuses to downgrade. It also
refuses to touch a copy it doesn't own (Homebrew, Nix, pipx, a `node_modules`
tree) and tells you which tool to use instead.

## `ml-dash version`

```bash
ml-dash version        # also: ml-dash --version, ml-dash -V
```

---

## Files and environment variables

| | Used for |
|---|---|
| `~/.dash/config.json` | `remote_url`, an optional `api_key` (used instead of the stored login), `auth_url`, and the device-flow secret |
| `~/.dash/tokens.encrypted`, `~/.dash/encryption.key` | The token, when no keychain is used |
| `ML_DASH_CONFIG_DIR` | Use a different directory than `~/.dash` for the CLI. The Python SDK always reads `~/.dash` |
| `ML_DASH_NO_KEYCHAIN=1` | Never use the OS keychain. Store and read the token in the encrypted file |

## Troubleshooting

**`ml-dash: command not found` after upgrading the SDK.** Since SDK 0.7.0, pip
doesn't install the CLI. Install it as shown in [Install](reference/overview.md#cli).
`python -m ml_dash.cli` no longer exists.

**Authentication errors.** Run `ml-dash profile`. It reports an expired token.
Then run `ml-dash logout && ml-dash login`.

**The SDK can't find your login but the CLI can.** The SDK and the CLI must
agree on where the token lives. See
[Using the token from Python](reference/get-started-authentication.md#using-the-token-from-python).
