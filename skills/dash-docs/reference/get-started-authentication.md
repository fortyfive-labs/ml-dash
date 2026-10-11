# Authentication

You log in once with the `ml-dash` CLI. The CLI saves a token on your
machine, and both the CLI and the Python SDK read it from there. Local mode
needs no login at all.

## Log in

Install the CLI first (see [Install](reference/overview.md#cli)), then:

```bash
ml-dash login
```

The CLI prints a short code and a QR code, and opens your browser at the
approval page. Approve the request there. You never type a password into the
terminal. Once you approve, the CLI saves the token and tells you where it
put it, for example `Your authentication token has been stored (keychain).`

| Flag | Use |
|---|---|
| `--no-browser` | Print the code and URL, but don't open a browser. Use this on a remote machine and approve from any other device. |
| `--dash-url URL` | Log in to a server other than `https://api.dash.ml` |
| `--auth-url URL` | Use a different authorization server (default `https://auth.vuer.ai`) |

The code expires after 10 minutes. Check who you are logged in as, or log out:

```bash
ml-dash profile          # fetches your profile from the server
ml-dash profile --cached # reads the stored token only, no network
ml-dash logout           # clears the token from every place it could be stored
```

## Where the token lives

The CLI stores the token in the most secure place the machine offers:

| Machine | Where `ml-dash login` stores the token |
|---|---|
| macOS | The login keychain (service `ml-dash`) |
| Linux with `secret-tool` (a desktop session) | The Secret Service keyring (service `ml-dash`) |
| Windows, or Linux without `secret-tool` (servers, clusters, containers) | `~/.dash/tokens.encrypted`, with its key in `~/.dash/encryption.key` |

Set `ML_DASH_NO_KEYCHAIN=1` before `ml-dash login` to skip the keychain and
always use the encrypted file.

## Using the token from Python

The SDK loads the stored token when an experiment has a `dash_url`. You don't
pass any credentials in code:

```python
from ml_dash import Experiment

with Experiment(prefix="alice/project/run-1", dash_url="https://api.dash.ml").run as exp:
    exp.log("Authenticated automatically")
```

The SDK reads the same places the CLI writes. If `keyring` is installed, it
uses the OS keychain. Otherwise it uses `~/.dash/tokens.encrypted`. So match
the SDK install to where your login went:

- **Token in the keychain** (macOS, desktop Linux): install `pip install
  "ml-dash[auth]"`, which adds `keyring`. Without it, the SDK can't see the
  token.
- **Token in `~/.dash/tokens.encrypted`**: a plain `pip install ml-dash`
  reads it.

> **Getting an AuthenticationError?** Run `ml-dash profile` to confirm the CLI is logged in, then check that the SDK
> install matches the list above. A common mismatch: the token is in
> `~/.dash/tokens.encrypted`, but another package pulled `keyring` into the
> Python environment, so the SDK looks only in the keychain. Uninstall
> `keyring` from that environment, or log in again on a machine whose keychain
> the SDK can read.

## Servers and configuration

Both the CLI and the SDK default to `https://api.dash.ml`.

- **CLI:** `--dash-url` (alias `--api-url`) on any command, or `remote_url` in
  `~/.dash/config.json`.
- **SDK:** the `dash_url` argument. `dash_url=True` uses the `ML_DASH_API_URL`
  environment variable, or `https://api.dash.ml` if it isn't set.

For scripts and CI, the CLI also accepts a token written directly into
`~/.dash/config.json`. It takes precedence over the stored login:

```json
{
  "remote_url": "https://api.dash.ml",
  "api_key": "<token>"
}
```

The SDK does not read `api_key` from this file. It reads only the stored login
described above.

## How login works

`ml-dash login` uses the OAuth 2.0 Device Authorization Grant
([RFC 8628](https://www.rfc-editor.org/rfc/rfc8628)) against
`auth.vuer.ai`, then exchanges the result for an ML-Dash token:

```
ml-dash CLI                  auth.vuer.ai                   Browser
     |-- POST /api/device/start -->|                               |
     |   { device_secret_hash }    |                               |
     |<-- { user_code, verification_uri, expires_in: 600 }         |
     |  [prints code + QR, opens browser]                          |
     |                             |<-- user enters code, approves |
     |-- POST /api/device/poll --->|                               |
     |<-- 202 authorization_pending   (every 5 s)                  |
     |<-- 200 { access_token }     |                               |
     |                                                             |
     |-- POST /api/auth/exchange ──→ ML-Dash server                |
     |<-- { ML-Dash token }                                        |
  [stores the token]
```

- The poll is tied to a random per-machine secret, stored in
  `~/.dash/config.json`. Only its SHA-256 hash is sent to the server, so there
  is no `device_code` that could be intercepted.
- The token from `auth.vuer.ai` is exchanged at the ML-Dash server's
  `/api/auth/exchange` for the token the CLI stores and the SDK sends.
