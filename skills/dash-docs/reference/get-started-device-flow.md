# ml-dash Login Flow (Device Authorization Grant)

The flow follows **OAuth 2.0 Device Authorization Grant (RFC 8628)** — designed for CLI apps that can't open a browser directly.

## Actors

- **CLI** — `ml-dash login` (Python)
- **vuer-auth** — `auth.vuer.ai` (our server)
- **Browser** — user opens manually or auto-launched

---

## Flow Diagram

```
CLI (ml-dash)                vuer-auth                    Browser (User)
     |                           |                               |
     |-- POST /api/device/start ->|                               |
     |   { device_secret_hash }  |                               |
     |<-- { user_code: "AB12CD34" |                               |
     |     verification_uri,     |                               |
     |     expires_in: 600 } ----|                               |
     |                           |                               |
     | [prints user_code + URL]  |                               |
     | [opens browser]           |                               |
     |                           |<-- GET /device/verification?code=AB12CD34
     |                           |                               |
     |                           |   [user enters code, sees     |
     |                           |    client + scope info]       |
     |                           |<-- POST /api/device/verify-code
     |                           |    { user_code }              |
     |                           |--{ clientId, scope } -------->|
     |                           |                               |
     |                           |   [user clicks "Authorize"]   |
     |                           |<-- POST /api/device/complete  |
     |                           |    { user_code } + session    |
     |                           |   [generates JWT, stores it]  |
     |                           |--{ success: true } ---------->|
     |                           |                               |
     |-- POST /api/device/poll -->|                    [done ✓]  |
     |   { device_secret_hash }  |                               |
     |<-- 202 authorization_pending (repeats every 5s)           |
     |                           |                               |
     |-- POST /api/device/poll -->|                               |
     |<-- 200 { access_token } --|                               |
     |                           |                               |
     |-- POST /api/auth/exchange (ml-dash server)                |
     |<-- { ml_dash_token }                                      |
     |                           |                               |
  [stores token locally]         |                               |
```

---

## API Endpoints

| Endpoint | Caller | Description |
|---|---|---|
| `POST /api/device/start` | CLI | Start device flow, returns `user_code` and `verification_uri` |
| `POST /api/device/verify-code` | Web UI | Check if user code is valid, returns `clientId` and `scope` |
| `POST /api/device/complete` | Web UI | Complete authorization using session cookie, generates JWT |
| `POST /api/device/poll` | CLI | Poll for authorization status, returns `access_token` when done |

### Poll Response States

| HTTP Status | Body | Meaning |
|---|---|---|
| `200` | `{ access_token, token_type }` | Authorization complete |
| `202` | `{ error: "authorization_pending" }` | Still waiting for user |
| `400` | `{ error: "expired_token" }` | Code expired (10 min limit) |
| `404` | `{ error: "access_denied" }` | Session not found |

---

## Key Points

1. **No password in CLI** — credentials never touch the terminal
2. **`device_secret_hash`** — a stable per-machine secret used to link the poll to the correct session (instead of a `device_code` that could be intercepted)
3. **Two-step web UI** — `verify-code` checks the code first (shows what's being authorized), then `complete` does the actual auth with the user's session cookie
4. **JWT from vuer-auth → exchanged for ml-dash token** — vuer-auth issues a short-lived JWT; the ml-dash server then issues its own permanent token via `/api/auth/exchange`
