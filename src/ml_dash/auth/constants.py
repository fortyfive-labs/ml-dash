"""Authentication constants for ml-dash."""

# Vuer-auth server URL
VUER_AUTH_URL = "https://auth.vuer.ai"

# OAuth client ID for ml-dash
CLIENT_ID = "ml-dash-client"

# Default OAuth scopes (no offline_access since we get permanent tokens)
DEFAULT_SCOPE = "openid profile email"
