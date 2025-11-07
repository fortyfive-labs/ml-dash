"""
Configuration management for ML-Dash.

Handles storing and retrieving authentication tokens and user preferences.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime, timezone


class ConfigManager:
    """
    Manages ML-Dash configuration stored in user's home directory.

    Configuration is stored in ~/.ml-dash/config.json and includes:
    - Authentication token
    - Auth server URL
    - Timestamps

    Thread-safe with atomic file writes.
    """

    DEFAULT_CONFIG_DIR = Path.home() / ".ml-dash"
    DEFAULT_AUTH_SERVER = "https://staging-auth.ml-dash.com"
    CONFIG_FILE = "config.json"

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Custom config directory (defaults to ~/.ml-dash)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILE

    def ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Configuration dict (empty dict if file doesn't exist)
        """
        if not self.config_path.exists():
            return {}

        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Return empty config if file is corrupted
            return {}

    def save_config(self, config: Dict[str, Any]):
        """
        Save configuration to file.

        Args:
            config: Configuration dict to save
        """
        self.ensure_config_dir()

        # Write atomically by writing to temp file then moving
        temp_path = self.config_path.with_suffix('.json.tmp')

        try:
            with open(temp_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Atomic rename
            temp_path.replace(self.config_path)
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    def get_token(self, auth_server: Optional[str] = None) -> Optional[str]:
        """
        Get saved authentication token.

        Args:
            auth_server: Optional auth server URL to match

        Returns:
            Token string if found and valid, None otherwise
        """
        config = self.load_config()

        if "token" not in config:
            return None

        # If auth_server specified, verify it matches
        if auth_server and config.get("auth_server") != auth_server:
            return None

        return config["token"]

    def save_token(self, token: str, auth_server: Optional[str] = None):
        """
        Save authentication token.

        Args:
            token: Token string to save
            auth_server: Optional auth server URL
        """
        config = self.load_config()

        config["token"] = token
        config["auth_server"] = auth_server or self.DEFAULT_AUTH_SERVER
        config["updated_at"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        self.save_config(config)

    def clear_token(self):
        """Clear saved authentication token."""
        config = self.load_config()

        if "token" in config:
            del config["token"]
        if "auth_server" in config:
            del config["auth_server"]

        config["updated_at"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        self.save_config(config)

    def get_auth_server(self) -> Optional[str]:
        """
        Get saved auth server URL.

        Returns:
            Auth server URL if found, None otherwise
        """
        config = self.load_config()
        return config.get("auth_server")

    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated (has saved token).

        Returns:
            True if token exists, False otherwise
        """
        return self.get_token() is not None
