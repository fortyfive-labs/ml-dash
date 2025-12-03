"""Configuration file management for ML-Dash CLI."""

from pathlib import Path
import json
from typing import Optional, Dict, Any


class Config:
    """
    Manages ML-Dash CLI configuration file.

    Configuration is stored in ~/.ml-dash/config.json with structure:
    {
        "remote_url": "https://api.dash.ml",
        "api_key": "token",
        "default_batch_size": 100
    }
    """

    DEFAULT_CONFIG_DIR = Path.home() / ".ml-dash"
    CONFIG_FILE = "config.json"

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.

        Args:
            config_dir: Config directory path (defaults to ~/.ml-dash)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILE
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load config from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If config is corrupted, return empty dict
                return {}
        return {}

    def save(self):
        """Save config to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value.

        Args:
            key: Config key
            default: Default value if key not found

        Returns:
            Config value or default
        """
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set config value and save.

        Args:
            key: Config key
            value: Config value
        """
        self._data[key] = value
        self.save()

    def delete(self, key: str):
        """
        Delete config key and save.

        Args:
            key: Config key to delete
        """
        if key in self._data:
            del self._data[key]
            self.save()

    def clear(self):
        """Clear all config and save."""
        self._data = {}
        self.save()

    @property
    def remote_url(self) -> Optional[str]:
        """Get default remote URL."""
        return self.get("remote_url")

    @remote_url.setter
    def remote_url(self, url: str):
        """Set default remote URL."""
        self.set("remote_url", url)

    @property
    def api_key(self) -> Optional[str]:
        """Get default API key."""
        return self.get("api_key")

    @api_key.setter
    def api_key(self, key: str):
        """Set default API key."""
        self.set("api_key", key)

    @property
    def batch_size(self) -> int:
        """Get default batch size for uploads."""
        return self.get("default_batch_size", 100)

    @batch_size.setter
    def batch_size(self, size: int):
        """Set default batch size."""
        self.set("default_batch_size", size)
