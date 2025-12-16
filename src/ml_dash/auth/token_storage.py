"""Token storage backends for ml-dash authentication."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .exceptions import StorageError


class TokenStorage(ABC):
    """Abstract base class for token storage backends."""

    @abstractmethod
    def store(self, key: str, value: str) -> None:
        """Store a token.

        Args:
            key: Storage key
            value: Token string to store
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[str]:
        """Load a token.

        Args:
            key: Storage key

        Returns:
            Token string or None if not found
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a token.

        Args:
            key: Storage key
        """
        pass


class KeyringStorage(TokenStorage):
    """OS keyring storage backend (macOS Keychain, Windows Credential Manager, Linux Secret Service)."""

    SERVICE_NAME = "ml-dash"

    def __init__(self):
        """Initialize keyring storage."""
        try:
            import keyring
            self.keyring = keyring
        except ImportError:
            raise StorageError(
                "keyring library not installed. "
                "Install with: pip install keyring"
            )

    def store(self, key: str, value: str) -> None:
        """Store token in OS keyring."""
        try:
            self.keyring.set_password(self.SERVICE_NAME, key, value)
        except Exception as e:
            raise StorageError(f"Failed to store token in keyring: {e}")

    def load(self, key: str) -> Optional[str]:
        """Load token from OS keyring."""
        try:
            return self.keyring.get_password(self.SERVICE_NAME, key)
        except Exception as e:
            raise StorageError(f"Failed to load token from keyring: {e}")

    def delete(self, key: str) -> None:
        """Delete token from OS keyring."""
        try:
            self.keyring.delete_password(self.SERVICE_NAME, key)
        except Exception:
            # Silently ignore if key doesn't exist
            pass


class EncryptedFileStorage(TokenStorage):
    """Encrypted file storage backend using Fernet symmetric encryption."""

    def __init__(self, config_dir: Path):
        """Initialize encrypted file storage.

        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = Path(config_dir)
        self.tokens_file = self.config_dir / "tokens.encrypted"
        self.key_file = self.config_dir / "encryption.key"

        try:
            from cryptography.fernet import Fernet
            self.Fernet = Fernet
        except ImportError:
            raise StorageError(
                "cryptography library not installed. "
                "Install with: pip install cryptography"
            )

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Generate or load encryption key
        if not self.key_file.exists():
            key = self.Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # User read/write only
        else:
            key = self.key_file.read_bytes()

        self.cipher = self.Fernet(key)

    def _load_all(self) -> dict:
        """Load all tokens from encrypted file."""
        if not self.tokens_file.exists():
            return {}

        try:
            encrypted = self.tokens_file.read_bytes()
            decrypted = self.cipher.decrypt(encrypted)
            return json.loads(decrypted)
        except Exception as e:
            raise StorageError(f"Failed to decrypt tokens file: {e}")

    def _save_all(self, data: dict) -> None:
        """Save all tokens to encrypted file."""
        try:
            plaintext = json.dumps(data).encode()
            encrypted = self.cipher.encrypt(plaintext)
            self.tokens_file.write_bytes(encrypted)
            self.tokens_file.chmod(0o600)  # User read/write only
        except Exception as e:
            raise StorageError(f"Failed to encrypt tokens file: {e}")

    def store(self, key: str, value: str) -> None:
        """Store token in encrypted file."""
        all_tokens = self._load_all()
        all_tokens[key] = value
        self._save_all(all_tokens)

    def load(self, key: str) -> Optional[str]:
        """Load token from encrypted file."""
        all_tokens = self._load_all()
        return all_tokens.get(key)

    def delete(self, key: str) -> None:
        """Delete token from encrypted file."""
        all_tokens = self._load_all()
        if key in all_tokens:
            del all_tokens[key]
            self._save_all(all_tokens)


class PlaintextFileStorage(TokenStorage):
    """Plaintext file storage backend (INSECURE - only for testing/fallback)."""

    _warning_shown = False

    def __init__(self, config_dir: Path):
        """Initialize plaintext file storage.

        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = Path(config_dir)
        self.tokens_file = self.config_dir / "tokens.json"

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Show security warning on first use
        if not PlaintextFileStorage._warning_shown:
            try:
                from rich.console import Console
                console = Console()
                console.print(
                    "\n[bold red]WARNING: Storing tokens in plaintext![/bold red]\n"
                    "[yellow]Your authentication tokens are being stored unencrypted.[/yellow]\n"
                    "[yellow]This is insecure and only recommended for testing.[/yellow]\n\n"
                    "To use secure storage:\n"
                    "  • Install keyring: pip install keyring\n"
                    "  • Or encrypted storage will be used automatically\n"
                )
            except ImportError:
                print("WARNING: Storing tokens in plaintext! This is insecure.")

            PlaintextFileStorage._warning_shown = True

    def _load_all(self) -> dict:
        """Load all tokens from file."""
        if not self.tokens_file.exists():
            return {}

        try:
            with open(self.tokens_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_all(self, data: dict) -> None:
        """Save all tokens to file."""
        with open(self.tokens_file, "w") as f:
            json.dump(data, f, indent=2)
        self.tokens_file.chmod(0o600)  # User read/write only

    def store(self, key: str, value: str) -> None:
        """Store token in plaintext file."""
        all_tokens = self._load_all()
        all_tokens[key] = value
        self._save_all(all_tokens)

    def load(self, key: str) -> Optional[str]:
        """Load token from plaintext file."""
        all_tokens = self._load_all()
        return all_tokens.get(key)

    def delete(self, key: str) -> None:
        """Delete token from plaintext file."""
        all_tokens = self._load_all()
        if key in all_tokens:
            del all_tokens[key]
            self._save_all(all_tokens)


def get_token_storage(config_dir: Optional[Path] = None) -> TokenStorage:
    """Auto-detect and return appropriate storage backend.

    Tries backends in order of security:
    1. KeyringStorage (OS keyring)
    2. EncryptedFileStorage (encrypted file)
    3. PlaintextFileStorage (plaintext file with warning)

    Args:
        config_dir: Configuration directory (defaults to ~/.ml-dash)

    Returns:
        TokenStorage instance
    """
    if config_dir is None:
        config_dir = Path.home() / ".ml-dash"

    # Try keyring first
    try:
        return KeyringStorage()
    except (ImportError, StorageError):
        pass

    # Try encrypted file storage
    try:
        return EncryptedFileStorage(config_dir)
    except (ImportError, StorageError):
        pass

    # Fallback to plaintext (with warning)
    return PlaintextFileStorage(config_dir)
