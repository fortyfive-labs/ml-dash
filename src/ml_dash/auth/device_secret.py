"""Device secret generation and management for ml-dash."""

import hashlib
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_dash.config import Config


def generate_device_secret() -> str:
    """Generate a cryptographically secure device secret.

    Returns:
        A 64-character hexadecimal string (256 bits of entropy)
    """
    return secrets.token_hex(32)  # 32 bytes = 256 bits


def hash_device_secret(secret: str) -> str:
    """Hash device secret using SHA256.

    Args:
        secret: The device secret to hash

    Returns:
        SHA256 hash as hexadecimal string
    """
    return hashlib.sha256(secret.encode()).hexdigest()


def get_or_create_device_secret(config: "Config") -> str:
    """Load device secret from config or generate a new one.

    Args:
        config: ML-Dash configuration object

    Returns:
        Device secret string
    """
    device_secret = config.device_secret

    if not device_secret:
        # Generate new device secret
        device_secret = generate_device_secret()
        config.set("device_secret", device_secret)
        config.save()

    return device_secret
