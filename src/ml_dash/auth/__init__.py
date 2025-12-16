"""Authentication module for ml-dash."""

from .constants import VUER_AUTH_URL, CLIENT_ID, DEFAULT_SCOPE
from .device_flow import DeviceFlowClient, DeviceFlowResponse
from .device_secret import (
    generate_device_secret,
    hash_device_secret,
    get_or_create_device_secret,
)
from .token_storage import (
    TokenStorage,
    KeyringStorage,
    EncryptedFileStorage,
    PlaintextFileStorage,
    get_token_storage,
)
from .exceptions import (
    AuthenticationError,
    NotAuthenticatedError,
    DeviceCodeExpiredError,
    AuthorizationDeniedError,
    TokenExchangeError,
    StorageError,
)

__all__ = [
    # Constants
    "VUER_AUTH_URL",
    "CLIENT_ID",
    "DEFAULT_SCOPE",
    # Device Flow
    "DeviceFlowClient",
    "DeviceFlowResponse",
    # Device Secret
    "generate_device_secret",
    "hash_device_secret",
    "get_or_create_device_secret",
    # Token Storage
    "TokenStorage",
    "KeyringStorage",
    "EncryptedFileStorage",
    "PlaintextFileStorage",
    "get_token_storage",
    # Exceptions
    "AuthenticationError",
    "NotAuthenticatedError",
    "DeviceCodeExpiredError",
    "AuthorizationDeniedError",
    "TokenExchangeError",
    "StorageError",
]
