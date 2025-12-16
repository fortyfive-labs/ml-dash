"""Custom exceptions for ml-dash authentication."""


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class NotAuthenticatedError(AuthenticationError):
    """Raised when user is not authenticated."""
    pass


class DeviceCodeExpiredError(AuthenticationError):
    """Raised when device code expires before authorization."""
    pass


class AuthorizationDeniedError(AuthenticationError):
    """Raised when user denies authorization request."""
    pass


class TokenExchangeError(AuthenticationError):
    """Raised when token exchange with ml-dash server fails."""
    pass


class StorageError(Exception):
    """Base exception for token storage errors."""
    pass
