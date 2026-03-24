"""
Public exception hierarchy for ml-dash.

Callers can catch broad categories::

    except ml_dash.NetworkError:
        retry_later()

or narrow types::

    except ml_dash.AuthenticationError:
        prompt_relogin()
"""


class MlDashError(Exception):
    """Base class for all ml-dash errors."""


class ConfigurationError(MlDashError):
    """Invalid arguments, missing settings, or unsupported options."""


class AuthenticationError(MlDashError):
    """Token missing, expired, or rejected by the server."""


class StorageError(MlDashError):
    """Disk I/O failure, metadata corruption, or checksum mismatch."""


class ExperimentError(MlDashError):
    """Experiment lifecycle violation (e.g. not started, write-protected)."""


class NetworkError(MlDashError):
    """HTTP or GraphQL failure when communicating with the remote server."""
