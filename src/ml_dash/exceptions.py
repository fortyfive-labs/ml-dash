"""
Public exception hierarchy for ml-dash.

Callers can catch broad categories::

    except ml_dash.NetworkError:
        retry_later()

or narrow types::

    except ml_dash.AuthenticationError:
        prompt_relogin()
"""

from typing import Optional


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


class MetricRowsError(NetworkError):
    """
    A raw metric rows read failed.

    Attributes:
        status_code: HTTP status of the response.
        code: The server's machine-readable code, e.g. ``"snapshot_changed"``
            (409: restart the read without a cursor), ``"metric_not_found"`` or
            ``"invalid_cursor"``. None when the server sent no code (a server
            without the rows route answers 404 with none), or when the SDK
            rejected a 2xx response as malformed.
    """

    def __init__(self, message: str, status_code: int, code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
