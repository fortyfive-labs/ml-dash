"""JobLogger - Job-based logging wrapper around ML_Logger.

This class provides a simple wrapper around ML_Logger for job-based logging.
"""

from typing import Optional

from .ml_logger import ML_Logger
from .backends.base import StorageBackend


class JobLogger(ML_Logger):
    """Job-based logger wrapper.

    This is a simple wrapper around ML_Logger that can be extended
    with job-specific functionality in the future.

    Args:
        prefix: Directory prefix for logging
        backend: Storage backend (optional)
        job_id: Optional job identifier
    """

    def __init__(
        self,
        prefix: str,
        backend: Optional[StorageBackend] = None,
        job_id: Optional[str] = None,
    ):
        """Initialize JobLogger.

        Args:
            prefix: Directory prefix for logging
            backend: Storage backend (optional)
            job_id: Optional job identifier
        """
        super().__init__(prefix, backend)
        self.job_id = job_id

    def __repr__(self) -> str:
        """String representation."""
        return f"JobLogger(prefix='{self.prefix}', job_id='{self.job_id}', entries={len(self.buffer)})"
