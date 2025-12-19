"""
RUN - Global run configuration object for ML-Dash.

This module provides a global RUN object that serves as the single source
of truth for run/experiment metadata. Uses params-proto for configuration.

Usage:
    from ml_dash import RUN

    # Configure the run
    EXP.name = "my-experiment"
    EXP.project = "my-project"

    # Use in templates (use dot notation for unique paths)
    folder = "/experiments/{EXP.name}.{EXP.id}".format(RUN=RUN)

    # With dxp singleton (RUN is auto-populated)
    from ml_dash import dxp
    with dxp.run:
        # EXP.name, EXP.project, EXP.id, EXP.timestamp are set
        dxp.log().info(f"Running {EXP.name}")
"""

import time
from datetime import datetime, timezone
from params_proto import proto


@proto.prefix
class EXP:
    """
    Global run configuration.

    This class is the single source of truth for run metadata.
    Configure it before starting an experiment, or let dxp auto-configure.

    Template variables available:
        {EXP.name}      - Experiment name
        {EXP.project}   - Project name
        {EXP.id}        - Numeric run ID (milliseconds since epoch)
        {EXP.date}      - Date string (YYYYMMDD)
        {EXP.time}      - Time string (HHMMSS)
        {EXP.datetime}  - DateTime string (YYYYMMDD.HHMMSS)
        {EXP.timestamp} - ISO timestamp

    Example:
        EXP.name = "exp.{EXP.date}"  # -> "exp.20251219"
        dxp.run.folder = "{EXP.project}/{EXP.name}.{EXP.id}"
    """
    # Core identifiers
    name: str = "untitled"  # Run/experiment name (can be a template)
    project: str = "scratch"  # Project name

    # Auto-generated identifiers (populated at run.start())
    id: int = None  # Unique run ID (numeric, milliseconds since epoch)
    timestamp: str = None  # ISO timestamp string

    # Optional configuration
    folder: str = None  # Folder path with optional templates
    description: str = None  # Run description

    # Internal: stores the original name template
    _name_template: str = None

    @classmethod
    def _generate_id(cls) -> int:
        """Generate a unique numeric run ID (milliseconds since epoch)."""
        return int(time.time() * 1000)

    @classmethod
    @property
    def date(cls) -> str:
        """Current date as YYYYMMDD."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    @classmethod
    @property
    def time(cls) -> str:
        """Current time as HHMMSS."""
        return datetime.now(timezone.utc).strftime("%H%M%S")

    @classmethod
    @property
    def datetime(cls) -> str:
        """Current datetime as YYYYMMDD.HHMMSS."""
        return datetime.now(timezone.utc).strftime("%Y%m%d.%H%M%S")

    @classmethod
    def _init_run(cls) -> None:
        """Initialize run ID and timestamp if not already set."""
        if cls.id is None:
            cls.id = cls._generate_id()
            cls.timestamp = datetime.now(timezone.utc).isoformat()

        # Expand name template if set
        if cls._name_template:
            cls.name = cls._name_template.format(RUN=cls)

    @classmethod
    def _format(cls, template: str) -> str:
        """
        Format a template string with RUN values.

        Args:
            template: String with {EXP.attr} placeholders

        Returns:
            Formatted string with placeholders replaced

        Example:
            EXP._format("/experiments/{EXP.name}.{EXP.id}")
            # -> "/experiments/my-exp.1734567890123"
        """
        return template.format(RUN=cls)

    @classmethod
    def _reset(cls) -> None:
        """Reset RUN to defaults (for testing or new runs)."""
        cls.name = "untitled"
        cls._name_template = None
        cls.project = "scratch"
        cls.id = None
        cls.timestamp = None
        cls.folder = None
        cls.description = None
