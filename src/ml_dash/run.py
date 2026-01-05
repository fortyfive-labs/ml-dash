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
    prefix = "/experiments/{EXP.name}.{EXP.id}".format(EXP=EXP)

    # With dxp singleton (EXP is auto-populated)
    from ml_dash import dxp
    with dxp.run:
        # EXP.name, EXP.project, EXP.id, EXP.timestamp are set
        dxp.log().info(f"Running {EXP.name}")
"""

import time
from datetime import datetime, timezone
from params_proto import proto, EnvVar


@proto.prefix
class EXP:
    """
    Global experiment configuration.

    This class is the single source of truth for experiment metadata.
    Configure it before starting an experiment, or let dxp auto-configure.

    Template variables available:
        {EXP.prefix}    - Experiment prefix = <project>/<folders...>/<exp_name>
        {EXP.id}        - Numeric experiment ID (milliseconds since epoch)
        {EXP.date}      - Date string (YYYYMMDD)
        {EXP.time}      - Time string (HHMMSS)
        {EXP.datetime}  - DateTime string (YYYYMMDD.HHMMSS)
        {EXP.timestamp} - ISO timestamp

    Example:
        EXP.name = "exp.{EXP.date}"  # -> "exp.20251219"
        dxp.run.prefix = "{EXP.project}/{EXP.name}.{EXP.id}"
    """
    #
    PREFIX: str = ""  # Prefix path with default templates

    # Core identifiers
    name: str = "scratch"  # Experiment name (can be a template)
    project: str = EnvVar @ "DASH_PROJECT" | "scratch"  # Project name

    # Auto-generated identifiers
    id: int = None  # Unique experiment ID (numeric, milliseconds since epoch)
    timestamp: str = None  # ISO timestamp string

    prefix: str = None  # Prefix path with optional templates
    description: str = None  # Experiment description

    # Internal: stores the original name template
    _name_template: str = None

    def __post_init__(self):
        pass

    @classmethod
    def _init_run(cls):
        """Initialize run with unique ID and timestamp."""
        if cls.id is None:
            # Generate unique Snowflake ID
            from .snowflake import generate_id
            cls.id = generate_id()

        if cls.timestamp is None:
            # Generate ISO timestamp
            from datetime import datetime, timezone
            cls.timestamp = datetime.now(timezone.utc).isoformat()

    @classmethod
    def _reset(cls):
        """Reset run state for next experiment."""
        cls.id = None
        cls.timestamp = None
        cls.prefix = None
        cls.description = None
        cls._name_template = None

