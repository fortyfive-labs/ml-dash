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
        {EXP.owner}     - Owner/user
        {EXP.id}        - Numeric experiment ID (milliseconds since epoch)
        {EXP.date}      - Date string (YYYYMMDD)
        {EXP.time}      - Time string (HHMMSS)
        {EXP.datetime}  - DateTime string (YYYYMMDD.HHMMSS)
        {EXP.timestamp} - ISO timestamp

    Example:
        EXP.name = "exp.{EXP.date}"  # -> "exp.20251219"
        dxp.run.prefix = "{EXP.project}/{EXP.name}.{EXP.id}"

    Auto-detection from file path:
        # In experiments/__init__.py
        from ml_dash import EXP
        EXP.project_root = "/path/to/my-project/experiments"

        # In experiments/vision/resnet/train.py
        from ml_dash import EXP
        EXP.__post_init__(entry=__file__)
        # Result: EXP.prefix = "vision/resnet", EXP.name = "resnet"
    """
    #
    PREFIX: str = ""  # Prefix path with default templates

    # Core identifiers
    owner: str = EnvVar @ "DASH_USER" | "scratch"  # Owner/user
    name: str = "scratch"  # Experiment name (can be a template)
    project: str = EnvVar @ "DASH_PROJECT" | "scratch"  # Project name

    # Auto-generated identifiers
    id: int = None  # Unique experiment ID (numeric, milliseconds since epoch)
    timestamp: str = None  # ISO timestamp string

    prefix: str = None  # Prefix path with optional templates
    description: str = None  # Experiment description

    # Project root for auto-detection
    project_root: str = None  # Root directory for experiment hierarchy
    entry: str = None  # Entry point file/directory path

    # Internal: stores the original name template
    _name_template: str = None

    def __post_init__(self, entry: str = None):
        """
        Initialize EXP with auto-detected prefix from entry path.

        Args:
            entry: Path to entry file/directory (e.g., __file__ or directory
                   containing sweep.jsonl). If not provided, uses caller's
                   __file__ automatically.

        Computes prefix as relative path from project_root to entry's directory.

        Example:
            # experiments/__init__.py
            from ml_dash import EXP
            EXP.project_root = "/path/to/my-project/experiments"

            # experiments/vision/resnet/train.py
            from ml_dash import EXP
            EXP.__post_init__(entry=__file__)
            # Result: EXP.prefix = "vision/resnet", EXP.name = "resnet"
        """
        from pathlib import Path

        # Use provided entry or try to auto-detect from caller
        if entry is None:
            import inspect
            frame = inspect.currentframe().f_back
            entry = frame.f_globals.get('__file__')

        if entry and self.project_root:
            entry_path = Path(entry).resolve()
            entry_dir = entry_path.parent if entry_path.is_file() else entry_path
            root = Path(self.project_root).resolve()

            try:
                relative = entry_dir.relative_to(root)
                self.prefix = str(relative)
                self.name = entry_dir.name
                self.entry = str(entry_path)
            except ValueError:
                # entry is not under project_root, keep current values
                pass

    @classmethod
    def _init_run(cls):
        """
        Initialize run-specific values (id, timestamp, date, time).
        Called when an experiment starts.
        """
        now = datetime.now(timezone.utc)
        cls.id = int(time.time() * 1000)  # milliseconds since epoch
        cls.timestamp = now.isoformat()

    @classmethod
    def _reset(cls):
        """
        Reset run-specific values after an experiment closes.
        Preserves project_root and other configuration.
        """
        cls.id = None
        cls.timestamp = None
        cls.entry = None

    @property
    def date(cls) -> str:
        """Date string in YYYYMMDD format."""
        now = datetime.now(timezone.utc)
        return now.strftime('%Y%m%d')

    @property
    def time(cls) -> str:
        """Time string in HHMMSS format."""
        now = datetime.now(timezone.utc)
        return now.strftime('%H%M%S')

    @property
    def datetime(cls) -> str:
        """DateTime string in YYYYMMDD.HHMMSS format."""
        now = datetime.now(timezone.utc)
        return now.strftime('%Y%m%d.%H%M%S')

