"""
EXP - Global experiment configuration object for ML-Dash.

This module provides a global EXP object that serves as the single source
of truth for experiment metadata. Uses params-proto for configuration.

Usage:
    from ml_dash import EXP

    # Configure via environment variable
    # export ML_DASH_PREFIX="ge/myproject/experiments/exp1"

    # Or set directly
    EXP.PREFIX = "ge/myproject/experiments/exp1"

    # Use in templates
    prefix = "{EXP.PREFIX}/{EXP.name}.{EXP.id}".format(EXP=EXP)

    # With Experiment (EXP is auto-populated)
    from ml_dash import Experiment
    with Experiment(prefix=EXP.PREFIX).run as exp:
        exp.log().info(f"Running {EXP.name}")
"""

from datetime import datetime, timezone
from params_proto import proto, EnvVar


@proto.prefix
class EXP:
    """
    Global experiment configuration.

    This class is the single source of truth for experiment metadata.
    Configure it before starting an experiment, or let Experiment auto-configure.

    Prefix format: {owner}/{project}/path.../[name]

    Template variables available:
        {EXP.PREFIX}    - Full experiment prefix from env or direct setting
        {EXP.name}      - Experiment name (last segment of prefix)
        {EXP.id}        - Unique experiment ID (snowflake)
        {EXP.date}      - Date string (YYYYMMDD)
        {EXP.time}      - Time string (HHMMSS)
        {EXP.datetime}  - DateTime string (YYYYMMDD.HHMMSS)
        {EXP.timestamp} - ISO timestamp

    Example:
        # Set prefix via env var or directly
        EXP.PREFIX = "ge/myproject/exp1"

        # Or use environment variable
        # export ML_DASH_PREFIX="ge/myproject/exp1"

    Auto-detection from file path:
        # In experiments/__init__.py
        from ml_dash import EXP
        EXP.project_root = "/path/to/my-project/experiments"

        # In experiments/vision/resnet/train.py
        from ml_dash import EXP
        EXP.__post_init__(entry=__file__)
        # Result: EXP.prefix = "vision/resnet", EXP.name = "resnet"
    """
    PREFIX: str = EnvVar @ "ML_DASH_PREFIX" | None
    """Full experiment path: {owner}/{project}/path.../[name]"""

    API_URL: str = EnvVar @ "ML_DASH_API_URL" | "https://api.dash.ml"
    """Remote API server URL"""

    name: str = "scratch"
    """Experiment name (last segment of prefix)"""

    description: str = None
    """Experiment description"""

    id: int = None
    """Unique experiment ID (snowflake, auto-generated at run start)"""

    timestamp: str = None
    """ISO timestamp (auto-generated at run start)"""

    prefix: str = None
    """Resolved prefix after template substitution"""

    project_root: str = None
    """Root directory for experiment hierarchy (for auto-detection)"""

    entry: str = None
    """Entry point file/directory path"""

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

