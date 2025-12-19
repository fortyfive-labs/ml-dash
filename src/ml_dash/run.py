"""
RUN - Global run configuration object for ML-Dash.

This module provides a global RUN object that serves as the single source
of truth for run/experiment metadata. Uses params-proto for configuration.

Usage:
    from ml_dash import RUN

    # Configure the run
    RUN.name = "my-experiment"
    RUN.project = "my-project"

    # Use in templates
    folder = "/experiments/{RUN.name}".format(RUN=RUN)

    # With dxp singleton (RUN is auto-populated)
    from ml_dash import dxp
    with dxp.run:
        # RUN.name, RUN.project, RUN.id, RUN.timestamp are set
        dxp.log().info(f"Running {RUN.name}")
"""

from datetime import datetime
from params_proto import proto


@proto.prefix
class RUN:
    """
    Global run configuration.

    This class is the single source of truth for run metadata.
    Configure it before starting an experiment, or let dxp auto-configure.
    """
    # Core identifiers
    name: str = "untitled"  # Run/experiment name
    project: str = "scratch"  # Project name

    # Auto-generated identifiers (populated at run.start())
    id: str = None  # Unique run ID (auto-generated)
    timestamp: str = None  # Run timestamp (same as id)

    # Optional configuration
    folder: str = None  # Folder path with optional templates
    description: str = None  # Run description

    @classmethod
    def _generate_id(cls) -> str:
        """Generate a unique run ID based on current timestamp."""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def _init_run(cls) -> None:
        """Initialize run ID and timestamp if not already set."""
        if cls.id is None:
            cls.id = cls._generate_id()
            cls.timestamp = cls.id

    @classmethod
    def _format(cls, template: str) -> str:
        """
        Format a template string with RUN values.

        Args:
            template: String with {RUN.attr} placeholders

        Returns:
            Formatted string with placeholders replaced

        Example:
            RUN._format("/experiments/{RUN.name}_{RUN.id}")
            # -> "/experiments/my-exp_20241219_143022"
        """
        return template.format(RUN=cls)

    @classmethod
    def _reset(cls) -> None:
        """Reset RUN to defaults (for testing or new runs)."""
        cls.name = "untitled"
        cls.project = "scratch"
        cls.id = None
        cls.timestamp = None
        cls.folder = None
        cls.description = None
