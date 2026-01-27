"""
RUN - Global experiment configuration object for ML-Dash.

This module provides a global RUN object that serves as the single source
of truth for experiment metadata. Uses params-proto for configuration.

Usage:
    from ml_dash import RUN

    # Configure via environment variable
    # export ML_DASH_PREFIX="ge/myproject/experiments/exp1"

    # Or set directly
    RUN.PREFIX = "ge/myproject/experiments/exp1"

    # Use in templates
    prefix = "{RUN.PREFIX}/{RUN.name}.{RUN.id}".format(RUN=RUN)

    # With Experiment (RUN is auto-populated)
    from ml_dash import Experiment
    with Experiment(prefix=RUN.PREFIX).run as exp:
        exp.logs.info(f"Running {RUN.name}")
"""

import functools
import os
import sys
import typing
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from params_proto import EnvVar, proto


def requires_open(func):
  """
  Decorator that ensures the experiment is open before executing a method.

  Raises:
      RuntimeError: If experiment is not started
  """

  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    if not self._is_open:
      raise RuntimeError(
        "Experiment not started. Use 'with experiment.run:' or call experiment.run.start() first.\n"
        "Example:\n"
        "  with dxp.run:\n"
        "      dxp.params.set(lr=0.001)"
      )
    return func(self, *args, **kwargs)

  return wrapper

if typing.TYPE_CHECKING:
  from .client import RemoteClient
  from .experiment import Experiment
  from .storage import LocalStorage

PROJECT_ROOT_FILES = ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg")


def find_project_root(
  start: Union[str, Path] = None,
  verbose: bool = False,
) -> str:
  """Find the nearest project root by looking for common project files.

  Walks up the directory tree from `start` until it finds a directory
  containing pyproject.toml, requirements.txt, setup.py, or setup.cfg.

  Args:
      start: Starting directory or file path. Defaults to cwd.
      verbose: If True, print search progress.

  Returns:
      String path to the project root directory, or cwd if not found.
  """
  if start is None:
    start = Path.cwd()
  else:
    start = Path(start)

  if start.is_file():
    start = start.parent

  if verbose:
    print(f"Searching for project root from: {start}")

  for parent in [start, *start.parents]:
    if verbose:
      print(f"  Checking: {parent}")
    for filename in PROJECT_ROOT_FILES:
      if (parent / filename).exists():
        if verbose:
          print(f"  Found: {parent / filename}")
        return str(parent)

  if verbose:
    print(f"  No project root found, using cwd: {Path.cwd()}")
  return str(Path.cwd())


@proto.prefix
class RUN:
  """
  Global Experiment Run Configuration and
  Lifecycle manager for experiments

  This class is the single source of truth for experiment metadata.
  Configure it before starting an experiment, or through the Experiment
  constructor.

  Supports three usage patterns:
  1. Method calls: experiment.run.start(), experiment.run.complete()
  2. Context manager: with Experiment(...).run as exp:
      3. Decorator: @exp.run or @Experiment(...).run


  Default prefix template:
      {project}/{now:%Y/%m-%d}/{path_stem}/{job_name}

  Example:
      # Set prefix via environment variable
      # export ML_DASH_PREFIX="ge/myproject/exp1"

      # Or configure directly
      from ml_dash.run import RUN

      RUN.project = "my-project"
      RUN.prefix = "{username}/{project}/{now:%Y-%m-%d}/{entry}"

  Auto-detection:
      project_root is auto-detected by searching for pyproject.toml,
      requirements.txt, setup.py, or setup.cfg in parent directories.
  """

  user: str = EnvVar @ "ML_DASH_USER" @ "USER"

  api_url: str = EnvVar @ "ML_DASH_API_URL" | "https://api.dash.ml"
  """Remote API server URL"""

  ### Experiment and project information
  project = "{user}/scratch"  # default project name

  prefix: str = (
    EnvVar @ "ML_DASH_PREFIX" | "{project}/{now:%Y/%m-%d}/{path_stem}/{job_name}"
  )
  """Full experiment path: {owner}/{project}/path.../[name]"""

  readme = None

  id: int = None
  """Unique experiment ID (snowflake, auto-generated at run start)"""

  now = datetime.now()
  """Timestamp at import time. Does not change during the session."""

  @property
  def date(self) -> str:
    """Date string in YYYYMMDD format."""
    return self.now.strftime("%Y%m%d")

  @property
  def datetime_str(self) -> str:
    """DateTime string in YYYYMMDD.HHMMSS format."""
    return self.now.strftime("%Y%m%d.%H%M%S")

  timestamp: str = None
  """Timestamp created at instantiation"""

  ### file properties
  project_root: str = None
  """Root directory for experiment hierarchy (for auto-detection)"""

  entry: Union[Path, str] = None
  """Entry point file/directory path"""

  path_stem: str = None

  job_counter: int = 1  # Default to 0. Use True to increment by 1.

  job_name: str = "{now:%H.%M.%S}/{job_counter:03d}"

  """ 
      Default to '{now:%H.%M.%S}'. use '{now:%H.%M.%S}/{job_counter:03d}'
      
      for multiple launches. You can do so by setting:

      ```python
      RUN.job_name += "/{job_counter}"

      for params in sweep:
         thunk = instr(main)
         jaynes.run(thun)
      jaynes.listen()
      ```
  """

  debug = "pydevd" in sys.modules
  "set to True automatically for pyCharm"

  _experiment: "Experiment" = None
  _client: Optional["RemoteClient"] = None
  _storage: Optional["LocalStorage"] = None

  # Prefix components (parsed from prefix)
  owner: Optional[str] = None
  name: Optional[str] = None
  _folder_path: Optional[str] = None

  def __post_init__(self):
    """

    Initialize RUN with auto-detected prefix from entry path.


    Args:
        entry: Path to entry file/directory (e.g., __file__ or directory
               containing sweep.jsonl). If not provided, uses caller's
               __file__ automatically.

    Computes prefix as relative path from project_root to entry's directory.

    Example:
        # experiments/__init__.py
        from ml_dash import RUN

        RUN.project_root = "/path/to/my-project/experiments"

        # experiments/vision/resnet/train.py
        from ml_dash import RUN

        RUN(entry=__file__)
        # Result: RUN.prefix = "vision/resnet", RUN.name = "resnet"
    """
    # Use provided entry or try to auto-detect from caller
    if self.entry is None:
      import inspect

      # Walk up the stack to find the actual caller (skip params_proto frames)
      frame = inspect.currentframe().f_back
      while frame:
        file_path = frame.f_globals.get("__file__", "")
        if "params_proto" not in file_path and "ml_dash/run.py" not in file_path:
          break
        frame = frame.f_back

      self.entry = frame.f_globals.get("__file__") if frame else None

    if not self.path_stem:

      def stem(path):
        return os.path.splitext(str(path))[0]

      def truncate(path, depth):
        return "/".join(str(path).split("/")[depth:])

      self.project_root = str(self.project_root or find_project_root(self.entry))
      script_root_depth = self.project_root.split("/").__len__()

      script_truncated = truncate(os.path.abspath(self.entry), depth=script_root_depth)

      self.path_stem = stem(script_truncated)

    if isinstance(RUN.job_counter, int) or isinstance(RUN.job_counter, float):
      RUN.job_counter += 1

    while "{" in self.prefix:
      data = vars(self)
      for k, v in data.items():
        if isinstance(v, str):
          setattr(self, k, v.format(**data))

    # for k, v in data.items():
    #   print(f"> {k:>30}: {v}")

    # Parse prefix into components: {owner}/{project}/path.../[name]
    if self.prefix:
      self._folder_path = self.prefix
      parts = self.prefix.strip("/").split("/")
      if len(parts) >= 2:
        self.owner = parts[0]
        self.project = parts[1]
        # self.name is the last segment
        self.name = parts[-1] if len(parts) > 2 else parts[1]

  def __setattr__(self, name: str, value):
    """
    Intercept attribute setting to expand {EXP.attr} templates in prefix.

    When prefix is set, expands any {EXP.name}, {EXP.id}, {EXP.date}, etc. templates
    using current instance's attributes. Also syncs back to class-level RUN attributes.
    """
    # Prevent prefix changes after experiment has started
    if name == "prefix" and isinstance(value, str):
      experiment = getattr(self, "_experiment", None)
      if experiment is not None and getattr(experiment, "_is_open", False):
        raise RuntimeError(
          "Cannot change prefix after experiment has been initialized. "
          "Set prefix before calling experiment.run.start() or entering the context manager."
        )

    # Expand templates if setting prefix
    if name == "prefix" and isinstance(value, str):
      # Check if value contains {EXP. templates
      if "{EXP." in value:
        import re

        def replace_match(match):
          attr_name = match.group(1)
          # Special handling for id - generate if needed
          if attr_name == "id" and not getattr(self, "id", None):
            from ml_dash.snowflake import generate_id
            object.__setattr__(self, "id", generate_id())

          # Get attribute, raising error if not found
          try:
            attr_value = getattr(self, attr_name)
            if attr_value is None:
              raise AttributeError(f"Attribute '{attr_name}' is None")
            return str(attr_value)
          except AttributeError:
            raise AttributeError(f"RUN has no attribute '{attr_name}'")

        # Match {EXP.attr_name} pattern
        pattern = r"\{EXP\.(\w+)\}"
        value = re.sub(pattern, replace_match, value)

      # Always update _folder_path when prefix changes
      object.__setattr__(self, "_folder_path", value)

      # Parse and update owner, project, name from new prefix
      parts = value.strip("/").split("/")
      if len(parts) >= 2:
        object.__setattr__(self, "owner", parts[0])
        object.__setattr__(self, "project", parts[1])
        object.__setattr__(self, "name", parts[-1] if len(parts) > 2 else parts[1])

    # Use object.__setattr__ to set the value
    object.__setattr__(self, name, value)

  def start(self) -> "Experiment":
    """
    Start the experiment (sets status to RUNNING).

    Returns:
        The experiment instance for chaining
    """
    return self._experiment._open()

  def complete(self) -> None:
    """Mark experiment as completed (status: COMPLETED)."""
    self._experiment._close(status="COMPLETED")

  def fail(self) -> None:
    """Mark experiment as failed (status: FAILED)."""
    self._experiment._close(status="FAILED")

  def cancel(self) -> None:
    """Mark experiment as cancelled (status: CANCELLED)."""
    self._experiment._close(status="CANCELLED")

  def __enter__(self) -> "Experiment":
    """Context manager entry - starts the experiment."""
    return self.start()

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - completes or fails the experiment."""
    if exc_type is not None:
      self.fail()
    else:
      self.complete()
    return False


if __name__ == "__main__":
  RUN.description = ""
  RUN.entry = __file__
  RUN.prefix = "you you"

  run = RUN()
  print(vars(run))
