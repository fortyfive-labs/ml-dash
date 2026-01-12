import re
from pathlib import Path


def setup_dir(tmpdir, path_string):
  """Set up local file structure from a tree-style string.

  Use trailing * (with space) to mark the starting location (returned as cwd).

  Example path_string:
      project/
      ├─ pyproject.toml
      ├─ src/
      │  └─ module/
      │     └─ train.py
      └─ test/
         └─ test_run_config_master_test.py  *   <-- * marks cwd

  Returns:
      Path to the marked (*) location, or None if not marked.
  """
  lines = path_string.strip().splitlines()
  path_stack = [Path(tmpdir)]
  cwd = None

  for line in lines:
    if not line.strip():
      continue

    # Strip tree characters and calculate depth (2 spaces per level)
    stripped = line.lstrip("│├└─ ")
    if not stripped:
      continue

    indent = len(line) - len(line.lstrip())
    depth = indent // 2

    # Check for * marker (trailing, with space before it)
    # Pattern: "filename  *" or "filename *"
    match = re.match(r"^(.+?)\s+\*$", stripped)
    is_cwd = match is not None
    if is_cwd:
      stripped = match.group(1)

    # Adjust stack to current depth
    path_stack = path_stack[: depth + 1]

    # Build full path
    name = stripped.rstrip("/")
    current_path = path_stack[-1] / name

    if stripped.endswith("/"):
      current_path.mkdir(parents=True, exist_ok=True)
      path_stack.append(current_path)
    else:
      current_path.parent.mkdir(parents=True, exist_ok=True)
      current_path.touch()

    if is_cwd:
      cwd = current_path

  return cwd


def test_find_project_root(tmpdir):
  """First we setup a complex file structure using a multi-line string
  we then start with cwd=rel/path/... , and show that the resulting path
  is correct.
  """
  from ml_dash.run import find_project_root

  # * marks our starting point (deep script)
  cwd = setup_dir(
    tmpdir,
    """
      myproject/
      ├─ pyproject.toml
      └─ src/
         └─ pkg/
            └─ module/
               └─ deep/
                  └─ script.py  *
  """,
  )

  expected_root = str(Path(tmpdir) / "myproject")

  # Starting from deep script, should find pyproject.toml at myproject/
  result = find_project_root(cwd)
  assert result == expected_root

  # Starting from directory should also work
  result = find_project_root(cwd.parent)
  assert result == expected_root


def test_expected_RUN_behavior(tmpdir):
  """Test RUN instantiation with deterministic now parameter."""
  from datetime import datetime

  from ml_dash.run import RUN

  # * marks the entry script
  cwd = setup_dir(
    tmpdir,
    """
      project/
      ├─ pyproject.toml
      └─ test/
         └─ test_run_config_master_test.py  *
  """,
  )

  fixed_time = datetime(2026, 1, 12, 3, 47, 54)
  project_root = str(Path(tmpdir) / "project")

  # Reset job counter (instance gets this value, then class is incremented)
  RUN.job_counter = 1

  run1 = RUN(entry=str(cwd), now=fixed_time, project_root=project_root)
  # Expected: ge/scratch/2026/01-12/test/test_run/03.47.54/001
  assert "2026/01-12" in run1.prefix
  assert "test/test_run" in run1.prefix
  assert "03.47.54" in run1.prefix
  assert run1.prefix.endswith("/001")
  # print(run1.prefix)

  run2 = RUN(entry=str(cwd), now=fixed_time, project_root=project_root)
  # Job counter increments to 002
  assert run2.prefix.endswith("/002")
  # print(run2.prefix)
