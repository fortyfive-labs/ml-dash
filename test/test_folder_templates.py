"""
Test prefix template syntax support with {EXP.attr} format.

Uses the global EXP ParamsProto object for clean template expansion.
- {EXP.name}: Experiment name (last segment of prefix)
- {EXP.id}: Unique run ID (snowflake)
- {EXP.date}: Date string (YYYYMMDD)
- {EXP.datetime}: DateTime string (YYYYMMDD.HHMMSS)
"""

import tempfile
from pathlib import Path

from ml_dash import RUN, Experiment


def test_template_with_run_name():
  """Test template with {EXP.name} variable."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test/project/my-experiment", dash_root=tmpdir)
    exp.run.prefix = "iclr_2024/{EXP.name}"

    # EXP.name is the last segment of the original prefix
    assert exp.run.prefix == "iclr_2024/my-experiment"

    with exp.run:
      assert exp._folder_path == "iclr_2024/my-experiment"
      assert RUN.name == "my-experiment"

  print("✓ Template with {EXP.name} works")


def test_template_with_run_id():
  """Test template with {EXP.id} for unique folders."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test/project/my-experiment", dash_root=tmpdir)
    exp.run.prefix = "iclr_2024/{EXP.name}.{EXP.id}"

    # Should include both name and numeric id
    assert exp.run.prefix.startswith("iclr_2024/my-experiment.")
    # EXP.id is an 18-digit Snowflake ID
    parts = exp.run.prefix.split(".")
    assert len(parts) == 2
    assert parts[1].isdigit()
    assert len(parts[1]) == 18  # Snowflake ID (64-bit distributed ID)

    with exp.run:
      assert RUN.id is not None
      assert isinstance(RUN.id, int)

  print("✓ Template with {EXP.name}.{EXP.id} works")


def test_template_with_multiple_vars():
  """Test template with multiple variables."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="vision/models/resnet50", dash_root=tmpdir)
    exp.run.prefix = "experiments/{EXP.name}/runs"

    # Verify name is present
    assert exp.run.prefix == "experiments/resnet50/runs"

    with exp.run:
      assert exp._folder_path == "experiments/resnet50/runs"

  print("✓ Template with multiple variables works")


def test_static_folder_still_works():
  """Test that static prefixes without templates still work."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test/project/test", dash_root=tmpdir)
    exp.run.prefix = "static/folder/path"

    assert exp.run.prefix == "static/folder/path"

    with exp.run:
      assert exp._folder_path == "static/folder/path"

  print("✓ Static prefix (no templates) still works")


def test_unknown_template_variable():
  """Test that unknown template variables raise AttributeError."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test/project/test", dash_root=tmpdir)

    # Unknown variables raise AttributeError
    try:
      exp.run.prefix = "experiments/{EXP.invalid_var}/path"
      assert False, "Should have raised AttributeError"
    except AttributeError as e:
      assert "invalid_var" in str(e)

  print("✓ Unknown template variables raise AttributeError")


def test_template_with_date():
  """Test template with {EXP.date} variable."""
  with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(prefix="test/project/experiment", dash_root=tmpdir)
    exp.run.prefix = "{EXP.name}.{EXP.date}"

    # Should have name and YYYYMMDD format date
    assert exp.run.prefix.startswith("experiment.")
    parts = exp.run.prefix.split(".")
    assert len(parts) == 2
    assert len(parts[1]) == 8  # YYYYMMDD
    assert parts[1].isdigit()

  print("✓ Template with {EXP.date} works")


def demo_folder_templates():
  """Demo showing prefix template usage."""
  print("\n" + "=" * 60)
  print("Prefix Templates Demo - {EXP.attr} Format")
  print("=" * 60)

  with tempfile.TemporaryDirectory() as tmpdir:
    ml_dash_dir = Path(tmpdir) / "ml-dash-test-templates-demo"

    # Reset EXP for clean demo
    RUN._reset()

    # Create experiment
    exp = Experiment(prefix="demo/project/dxp", dash_root=str(ml_dash_dir))

    # Example 1: Using {EXP.name}
    print("\n1. Template with {EXP.name}:")
    exp.run.prefix = "iclr_2024/{EXP.name}"
    print("   Template: 'iclr_2024/{EXP.name}'")
    print(f"   Expanded: '{exp.run.prefix}'")

    with exp.run:
      print(f"   ✓ Experiment prefix: {exp._folder_path}")

    # Example 2: Using EXP.id for unique folders (dot notation)
    RUN._reset()
    exp2 = Experiment(prefix="demo/project/dxp", dash_root=str(ml_dash_dir))
    print("\n2. Template with {EXP.name}.{EXP.id}:")
    exp2.run.prefix = "runs/{EXP.name}.{EXP.id}"
    print("   Template: 'runs/{EXP.name}.{EXP.id}'")
    print(f"   Expanded: '{exp2.run.prefix}'")

    with exp2.run:
      print(f"   ✓ Experiment prefix: {exp2._folder_path}")

    # Example 3: Using date/time templates
    RUN._reset()
    exp3 = Experiment(prefix="demo/project/dxp", dash_root=str(ml_dash_dir))
    print("\n3. Template with {EXP.date}:")
    exp3.run.prefix = "{EXP.name}.{EXP.date}"
    print("   Template: '{EXP.name}.{EXP.date}'")
    print(f"   Expanded: '{exp3.run.prefix}'")

    with exp3.run:
      print(f"   ✓ Experiment prefix: {exp3._folder_path}")

    print("\n" + "=" * 60)
    print("EXP attributes:")
    print(f"  • EXP.name = '{RUN.name}' (experiment name)")
    print(f"  • EXP.id = {RUN.id} (numeric run ID)")
    print(f"  • EXP.date = '{RUN.date}' (YYYYMMDD)")
    print(f"  • EXP.datetime = '{RUN.datetime}' (YYYYMMDD.HHMMSS)")
    print(f"  • EXP.timestamp = '{RUN.timestamp}' (ISO)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
  # Run demo
  demo_folder_templates()

  # Run tests
  print("Running tests...\n")
  test_template_with_run_name()
  test_template_with_run_id()
  test_template_with_multiple_vars()
  test_static_folder_still_works()
  test_unknown_template_variable()
  test_template_with_date()
  print("\n✅ All tests passed!")
