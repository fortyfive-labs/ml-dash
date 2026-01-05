"""
Test prefix template syntax support with {EXP.attr} format.

Uses the global EXP ParamsProto object for clean template expansion.
- {EXP.name}: Experiment name (no timestamp - just the name)
- {EXP.project}: Project name
- {EXP.id}: Unique run ID (timestamp-based)
- {EXP.timestamp}: Same as id
"""

from ml_dash import Experiment, EXP, dxp
from ml_dash.storage import LocalStorage
import tempfile
import shutil
from pathlib import Path


def test_template_with_run_name():
    """Test template with {EXP.name} variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="test", prefix="my-experiment", local_path=tmpdir)
        exp.run.prefix = "iclr_2024/{EXP.name}"

        # EXP.name is just the experiment name (no timestamp)
        assert exp.run.prefix == "iclr_2024/my-experiment"

        with exp.run:
            assert exp._folder_path == "iclr_2024/my-experiment"
            assert EXP.name == "my-experiment"

    print("✓ Template with {EXP.name} works")


def test_template_with_run_id():
    """Test template with {EXP.id} for unique folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="test", prefix="my-experiment", local_path=tmpdir)
        exp.run.prefix = "iclr_2024/{EXP.name}.{EXP.id}"

        # Should include both name and numeric id
        assert exp.run.prefix.startswith("iclr_2024/my-experiment.")
        # EXP.id is an 18-digit Snowflake ID
        parts = exp.run.prefix.split(".")
        assert len(parts) == 2
        assert parts[1].isdigit()
        assert len(parts[1]) == 18  # Snowflake ID (64-bit distributed ID)

        with exp.run:
            assert EXP.id is not None
            assert isinstance(EXP.id, int)

    print("✓ Template with {EXP.name}.{EXP.id} works")


def test_template_with_run_project():
    """Test template with {EXP.project} variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="vision", prefix="exp1", local_path=tmpdir)
        exp.run.prefix = "{EXP.project}/experiments/{EXP.name}"

        # Verify template was expanded
        assert exp.run.prefix == "vision/experiments/exp1"

        with exp.run:
            assert exp._folder_path == "vision/experiments/exp1"
            assert EXP.project == "vision"
            assert EXP.name == "exp1"

    print("✓ Template with {EXP.project} and {EXP.name} works")


def test_template_with_dxp():
    """Test template with dxp singleton."""
    ml_dash_dir = Path(".ml-dash-test-templates")

    # Clean up first
    if dxp._is_open:
        dxp.run.complete()
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)

    # Configure dxp for local mode testing
    dxp._storage = LocalStorage(root_path=ml_dash_dir)
    dxp._client = None

    # Reset EXP.id for fresh timestamp
    EXP.id = None
    EXP.timestamp = None

    dxp.run.prefix = "iclr_2024/{EXP.name}"

    # Should be formatted with dxp's name
    assert dxp.run.prefix == "iclr_2024/dxp"

    with dxp.run:
        assert dxp._folder_path == "iclr_2024/dxp"
        assert EXP.name == "dxp"

    # Clean up
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)

    print("✓ Template with dxp and {EXP.name} works")


def test_template_with_multiple_vars():
    """Test template with multiple variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="vision", prefix="resnet50", local_path=tmpdir)
        exp.run.prefix = "experiments/{EXP.project}/models/{EXP.name}/runs"

        # Verify both project and name are present
        assert exp.run.prefix == "experiments/vision/models/resnet50/runs"

        with exp.run:
            assert exp._folder_path == "experiments/vision/models/resnet50/runs"

    print("✓ Template with multiple variables works")


def test_static_folder_still_works():
    """Test that static prefixes without templates still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="test", prefix="test", local_path=tmpdir)
        exp.run.prefix = "static/folder/path"

        assert exp.run.prefix == "static/folder/path"

        with exp.run:
            assert exp._folder_path == "static/folder/path"

    print("✓ Static prefix (no templates) still works")


def test_unknown_template_variable():
    """Test that unknown template variables raise AttributeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(project="test", prefix="test", local_path=tmpdir)

        # Unknown variables raise AttributeError
        try:
            exp.run.prefix = "experiments/{EXP.invalid_var}/path"
            assert False, "Should have raised AttributeError"
        except AttributeError as e:
            assert "invalid_var" in str(e)

    print("✓ Unknown template variables raise AttributeError")


def demo_folder_templates():
    """Demo showing prefix template usage."""
    print("\n" + "="*60)
    print("Prefix Templates Demo - {EXP.attr} Format")
    print("="*60)

    ml_dash_dir = Path(".ml-dash-test-templates-demo")

    # Clean up first
    if dxp._is_open:
        dxp.run.complete()
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)

    # Configure dxp for local mode testing
    dxp._storage = LocalStorage(root_path=ml_dash_dir)
    dxp._client = None

    # Reset EXP for clean demo
    EXP._reset()

    # Example 1: Using {EXP.name}
    print("\n1. Template with {EXP.name}:")
    dxp.run.prefix = "iclr_2024/{EXP.name}"
    print(f"   Template: 'iclr_2024/{{EXP.name}}'")
    print(f"   Expanded: '{dxp.run.prefix}'")

    with dxp.run:
        print(f"   ✓ Experiment prefix: {dxp._folder_path}")

    # Example 2: Using EXP.id for unique folders (dot notation)
    EXP._reset()
    print("\n2. Template with {EXP.name}.{EXP.id}:")
    dxp.run.prefix = "runs/{EXP.name}.{EXP.id}"
    print(f"   Template: 'runs/{{EXP.name}}.{{EXP.id}}'")
    print(f"   Expanded: '{dxp.run.prefix}'")

    with dxp.run:
        print(f"   ✓ Experiment prefix: {dxp._folder_path}")

    # Example 3: Using date/time templates
    EXP._reset()
    print("\n3. Template with {EXP.date}:")
    dxp.run.prefix = "{EXP.project}/{EXP.name}.{EXP.date}"
    print(f"   Template: '{{EXP.project}}/{{EXP.name}}.{{EXP.date}}'")
    print(f"   Expanded: '{dxp.run.prefix}'")

    with dxp.run:
        print(f"   ✓ Experiment prefix: {dxp._folder_path}")

    print("\n" + "="*60)
    print("EXP attributes:")
    print(f"  • EXP.name = '{EXP.name}' (experiment name)")
    print(f"  • EXP.project = '{EXP.project}' (project name)")
    print(f"  • EXP.id = {EXP.id} (numeric run ID)")
    print(f"  • EXP.date = '{EXP.date}' (YYYYMMDD)")
    print(f"  • EXP.datetime = '{EXP.datetime}' (YYYYMMDD.HHMMSS)")
    print(f"  • EXP.timestamp = '{EXP.timestamp}' (ISO)")
    print("="*60 + "\n")

    # Clean up
    if ml_dash_dir.exists():
        shutil.rmtree(ml_dash_dir)


if __name__ == "__main__":
    # Run demo
    demo_folder_templates()

    # Run tests
    print("Running tests...\n")
    test_template_with_run_name()
    test_template_with_run_id()
    test_template_with_run_project()
    test_template_with_dxp()
    test_template_with_multiple_vars()
    test_static_folder_still_works()
    test_unknown_template_variable()
    print("\n✅ All tests passed!")
