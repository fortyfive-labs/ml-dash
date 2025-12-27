"""
Test folder template syntax support with {EXP.attr} format.

Uses the global RUN ParamsProto object for clean template expansion.
- {EXP.name}: Experiment name (no timestamp - just the name)
- {EXP.project}: Project name
- {EXP.id}: Unique run ID (timestamp-based)
- {EXP.timestamp}: Same as id
"""

from ml_dash import Experiment, EXP, dxp
import tempfile


def test_template_with_run_name():
    """Test template with {EXP.name} variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("my-experiment", project="test", local_path=tmpdir)
        exp.run.folder = "iclr_2024/{EXP.name}"

        # EXP.name is just the experiment name (no timestamp)
        assert exp.run.folder == "iclr_2024/my-experiment"

        with exp.run:
            assert exp.folder == "iclr_2024/my-experiment"
            assert EXP.name == "my-experiment"

    print("✓ Template with {EXP.name} works")


def test_template_with_run_id():
    """Test template with {EXP.id} for unique folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("my-experiment", project="test", local_path=tmpdir)
        exp.run.folder = "iclr_2024/{EXP.name}.{EXP.id}"

        # Should include both name and numeric id
        assert exp.run.folder.startswith("iclr_2024/my-experiment.")
        # EXP.id is a 13-digit millisecond timestamp
        parts = exp.run.folder.split(".")
        assert len(parts) == 2
        assert parts[1].isdigit()
        assert len(parts[1]) == 13  # milliseconds since epoch

        with exp.run:
            assert EXP.id is not None
            assert isinstance(EXP.id, int)

    print("✓ Template with {EXP.name}.{EXP.id} works")


def test_template_with_run_project():
    """Test template with {EXP.project} variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("exp1", project="vision", local_path=tmpdir)
        exp.run.folder = "{EXP.project}/experiments/{EXP.name}"

        # Verify template was expanded
        assert exp.run.folder == "vision/experiments/exp1"

        with exp.run:
            assert exp.folder == "vision/experiments/exp1"
            assert EXP.project == "vision"
            assert EXP.name == "exp1"

    print("✓ Template with {EXP.project} and {EXP.name} works")


def test_template_with_dxp():
    """Test template with dxp singleton."""
    # Reset EXP.id for fresh timestamp, but keep dxp's name/project
    EXP.id = None
    EXP.timestamp = None

    dxp.run.folder = "/iclr_2024/{EXP.name}"

    # Should be formatted with dxp's name
    assert dxp.run.folder == "/iclr_2024/dxp"

    with dxp.run:
        assert dxp.folder == "/iclr_2024/dxp"
        assert EXP.name == "dxp"

    print("✓ Template with dxp and {EXP.name} works")


def test_template_with_multiple_vars():
    """Test template with multiple variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("resnet50", project="vision", local_path=tmpdir)
        exp.run.folder = "experiments/{EXP.project}/models/{EXP.name}/runs"

        # Verify both project and name are present
        assert exp.run.folder == "experiments/vision/models/resnet50/runs"

        with exp.run:
            assert exp.folder == "experiments/vision/models/resnet50/runs"

    print("✓ Template with multiple variables works")


def test_static_folder_still_works():
    """Test that static folders without templates still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("test", project="test", local_path=tmpdir)
        exp.run.folder = "static/folder/path"

        assert exp.run.folder == "static/folder/path"

        with exp.run:
            assert exp.folder == "static/folder/path"

    print("✓ Static folder (no templates) still works")


def test_unknown_template_variable():
    """Test that unknown template variables raise AttributeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment("test", project="test", local_path=tmpdir)

        # Unknown variables raise AttributeError
        try:
            exp.run.folder = "experiments/{EXP.invalid_var}/path"
            assert False, "Should have raised AttributeError"
        except AttributeError as e:
            assert "invalid_var" in str(e)

    print("✓ Unknown template variables raise AttributeError")


def demo_folder_templates():
    """Demo showing folder template usage."""
    print("\n" + "="*60)
    print("Folder Templates Demo - {EXP.attr} Format")
    print("="*60)

    # Reset RUN for clean demo
    EXP._reset()

    # Example 1: Using {EXP.name}
    print("\n1. Template with {EXP.name}:")
    dxp.run.folder = "iclr_2024/{EXP.name}"
    print(f"   Template: 'iclr_2024/{{EXP.name}}'")
    print(f"   Expanded: '{dxp.run.folder}'")

    with dxp.run:
        print(f"   ✓ Experiment folder: {dxp.folder}")

    # Example 2: Using EXP.id for unique folders (dot notation)
    EXP._reset()
    print("\n2. Template with {EXP.name}.{EXP.id}:")
    dxp.run.folder = "runs/{EXP.name}.{EXP.id}"
    print(f"   Template: 'runs/{{EXP.name}}.{{EXP.id}}'")
    print(f"   Expanded: '{dxp.run.folder}'")

    with dxp.run:
        print(f"   ✓ Experiment folder: {dxp.folder}")

    # Example 3: Using date/time templates
    EXP._reset()
    print("\n3. Template with {EXP.date}:")
    dxp.run.folder = "{EXP.project}/{EXP.name}.{EXP.date}"
    print(f"   Template: '{{EXP.project}}/{{EXP.name}}.{{EXP.date}}'")
    print(f"   Expanded: '{dxp.run.folder}'")

    with dxp.run:
        print(f"   ✓ Experiment folder: {dxp.folder}")

    print("\n" + "="*60)
    print("RUN attributes:")
    print(f"  • EXP.name = '{EXP.name}' (experiment name)")
    print(f"  • EXP.project = '{EXP.project}' (project name)")
    print(f"  • EXP.id = {EXP.id} (numeric run ID)")
    print(f"  • EXP.date = '{EXP.date}' (YYYYMMDD)")
    print(f"  • EXP.datetime = '{EXP.datetime}' (YYYYMMDD.HHMMSS)")
    print(f"  • EXP.timestamp = '{EXP.timestamp}' (ISO)")
    print("="*60 + "\n")


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
