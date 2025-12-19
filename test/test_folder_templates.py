"""
Test folder template syntax support with {RUN.name} format.
"""

from ml_dash import Experiment, dxp
import tempfile


def test_template_with_run_name():
    """Test template with {RUN.name} variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="my-experiment", project="test", local_path=tmpdir)
        exp.run.folder = "iclr_2024/{RUN.name}"

        # Verify template was expanded with timestamp
        # Format: iclr_2024/my-experiment_YYYYMMDD_HHMMSS
        assert exp.run.folder.startswith("iclr_2024/my-experiment_")
        assert len(exp.run.folder) == len("iclr_2024/my-experiment_20251217_093440")

        with exp.run:
            assert exp.folder.startswith("iclr_2024/my-experiment_")

    print("✓ Template with {RUN.name} works")


def test_template_with_run_project():
    """Test template with {RUN.project} variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="exp1", project="vision", local_path=tmpdir)
        exp.run.folder = "{RUN.project}/experiments/{RUN.name}"

        # Verify template was expanded with project and timestamped name
        assert exp.run.folder.startswith("vision/experiments/exp1_")
        assert "vision" in exp.run.folder

        with exp.run:
            assert exp.folder.startswith("vision/experiments/exp1_")

    print("✓ Template with {RUN.project} and {RUN.name} works")


def test_template_with_dxp():
    """Test template with dxp singleton."""
    dxp.run.folder = "/iclr_2024/{RUN.name}"

    # Verify template was expanded with timestamp
    assert dxp.run.folder.startswith("/iclr_2024/dxp_")

    with dxp.run:
        assert dxp.folder.startswith("/iclr_2024/dxp_")

    print("✓ Template with dxp and {RUN.name} works")


def test_template_with_multiple_vars():
    """Test template with multiple variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="resnet50", project="vision", local_path=tmpdir)
        exp.run.folder = "experiments/{RUN.project}/models/{RUN.name}/runs"

        # Verify both project and timestamped name are present
        assert exp.run.folder.startswith("experiments/vision/models/resnet50_")
        assert exp.run.folder.endswith("/runs")

        with exp.run:
            assert exp.folder.startswith("experiments/vision/models/resnet50_")
            assert exp.folder.endswith("/runs")

    print("✓ Template with multiple variables works")


def test_static_folder_still_works():
    """Test that static folders without templates still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", project="test", local_path=tmpdir)
        exp.run.folder = "static/folder/path"

        assert exp.run.folder == "static/folder/path"

        with exp.run:
            assert exp.folder == "static/folder/path"

    print("✓ Static folder (no templates) still works")


def test_unknown_template_variable():
    """Test that unknown template variables are left as-is."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", project="test", local_path=tmpdir)

        # Unknown variables are just left as-is (not replaced)
        exp.run.folder = "experiments/{RUN.invalid_var}/path"

        # Should not be replaced since it's not a known variable
        assert exp.run.folder == "experiments/{RUN.invalid_var}/path"

    print("✓ Unknown template variables are left as-is")


def demo_folder_templates():
    """Demo showing folder template usage."""
    from datetime import datetime

    print("\n" + "="*60)
    print("Folder Templates Demo - {RUN.name} Format")
    print("="*60)

    # Example 1: Using {RUN.name}
    print("\n1. Template with {RUN.name}:")
    dxp.run.folder = "iclr_2024/{RUN.name}"
    print(f"   Template: 'iclr_2024/{{RUN.name}}'")
    print(f"   Expanded: '{dxp.run.folder}'")

    with dxp.run:
        print(f"   ✓ Experiment folder: {dxp.folder}")

    # Example 2: Using absolute path
    print("\n2. Template with absolute path:")
    dxp.run.folder = "/conferences/{RUN.name}"
    print(f"   Template: '/conferences/{{RUN.name}}'")
    print(f"   Expanded: '{dxp.run.folder}'")

    with dxp.run:
        print(f"   ✓ Experiment folder: {dxp.folder}")

    # Example 3: Multiple variables
    print("\n3. Template with {RUN.project} and {RUN.name}:")
    dxp.run.folder = "{RUN.project}/experiments/{RUN.name}/results"
    print(f"   Template: '{{RUN.project}}/experiments/{{RUN.name}}/results'")
    print(f"   Expanded: '{dxp.run.folder}'")

    with dxp.run:
        print(f"   ✓ Experiment folder: {dxp.folder}")

    print("\n" + "="*60)
    print("Benefits:")
    print("  • Dynamic folder names based on experiment metadata")
    print("  • Cleaner, more maintainable code")
    print("  • Automatic organization without manual string formatting")
    print("  • Supported variables: {RUN.name}, {RUN.project}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_folder_templates()

    # Run tests
    print("Running tests...\n")
    test_template_with_run_name()
    test_template_with_run_project()
    test_template_with_dxp()
    test_template_with_multiple_vars()
    test_static_folder_still_works()
    test_unknown_template_variable()
    print("\n✅ All tests passed!")
