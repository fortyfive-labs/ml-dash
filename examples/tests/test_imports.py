"""Test that packages can be imported."""

def test_package_imports():
    """Test that ml_experiments package can be imported."""
    import ml_experiments
    assert hasattr(ml_experiments, '__version__')
    assert ml_experiments.__version__ == "0.1.0"


def test_subpackage_imports():
    """Test that subpackages can be imported."""
    import ml_experiments.sweeps
    import ml_experiments.baselines

    # Should not raise any errors
    assert ml_experiments.sweeps is not None
    assert ml_experiments.baselines is not None
