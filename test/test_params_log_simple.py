"""
Test params.log() method - alias for params.set().
"""

import tempfile
from pathlib import Path
from ml_dash import Experiment


def test_params_log_is_alias_for_set():
    """Test that params.log() behaves exactly like params.set()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with set()
        exp1 = Experiment(prefix="test/project/test-set", dash_root=tmpdir)
        with exp1.run:
            exp1.params.set(a=1, b=2, c=3)
            params1 = exp1.params.get()

        # Test with log()
        exp2 = Experiment(prefix="test/project/test-log", dash_root=tmpdir)
        with exp2.run:
            exp2.params.log(a=1, b=2, c=3)
            params2 = exp2.params.get()

        # Both should produce identical parameters
        assert params1 == params2
        assert params1 == {"a": 1, "b": 2, "c": 3}


def test_params_log_chaining():
    """Test that params.log() supports chaining."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/test-chain", dash_root=tmpdir)
        with exp.run:
            # Should support chaining
            result = exp.params.log(a=1).log(b=2)

            # Verify it returns ParametersBuilder
            from ml_dash.params import ParametersBuilder
            assert isinstance(result, ParametersBuilder)

            # Verify both parameters were set
            params = exp.params.get()
            assert params["a"] == 1
            assert params["b"] == 2


def test_params_log_nested():
    """Test that params.log() handles nested dicts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(prefix="test/project/test-nested", dash_root=tmpdir)
        with exp.run:
            exp.params.log(
                model={"type": "resnet50", "layers": 50},
                training={"lr": 0.001, "epochs": 100}
            )

            params = exp.params.get()
            assert params["model.type"] == "resnet50"
            assert params["model.layers"] == 50
            assert params["training.lr"] == 0.001
            assert params["training.epochs"] == 100


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
