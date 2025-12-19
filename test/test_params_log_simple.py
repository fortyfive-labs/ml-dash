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
        exp1 = Experiment(name="test-set", project="test", local_path=tmpdir)
        with exp1.run:
            exp1.params.set(a=1, b=2, c=3)
            params1 = exp1.params.get()

        # Test with log()
        exp2 = Experiment(name="test-log", project="test", local_path=tmpdir)
        with exp2.run:
            exp2.params.log(a=1, b=2, c=3)
            params2 = exp2.params.get()

        # Both should produce identical parameters
        assert params1 == params2
        assert params1 == {"a": 1, "b": 2, "c": 3}

        print("✓ params.log() behaves exactly like params.set()")


def test_params_log_chaining():
    """Test that params.log() supports chaining."""

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test-chain", project="test", local_path=tmpdir)
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

        print("✓ params.log() supports method chaining")


def test_params_log_nested():
    """Test that params.log() handles nested dicts."""

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test-nested", project="test", local_path=tmpdir)
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

        print("✓ params.log() flattens nested dicts")


def demo_params_log():
    """Demo showing params.log() usage."""
    print("\n" + "="*60)
    print("params.log() Demo - Better Parameter Organization")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="demo", project="demo", local_path=tmpdir)

        with exp.run:
            print("\n1. Setting initial parameters with params.log():")
            exp.params.log(
                learning_rate=0.001,
                batch_size=32,
                model="resnet50",
                optimizer="adam"
            )
            print("   ✓ Parameters set")

            print("\n2. Updating learning rate (simulating LR decay):")
            exp.params.log(learning_rate=0.0001)
            print("   ✓ Learning rate updated")

            print("\n3. Final parameters:")
            params = exp.params.get()
            for key, value in params.items():
                print(f"   {key}: {value}")

            print("\n4. params.log() vs params.set():")
            print("   • params.log() is an alias for params.set()")
            print("   • Both behave exactly the same")
            print("   • Use whichever name makes more semantic sense")

    print("\n" + "="*60)
    print("✓ Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_params_log()

    # Run tests
    print("Running tests...\n")
    test_params_log_is_alias_for_set()
    test_params_log_chaining()
    test_params_log_nested()
    print("\n✅ All tests passed!")
