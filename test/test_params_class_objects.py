"""
Test params.log() with class objects (for params_proto support).
"""

from ml_dash import dxp


def test_params_with_class_object():
    """Test that params.log() can accept class objects."""

    class Args:
        batch_size = 64
        lr = 0.001
        epochs = 10

    with dxp.run:
        # Should automatically extract class attributes
        dxp.params.log(Args=Args)

        params = dxp.params.get()

        assert params["Args.batch_size"] == 64
        assert params["Args.lr"] == 0.001
        assert params["Args.epochs"] == 10

    print("✓ params.log(Args=Args) works with class objects")


def test_params_with_nested_config():
    """Test nested config classes."""

    class ModelConfig:
        hidden_size = 128
        num_layers = 3

    class OptimizerConfig:
        lr = 0.001
        beta1 = 0.9

    with dxp.run:
        dxp.params.log(
            model=ModelConfig,
            optimizer=OptimizerConfig
        )

        params = dxp.params.get()

        assert params["model.hidden_size"] == 128
        assert params["model.num_layers"] == 3
        assert params["optimizer.lr"] == 0.001
        assert params["optimizer.beta1"] == 0.9

    print("✓ Nested config classes work")


def test_params_mixed_class_and_dict():
    """Test mixing class objects and regular dicts."""

    class Args:
        batch_size = 64
        lr = 0.001

    with dxp.run:
        dxp.params.log(
            Args=Args,
            runtime={"gpu": "cuda:0", "workers": 4}
        )

        params = dxp.params.get()

        assert params["Args.batch_size"] == 64
        assert params["Args.lr"] == 0.001
        assert params["runtime.gpu"] == "cuda:0"
        assert params["runtime.workers"] == 4

    print("✓ Mixed class and dict params work")


def test_params_skips_private_attributes():
    """Test that private attributes are skipped."""

    class Args:
        batch_size = 64
        _private = "should be skipped"
        __magic__ = "should be skipped"

    with dxp.run:
        dxp.params.log(Args=Args)

        params = dxp.params.get()

        assert params["Args.batch_size"] == 64
        assert "Args._private" not in params
        assert "Args.__magic__" not in params

    print("✓ Private attributes are correctly skipped")


def demo_params_class_objects():
    """Demo showing class object support."""
    print("\n" + "="*60)
    print("Params Class Objects Demo")
    print("="*60)

    class TrainingConfig:
        """Training hyperparameters."""
        batch_size = 32
        learning_rate = 0.001
        epochs = 100
        warmup_steps = 1000

    class ModelConfig:
        """Model architecture."""
        hidden_size = 768
        num_layers = 12
        num_heads = 12

    print("\n1. Single config class:")
    with dxp.run:
        dxp.params.log(Args=TrainingConfig)
        params = dxp.params.get()

        for key in sorted(params.keys()):
            if key.startswith("Args."):
                print(f"   {key} = {params[key]}")

    print("\n2. Multiple config classes:")
    with dxp.run:
        dxp.params.log(
            training=TrainingConfig,
            model=ModelConfig
        )
        params = dxp.params.get()

        print("   Training config:")
        for key in sorted(k for k in params.keys() if k.startswith("training.")):
            print(f"     {key} = {params[key]}")

        print("   Model config:")
        for key in sorted(k for k in params.keys() if k.startswith("model.")):
            print(f"     {key} = {params[key]}")

    print("\n" + "="*60)
    print("Benefits:")
    print("  • Pass config classes directly (no manual conversion)")
    print("  • Works with params_proto and similar libraries")
    print("  • Automatic dot-notation: Args.batch_size, model.lr, etc.")
    print("  • Private attributes (_var, __var__) auto-skipped")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_params_class_objects()

    # Run tests
    print("Running tests...\n")
    test_params_with_class_object()
    test_params_with_nested_config()
    test_params_mixed_class_and_dict()
    test_params_skips_private_attributes()
    print("\n✅ All tests passed!")
