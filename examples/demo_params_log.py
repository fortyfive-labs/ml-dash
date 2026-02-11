"""
Demo: params.log() - Better parameter organization

params.log() is an alias for params.set(), providing semantic clarity
when you want to emphasize that you're logging/recording parameters.
"""

from ml_dash import dxp

print("Demo: params.log() - Better Parameter Organization")
print("=" * 60)

with dxp.run:
    # Example 1: Initial setup
    print("\n1. Setting initial hyperparameters:")
    dxp.params.log(
        learning_rate=0.001,
        batch_size=32,
        model="resnet50",
        optimizer="adam",
        epochs=100
    )
    print("   ✓ Parameters logged")

    # Example 2: Nested parameters
    print("\n2. Logging nested model configuration:")
    dxp.params.log(
        model={
            "architecture": "resnet50",
            "pretrained": True,
            "num_classes": 1000
        },
        training={
            "lr": 0.001,
            "scheduler": "cosine",
            "warmup_epochs": 5
        }
    )
    print("   ✓ Nested parameters flattened and logged")

    # Example 3: Parameter updates during training
    print("\n3. Simulating parameter updates during training:")
    for epoch in [0, 5, 10]:
        if epoch == 5:
            print(f"   Epoch {epoch}: Decaying learning rate")
            dxp.params.log(learning_rate=0.0001)
        elif epoch == 10:
            print(f"   Epoch {epoch}: Further decay")
            dxp.params.log(learning_rate=0.00001)

    print("\n4. Final parameters:")
    final_params = dxp.params.get()
    for key, value in sorted(final_params.items()):
        print(f"   {key}: {value}")

print("\n" + "=" * 60)
print("Key Points:")
print("  • params.log() is identical to params.set()")
print("  • Both methods set parameters in the same way")
print("  • Use .log() for semantic clarity in your code")
print("  • Use .set() when you prefer explicit naming")
print("  • Both support chaining and nested dicts")
print("=" * 60)
