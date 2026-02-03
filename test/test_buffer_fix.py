"""Test that all buffered metrics are flushed on shutdown."""
import getpass
import random
from ml_dash import Experiment

def test_buffer_flush():
    """Test that large numbers of metrics are all flushed."""

    owner = getpass.getuser()

    with Experiment(
        prefix=f"tom/buffer-test/flush-fix-test",
        readme="Testing buffer flush fix",
        tags=["test", "buffer"],
        dash_url='http://localhost:3000',
    ).run as experiment:

        # Log 500 data points (5 batches of 100)
        # This tests that multiple batches are flushed on shutdown
        print("Logging 500 data points per metric...")
        for i in range(500):
            experiment.metrics("train").log(
                epoch=i,
                loss=2.5 * (0.95**i) + random.uniform(-0.05, 0.05),
                accuracy=min(0.99, 0.3 + i * 0.001) + random.uniform(-0.01, 0.01)
            )

            experiment.metrics("validation").log(
                epoch=i,
                loss=2.6 * (0.95**i) + random.uniform(-0.08, 0.08),
                accuracy=min(0.96, 0.28 + i * 0.001) + random.uniform(-0.02, 0.02)
            )

        print("Done logging. Experiment will now close and flush all data...")

    print("✓ Experiment closed. All data should be flushed.")
    print(f"Check database - should have 500 points in 'train' and 'validation' metrics")

if __name__ == "__main__":
    test_buffer_flush()
