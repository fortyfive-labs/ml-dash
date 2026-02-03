"""Test OOM protection with bounded queues and backpressure."""
import getpass
import time
from ml_dash import Experiment

def test_oom_protection():
    """Test that queue backpressure prevents OOM."""

    owner = getpass.getuser()

    print("Testing OOM protection with rapid metric generation...")
    print("This test generates 15,000 points rapidly (more than queue capacity)")
    print("The system should:")
    print("  1. Warn when queue reaches 80% capacity")
    print("  2. Trigger aggressive flushing at 50% capacity")
    print("  3. Block (backpressure) if queue fills completely")
    print("  4. Successfully flush all data on shutdown\n")

    with Experiment(
        prefix=f"tom/oom-test/protection-test",
        readme="Testing OOM protection with bounded queues",
        tags=["test", "oom"],
        dash_url='http://localhost:3000',
    ).run as experiment:

        start_time = time.time()

        # Generate 15,000 data points rapidly (exceeds queue capacity of 10,000)
        # This should trigger warnings and backpressure
        for i in range(15000):
            experiment.metrics("train").log(
                epoch=i,
                loss=1.0 / (i + 1),
                accuracy=min(0.99, i * 0.0001)
            )

            # Print progress every 1000 points
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Generated {i + 1}/15000 points ({rate:.0f} points/sec)")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n✓ Generated all 15,000 points in {total_time:.1f} seconds")
        print(f"  Average rate: {15000 / total_time:.0f} points/sec")
        print("\nClosing experiment and flushing all data...")

    print("✓ Experiment closed successfully")
    print("✓ All data should be flushed to database")
    print("\nNote: If you saw warnings about queue filling up, that's expected!")
    print("The backpressure mechanism prevents OOM by blocking when necessary.")

if __name__ == "__main__":
    test_oom_protection()
