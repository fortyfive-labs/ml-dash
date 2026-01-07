"""Basic experiment example - Your first ML-Dash experiment."""
import sys
sys.path.insert(0, '../../src')

from ml_dash import Experiment

def main():
    print("=" * 60)
    print("Basic Experiment Example")
    print("=" * 60)

    # Create a experiment in local mode
    with Experiment(
        prefix="hello-ml-dash",
        project="tutorials",
        
        description="My first ML-Dash experiment",
        tags=["tutorial", "basic"]
    ).run as experiment:
        # Log a message
        experiment.log("Hello from ML-Dash!", level="info")

        # Metric parameters
        experiment.params.set(message="Hello World", version="1.0")

        print("\n✓ Experiment created successfully!")
        print(f"Data stored in: {experiment._storage.root_path}")
        print(f"Experiment: {experiment.project}/{experiment.name}")

    print("\n" + "=" * 60)
    print("Check your data:")
    print("  cat tutorial_data/tutorials/hello-ml-dash/logs/logs.jsonl")
    print("  cat tutorial_data/tutorials/hello-ml-dash/parameters.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
