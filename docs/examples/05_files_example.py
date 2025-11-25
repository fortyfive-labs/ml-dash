"""Files example - Upload models and artifacts."""
import sys
sys.path.insert(0, '../../src')

from ml_dash import Experiment
import os
import json

def main():
    print("=" * 60)
    print("Files Example - Uploading Artifacts")
    print("=" * 60)

    # Create some sample files to upload
    os.makedirs("temp_files", exist_ok=True)

    # Create a fake model file
    with open("temp_files/model.txt", "w") as f:
        f.write("Simulated model weights\n" * 10)

    # Create a config file
    config = {
        "model": "resnet50",
        "learning_rate": 0.001,
        "batch_size": 32
    }
    with open("temp_files/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create a results file
    with open("temp_files/results.txt", "w") as f:
        f.write("Epoch,Loss,Accuracy\n")
        for i in range(10):
            f.write(f"{i+1},{1.0/(i+1):.4f},{0.5+i*0.05:.4f}\n")

    with Experiment(
        name="files-demo",
        project="tutorials",
        local_path="./tutorial_data"
    ).run as experiment:
        print("\n1. Uploading model file...")

        # Upload model
        result = experiment.files(
            file_path="temp_files/model.txt",
            prefix="/models",
            description="Trained model weights",
            tags=["model", "final"]
        ).save()
        print(f"   ✓ Uploaded: {result['filename']} ({result['sizeBytes']} bytes)")
        print(f"   Checksum: {result['checksum']}")

        print("\n2. Uploading configuration...")

        # Upload config
        result = experiment.files(
            file_path="temp_files/config.json",
            prefix="/config",
            description="Training configuration",
            tags=["config"]
        ).save()
        print(f"   ✓ Uploaded: {result['filename']}")

        print("\n3. Uploading results...")

        # Upload results
        result = experiment.files(
            file_path="temp_files/results.txt",
            prefix="/results",
            description="Training results per epoch",
            tags=["results", "metrics"],
            metadata={"epochs": 10, "format": "csv"}
        ).save()
        print(f"   ✓ Uploaded: {result['filename']}")

        print("\n4. Listing all files...")

        # List all files
        files = experiment.files().list()
        print(f"   Found {len(files)} files:")
        for file in files:
            print(f"     - {file['path']}/{file['filename']} ({file['sizeBytes']} bytes)")

        experiment.log("Files uploaded successfully", level="info")

    # Clean up temp files
    import shutil
    shutil.rmtree("temp_files")

    print("\n✓ All files uploaded!")
    print("\n" + "=" * 60)
    print("View uploaded files:")
    print("  ls -lR tutorial_data/.ml-dash/tutorials/files-demo/files/")
    print("=" * 60)

if __name__ == "__main__":
    main()
