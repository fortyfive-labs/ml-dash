"""Download experiment data with proper hierarchy."""

import json
from pathlib import Path
from ml_dash.client import RemoteClient
from ml_dash.config import Config

# Setup
config = Config()
client = RemoteClient(
    base_url="https://api.dash.ml",
    api_key=config.api_key
)

# Get experiment by path
exp_data = client.get_experiment_by_path_graphql(
    project_slug="robot9",
    experiment_path="examples/tracking-data-collection1",
    namespace_slug="tom_tao_e4c2c9"
)

if not exp_data:
    print("Experiment not found!")
    exit(1)

experiment_id = exp_data["id"]
print(f"Found experiment: {exp_data['name']} (ID: {experiment_id})")

# Create output directory
output_dir = Path("test/downloads/tracking-data-collection1")
output_dir.mkdir(parents=True, exist_ok=True)

# Download parameters
print("\n📋 Downloading parameters...")
params = client.get_parameters(experiment_id)
if params:
    params_file = output_dir / "parameters.json"
    params_file.write_text(json.dumps(params, indent=2))
    print(f"  ✓ Saved to {params_file}")

# Download logs
print("\n📝 Downloading logs...")
logs_result = client.query_logs(experiment_id, limit=1000)
if logs_result and "logs" in logs_result:
    logs = logs_result["logs"]
    logs_file = output_dir / "logs.jsonl"
    with open(logs_file, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    print(f"  ✓ Saved {len(logs)} logs to {logs_file}")

# Download metrics (preserve hierarchy: metrics/train/data.jsonl)
print("\n📊 Downloading metrics...")
metrics_list = client.list_metrics(experiment_id)
print(f"  Found {len(metrics_list)} metric topics")

for metric in metrics_list:
    metric_name = metric["name"]
    print(f"  Downloading metric: {metric_name}")

    data_result = client.read_metric_data(experiment_id, metric_name, limit=10000)
    if data_result and "data" in data_result:
        data = data_result["data"]
        # Preserve hierarchy: metrics/train/data.jsonl
        metric_dir = output_dir / "metrics" / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_file = metric_dir / "data.jsonl"
        with open(metric_file, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"    ✓ Saved {len(data)} datapoints to {metric_file}")

# Download files (organize by semantic path: robot/position)
print("\n📁 Downloading files...")
files = client.list_files(experiment_id)
print(f"  Found {len(files)} files")

# For this test, we know files are image frames uploaded to "robot/position"
files_dir = output_dir / "files" / "robot" / "position"
files_dir.mkdir(parents=True, exist_ok=True)

downloaded_count = 0
for file_info in files:
    file_id = file_info["id"]
    file_name = file_info.get("name", "unknown")

    # Skip track data files
    if file_name == "tracks":
        continue

    try:
        dest_path = files_dir / file_name
        client.download_file(experiment_id, file_id, dest_path=str(dest_path))
        downloaded_count += 1

        if downloaded_count <= 5:
            print(f"  ✓ {file_name}")
    except Exception as e:
        print(f"  ⚠ Error downloading {file_name}: {e}")

print(f"  ✓ Downloaded {downloaded_count} files to {files_dir}")

# Download tracks using CLI command
print("\n📊 Downloading tracks...")
tracks_dir = output_dir / "tracks" / "robot" / "position"
tracks_dir.mkdir(parents=True, exist_ok=True)

import subprocess
try:
    subprocess.run([
        "ml-dash", "download", "--tracks",
        "tom_tao_e4c2c9/robot9/examples/tracking-data-collection1/robot/position",
        "-f", "json",
        "-o", str(tracks_dir / "tracks.json")
    ], check=True, capture_output=True)
    print(f"  ✓ Saved tracks to {tracks_dir / 'tracks.json'}")
except Exception as e:
    print(f"  ⚠ Error: {e}")

print(f"\n✅ Download complete!")
print(f"📂 Output directory: {output_dir.absolute()}")
