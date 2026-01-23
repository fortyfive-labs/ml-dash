# Tracks API

Tracks are for storing sparse timestamped data like robot trajectories, camera poses, sensor readings, and other multi-modal time series data. Unlike metrics (which use index-based time series), tracks use explicit timestamps and support arbitrary data fields per entry.

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [TracksManager](#tracksmanager)
- [TrackBuilder](#trackbuilder)
- [Reading Track Data](#reading-track-data)
- [Export Formats](#export-formats)
- [Storage Format](#storage-format)
- [Complete Examples](#complete-examples)

---

## Overview

Tracks are designed for:

- **Sparse data**: Unlike metrics that log every step, tracks log data at irregular intervals
- **Timestamped entries**: Each entry requires an explicit `_ts` timestamp
- **Multi-modal data**: Store different data fields (positions, velocities, sensors) together
- **Automatic merging**: Entries with the same timestamp are automatically merged

### When to Use Tracks vs Metrics

| Use Tracks For | Use Metrics For |
|----------------|-----------------|
| Robot trajectories | Training loss per epoch |
| Sensor readings at irregular intervals | Accuracy at regular steps |
| Camera poses | Learning rate schedules |
| Motion capture data | Batch statistics |
| Event-driven data | Index-based time series |

---

## Basic Usage

```python
from ml_dash import Experiment

with Experiment(prefix="alice/robotics/manipulation").run as exp:
    # Log robot state with timestamp
    exp.tracks("robot/position").append(
        q=[0.1, 0.2, 0.3],      # joint positions
        e=[0.5, 0.0, 0.6],      # end effector position
        _ts=1.0                  # timestamp in seconds (required)
    )

    # Log additional data at same timestamp (will be merged)
    exp.tracks("robot/position").append(
        v=[0.01, 0.02, 0.03],   # velocities
        _ts=1.0                  # same timestamp
    )
    # Result: {timestamp: 1.0, q: [...], e: [...], v: [...]}

    # Log at different timestamp
    exp.tracks("robot/position").append(
        q=[0.2, 0.3, 0.4],
        e=[0.6, 0.1, 0.7],
        _ts=2.0
    )
```

### Timestamp Requirements

The `_ts` parameter is **required** for all track entries:

```python
# Valid - numeric timestamp
experiment.tracks("sensor").append(temp=25.5, _ts=0.0)
experiment.tracks("sensor").append(temp=26.0, _ts=0.5)
experiment.tracks("sensor").append(temp=26.5, _ts=1.0)

# Invalid - missing timestamp (raises ValueError)
experiment.tracks("sensor").append(temp=27.0)  # Error!

# Invalid - non-numeric timestamp (raises ValueError)
experiment.tracks("sensor").append(temp=27.0, _ts="1.0")  # Error!
```

---

## TracksManager

The `TracksManager` is accessed via `experiment.tracks` and manages all track topics.

### Getting a TrackBuilder

```python
# Get TrackBuilder for a specific topic
builder = experiment.tracks("robot/position")

# Topic names can use "/" for hierarchical organization
experiment.tracks("robot/joints")
experiment.tracks("robot/gripper")
experiment.tracks("camera/rgb")
experiment.tracks("camera/depth")
experiment.tracks("sensors/imu")
```

### Flushing All Topics

```python
# Flush all buffered track data to storage
experiment.tracks.flush()
```

---

## TrackBuilder

The `TrackBuilder` provides methods for appending and reading track data.

### append(**kwargs)

Append a single timestamped entry to the track.

```python
experiment.tracks("robot/state").append(
    q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # joint positions (6 DOF)
    qd=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # joint velocities
    tau=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # joint torques
    gripper_pos=0.04,  # gripper position
    gripper_force=10.5,  # gripper force
    _ts=1.5  # timestamp in seconds
)
```

**Parameters:**
- `_ts` (float, required): Timestamp for the entry
- `**kwargs`: Arbitrary data fields (lists, dicts, scalars)

**Returns:** `TrackBuilder` (for method chaining)

### flush()

Flush this topic's buffered entries to storage.

```python
# Flush specific topic only
experiment.tracks("robot/position").flush()
```

**Returns:** `TrackBuilder` (for method chaining)

### Method Chaining

```python
# Chain multiple appends
experiment.tracks("events").append(event="start", _ts=0.0).append(event="calibration", _ts=1.0).append(event="ready", _ts=2.0).flush()
```

---

## Reading Track Data

### read()

Read track data with optional filtering.

```python
# Read all data as JSON
data = experiment.tracks("robot/position").read()

# Read with time range filter
data = experiment.tracks("robot/position").read(
    start_timestamp=0.0,
    end_timestamp=10.0
)

# Read specific columns only
data = experiment.tracks("robot/position").read(
    columns=["q", "e"]
)

# Combine filters
data = experiment.tracks("robot/position").read(
    start_timestamp=5.0,
    end_timestamp=15.0,
    columns=["q", "qd"]
)
```

**Parameters:**
- `start_timestamp` (float, optional): Start time filter (inclusive)
- `end_timestamp` (float, optional): End time filter (inclusive)
- `columns` (List[str], optional): Specific columns to retrieve
- `format` (str, default="json"): Export format

**Returns:** Data in the requested format

### list_entries()

Get all entries as a list of dictionaries.

```python
entries = experiment.tracks("robot/position").list_entries()
for entry in entries:
    print(f"t={entry['timestamp']}: q={entry.get('q')}")
```

---

## Export Formats

Track data can be exported in multiple formats:

### JSON (default)

```python
data = experiment.tracks("robot/position").read(format="json")
# Returns: {"entries": [{"timestamp": 1.0, "q": [...], ...}, ...]}
```

### JSONL (JSON Lines)

```python
jsonl_bytes = experiment.tracks("robot/position").read(format="jsonl")
# Returns: bytes with one JSON object per line
```

### Parquet

```python
parquet_bytes = experiment.tracks("robot/position").read(format="parquet")
# Returns: bytes in Apache Parquet format
# Use with pandas: pd.read_parquet(io.BytesIO(parquet_bytes))
```

### Mocap (Motion Capture JSON)

```python
mocap_data = experiment.tracks("robot/position").read(format="mocap")
# Returns: Motion capture specific JSON format
```

---

## Storage Format

### Local Mode

Tracks are stored as JSONL files:

```
.dash/
└── owner/
    └── project/
        └── experiment/
            └── tracks/
                ├── robot_position/    # "/" replaced with "_"
                │   └── data.jsonl
                ├── camera_rgb/
                │   └── data.jsonl
                └── sensors_imu/
                    └── data.jsonl
```

Each line in `data.jsonl` contains one entry:

```json
{"timestamp": 1.0, "q": [0.1, 0.2, 0.3], "e": [0.5, 0.0, 0.6]}
{"timestamp": 2.0, "q": [0.2, 0.3, 0.4], "e": [0.6, 0.1, 0.7]}
```

### Remote Mode

- Entries are stored in MongoDB with timestamp indexing
- Large datasets may be archived to S3
- Supports efficient time-range queries

---

## Complete Examples

### Robot Trajectory Logging

```python
from ml_dash import Experiment
import numpy as np

with Experiment(prefix="alice/robotics/pick-place").run as exp:
    exp.params.set(
        robot="franka_panda",
        task="pick_and_place",
        controller="impedance"
    )

    # Log trajectory at 100Hz
    dt = 0.01
    for i in range(1000):
        t = i * dt

        # Get robot state (from your controller)
        q = get_joint_positions()
        qd = get_joint_velocities()
        ee_pos = get_end_effector_position()
        ee_quat = get_end_effector_orientation()

        # Log to track
        exp.tracks("robot/state").append(
            q=q.tolist(),
            qd=qd.tolist(),
            ee_pos=ee_pos.tolist(),
            ee_quat=ee_quat.tolist(),
            _ts=t
        )

        # Log gripper separately
        exp.tracks("robot/gripper").append(
            position=get_gripper_position(),
            force=get_gripper_force(),
            _ts=t
        )

    # Flush all track data
    exp.tracks.flush()

    # Export trajectory for analysis
    trajectory = exp.tracks("robot/state").read()
    exp.log(f"Logged {len(trajectory['entries'])} trajectory points")
```

### Multi-Sensor Logging

```python
from ml_dash import Experiment

with Experiment(prefix="alice/sensors/calibration").run as exp:
    # Log different sensors at different rates

    # IMU at 200Hz
    for i in range(2000):
        exp.tracks("sensors/imu").append(
            accel=[ax, ay, az],
            gyro=[gx, gy, gz],
            _ts=i * 0.005
        )

    # Camera at 30Hz
    for i in range(300):
        exp.tracks("sensors/camera").append(
            frame_id=i,
            exposure=0.01,
            _ts=i * 0.033
        )

    # Force sensor at 1kHz
    for i in range(10000):
        exp.tracks("sensors/force").append(
            fx=fx, fy=fy, fz=fz,
            tx=tx, ty=ty, tz=tz,
            _ts=i * 0.001
        )

    exp.tracks.flush()
```

### Time-Range Analysis

```python
from ml_dash import Experiment

# Resume experiment and analyze
exp = Experiment(prefix="alice/robotics/experiment-1")

with exp.run:
    # Read specific time window
    contact_data = exp.tracks("sensors/force").read(
        start_timestamp=5.0,
        end_timestamp=7.0
    )

    # Analyze contact forces during grasp
    forces = [e["fx"] for e in contact_data["entries"]]
    max_force = max(forces)
    exp.log(f"Max contact force: {max_force:.2f}N")

    # Export to Parquet for further analysis
    parquet_data = exp.tracks("robot/state").read(format="parquet")
    with open("trajectory.parquet", "wb") as f:
        f.write(parquet_data)
```

### Merging Data at Same Timestamp

```python
from ml_dash import Experiment

with Experiment(prefix="alice/fusion/experiment").run as exp:
    # Log from different sources at same timestamps
    # Entries are automatically merged

    for t in np.arange(0, 10, 0.1):
        # From position sensor
        experiment.tracks("fused").append(x=x, y=y, z=z, _ts=t)

        # From velocity estimator (same timestamp)
        experiment.tracks("fused").append(vx=vx, vy=vy, vz=vz, _ts=t)

        # From orientation filter (same timestamp)
        experiment.tracks("fused").append(qw=qw, qx=qx, qy=qy, qz=qz, _ts=t)

    # Result: each timestamp has all fields merged
    # {timestamp: 0.0, x: ..., y: ..., z: ..., vx: ..., vy: ..., vz: ..., qw: ..., ...}

    experiment.tracks.flush()
```

---

## API Summary

### TracksManager (`experiment.tracks`)

| Method | Description |
|--------|-------------|
| `tracks(topic)` | Get TrackBuilder for a topic |
| `tracks.flush()` | Flush all topics to storage |

### TrackBuilder (`experiment.tracks(topic)`)

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `append(**kwargs)` | `_ts` (required), data fields | `TrackBuilder` | Append timestamped entry |
| `flush()` | - | `TrackBuilder` | Flush this topic |
| `read(...)` | start_timestamp, end_timestamp, columns, format | varies | Read track data |
| `list_entries()` | - | `List[dict]` | Get all entries as list |

### Quick Reference

```python
# Append data (timestamp required)
experiment.tracks("topic").append(field=value, _ts=1.0)

# Flush specific topic
experiment.tracks("topic").flush()

# Flush all topics
experiment.tracks.flush()

# Read all data
data = experiment.tracks("topic").read()

# Read with filters
data = experiment.tracks("topic").read(
    start_timestamp=0.0,
    end_timestamp=10.0,
    columns=["field1", "field2"],
    format="json"  # or "jsonl", "parquet", "mocap"
)

# Get entries as list
entries = experiment.tracks("topic").list_entries()
```
