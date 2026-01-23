---
description: Timestamped multi-modal data logging for robotics, sensors, and trajectories
globs:
  - "**/*.py"
  - "**/robot*.py"
  - "**/trajectory*.py"
  - "**/sensor*.py"
  - "**/control*.py"
  - "**/mocap*.py"
keywords:
  - tracks
  - trajectory
  - trajectories
  - robot
  - robotics
  - sensor
  - sensors
  - timestamp
  - timestamped
  - motion capture
  - mocap
  - pose
  - position
  - velocity
  - joint
  - end effector
  - gripper
  - imu
  - sparse data
---

# ML-Dash Tracks API

Tracks are for storing sparse timestamped data like robot trajectories, camera poses, sensor readings, and multi-modal time series data.

## When to Use Tracks

- Robot trajectories and joint states
- Sensor readings at irregular intervals
- Motion capture data
- Camera poses and transforms
- Event-driven data with timestamps
- Any data that needs explicit timestamps (not index-based)

## Basic Usage

```python
from ml_dash import Experiment

with Experiment(prefix="alice/robotics/experiment").run as exp:
    # Timestamp (_ts) is required
    experiment.tracks("robot/position").append(
        q=[0.1, 0.2, 0.3],      # joint positions
        e=[0.5, 0.0, 0.6],      # end effector
        _ts=1.0                  # timestamp in seconds
    )
```

## Key Concepts

### Timestamps are Required

```python
# Valid
experiment.tracks("sensor").append(temp=25.5, _ts=0.0)

# Invalid - raises ValueError
experiment.tracks("sensor").append(temp=25.5)  # Missing _ts!
```

### Automatic Merging

Entries with the same timestamp are automatically merged:

```python
experiment.tracks("robot/state").append(q=[0.1, 0.2], _ts=1.0)
experiment.tracks("robot/state").append(v=[0.01, 0.02], _ts=1.0)
# Result: {timestamp: 1.0, q: [0.1, 0.2], v: [0.01, 0.02]}
```

### Topic Organization

Use "/" for hierarchical topic names:

```python
experiment.tracks("robot/joints")
experiment.tracks("robot/gripper")
experiment.tracks("camera/rgb")
experiment.tracks("sensors/imu")
```

---

## TracksManager API

Access via `experiment.tracks`:

```python
# Get TrackBuilder for a topic
builder = experiment.tracks("robot/position")

# Flush all topics to storage
experiment.tracks.flush()
```

---

## TrackBuilder API

### append(**kwargs)

Append a timestamped entry:

```python
experiment.tracks("robot/state").append(
    q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # joint positions
    qd=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # velocities
    gripper_pos=0.04,
    _ts=1.5  # timestamp (required)
)
```

### flush()

Flush specific topic:

```python
experiment.tracks("robot/position").flush()
```

### read()

Read track data with optional filtering:

```python
# Read all data
data = experiment.tracks("robot/position").read()

# Read time range
data = experiment.tracks("robot/position").read(
    start_timestamp=0.0,
    end_timestamp=10.0
)

# Read specific columns
data = experiment.tracks("robot/position").read(
    columns=["q", "e"]
)

# Export formats: json (default), jsonl, parquet, mocap
parquet_data = experiment.tracks("robot/position").read(format="parquet")
```

### list_entries()

Get entries as a list:

```python
entries = experiment.tracks("robot/position").list_entries()
for entry in entries:
    print(f"t={entry['timestamp']}: q={entry.get('q')}")
```

---

## Complete Examples

### Robot Trajectory Logging

```python
from ml_dash import Experiment

with Experiment(prefix="alice/robotics/pick-place").run as exp:
    exp.params.set(robot="franka_panda", task="pick_and_place")

    # Log trajectory at 100Hz
    dt = 0.01
    for i in range(1000):
        t = i * dt
        q = get_joint_positions()
        ee_pos = get_end_effector_position()

        experiment.tracks("robot/state").append(
            q=q.tolist(),
            ee_pos=ee_pos.tolist(),
            _ts=t
        )

    experiment.tracks.flush()
```

### Multi-Sensor Logging

```python
with Experiment(prefix="alice/sensors/calibration").run as exp:
    # IMU at 200Hz
    for i in range(2000):
        experiment.tracks("sensors/imu").append(
            accel=[ax, ay, az],
            gyro=[gx, gy, gz],
            _ts=i * 0.005
        )

    # Force sensor at 1kHz
    for i in range(10000):
        experiment.tracks("sensors/force").append(
            fx=fx, fy=fy, fz=fz,
            _ts=i * 0.001
        )

    experiment.tracks.flush()
```

### Time-Range Analysis

```python
# Read specific time window
contact_data = experiment.tracks("sensors/force").read(
    start_timestamp=5.0,
    end_timestamp=7.0
)

# Export to Parquet
parquet_data = experiment.tracks("robot/state").read(format="parquet")
with open("trajectory.parquet", "wb") as f:
    f.write(parquet_data)
```

---

## Storage Format

### Local Mode

```
.dash/owner/project/experiment/tracks/{topic_safe}/data.jsonl
```

Each line is a JSON entry:
```json
{"timestamp": 1.0, "q": [0.1, 0.2, 0.3], "e": [0.5, 0.0, 0.6]}
```

### Remote Mode

MongoDB with timestamp indexing, supporting efficient time-range queries.

---

## Quick Reference

```python
# Append (timestamp required)
experiment.tracks("topic").append(field=value, _ts=1.0)

# Flush specific topic
experiment.tracks("topic").flush()

# Flush all topics
experiment.tracks.flush()

# Read data
data = experiment.tracks("topic").read()
data = experiment.tracks("topic").read(
    start_timestamp=0.0,
    end_timestamp=10.0,
    columns=["field1"],
    format="json"  # json, jsonl, parquet, mocap
)

# Get entries as list
entries = experiment.tracks("topic").list_entries()
```
