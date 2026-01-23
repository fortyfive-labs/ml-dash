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
  - robot
  - robotics
  - sensor
  - timestamp
  - motion capture
  - mocap
  - pose
  - imu
  - joint
  - end effector
---

# ML-Dash Tracks API

Tracks store timestamped multi-modal data. The topic is an ID/prefix, and kwargs are the data fields logged together at each timestamp.

## Realistic Examples

### Robot State Logging

Log full robot state in a single call:

```python
from ml_dash import Experiment

with Experiment(prefix="alice/franka/pick-place").run as exp:
    for t in range(1000):
        ts = t * 0.01  # 100Hz

        exp.tracks("robot").append(
            q=robot.joint_positions.tolist(),      # [7] joint positions
            qd=robot.joint_velocities.tolist(),    # [7] joint velocities
            tau=robot.joint_torques.tolist(),      # [7] joint torques
            ee_pos=robot.ee_position.tolist(),     # [3] end effector xyz
            ee_quat=robot.ee_quaternion.tolist(),  # [4] end effector orientation
            gripper=robot.gripper_width,           # scalar
            _ts=ts
        )
```

### IMU Sensor Logging

Log all IMU channels together:

```python
with Experiment(prefix="alice/drone/flight-01").run as exp:
    while running:
        exp.tracks("imu").append(
            accel=[ax, ay, az],      # accelerometer [m/s²]
            gyro=[gx, gy, gz],       # gyroscope [rad/s]
            mag=[mx, my, mz],        # magnetometer [μT]
            temp=imu.temperature,    # temperature [°C]
            _ts=time.time()
        )
```

### Multi-Camera Setup

```python
with Experiment(prefix="alice/manipulation/demo").run as exp:
    for frame_id in range(num_frames):
        ts = frame_id / fps

        # Each camera is a separate track
        exp.tracks("cam/wrist").append(
            path=f"frames/wrist_{frame_id:06d}.png",
            intrinsics=K_wrist.tolist(),
            pose=T_wrist.tolist(),
            _ts=ts
        )

        exp.tracks("cam/overhead").append(
            path=f"frames/overhead_{frame_id:06d}.png",
            intrinsics=K_overhead.tolist(),
            pose=T_overhead.tolist(),
            _ts=ts
        )
```

### Force-Torque Sensor

```python
with Experiment(prefix="alice/assembly/insertion").run as exp:
    while inserting:
        exp.tracks("ft_sensor").append(
            force=[fx, fy, fz],      # [N]
            torque=[tx, ty, tz],     # [Nm]
            contact=is_contact,      # bool
            _ts=time.time()
        )
```

---

## API Reference

### tracks(topic)

Get a track builder for a topic ID:

```python
builder = experiment.tracks("robot")
builder = experiment.tracks("imu")
builder = experiment.tracks("cam/wrist")
```

### append(**kwargs, _ts=)

Log data fields at a timestamp. **`_ts` is required.**

```python
experiment.tracks("robot").append(
    q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    qd=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
    gripper=0.04,
    _ts=1.5
)
```

### flush()

Flush buffered data to storage:

```python
experiment.tracks("robot").flush()  # Flush one track
experiment.tracks.flush()           # Flush all tracks
```

### read()

Read track data:

```python
# All data
data = experiment.tracks("robot").read()

# Time range
data = experiment.tracks("robot").read(
    start_timestamp=5.0,
    end_timestamp=10.0
)

# Specific columns only
data = experiment.tracks("robot").read(columns=["q", "ee_pos"])

# Export formats: json, jsonl, parquet, mocap
parquet = experiment.tracks("robot").read(format="parquet")
```

---

## Key Behaviors

### Timestamp Required

```python
# Valid
experiment.tracks("imu").append(accel=[0, 0, 9.8], _ts=0.0)

# Invalid - raises ValueError
experiment.tracks("imu").append(accel=[0, 0, 9.8])  # Missing _ts!
```

### Same-Timestamp Merging

Multiple appends at same `_ts` merge into one entry:

```python
experiment.tracks("robot").append(q=[0.1, 0.2], _ts=1.0)
experiment.tracks("robot").append(gripper=0.04, _ts=1.0)
# Result: {_ts: 1.0, q: [0.1, 0.2], gripper: 0.04}
```

### Background Buffering

Data is buffered and flushed automatically:
- On configurable interval
- When batch size reached
- When experiment exits
- When `.flush()` called

### Method Chaining

```python
experiment.tracks("robot") \
    .append(q=[0.1], _ts=0.0) \
    .append(q=[0.2], _ts=0.1) \
    .flush()
```

---

## Complete Robotics Example

```python
from ml_dash import Experiment
import numpy as np

with Experiment(prefix="alice/franka/demo-001").run as exp:
    exp.params.set(
        robot="franka_panda",
        task="pick_and_place",
        control_freq=100
    )

    dt = 0.01
    for step in range(10000):
        ts = step * dt

        # Robot state - all fields in one call
        exp.tracks("robot").append(
            q=env.robot.q.tolist(),
            qd=env.robot.qd.tolist(),
            tau=env.robot.tau.tolist(),
            ee_pos=env.robot.ee_pos.tolist(),
            ee_quat=env.robot.ee_quat.tolist(),
            gripper_pos=env.robot.gripper_pos,
            _ts=ts
        )

        # Action commanded
        exp.tracks("action").append(
            target_q=action[:7].tolist(),
            target_gripper=action[7],
            _ts=ts
        )

        # Observations
        if step % 10 == 0:  # 10Hz camera
            exp.tracks("obs").append(
                rgb_path=f"rgb/{step:06d}.png",
                depth_path=f"depth/{step:06d}.png",
                _ts=ts
            )

    # Read back for analysis
    robot_data = exp.tracks("robot").read(
        start_timestamp=0.0,
        end_timestamp=5.0,
        columns=["q", "ee_pos"]
    )
```

---

## Tracks vs Metrics

| | Tracks | Metrics |
|---|---|---|
| **Purpose** | Timestamped multi-modal data | Training metrics |
| **Timestamp** | Required (`_ts=`) | Auto-indexed |
| **Use case** | Robot state, IMU, poses | Loss, accuracy |
| **Query** | Time-range | Index-range |

```python
# TRACKS: robotics data with explicit timestamps
experiment.tracks("robot").append(q=[0.1, 0.2], ee_pos=[0.5, 0.0, 0.3], _ts=1.5)

# METRICS: training metrics (auto-indexed)
experiment.metrics("train").log(loss=0.5, accuracy=0.92, epoch=10)
```
