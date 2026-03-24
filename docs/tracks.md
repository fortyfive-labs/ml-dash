# Track API

The Track API provides efficient time-series data tracking for robotics, reinforcement learning, and any sequential experiments.

## Overview

Tracks are perfect for:
- **Robotics**: Robot positions, joint angles, sensor readings
- **Reinforcement Learning**: Agent trajectories, rewards, states
- **Simulations**: Physics parameters over time
- **Sequential Data**: Any data that changes over time

## Basic Usage

```python
from ml_dash import Experiment

with Experiment("robotics/training").run as experiment:
    for step in range(1000):
        # Append data to a track
        experiment.track("robot/position").append({
            "step": step,
            "x": position[0],
            "y": position[1],
            "z": position[2]
        })
```

## Timestamp-Based Tracking

Tracks support three timestamp modes:

### 1. Auto-Generated Timestamps (Recommended)

When `_ts` is not provided, timestamps are automatically generated:

```python
with Experiment("my-project/exp").run as experiment:
    for i in range(100):
        # Auto-generate unique timestamps
        experiment.tracks("sensors/temperature").append(value=temperature)
        experiment.tracks("sensors/pressure").append(value=pressure)
```

### 2. Explicit Timestamps

Provide explicit timestamps when you need precise control:

```python
import time

with Experiment("my-project/exp").run as experiment:
    for i in range(100):
        timestamp = time.time()

        # Track with explicit timestamp
        experiment.tracks("sensors/temperature").append(value=temperature, _ts=timestamp)

        # Track with same timestamp (entries are merged)
        experiment.tracks("sensors/pressure").append(value=pressure, _ts=timestamp)
```

### 3. Timestamp Inheritance with `_ts=-1`

Use `_ts=-1` to inherit the last timestamp from the previous `tracks.append()` or `metrics.log()` call in the same thread. This is perfect for synchronizing multi-modal data:

```python
with Experiment("robotics/multi-modal").run as experiment:
    for step in range(1000):
        # First append - auto-generates timestamp
        experiment.tracks("robot/pose").append(position=[1.0, 2.0, 3.0])

        # Following appends - inherit the same timestamp
        experiment.tracks("camera/left").append(width=640, height=480, _ts=-1)
        experiment.tracks("camera/right").append(width=640, height=480, _ts=-1)
        experiment.tracks("robot/velocity").append(linear=[0.1, 0.0, 0.0], _ts=-1)
        experiment.tracks("sensors/lidar").append(ranges=[1.5, 2.0, 2.5], _ts=-1)

        # All 5 tracks now share the exact same timestamp!
```

`_ts=-1` also inherits across **metrics and tracks** — they share the same last timestamp per thread:

```python
# Set timestamp via track, inherit in metric
experiment.tracks("robot/pose").append(position=[1.0, 2.0, 3.0], _ts=100.0)
experiment.metrics("train").log(loss=0.5, _ts=-1)   # inherits 100.0

# Set timestamp via metric, inherit in track
experiment.metrics("train").log(loss=0.5)            # auto _ts, sets _last_timestamp
experiment.tracks("robot/pose").append(x=1.0, _ts=-1)  # inherits it
```

**Benefits of `_ts=-1`:**
- Cleaner code - no need to manually pass timestamps around
- Less error-prone - can't accidentally use wrong timestamp
- Perfect for robotics/ML multi-modal data (poses, images, sensors at same instant)
- Works across ALL tracks and metrics globally (per thread)

## Multiple Tracks

Track different aspects of your experiment:

```python
with Experiment("robotics/walker").run as experiment:
    for step in range(1000):
        # Robot position
        experiment.track("robot/position").append({
            "step": step,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2]
        })

        # Joint angles
        experiment.track("robot/joints").append({
            "step": step,
            "joint1": angles[0],
            "joint2": angles[1],
            "joint3": angles[2]
        })

        # Control signals
        experiment.track("robot/control").append({
            "step": step,
            "motor1": ctrl[0],
            "motor2": ctrl[1]
        })

        # Sensor readings
        experiment.track("robot/sensors").append({
            "step": step,
            "imu_x": imu[0],
            "imu_y": imu[1],
            "imu_z": imu[2],
            "force": force_sensor
        })
```

## Numpy Array Serialization

Tracks automatically serialize numpy arrays:

```python
import numpy as np

with Experiment("rl/training").run as experiment:
    for episode in range(100):
        state = env.reset()  # numpy array

        # Numpy arrays are automatically converted to lists
        experiment.track("agent/states").append({
            "episode": episode,
            "state": state,  # numpy array - auto-serialized!
            "reward": 0.0
        })
```

## MuJoCo Example

Perfect for tracking robot simulations:

```python
import mujoco
import numpy as np

with Experiment("robotics/mujoco-sim").run as experiment:
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    for i in range(1000):
        # Apply control
        data.ctrl[0] = 4.0 * np.sin(i * 0.03)
        data.ctrl[1] = 2.5 * np.sin(i * 0.05)

        # Step simulation
        mujoco.mj_step(model, data)

        if i % 4 == 0:
            # Get end effector position
            ee_pos = data.site_xpos[model.site('end_effector').id].copy()

            # Track position
            experiment.track("robot/position").append({
                "step": i,
                "x": ee_pos[0],
                "y": ee_pos[1],
                "z": ee_pos[2]
            })

            # Track control
            experiment.track("robot/control").append({
                "step": i,
                "ctrl_1": float(data.ctrl[0]),
                "ctrl_2": float(data.ctrl[1])
            })

            # Save frame aligned with track
            renderer.update_scene(data)
            pixels = renderer.render()
            experiment.files("frames").save_image(
                pixels,
                to=f"frame_{i:05d}.jpg"
            )
```

## Aligning Frames with Tracks

Use consistent step indices to align frames with track data:

```python
with Experiment("vision/tracking").run as experiment:
    for step in range(1000):
        # Track data
        experiment.track("agent/position").append({
            "step": step,
            "x": x,
            "y": y
        })

        # Save frame with same step index
        experiment.files("frames").save_image(
            frame,
            to=f"frame_{step:05d}.jpg"
        )

        # Now frames and tracks are aligned by step index!
```

## Reinforcement Learning Example

```python
import gymnasium as gym

with Experiment("rl/cartpole").run as experiment:
    env = gym.make("CartPole-v1")

    for episode in range(100):
        state, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track episode trajectory
            experiment.track("agent/trajectory").append({
                "episode": episode,
                "step": step,
                "state": state.tolist(),  # numpy to list
                "action": int(action),
                "reward": float(reward),
                "next_state": next_state.tolist()
            })

            state = next_state
            total_reward += reward
            step += 1

        # Track episode summary
        experiment.track("agent/episodes").append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": step
        })
```

## Buffering and Performance

Tracks use the background buffering system:

```python
with Experiment("high-freq/tracking").run as experiment:
    # All track appends are non-blocking
    for i in range(100000):
        experiment.track("sensor/data").append({
            "step": i,
            "value": sensor.read()
        })
        # Returns immediately! No blocking I/O

    # Automatically batched and uploaded in background
```

Configure track buffering:

```bash
export ML_DASH_TRACK_BATCH_SIZE=100  # Batch 100 track entries
export ML_DASH_FLUSH_INTERVAL=5.0    # Flush every 5 seconds
```

## Timestamp Merging

Entries with the same timestamp are automatically merged:

```python
# Using explicit timestamps
experiment.tracks("multi-sensor").append(temperature=25.0, _ts=12345.67)
experiment.tracks("multi-sensor").append(pressure=1013.0, _ts=12345.67)

# Result: Single entry with both fields
# {"timestamp": 12345.67, "temperature": 25.0, "pressure": 1013.0}
```

You can also use `_ts=-1` for merging:

```python
# First append sets the timestamp
experiment.tracks("multi-sensor").append(temperature=25.0, _ts=12345.67)

# Following appends inherit and merge
experiment.tracks("multi-sensor").append(pressure=1013.0, _ts=-1)
experiment.tracks("multi-sensor").append(humidity=60.0, _ts=-1)

# Result: Single entry with all three fields at timestamp 12345.67
```

## Best Practices

### 1. Use Consistent Indexing

```python
# Good: Use step index consistently
for step in range(1000):
    experiment.track("data").append({"step": step, "value": v})
    experiment.files("frames").save_image(frame, to=f"frame_{step:05d}.jpg")

# Bad: Inconsistent indexing makes alignment hard
for i in range(1000):
    experiment.track("data").append({"index": i * 2, "value": v})
    experiment.files("frames").save_image(frame, to=f"frame_{i}.jpg")
```

### 2. Include Step/Episode in Data

```python
# Good: Always include step/episode for easy querying
experiment.track("agent/rewards").append({
    "episode": episode,
    "step": step,
    "reward": reward
})

# Less useful: Missing context
experiment.track("agent/rewards").append({
    "reward": reward
})
```

### 3. Organize by Topic

```python
# Good: Clear topic hierarchy
experiment.track("robot/joints/arm").append(data)
experiment.track("robot/joints/leg").append(data)
experiment.track("robot/sensors/imu").append(data)

# Less clear: Flat structure
experiment.track("arm_data").append(data)
experiment.track("leg_data").append(data)
```

### 4. Zero-Pad Filenames for Alignment

```python
# Good: Files sort correctly
experiment.files("frames").save_image(frame, to=f"frame_{i:05d}.jpg")
# frame_00000.jpg, frame_00001.jpg, frame_00002.jpg

# Bad: Sorting breaks
experiment.files("frames").save_image(frame, to=f"frame_{i}.jpg")
# frame_1.jpg, frame_10.jpg, frame_2.jpg  (wrong order!)
```

### 5. Store Frame References in Tracks

```python
for step in range(1000):
    frame_filename = f"frame_{step:05d}.jpg"

    # Save frame
    experiment.files("frames").save_image(frame, to=frame_filename)

    # Reference frame in track
    experiment.track("robot/position").append({
        "step": step,
        "frame": frame_filename,  # Link to frame
        "x": x,
        "y": y,
        "z": z
    })
```

## Track Slicing and Iteration

The `slice()` method returns an iterable view of track data with timestamp-based indexing using floor matching.

### Basic Iteration

```python
with Experiment("robotics/analysis").run as experiment:
    # ... append data to tracks ...
    experiment.tracks.flush()

    # Create slice
    track_slice = experiment.tracks("robot/pose").slice()

    # Iterate through all entries
    for entry in track_slice:
        timestamp = entry["timestamp"]
        position = entry["position"]
        print(f"t={timestamp}: pos={position}")
```

### Timestamp Range Filtering

```python
# Slice specific time range
track_slice = experiment.tracks("robot/pose").slice(
    start_timestamp=10.0,
    end_timestamp=20.0
)

# Iterate through entries in range [10.0, 20.0]
for entry in track_slice:
    process(entry)
```

### Floor-Match Timestamp Queries

The slice object supports `findByTime()` with **floor matching**: returns the entry with the largest timestamp ≤ queried timestamp.

```python
# Create slice
track_slice = experiment.tracks("robot/pose").slice()

# If timestamps are [1.0, 3.0, 5.0, 7.0, 9.0]:

# Single timestamp queries
entry = track_slice.findByTime(5.5)  # Returns entry at timestamp 5.0 (floor match)
entry = track_slice.findByTime(5.0)  # Returns entry at timestamp 5.0 (exact match)
entry = track_slice.findByTime(6.9)  # Returns entry at timestamp 5.0 (floor match)
entry = track_slice.findByTime(7.0)  # Returns entry at timestamp 7.0 (exact match)

# Batch queries with list of timestamps
entries = track_slice.findByTime([1.0, 3.5, 7.0])  # Returns list of 3 entries
# entries[0] -> timestamp 1.0
# entries[1] -> timestamp 3.0 (floor match for 3.5)
# entries[2] -> timestamp 7.0

# Query before first timestamp raises KeyError
# track_slice.findByTime(0.5)  # KeyError: No entry found with timestamp <= 0.5

# Query after last timestamp returns last entry
entry = track_slice.findByTime(100.0)  # Returns entry at timestamp 9.0
```

### Practical Example: Robot Trajectory Analysis

```python
with Experiment("robotics/analysis").run as experiment:
    # ... data already logged ...

    # Analyze trajectory in specific time window
    trajectory = experiment.tracks("robot/pose").slice(0.0, 30.0)

    # Iterate through trajectory
    positions = []
    for entry in trajectory:
        positions.append(entry["position"])

    # Get robot state at specific times (single queries)
    state_at_5s = trajectory.findByTime(5.0)
    state_at_10s = trajectory.findByTime(10.0)
    state_at_15_5s = trajectory.findByTime(15.5)  # Floor match to closest ≤ 15.5

    # Or batch query multiple times at once
    states = trajectory.findByTime([5.0, 10.0, 15.5, 20.0, 25.0])
    for state in states:
        print(f"t={state['timestamp']}: pos={state['position']}")

    # Calculate metrics
    total_distance = calculate_distance(positions)
    print(f"Total distance: {total_distance:.2f}m")
    print(f"Number of waypoints: {len(trajectory)}")
```

### Synchronizing Multi-Modal Data

Combine slicing with timestamp inheritance for synchronized queries:

```python
with Experiment("robotics/sync").run as experiment:
    # Read multiple synchronized tracks
    pose_slice = experiment.tracks("robot/pose").slice(0.0, 10.0)
    camera_slice = experiment.tracks("camera/left").slice(0.0, 10.0)
    lidar_slice = experiment.tracks("sensors/lidar").slice(0.0, 10.0)

    # Query all tracks at same timestamp (floor match ensures sync)
    for pose_entry in pose_slice:
        t = pose_entry["timestamp"]
        camera_entry = camera_slice.findByTime(t)
        lidar_entry = lidar_slice.findByTime(t)

        # All entries at same timestamp (or floor-matched)
        process_multimodal(pose_entry, camera_entry, lidar_entry)
```

### Slice Features

**Iterator Protocol:**
- `for entry in slice: ...` - Iterate through entries
- `iter(slice)` - Get iterator
- Can iterate multiple times (data is cached)

**Timestamp Queries:**
- `slice.findByTime(timestamp)` - Get entry by timestamp (floor match)
- Optimized for sequential queries using internal index
- Raises `KeyError` if query timestamp is before first entry

**Length:**
- `len(slice)` - Get number of entries in slice

**Representation:**
- `repr(slice)` - Shows topic, start, and end timestamps

## API Reference

### `experiment.track(topic: str) -> TrackBuilder`

Returns a TrackBuilder for the specified topic.

**Parameters:**
- `topic` (str): Topic name (e.g., "robot/position")

**Returns:**
- TrackBuilder instance

### `TrackBuilder.append(data: Dict[str, Any] = None, *, timestamp: float = None) -> None`

Append a data entry to the track.

**Parameters:**
- `data` (dict, optional): Data fields to track
- `timestamp` (float, optional): Explicit timestamp (uses current time if not provided)

**Examples:**

```python
# With automatic timestamp
experiment.track("sensor/temp").append({"value": 25.0})

# With explicit timestamp
experiment.track("sensor/temp").append(
    timestamp=time.time(),
    data={"value": 25.0}
)

# Positional data argument
experiment.track("sensor/temp").append({
    "step": 100,
    "value": 25.0
})
```

### `TrackBuilder.slice(start_timestamp: float = None, end_timestamp: float = None) -> TrackSlice`

Create an iterable slice of track data with timestamp-based indexing.

**Parameters:**
- `start_timestamp` (float, optional): Start timestamp (inclusive)
- `end_timestamp` (float, optional): End timestamp (inclusive)

**Returns:**
- `TrackSlice` object supporting iteration and timestamp indexing

**Examples:**

```python
# Full track slice
track_slice = experiment.tracks("robot/pose").slice()

# Time range slice
track_slice = experiment.tracks("robot/pose").slice(
    start_timestamp=0.0,
    end_timestamp=10.0
)

# Iterate
for entry in track_slice:
    print(entry["timestamp"], entry["position"])

# Floor-match timestamp query
entry = track_slice.findByTime(5.5)  # Returns entry with largest timestamp <= 5.5

# Get length
count = len(track_slice)
```

## Comparison with Metrics

| Feature | Tracks | Metrics |
|---------|--------|---------|
| Use Case | Time-series data, trajectories | Training metrics, losses |
| Structure | Flexible dict per entry | Flat key-value pairs |
| Timestamp | `_ts` field, explicit or auto | `_ts` field, explicit or auto |
| `_ts=-1` inheritance | Yes, shared with metrics | Yes, shared with tracks |
| Merging | Timestamp-based merging | No merging |
| Best For | Robotics, RL, sensors | Training loss, accuracy |

## Example: Complete Robot Tracking

```python
import mujoco
import numpy as np
from ml_dash import Experiment

with Experiment(
    prefix="robotics/complete-tracking",
    tags=["mujoco", "tracking"],
    dash_url="http://localhost:3000"
).run as experiment:
    experiment.log("Starting robot tracking experiment")

    # Initialize MuJoCo
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    steps = 1000
    record_interval = 4

    for i in range(steps):
        # Control
        ctrl_1 = 4.0 * np.sin(i * 0.03)
        ctrl_2 = 2.5 * np.sin(i * 0.05)
        data.ctrl[0] = ctrl_1
        data.ctrl[1] = ctrl_2

        # Simulate
        mujoco.mj_step(model, data)

        # Record
        if i % record_interval == 0:
            # Render
            renderer.update_scene(data)
            pixels = renderer.render()

            # Get state
            ee_pos = data.site_xpos[model.site('end_effector').id].copy()

            # Save frame
            experiment.files("robot/frames").save_image(
                pixels,
                to=f"frame_{i:05d}.jpg",
                quality=85
            )

            # Track position
            experiment.track("robot/position").append({
                "step": i,
                "x": float(ee_pos[0]),
                "y": float(ee_pos[1]),
                "z": float(ee_pos[2])
            })

            # Track control
            experiment.track("robot/control").append({
                "step": i,
                "ctrl_1": float(ctrl_1),
                "ctrl_2": float(ctrl_2)
            })

            experiment.log(
                f"Step {i:3d}: pos=({ee_pos[0]:.4f}, "
                f"{ee_pos[1]:.4f}, {ee_pos[2]:.4f})"
            )

    experiment.log("Tracking complete!")
```

This creates a complete tracking dataset with aligned frames, positions, and control signals!
