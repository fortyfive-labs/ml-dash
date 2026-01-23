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

Tracks automatically handle timestamps:

```python
import time

with Experiment("my-project/exp").run as experiment:
    for i in range(100):
        timestamp = time.time()

        # Track with explicit timestamp
        experiment.track("sensors/temperature").append(
            timestamp=timestamp,
            data={"value": temperature}
        )

        # Track with same timestamp (entries are merged)
        experiment.track("sensors/pressure").append(
            timestamp=timestamp,
            data={"value": pressure}
        )
```

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
timestamp = 12345.67

# First append
experiment.track("multi-sensor").append(
    timestamp=timestamp,
    data={"temperature": 25.0}
)

# Second append with same timestamp - merged!
experiment.track("multi-sensor").append(
    timestamp=timestamp,
    data={"pressure": 1013.0}
)

# Result: Single entry with both fields
# {"timestamp": 12345.67, "temperature": 25.0, "pressure": 1013.0}
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

## Comparison with Metrics

| Feature | Tracks | Metrics |
|---------|--------|---------|
| Use Case | Time-series data, trajectories | Training metrics, losses |
| Structure | Flexible dict per entry | Flat key-value pairs |
| Timestamp | Explicit or auto | Implicit (step-based) |
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
