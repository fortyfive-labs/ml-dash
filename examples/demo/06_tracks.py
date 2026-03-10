"""
06 - Tracks: Time-stamped multi-modal sensor / trajectory data.

Tracks are designed for time-series data where each entry has a wall-clock
or simulation timestamp (_ts).  Typical use cases: robot joint states,
camera frames, LiDAR scans, or any other sensor data.

Covers:
  - tracks("topic").append()   – append a single entry with _ts
  - tracks("topic").flush()    – flush the topic buffer to storage
  - tracks.flush()             – flush all topics
  - tracks("topic").read()     – read back as JSON or mocap format
  - tracks("topic").slice()    – time-range slice
  - tracks("topic").list_entries() – iterate entries
"""

from ml_dash import Experiment

DASH_ROOT = "/tmp/ml-dash-demo"

exp = Experiment(prefix="alice/robotics/track-demo", dash_root=DASH_ROOT)
exp.run.start()

# ---------------------------------------------------------------------------
# 1. Append entries to named topics
#    _ts is the timestamp (float, seconds).  Omit _ts to use wall-clock time.
# ---------------------------------------------------------------------------
dt = 1.0 / 30  # 30 Hz

for i in range(60):
    t = i * dt

    # Robot joint positions
    exp.tracks("robot/joints").append(
        q=[0.1 * i, -0.2 * i, 0.05 * i, 0.0, 0.0, 0.0],  # joint angles
        dq=[0.01 * i, -0.02 * i, 0.0, 0.0, 0.0, 0.0],     # velocities
        _ts=t,
    )

    # End-effector pose
    exp.tracks("robot/eef").append(
        pos=[0.3 + 0.01 * i, 0.0, 0.5],
        quat=[0.0, 0.0, 0.0, 1.0],
        _ts=t,
    )

    # Camera frame metadata (every 10th frame)
    if i % 10 == 0:
        exp.tracks("camera/rgb").append(
            frame_id=i,
            path=f"frames/frame_{i:04d}.png",
            exposure_ms=8.3,
            _ts=t,
        )

# ---------------------------------------------------------------------------
# 2. Flush a specific topic
# ---------------------------------------------------------------------------
exp.tracks("robot/joints").flush()

# ---------------------------------------------------------------------------
# 3. Flush all topics at once
# ---------------------------------------------------------------------------
exp.tracks.flush()

# ---------------------------------------------------------------------------
# 4. Read back track data (JSON format)
# ---------------------------------------------------------------------------
data = exp.tracks("robot/joints").read(format="json")
print(f"robot/joints: {data['count']} entries")
if data["entries"]:
    print("  first entry:", data["entries"][0])

# ---------------------------------------------------------------------------
# 5. Read back as mocap format (structured for motion-capture pipelines)
# ---------------------------------------------------------------------------
mocap = exp.tracks("robot/joints").read(format="mocap")
print("mocap keys:", list(mocap.keys()))

# ---------------------------------------------------------------------------
# 6. Time-range slice
# ---------------------------------------------------------------------------
# Entries between t=0.5 s and t=1.0 s
sliced = exp.tracks("robot/joints").slice(start_timestamp=0.5, end_timestamp=1.0)
slice_entries = list(sliced)
print(f"slice(0.5, 1.0): {len(slice_entries)} entries")

# ---------------------------------------------------------------------------
# 7. List all entries as an iterable
# ---------------------------------------------------------------------------
entries = exp.tracks("robot/eef").list_entries()
print("eef entry count:", len(entries))

# ---------------------------------------------------------------------------
# 8. Multiple topics in one experiment
# ---------------------------------------------------------------------------
for i in range(5):
    exp.tracks("lidar/points").append(
        points=[[float(j) for j in range(10)] for _ in range(3)],
        intensity=[0.9, 0.8, 0.7],
        _ts=float(i),
    )

exp.tracks.flush()

exp.run.complete()
