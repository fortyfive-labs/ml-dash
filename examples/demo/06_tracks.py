"""
06 - Tracks: Time-stamped multi-modal sensor / trajectory data.

Tracks are designed for time-series data where each entry has a wall-clock
or simulation timestamp (_ts).  Typical use cases: robot joint states,
camera frames, LiDAR scans, or any other sensor data.

Covers:
  - tracks("topic").append()                    – append a single entry with _ts
  - tracks("topic").flush()                     – flush the topic buffer to storage
  - tracks.flush()                              – flush all topics
  - tracks("topic").read()                      – read back (JSON format)
  - tracks("topic").read(format="mocap")        – mocap export format
  - tracks("topic").read(format="jsonl")        – JSONL bytes
  - tracks("topic").read(format="parquet")      – Parquet bytes
  - tracks("topic").read(start_timestamp=, end_timestamp=)  – time range filter
  - tracks("topic").read(columns=[...])         – column projection
  - tracks("topic").slice()                     – iterable time-range slice
  - TrackSlice.findByTime()                     – floor-match lookup by timestamp
  - tracks("topic").list_entries()              – list all entries
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
# 5. Read with time-range filter and column projection
# ---------------------------------------------------------------------------
windowed = exp.tracks("robot/joints").read(
    start_timestamp=0.5,
    end_timestamp=1.0,
    format="json",
)
print(f"entries in [0.5, 1.0]: {windowed['count']}")

# Project only specific columns (field names)
projected = exp.tracks("robot/joints").read(
    columns=["timestamp", "q"],
    format="json",
)
if projected["entries"]:
    print("projected columns:", list(projected["entries"][0].keys()))

# ---------------------------------------------------------------------------
# 6. Other read formats
# ---------------------------------------------------------------------------
# Mocap format (structured for motion-capture pipelines)
mocap = exp.tracks("robot/joints").read(format="mocap")
print("mocap keys:", list(mocap.keys()))

# JSONL (bytes) — each line is a JSON record
jsonl_bytes = exp.tracks("robot/joints").read(format="jsonl")
print(f"JSONL bytes: {len(jsonl_bytes)} bytes")

# Parquet (bytes) — columnar binary format (requires pyarrow)
try:
    parquet_bytes = exp.tracks("robot/joints").read(format="parquet")
    print(f"Parquet bytes: {len(parquet_bytes)} bytes")
except Exception:
    pass  # pyarrow may not be installed

# ---------------------------------------------------------------------------
# 7. Time-range slice  (iterable)
# ---------------------------------------------------------------------------
sliced = exp.tracks("robot/joints").slice(start_timestamp=0.5, end_timestamp=1.0)
slice_entries = list(sliced)
print(f"slice(0.5, 1.0): {len(slice_entries)} entries")

# ---------------------------------------------------------------------------
# 8. TrackSlice.findByTime()  — floor-match lookup by timestamp
#    Returns the entry whose timestamp is the largest value <= query
# ---------------------------------------------------------------------------
full_slice = exp.tracks("robot/joints").slice()   # no filter = full dataset
if list(full_slice):
    full_slice2 = exp.tracks("robot/joints").slice()
    entry = full_slice2.findByTime(0.5)            # single query
    print("findByTime(0.5):", entry.get("timestamp"))

    full_slice3 = exp.tracks("robot/joints").slice()
    entries = full_slice3.findByTime([0.1, 0.5, 1.0])  # batch query
    print("findByTime batch timestamps:", [e.get("timestamp") for e in entries])

# ---------------------------------------------------------------------------
# 9. List all entries as a plain list
# ---------------------------------------------------------------------------
entries = exp.tracks("robot/eef").list_entries()
print("eef entry count:", len(entries))

# ---------------------------------------------------------------------------
# 10. Multiple topics in one experiment
# ---------------------------------------------------------------------------
for i in range(5):
    exp.tracks("lidar/points").append(
        points=[[float(j) for j in range(10)] for _ in range(3)],
        intensity=[0.9, 0.8, 0.7],
        _ts=float(i),
    )

exp.tracks.flush()

exp.run.complete()
