"""Test combined timestamp inheritance + slicing features."""

import tempfile
from pathlib import Path

from ml_dash import Experiment


def test_timestamp_inheritance_with_slicing():
    """Test using timestamp inheritance with slicing for multi-modal analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="robotics/multimodal",
            dash_root=str(local_path)
        ).run as experiment:
            # Log 50 timesteps of synchronized multi-modal data
            for step in range(50):
                # Robot pose - auto-generates timestamp
                experiment.tracks("robot/pose").append(
                    step=step,
                    position=[1.0 + step * 0.1, 2.0, 3.0],
                    orientation=[0.0, 0.0, step * 0.01]
                )

                # Camera, velocity, and sensors inherit same timestamp
                experiment.tracks("camera/left").append(
                    step=step,
                    width=640,
                    height=480,
                    _ts=-1
                )

                experiment.tracks("robot/velocity").append(
                    step=step,
                    linear=[0.1, 0.0, 0.0],
                    angular=[0.0, 0.0, 0.05],
                    _ts=-1
                )

                experiment.tracks("sensors/lidar").append(
                    step=step,
                    ranges=[1.5 + step * 0.01, 2.0, 2.5],
                    _ts=-1
                )

            experiment.tracks.flush()

            # Create slices for analysis window [steps 10-30]
            # Since we don't know exact timestamps, slice entire range and filter by step
            pose_slice = experiment.tracks("robot/pose").slice()
            camera_slice = experiment.tracks("camera/left").slice()
            velocity_slice = experiment.tracks("robot/velocity").slice()
            lidar_slice = experiment.tracks("sensors/lidar").slice()

            # Extract all entries for verification
            pose_entries = [e for e in pose_slice if 10 <= e["step"] <= 30]
            camera_entries = [e for e in camera_slice if 10 <= e["step"] <= 30]
            velocity_entries = [e for e in velocity_slice if 10 <= e["step"] <= 30]
            lidar_entries = [e for e in lidar_slice if 10 <= e["step"] <= 30]

        # Should have 21 entries (steps 10-30 inclusive)
        assert len(pose_entries) == 21
        assert len(camera_entries) == 21
        assert len(velocity_entries) == 21
        assert len(lidar_entries) == 21

        # Verify synchronized timestamps for each step
        for i in range(21):
            step = 10 + i
            # All tracks at this step should have same timestamp
            pose_ts = pose_entries[i]["timestamp"]
            camera_ts = camera_entries[i]["timestamp"]
            velocity_ts = velocity_entries[i]["timestamp"]
            lidar_ts = lidar_entries[i]["timestamp"]

            assert pose_ts == camera_ts == velocity_ts == lidar_ts, \
                f"Step {step}: Timestamps not synchronized"

            # Verify step values
            assert pose_entries[i]["step"] == step
            assert camera_entries[i]["step"] == step
            assert velocity_entries[i]["step"] == step
            assert lidar_entries[i]["step"] == step


def test_floor_match_with_inherited_timestamps():
    """Test floor matching works correctly with inherited timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/floor-inherit",
            dash_root=str(local_path)
        ).run as experiment:
            # Create synchronized data at specific timestamps
            timestamps = [1.0, 3.0, 5.0, 7.0, 9.0]

            for ts in timestamps:
                # First track sets timestamp
                experiment.tracks("track_a").append(value_a=ts * 10, _ts=ts)

                # Second track inherits
                experiment.tracks("track_b").append(value_b=ts * 100, _ts=-1)

            experiment.tracks.flush()

            # Create slices
            slice_a = experiment.tracks("track_a").slice()
            slice_b = experiment.tracks("track_b").slice()

            # Test floor matching at various timestamps
            test_timestamps = [
                (1.0, 1.0),   # Exact
                (2.5, 1.0),   # Floor to 1.0
                (3.0, 3.0),   # Exact
                (4.0, 3.0),   # Floor to 3.0
                (5.5, 5.0),   # Floor to 5.0
                (8.0, 7.0),   # Floor to 7.0
                (10.0, 9.0),  # Floor to 9.0 (after last)
            ]

            for query_ts, expected_ts in test_timestamps:
                entry_a = slice_a.findByTime(query_ts)
                entry_b = slice_b.findByTime(query_ts)

                # Both should floor to same timestamp (synchronized)
                assert entry_a["timestamp"] == expected_ts
                assert entry_b["timestamp"] == expected_ts

                # Verify data values
                assert entry_a["value_a"] == expected_ts * 10
                assert entry_b["value_b"] == expected_ts * 100


def test_slice_iteration_with_timestamp_sync():
    """Test iterating through synchronized slices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/iter-sync",
            dash_root=str(local_path)
        ).run as experiment:
            # Create 100 synchronized entries
            for i in range(100):
                experiment.tracks("sensor_a").append(value=i, _ts=float(i))
                experiment.tracks("sensor_b").append(value=i * 2, _ts=-1)
                experiment.tracks("sensor_c").append(value=i * 3, _ts=-1)

            experiment.tracks.flush()

            # Create slices for middle portion
            slice_a = experiment.tracks("sensor_a").slice(25.0, 75.0)
            slice_b = experiment.tracks("sensor_b").slice(25.0, 75.0)
            slice_c = experiment.tracks("sensor_c").slice(25.0, 75.0)

            # Iterate and verify synchronization
            entries_a = list(slice_a)
            entries_b = list(slice_b)
            entries_c = list(slice_c)

        # Should have 51 entries (25-75 inclusive)
        assert len(entries_a) == 51
        assert len(entries_b) == 51
        assert len(entries_c) == 51

        # Verify all synchronized
        for i in range(51):
            step = 25 + i
            assert entries_a[i]["timestamp"] == float(step)
            assert entries_b[i]["timestamp"] == float(step)
            assert entries_c[i]["timestamp"] == float(step)

            assert entries_a[i]["value"] == step
            assert entries_b[i]["value"] == step * 2
            assert entries_c[i]["value"] == step * 3


if __name__ == "__main__":
    test_timestamp_inheritance_with_slicing()
    test_floor_match_with_inherited_timestamps()
    test_slice_iteration_with_timestamp_sync()
    print("✓ All combined feature tests passed!")
