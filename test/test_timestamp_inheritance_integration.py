"""Integration test for timestamp inheritance feature."""

import tempfile
from pathlib import Path

from ml_dash import Experiment


def test_end_to_end_robotics_example():
    """Test complete robotics example with timestamp inheritance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="robotics/walker",
            dash_root=str(local_path)
        ).run as experiment:
            # Simulate 10 timesteps of robot data
            for step in range(10):
                # First track - auto-generates timestamp
                experiment.tracks("robot/pose").append(
                    step=step,
                    position=[1.0 + step * 0.1, 2.0, 3.0]
                )

                # Other tracks inherit the same timestamp
                experiment.tracks("camera/left").append(
                    step=step,
                    width=640,
                    height=480,
                    _ts=-1
                )

                experiment.tracks("camera/right").append(
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
                    ranges=[1.5, 2.0, 2.5, 3.0],
                    _ts=-1
                )

            # Flush all tracks
            experiment.tracks.flush()

            # Read back all tracks
            pose_data = experiment.tracks("robot/pose").read(format="json")
            camera_left_data = experiment.tracks("camera/left").read(format="json")
            camera_right_data = experiment.tracks("camera/right").read(format="json")
            velocity_data = experiment.tracks("robot/velocity").read(format="json")
            lidar_data = experiment.tracks("sensors/lidar").read(format="json")

        # Verify all tracks have 10 entries
        assert len(pose_data["entries"]) == 10
        assert len(camera_left_data["entries"]) == 10
        assert len(camera_right_data["entries"]) == 10
        assert len(velocity_data["entries"]) == 10
        assert len(lidar_data["entries"]) == 10

        # Verify timestamps are synchronized for each step
        for i in range(10):
            pose_ts = pose_data["entries"][i]["timestamp"]
            left_ts = camera_left_data["entries"][i]["timestamp"]
            right_ts = camera_right_data["entries"][i]["timestamp"]
            vel_ts = velocity_data["entries"][i]["timestamp"]
            lidar_ts = lidar_data["entries"][i]["timestamp"]

            # All timestamps for this step should match
            assert pose_ts == left_ts == right_ts == vel_ts == lidar_ts, \
                f"Step {i}: Timestamps should be synchronized"

        # Verify data content
        assert pose_data["entries"][0]["position"] == [1.0, 2.0, 3.0]
        assert pose_data["entries"][9]["position"] == [1.9, 2.0, 3.0]
        assert camera_left_data["entries"][0]["width"] == 640
        assert velocity_data["entries"][0]["linear"] == [0.1, 0.0, 0.0]
        assert lidar_data["entries"][0]["ranges"] == [1.5, 2.0, 2.5, 3.0]


def test_no_timestamp_provided_generates_unique():
    """Test that omitting _ts generates unique timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/auto-gen",
            dash_root=str(local_path)
        ).run as experiment:
            # Append multiple entries without _ts
            for i in range(100):
                experiment.tracks("fast_sensor").append(value=i)

            experiment.tracks.flush()
            data = experiment.tracks("fast_sensor").read(format="json")

        # Should have 100 unique timestamps
        entries = data["entries"]
        assert len(entries) == 100

        timestamps = [e["timestamp"] for e in entries]
        unique_timestamps = set(timestamps)
        assert len(unique_timestamps) == 100, "All auto-generated timestamps should be unique"

        # Timestamps should be monotonically increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], \
                f"Timestamp at index {i} should be greater than previous"


if __name__ == "__main__":
    test_end_to_end_robotics_example()
    test_no_timestamp_provided_generates_unique()
    print("✓ All integration tests passed!")
