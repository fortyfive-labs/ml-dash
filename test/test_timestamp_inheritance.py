"""Test timestamp inheritance features (_ts=-1 and auto-generation)."""

import tempfile
from pathlib import Path

from ml_dash import Experiment


def test_auto_generated_timestamps():
    """Test that timestamps are auto-generated when not provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/auto-ts-test",
            dash_root=str(local_path)
        ).run as experiment:
            # Append without _ts (should auto-generate)
            experiment.tracks("metric").append(value=0.5, step=1)
            experiment.tracks("metric").append(value=0.6, step=2)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("metric").read(format="json")

        # Verify _ts was added
        entries = data.get("entries", [])
        assert len(entries) == 2
        assert "timestamp" in entries[0]
        assert "timestamp" in entries[1]

        # Verify timestamps are different (unique)
        ts1 = entries[0]["timestamp"]
        ts2 = entries[1]["timestamp"]
        assert ts1 != ts2, "Auto-generated timestamps should be unique"


def test_explicit_timestamps():
    """Test using explicit timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/explicit-ts-test",
            dash_root=str(local_path)
        ).run as experiment:
            # Append with explicit timestamps
            experiment.tracks("robot/position").append(q=[0.1, 0.2], _ts=1.0)
            experiment.tracks("robot/position").append(q=[0.2, 0.3], _ts=2.0)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("robot/position").read(format="json")

        # Verify explicit timestamps were used
        entries = data.get("entries", [])
        assert entries[0]["timestamp"] == 1.0
        assert entries[1]["timestamp"] == 2.0


def test_timestamp_inheritance_same_track():
    """Test timestamp inheritance with _ts=-1 (same track)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/inherit-ts-test",
            dash_root=str(local_path)
        ).run as experiment:
            # First append with explicit timestamp
            experiment.tracks("robot/state").append(q=[0.1, 0.2], _ts=1.5)

            # Second append inherits timestamp (will merge with first)
            experiment.tracks("robot/state").append(v=[0.01, 0.02], _ts=-1)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("robot/state").read(format="json")

        # Should be merged into single point with both fields
        entries = data.get("entries", [])
        assert len(entries) == 1, "Data with inherited _ts should merge"
        merged = entries[0]
        assert merged["timestamp"] == 1.5
        assert merged["q"] == [0.1, 0.2]
        assert merged["v"] == [0.01, 0.02]


def test_timestamp_inheritance_across_tracks():
    """Test timestamp inheritance with _ts=-1 across different tracks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/cross-track-ts-test",
            dash_root=str(local_path)
        ).run as experiment:
            # First append - auto-generates timestamp
            experiment.tracks("robot/pose").append(position=[1.0, 2.0, 3.0])

            # Second append on different track - inherits same timestamp
            experiment.tracks("camera/left/image").append(width=640, height=480, _ts=-1)

            # Third append on another track - also inherits same timestamp
            experiment.tracks("robot/velocity").append(linear=[0.1, 0.2, 0.3], _ts=-1)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back from all tracks
            pose_data = experiment.tracks("robot/pose").read(format="json")
            image_data = experiment.tracks("camera/left/image").read(format="json")
            velocity_data = experiment.tracks("robot/velocity").read(format="json")

        # All three tracks should have same timestamp
        pose_ts = pose_data["entries"][0]["timestamp"]
        image_ts = image_data["entries"][0]["timestamp"]
        velocity_ts = velocity_data["entries"][0]["timestamp"]

        assert pose_ts == image_ts == velocity_ts, \
            "All tracks with _ts=-1 should share the same timestamp"

        # Verify data is correct
        assert pose_data["entries"][0]["position"] == [1.0, 2.0, 3.0]
        assert image_data["entries"][0]["width"] == 640
        assert velocity_data["entries"][0]["linear"] == [0.1, 0.2, 0.3]


def test_timestamp_merging():
    """Test that data points with same _ts are merged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/merge-ts-test",
            dash_root=str(local_path)
        ).run as experiment:
            # Append multiple fields with same timestamp
            experiment.tracks("robot/state").append(q=[0.1, 0.2], _ts=1.0)
            experiment.tracks("robot/state").append(v=[0.01, 0.02], _ts=1.0)
            experiment.tracks("robot/state").append(e=[0.5, 0.6, 0.7], _ts=1.0)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("robot/state").read(format="json")

        # Should be merged into single data point
        entries = data.get("entries", [])
        assert len(entries) == 1, "Data points with same _ts should merge"

        merged_point = entries[0]
        assert merged_point["timestamp"] == 1.0
        assert merged_point["q"] == [0.1, 0.2]
        assert merged_point["v"] == [0.01, 0.02]
        assert merged_point["e"] == [0.5, 0.6, 0.7]


def test_timestamp_merging_with_inheritance():
    """Test merging when using _ts=-1 to inherit timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/merge-inherit-test",
            dash_root=str(local_path)
        ).run as experiment:
            # First point with explicit timestamp
            experiment.tracks("robot/state").append(q=[0.1, 0.2], _ts=2.0)

            # Subsequent points inherit and merge
            experiment.tracks("robot/state").append(v=[0.01, 0.02], _ts=-1)
            experiment.tracks("robot/state").append(e=[0.5, 0.6, 0.7], _ts=-1)

            # Different timestamp - new point
            experiment.tracks("robot/state").append(q=[0.3, 0.4], _ts=3.0)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("robot/state").read(format="json")

        # Should have 2 data points
        entries = data.get("entries", [])
        assert len(entries) == 2

        # First point has all merged fields
        point1 = entries[0]
        assert point1["timestamp"] == 2.0
        assert point1["q"] == [0.1, 0.2]
        assert point1["v"] == [0.01, 0.02]
        assert point1["e"] == [0.5, 0.6, 0.7]

        # Second point is separate
        point2 = entries[1]
        assert point2["timestamp"] == 3.0
        assert point2["q"] == [0.3, 0.4]


def test_method_chaining():
    """Test that append() returns self for method chaining."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/chain-test",
            dash_root=str(local_path)
        ).run as experiment:
            # Method chaining
            experiment.tracks("loss").append(value=0.5, epoch=1, _ts=1.0).append(value=0.4, epoch=2, _ts=2.0)

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("loss").read(format="json")

        # Should have 2 data points
        entries = data.get("entries", [])
        assert len(entries) == 2
        assert entries[0]["value"] == 0.5
        assert entries[1]["value"] == 0.4
