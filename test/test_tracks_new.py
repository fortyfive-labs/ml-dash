"""Comprehensive tests for tracks (timestamped multi-modal data) functionality in both local and remote modes."""

import getpass
import json
from pathlib import Path

import pytest


class TestBasicTracksLocal:
    """Tests for basic track operations in local mode."""

    def test_track_append_single_entry_local(self, local_experiment, tmp_proj):
        """Test appending single entry to a track in local mode."""
        with local_experiment("test-user/test/track-test-local").run as experiment:
            # Log robot position at timestamp 0.0
            experiment.tracks("robot/position").append(
                q=[0.1, -0.22, 0.45],
                e=[0.5, 0.0, 0.6],
                _ts=0.0
            )

            # Log another entry at timestamp 0.033
            experiment.tracks("robot/position").append(
                q=[0.2, -0.23, 0.46],
                e=[0.5, 0.1, 0.6],
                _ts=0.033
            )

        # Verify file exists
        track_dir = tmp_proj / getpass.getuser() / "test" / "track-test-local" / "tracks" / "robot_position"
        assert track_dir.exists()
        assert (track_dir / "data.jsonl").exists()
        assert (track_dir / "metadata.json").exists()

    def test_track_read_local(self, local_experiment, tmp_proj):
        """Test reading track data in local mode."""
        with local_experiment("test-user/test/track-read-local").run as experiment:
            # Write data
            for i in range(5):
                experiment.tracks("robot/position").append(
                    x=float(i),
                    y=float(i * 2),
                    z=float(i * 3),
                    _ts=float(i) * 0.033
                )

            # Read data (within same session, before auto-flush)
            experiment.flush()  # Force flush

            # Read data
            data = experiment.tracks("robot/position").read(format="json")

            # Verify
            assert data["count"] == 5
            assert len(data["entries"]) == 5
            assert data["entries"][0]["x"] == 0.0
            assert data["entries"][4]["x"] == 4.0

    def test_track_jsonl_roundtrip_local(self, local_experiment, tmp_proj):
        """Test JSONL round-trip: write → read → write should be identical."""
        with local_experiment("test-user/test/track-roundtrip").run as experiment:
            # Write test data
            test_entries = [
                {"timestamp": 0.0, "x": 1.0, "y": 2.0, "z": 3.0},
                {"timestamp": 0.033, "x": 1.1, "y": 2.1, "z": 3.1},
                {"timestamp": 0.066, "x": 1.2, "y": 2.2, "z": 3.2},
            ]

            for entry in test_entries:
                ts = entry.pop("timestamp")
                experiment.tracks("robot/position").append(_ts=ts, **entry)

            experiment.flush()

            # Read as JSONL
            jsonl_data = experiment.tracks("robot/position").read(format="jsonl")

        # Parse JSONL
        lines = jsonl_data.decode('utf-8').strip().split('\n')
        read_entries = [json.loads(line) for line in lines]

        # Verify data matches
        assert len(read_entries) == 3
        for i, entry in enumerate(read_entries):
            assert entry["timestamp"] == test_entries[i]["timestamp"]
            assert entry["x"] == test_entries[i]["x"]
            assert entry["y"] == test_entries[i]["y"]
            assert entry["z"] == test_entries[i]["z"]

    def test_track_timestamp_merge_local(self, local_experiment):
        """Test that entries with same timestamp are merged in local mode."""
        with local_experiment("test-user/test/track-merge-local").run as experiment:
            # Log different fields at same timestamp
            experiment.tracks("camera/data").append(frame_id=0, _ts=0.0)
            experiment.tracks("camera/data").append(path="frame_0.png", _ts=0.0)
            experiment.tracks("camera/data").append(camera={"pos": [0, 0, 1]}, _ts=0.0)

            experiment.flush()

            # Read back data
            data = experiment.tracks("camera/data").read(format="json")

            # Should have merged into single entry
            assert data["count"] == 1
            entry = data["entries"][0]
            assert entry["timestamp"] == 0.0
            assert entry["frame_id"] == 0
            assert entry["path"] == "frame_0.png"
            assert "camera.pos" in entry  # Nested dict flattened


class TestBasicTracks:
    """Tests for basic track operations."""

    @pytest.mark.remote
    def test_track_append_single_entry_remote(self, remote_experiment):
        """Test appending single entry to a track."""
        with remote_experiment("test-user/test/track-test-basic").run as experiment:
            # Log robot position at timestamp 0.0
            experiment.tracks("robot/position").append(
                q=[0.1, -0.22, 0.45],
                e=[0.5, 0.0, 0.6],
                _ts=0.0
            )

            # Log another entry at timestamp 0.033
            experiment.tracks("robot/position").append(
                q=[0.2, -0.23, 0.46],
                e=[0.5, 0.1, 0.6],
                _ts=0.033
            )

    @pytest.mark.remote
    def test_track_append_multiple_topics_remote(self, remote_experiment):
        """Test logging to multiple topics."""
        with remote_experiment("test-user/test/track-multi-topic").run as experiment:
            # Log to robot/position
            experiment.tracks("robot/position").append(x=1.0, y=2.0, z=3.0, _ts=0.0)

            # Log to camera/rgb
            experiment.tracks("camera/rgb").append(frame_id=0, path="frame_0.png", _ts=0.0)

            # Log to sensors/imu
            experiment.tracks("sensors/imu").append(ax=0.1, ay=0.2, az=9.8, _ts=0.0)

    @pytest.mark.remote
    def test_track_timestamp_validation_remote(self, remote_experiment):
        """Test that timestamp validation works."""
        with remote_experiment("test-user/test/track-validation").run as experiment:
            # Valid numeric timestamp
            experiment.tracks("robot/position").append(x=1.0, _ts=0.0)
            experiment.tracks("robot/position").append(x=2.0, _ts=1.5)

            # Missing timestamp should raise error
            with pytest.raises(ValueError, match="Timestamp '_ts' is required"):
                experiment.tracks("robot/position").append(x=3.0)

            # Invalid timestamp type should raise error
            with pytest.raises(ValueError, match="must be numeric"):
                experiment.tracks("robot/position").append(x=4.0, _ts="invalid")


class TestTimestampMerging:
    """Tests for timestamp-based entry merging."""

    @pytest.mark.remote
    def test_timestamp_merge_remote(self, remote_experiment):
        """Test that entries with same timestamp are merged."""
        with remote_experiment("test-user/test/track-merge").run as experiment:
            # Log different fields at same timestamp
            experiment.tracks("camera/data").append(frame_id=0, _ts=0.0)
            experiment.tracks("camera/data").append(path="frame_0.png", _ts=0.0)
            experiment.tracks("camera/data").append(camera={"pos": [0, 0, 1]}, _ts=0.0)

            # Manually flush to ensure merge happens
            experiment.tracks.flush()

            # Read back data
            data = experiment.tracks("camera/data").read(format="json")

            # Should have merged into single entry
            assert data["count"] == 1
            entry = data["entries"][0]
            assert entry["timestamp"] == 0.0
            assert entry["frame_id"] == 0
            assert entry["path"] == "frame_0.png"
            assert "camera.pos" in entry  # Nested dict flattened


class TestTrackFlushing:
    """Tests for track flushing operations."""

    @pytest.mark.remote
    def test_global_flush_remote(self, remote_experiment):
        """Test flushing all topics."""
        with remote_experiment("test-user/test/track-global-flush").run as experiment:
            # Log to multiple topics
            experiment.tracks("robot/position").append(x=1.0, _ts=0.0)
            experiment.tracks("camera/rgb").append(frame_id=0, _ts=0.0)

            # Global flush
            experiment.tracks.flush()

    @pytest.mark.remote
    def test_topic_specific_flush_remote(self, remote_experiment):
        """Test flushing specific topic."""
        with remote_experiment("test-user/test/track-topic-flush").run as experiment:
            # Log to multiple topics
            experiment.tracks("robot/position").append(x=1.0, _ts=0.0)
            experiment.tracks("camera/rgb").append(frame_id=0, _ts=0.0)

            # Flush only robot/position
            experiment.tracks("robot/position").flush()


class TestTrackRead:
    """Tests for reading track data."""

    @pytest.mark.remote
    def test_read_track_json_remote(self, remote_experiment):
        """Test reading track data in JSON format."""
        with remote_experiment("test-user/test/track-read-json").run as experiment:
            # Write data
            for i in range(5):
                experiment.tracks("robot/position").append(
                    x=float(i),
                    y=float(i * 2),
                    z=float(i * 3),
                    _ts=float(i) * 0.033
                )

            # Flush to ensure data is written
            experiment.tracks.flush()

            # Read data
            data = experiment.tracks("robot/position").read(format="json")

            # Verify
            assert data["count"] == 5
            assert len(data["entries"]) == 5
            assert data["entries"][0]["x"] == 0.0
            assert data["entries"][4]["x"] == 4.0

    @pytest.mark.remote
    def test_read_track_jsonl_remote(self, remote_experiment):
        """Test reading track data in JSONL format."""
        with remote_experiment("test-user/test/track-read-jsonl").run as experiment:
            # Write data
            for i in range(3):
                experiment.tracks("robot/position").append(x=float(i), _ts=float(i))

            experiment.tracks.flush()

            # Read as JSONL
            jsonl_data = experiment.tracks("robot/position").read(format="jsonl")

            # Verify it's bytes
            assert isinstance(jsonl_data, bytes)

            # Parse lines
            lines = jsonl_data.decode('utf-8').strip().split('\n')
            assert len(lines) == 3

            # Parse first entry
            first_entry = json.loads(lines[0])
            assert first_entry["timestamp"] == 0.0
            assert first_entry["x"] == 0.0

    @pytest.mark.remote
    def test_read_track_parquet_remote(self, remote_experiment):
        """Test reading track data in Parquet format."""
        with remote_experiment("test-user/test/track-read-parquet").run as experiment:
            # Write data
            for i in range(3):
                experiment.tracks("robot/position").append(x=float(i), _ts=float(i))

            experiment.tracks.flush()

            # Read as Parquet
            parquet_data = experiment.tracks("robot/position").read(format="parquet")

            # Verify it's bytes
            assert isinstance(parquet_data, bytes)
            assert len(parquet_data) > 0

    @pytest.mark.remote
    def test_read_track_mocap_remote(self, remote_experiment):
        """Test reading track data in Mocap format."""
        with remote_experiment("test-user/test/track-read-mocap").run as experiment:
            # Write data with metadata
            for i in range(3):
                experiment.tracks("robot/position").append(
                    x=float(i),
                    y=float(i * 2),
                    _ts=float(i) * 0.033
                )

            experiment.tracks.flush()

            # Read as Mocap JSON
            mocap_data = experiment.tracks("robot/position").read(format="mocap")

            # Verify structure
            assert isinstance(mocap_data, dict)
            assert mocap_data["version"] == "1.0"
            assert "metadata" in mocap_data
            assert "channels" in mocap_data
            assert "frames" in mocap_data

            # Verify channels
            assert "x" in mocap_data["channels"]
            assert "y" in mocap_data["channels"]

            # Verify frames
            assert len(mocap_data["frames"]) == 3
            assert mocap_data["frames"][0]["time"] == 0.0


class TestTrackFiltering:
    """Tests for filtering track data."""

    @pytest.mark.remote
    def test_read_with_timestamp_filter_remote(self, remote_experiment):
        """Test reading track data with timestamp filter."""
        with remote_experiment("test-user/test/track-filter-time").run as experiment:
            # Write data across time range
            for i in range(10):
                experiment.tracks("robot/position").append(x=float(i), _ts=float(i) * 0.1)

            experiment.tracks.flush()

            # Read subset by timestamp
            data = experiment.tracks("robot/position").read(
                start_timestamp=0.2,
                end_timestamp=0.5,
                format="json"
            )

            # Should only get timestamps 0.2, 0.3, 0.4, 0.5
            assert data["count"] == 4
            assert data["entries"][0]["timestamp"] == 0.2
            assert data["entries"][-1]["timestamp"] == 0.5

    @pytest.mark.remote
    def test_read_with_column_filter_remote(self, remote_experiment):
        """Test reading track data with column filter."""
        with remote_experiment("test-user/test/track-filter-cols").run as experiment:
            # Write data with multiple columns
            for i in range(3):
                experiment.tracks("robot/position").append(
                    x=float(i),
                    y=float(i * 2),
                    z=float(i * 3),
                    _ts=float(i)
                )

            experiment.tracks.flush()

            # Read only x and y columns
            data = experiment.tracks("robot/position").read(
                columns=["x", "y"],
                format="json"
            )

            # Verify only requested columns
            assert data["count"] == 3
            entry = data["entries"][0]
            assert "x" in entry
            assert "y" in entry
            assert "z" not in entry


class TestNestedData:
    """Tests for nested data structures."""

    @pytest.mark.remote
    def test_nested_dict_flattening_remote(self, remote_experiment):
        """Test that nested dicts are flattened with dot notation."""
        with remote_experiment("test-user/test/track-nested").run as experiment:
            # Log nested structure
            experiment.tracks("robot/state").append(
                camera={"pos": [0, 0, 1], "rot": [0, 0, 0, 1]},
                body={"joints": [0.1, 0.2, 0.3]},
                _ts=0.0
            )

            experiment.tracks.flush()

            # Read data
            data = experiment.tracks("robot/state").read(format="json")

            # Verify flattening
            entry = data["entries"][0]
            assert "camera.pos" in entry
            assert "camera.rot" in entry
            assert "body.joints" in entry
            assert entry["camera.pos"] == [0, 0, 1]


class TestSparseData:
    """Tests for sparse column handling."""

    @pytest.mark.remote
    def test_sparse_columns_remote(self, remote_experiment):
        """Test that sparse columns (appearing at different timestamps) work."""
        with remote_experiment("test-user/test/track-sparse").run as experiment:
            # Entry 1: only x, y
            experiment.tracks("robot/position").append(x=1.0, y=2.0, _ts=0.0)

            # Entry 2: only x, z (different columns)
            experiment.tracks("robot/position").append(x=1.5, z=3.0, _ts=0.033)

            # Entry 3: all columns
            experiment.tracks("robot/position").append(x=2.0, y=3.0, z=4.0, _ts=0.066)

            experiment.tracks.flush()

            # Read data
            data = experiment.tracks("robot/position").read(format="json")

            # Verify sparse handling
            assert data["count"] == 3

            # Entry 1 has x, y but not z
            assert data["entries"][0]["x"] == 1.0
            assert data["entries"][0]["y"] == 2.0
            assert "z" not in data["entries"][0]

            # Entry 2 has x, z but not y
            assert data["entries"][1]["x"] == 1.5
            assert data["entries"][1]["z"] == 3.0
            assert "y" not in data["entries"][1]

            # Entry 3 has all
            assert data["entries"][2]["x"] == 2.0
            assert data["entries"][2]["y"] == 3.0
            assert data["entries"][2]["z"] == 4.0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.remote
    def test_empty_track_remote(self, remote_experiment):
        """Test reading from non-existent track."""
        with remote_experiment("test-user/test/track-empty").run as experiment:
            # Try to read from track that doesn't exist
            with pytest.raises(Exception):  # Should raise error
                experiment.tracks("nonexistent/topic").read(format="json")

    @pytest.mark.remote
    def test_track_with_special_characters_remote(self, remote_experiment):
        """Test tracks with special characters in data."""
        with remote_experiment("test-user/test/track-special").run as experiment:
            # Log data with special characters
            experiment.tracks("robot/position").append(
                status="running✓",
                message="Hello 世界",
                _ts=0.0
            )

            experiment.tracks.flush()

            # Read back
            data = experiment.tracks("robot/position").read(format="json")
            entry = data["entries"][0]
            assert entry["status"] == "running✓"
            assert entry["message"] == "Hello 世界"

    @pytest.mark.remote
    def test_track_with_null_values_remote(self, remote_experiment):
        """Test tracks with None/null values."""
        with remote_experiment("test-user/test/track-null").run as experiment:
            experiment.tracks("robot/position").append(x=1.0, y=None, z=3.0, _ts=0.0)

            experiment.tracks.flush()

            data = experiment.tracks("robot/position").read(format="json")
            entry = data["entries"][0]
            assert entry["x"] == 1.0
            assert entry["y"] is None
            assert entry["z"] == 3.0


class TestBuffering:
    """Tests for background buffering."""

    @pytest.mark.remote
    def test_auto_buffering_remote(self, remote_experiment):
        """Test that buffering happens automatically."""
        with remote_experiment("test-user/test/track-buffer").run as experiment:
            # Log many entries without manual flush
            for i in range(50):
                experiment.tracks("robot/position").append(x=float(i), _ts=float(i) * 0.01)

        # Auto-flush should happen on context exit
        # Verify by reading (requires new session)
        with remote_experiment("test-user/test/track-buffer").run as experiment:
            # Note: This will create a new experiment, so this test is more conceptual
            pass


if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys

    sys.exit(pytest.main([__file__, "-v", "-m", "remote"]))
