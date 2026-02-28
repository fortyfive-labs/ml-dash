"""Test TrackSlice functionality (iteration and timestamp indexing)."""

import tempfile
from pathlib import Path

import pytest

from ml_dash import Experiment


def test_slice_basic_iteration():
    """Test basic iteration through track slice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/slice-iter",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with known timestamps
            for i in range(10):
                experiment.tracks("sensor").append(
                    value=i * 10,
                    step=i,
                    _ts=float(i)
                )

            experiment.tracks.flush()

            # Create slice
            track_slice = experiment.tracks("sensor").slice()

            # Iterate and collect
            entries = list(track_slice)

        # Verify all entries
        assert len(entries) == 10
        for i, entry in enumerate(entries):
            assert entry["timestamp"] == float(i)
            assert entry["value"] == i * 10
            assert entry["step"] == i


def test_slice_timestamp_range():
    """Test slicing with timestamp range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/slice-range",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps 0.0 to 9.0
            for i in range(10):
                experiment.tracks("sensor").append(value=i, _ts=float(i))

            experiment.tracks.flush()

            # Slice range [3.0, 7.0]
            track_slice = experiment.tracks("sensor").slice(
                start_timestamp=3.0,
                end_timestamp=7.0
            )

            entries = list(track_slice)

        # Should have entries at timestamps 3, 4, 5, 6, 7
        assert len(entries) == 5
        assert entries[0]["timestamp"] == 3.0
        assert entries[-1]["timestamp"] == 7.0
        assert [e["value"] for e in entries] == [3, 4, 5, 6, 7]


def test_slice_floor_match_exact():
    """Test floor matching with exact timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/floor-exact",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps [1.0, 3.0, 5.0, 7.0, 9.0]
            for i in [1, 3, 5, 7, 9]:
                experiment.tracks("sensor").append(value=i * 10, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query exact timestamps
            entry = track_slice.findByTime(5.0)

        assert entry["timestamp"] == 5.0
        assert entry["value"] == 50


def test_slice_floor_match_interpolated():
    """Test floor matching with timestamps between entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/floor-interp",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps [1.0, 3.0, 5.0, 7.0, 9.0]
            for i in [1, 3, 5, 7, 9]:
                experiment.tracks("sensor").append(value=i * 10, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query timestamps between entries (floor match)
            entry_5_5 = track_slice.findByTime(5.5)  # Should return entry at 5.0
            entry_6_9 = track_slice.findByTime(6.9)  # Should return entry at 5.0
            entry_7_0 = track_slice.findByTime(7.0)  # Should return entry at 7.0
            entry_8_5 = track_slice.findByTime(8.5)  # Should return entry at 7.0

        # Floor match: return largest timestamp <= query
        assert entry_5_5["timestamp"] == 5.0
        assert entry_5_5["value"] == 50

        assert entry_6_9["timestamp"] == 5.0
        assert entry_6_9["value"] == 50

        assert entry_7_0["timestamp"] == 7.0
        assert entry_7_0["value"] == 70

        assert entry_8_5["timestamp"] == 7.0
        assert entry_8_5["value"] == 70


def test_slice_floor_match_before_first():
    """Test floor matching with timestamp before first entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/floor-before",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps [5.0, 10.0, 15.0]
            for i in [5, 10, 15]:
                experiment.tracks("sensor").append(value=i, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query timestamp before first entry
            with pytest.raises(KeyError, match="No entry found with timestamp"):
                _ = track_slice.findByTime(3.0)


def test_slice_floor_match_after_last():
    """Test floor matching with timestamp after last entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/floor-after",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps [5.0, 10.0, 15.0]
            for i in [5, 10, 15]:
                experiment.tracks("sensor").append(value=i, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query timestamp after last entry (should return last entry)
            entry = track_slice.findByTime(20.0)

        assert entry["timestamp"] == 15.0
        assert entry["value"] == 15


def test_slice_len():
    """Test len() on track slice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/slice-len",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with 100 entries
            for i in range(100):
                experiment.tracks("sensor").append(value=i, _ts=float(i))

            experiment.tracks.flush()

            # Full slice
            full_slice = experiment.tracks("sensor").slice()
            assert len(full_slice) == 100

            # Partial slice
            partial_slice = experiment.tracks("sensor").slice(10.0, 19.0)
            assert len(partial_slice) == 10


def test_slice_empty_track():
    """Test slicing an empty track."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/slice-empty",
            dash_root=str(local_path)
        ).run as experiment:
            # Don't add any data
            experiment.tracks.flush()

            track_slice = experiment.tracks("empty_sensor").slice()

            # Should iterate 0 times
            entries = list(track_slice)
            assert len(entries) == 0
            assert len(track_slice) == 0

            # Indexing should raise KeyError
            with pytest.raises(KeyError, match="No entries in track slice"):
                _ = track_slice.findByTime(5.0)


def test_slice_reuse():
    """Test reusing same slice object multiple times."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/slice-reuse",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track
            for i in range(10):
                experiment.tracks("sensor").append(value=i, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Iterate multiple times
            entries1 = list(track_slice)
            entries2 = list(track_slice)

            # Both should have same data
            assert len(entries1) == 10
            assert len(entries2) == 10

            # Query multiple times
            entry1 = track_slice.findByTime(5.5)
            entry2 = track_slice.findByTime(5.5)
            assert entry1 == entry2


def test_slice_repr():
    """Test string representation of TrackSlice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/slice-repr",
            dash_root=str(local_path)
        ).run as experiment:
            track_slice = experiment.tracks("robot/pose").slice(0.0, 10.0)
            repr_str = repr(track_slice)

            assert "TrackSlice" in repr_str
            assert "robot/pose" in repr_str
            assert "start=0.0" in repr_str
            assert "end=10.0" in repr_str


def test_slice_floor_match_edge_cases():
    """Test floor matching edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/floor-edges",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with specific timestamps
            timestamps = [1.0, 1.5, 2.0, 5.0, 10.0]
            for ts in timestamps:
                experiment.tracks("sensor").append(value=ts * 100, _ts=ts)

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Test various queries
            assert track_slice.findByTime(1.0)["timestamp"] == 1.0
            assert track_slice.findByTime(1.25)["timestamp"] == 1.0  # Floor to 1.0
            assert track_slice.findByTime(1.5)["timestamp"] == 1.5
            assert track_slice.findByTime(1.75)["timestamp"] == 1.5  # Floor to 1.5
            assert track_slice.findByTime(2.0)["timestamp"] == 2.0
            assert track_slice.findByTime(3.0)["timestamp"] == 2.0  # Floor to 2.0
            assert track_slice.findByTime(4.999)["timestamp"] == 2.0  # Floor to 2.0
            assert track_slice.findByTime(5.0)["timestamp"] == 5.0
            assert track_slice.findByTime(100.0)["timestamp"] == 10.0  # Floor to 10.0


def test_slice_batch_queries_with_list():
    """Test batch timestamp queries using list of timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/batch-queries",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps [1.0, 3.0, 5.0, 7.0, 9.0]
            for i in [1, 3, 5, 7, 9]:
                experiment.tracks("sensor").append(value=i * 10, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query multiple timestamps at once
            entries = track_slice.findByTime([1.0, 3.0, 5.0])

        # Should return list of 3 entries
        assert isinstance(entries, list)
        assert len(entries) == 3

        # Verify each entry
        assert entries[0]["timestamp"] == 1.0
        assert entries[0]["value"] == 10
        assert entries[1]["timestamp"] == 3.0
        assert entries[1]["value"] == 30
        assert entries[2]["timestamp"] == 5.0
        assert entries[2]["value"] == 50


def test_slice_batch_queries_with_floor_matching():
    """Test batch queries with floor matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/batch-floor",
            dash_root=str(local_path)
        ).run as experiment:
            # Create track with timestamps [1.0, 3.0, 5.0, 7.0, 9.0]
            for i in [1, 3, 5, 7, 9]:
                experiment.tracks("sensor").append(value=i * 10, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query with floor matching (between exact timestamps)
            entries = track_slice.findByTime([2.5, 4.0, 6.5, 8.0])

        # Should return list of 4 entries with floor-matched timestamps
        assert len(entries) == 4
        assert entries[0]["timestamp"] == 1.0  # Floor match for 2.5
        assert entries[0]["value"] == 10
        assert entries[1]["timestamp"] == 3.0  # Floor match for 4.0
        assert entries[1]["value"] == 30
        assert entries[2]["timestamp"] == 5.0  # Floor match for 6.5
        assert entries[2]["value"] == 50
        assert entries[3]["timestamp"] == 7.0  # Floor match for 8.0
        assert entries[3]["value"] == 70


def test_slice_batch_queries_empty_list():
    """Test batch queries with empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/batch-empty",
            dash_root=str(local_path)
        ).run as experiment:
            for i in [1, 3, 5]:
                experiment.tracks("sensor").append(value=i * 10, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query with empty list
            entries = track_slice.findByTime([])

        # Should return empty list
        assert isinstance(entries, list)
        assert len(entries) == 0


def test_slice_batch_queries_mixed_types():
    """Test batch queries with mixed int and float timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / ".dash"

        with Experiment(
            prefix="test/batch-mixed",
            dash_root=str(local_path)
        ).run as experiment:
            for i in [1, 3, 5, 7]:
                experiment.tracks("sensor").append(value=i * 10, _ts=float(i))

            experiment.tracks.flush()

            track_slice = experiment.tracks("sensor").slice()

            # Query with mixed int and float
            entries = track_slice.findByTime([1, 3.5, 5.0, 7])

        assert len(entries) == 4
        assert entries[0]["timestamp"] == 1.0
        assert entries[1]["timestamp"] == 3.0  # Floor match for 3.5
        assert entries[2]["timestamp"] == 5.0
        assert entries[3]["timestamp"] == 7.0
