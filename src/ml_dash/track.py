"""
Track API - Timestamped multi-modal data logging for ML experiments.

Tracks are used for storing sparse timestamped data like robot trajectories,
camera poses, sensor readings, etc. Each track has a topic (e.g., "robot/position")
and stores entries with timestamps and arbitrary data fields.
"""

import bisect
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from .exceptions import ConfigurationError, ExperimentError, StorageError

if TYPE_CHECKING:
    from .experiment import Experiment


class TrackSlice:
    """
    Iterator for track data with timestamp-based queries using floor matching.

    Provides:
    - Iterator interface (__iter__, __next__)
    - Timestamp-based queries (findByTime) with floor match
    - Optimized sequential access using internal index

    Floor match: Returns the entry with the largest timestamp <= queried timestamp.

    Usage:
        # Create slice
        track_slice = experiment.tracks("robot/pose").slice(start_ts=0.0, end_ts=10.0)

        # Iterate through entries
        for entry in track_slice:
            print(entry["timestamp"], entry["position"])

        # Find entry at specific timestamp (floor match)
        entry = track_slice.findByTime(5.5)  # Returns entry with largest timestamp <= 5.5
    """

    def __init__(
        self,
        track_builder: 'TrackBuilder',
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ):
        """
        Initialize TrackSlice.

        Args:
            track_builder: Parent TrackBuilder instance
            start_timestamp: Optional start timestamp filter
            end_timestamp: Optional end timestamp filter
        """
        self._track_builder = track_builder
        self._start_timestamp = start_timestamp
        self._end_timestamp = end_timestamp
        self._data: Optional[List[Dict[str, Any]]] = None
        self._timestamps: Optional[List[float]] = None  # Sorted list for binary search
        self._iter_index = 0  # For iteration
        self._search_index = 0  # For optimized sequential findByTime queries

    def _load_data(self) -> None:
        """Lazy load data from track if not already loaded."""
        if self._data is None:
            data = self._track_builder.read(
                start_timestamp=self._start_timestamp,
                end_timestamp=self._end_timestamp,
                format="json"
            )
            self._data = data.get("entries", [])
            # Extract sorted timestamps for binary search
            self._timestamps = [entry["timestamp"] for entry in self._data]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return iterator for track entries."""
        self._load_data()
        self._iter_index = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next entry in iteration."""
        self._load_data()
        if self._iter_index >= len(self._data):
            raise StopIteration
        entry = self._data[self._iter_index]
        self._iter_index += 1
        return entry

    def findByTime(
        self,
        timestamp: Union[float, int, List[Union[float, int]]],
        neighborhood_size: int = 10
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Find entry by timestamp using floor match with optimized sequential search.

        Floor match: Returns the entry with the largest timestamp <= queried timestamp.

        Optimization: When queries are sequential (increasing timestamps), searches in
        the neighborhood of the last found index instead of doing full binary search.

        Args:
            timestamp: Query timestamp (single number or list of numbers)
            neighborhood_size: Number of entries to search around last index (default: 10)

        Returns:
            Entry dict (if single timestamp) or list of entry dicts (if list of timestamps)

        Raises:
            KeyError: If no entry found with timestamp <= query

        Example:
            # If data has timestamps [1.0, 3.0, 5.0, 7.0]
            track_slice.findByTime(5.5)  # Returns entry at timestamp 5.0
            track_slice.findByTime(3.0)  # Returns entry at timestamp 3.0
            track_slice.findByTime(0.5)  # Raises KeyError (no timestamp <= 0.5)

            # Batch queries
            entries = track_slice.findByTime([1.0, 3.5, 7.0])  # Returns list of 3 entries
        """
        # Handle list of timestamps
        if isinstance(timestamp, list):
            return [self._find_single_time(ts, neighborhood_size) for ts in timestamp]

        # Handle single timestamp
        return self._find_single_time(timestamp, neighborhood_size)

    def _find_single_time(
        self,
        timestamp: Union[float, int],
        neighborhood_size: int = 10
    ) -> Dict[str, Any]:
        """
        Find entry by single timestamp using floor match.

        Args:
            timestamp: Query timestamp
            neighborhood_size: Number of entries to search around last index

        Returns:
            Entry dict with largest timestamp <= query

        Raises:
            KeyError: If no entry found with timestamp <= query
        """
        self._load_data()

        if not self._timestamps:
            raise StorageError(f"No entries in track slice")

        # Fast path: exact match at current search index
        if (self._search_index < len(self._timestamps) and
                self._timestamps[self._search_index] == timestamp):
            return self._data[self._search_index]

        # Try optimized neighborhood search first (for sequential queries)
        if self._search_index < len(self._timestamps):
            # Define neighborhood bounds
            start_idx = max(0, self._search_index - neighborhood_size)
            end_idx = min(len(self._timestamps), self._search_index + neighborhood_size)

            # Search in neighborhood
            for i in range(start_idx, end_idx):
                if i + 1 < len(self._timestamps):
                    # Check if timestamp falls in [timestamps[i], timestamps[i+1])
                    if self._timestamps[i] <= timestamp < self._timestamps[i + 1]:
                        self._search_index = i
                        return self._data[i]
                else:
                    # Last entry - check if timestamp >= last timestamp
                    if timestamp >= self._timestamps[i]:
                        self._search_index = i
                        return self._data[i]

        # Fallback to binary search if neighborhood search failed
        # bisect_right returns index where timestamp would be inserted to maintain sorted order
        idx = bisect.bisect_right(self._timestamps, timestamp)

        # idx is where timestamp would be inserted, so idx-1 is the floor match
        if idx == 0:
            # All timestamps are > query timestamp
            raise StorageError(
                f"No entry found with timestamp <= {timestamp}. "
                f"Earliest timestamp: {self._timestamps[0]}"
            )

        # Update search index for next query and return entry
        self._search_index = idx - 1
        return self._data[self._search_index]

    def __len__(self) -> int:
        """Return number of entries in slice."""
        self._load_data()
        return len(self._data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrackSlice(topic='{self._track_builder._topic}', "
            f"start={self._start_timestamp}, end={self._end_timestamp})"
        )


class TracksManager:
    """
    Manager for track operations with support for global and per-topic flush.

    Usage:
        # Append to specific topic
        experiment.tracks("robot/position").append(q=[0.1, 0.2], _ts=1.0)

        # Flush all topics
        experiment.tracks.flush()

        # Flush specific topic
        experiment.tracks("robot/position").flush()
    """

    def __init__(self, experiment: 'Experiment'):
        """
        Initialize TracksManager.

        Args:
            experiment: Parent Experiment instance
        """
        self._experiment = experiment
        self._track_builders: Dict[str, 'TrackBuilder'] = {}  # Cache for TrackBuilder instances

    def __call__(self, topic: str) -> 'TrackBuilder':
        """
        Get TrackBuilder for a specific topic.

        Args:
            topic: Track topic (e.g., "robot/position", "camera/rgb")

        Returns:
            TrackBuilder instance for the topic

        Example:
            experiment.tracks("robot/position").append(x=1.0, y=2.0, _ts=0.5)
        """
        if topic not in self._track_builders:
            self._track_builders[topic] = TrackBuilder(self._experiment, topic, tracks_manager=self)

        return self._track_builders[topic]

    def flush(self) -> None:
        """
        Flush all topics to storage (remote or local).

        This will write all buffered track entries to the server/filesystem.

        Example:
            experiment.tracks.flush()
        """
        # Flush all topics via background buffer manager
        if self._experiment._buffer_manager:
            self._experiment._buffer_manager.flush_tracks()


class TrackBuilder:
    """
    Builder for track operations.

    Provides fluent API for appending timestamped data to tracks.

    Usage:
        # Append single entry
        experiment.tracks("robot/position").append(q=[0.1, 0.2], e=[0.5, 0.6], _ts=1.0)

        # Flush specific topic
        experiment.tracks("robot/position").flush()
    """

    def __init__(
        self,
        experiment: 'Experiment',
        topic: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tracks_manager: Optional['TracksManager'] = None
    ):
        """
        Initialize TrackBuilder.

        Args:
            experiment: Parent Experiment instance
            topic: Track topic (e.g., "robot/position")
            description: Optional track description
            tags: Optional tags for categorization
            metadata: Optional structured metadata (fps, units, etc.)
            tracks_manager: Parent TracksManager (for global flush)
        """
        self._experiment = experiment
        self._topic = topic
        self._description = description
        self._tags = tags
        self._metadata = metadata
        self._tracks_manager = tracks_manager

    def append(self, **kwargs) -> 'TrackBuilder':
        """
        Append a single timestamped entry to the track.

        The _ts parameter controls timestamping:
        - Not provided: Auto-generate timestamp using time.time()
        - _ts=-1: Inherit timestamp from previous append (across ALL tracks)
        - _ts=<number>: Use explicit timestamp (seconds since epoch)

        Entries with the same _ts will be merged when flushed.

        Args:
            _ts: Timestamp (optional). Use -1 to inherit from previous append.
            **kwargs: Data fields (e.g., q=[0.1, 0.2], e=[0.5, 0.6])

        Returns:
            Self for method chaining

        Raises:
            ValueError: If _ts is invalid or inheritance fails

        Example:
            # Auto-generate timestamp
            experiment.tracks("robot/position").append(q=[0.1, -0.22, 0.45])

            # Explicit timestamp
            experiment.tracks("robot/position").append(q=[0.1, -0.22, 0.45], _ts=2.0)

            # Inherit timestamp from previous append (across all tracks)
            experiment.tracks("robot/position").append(q=[0.1, -0.22, 0.45])
            experiment.tracks("camera/left").append(width=640, _ts=-1)  # Same timestamp!
        """
        # Extract and handle timestamp
        if '_ts' not in kwargs:
            # Auto-generate unique timestamp with collision avoidance (thread-safe)
            with self._experiment._track_timestamp_lock:
                timestamp = time.time()
                if timestamp <= self._experiment._track_last_auto_timestamp:
                    timestamp = self._experiment._track_last_auto_timestamp + 0.000001
                self._experiment._track_last_auto_timestamp = timestamp
        else:
            # Validate before popping so _ts remains in kwargs if invalid
            try:
                float(kwargs['_ts'])
            except (TypeError, ValueError):
                raise ConfigurationError(f"Timestamp '_ts' must be numeric, got: {type(kwargs['_ts'])}")

            timestamp = kwargs.pop('_ts')

            # Handle _ts=-1 (timestamp inheritance)
            if timestamp == -1:
                if self._experiment._last_timestamp is not None:
                    timestamp = self._experiment._last_timestamp
                else:
                    # No previous timestamp, auto-generate
                    timestamp = time.time()
                    if timestamp <= self._experiment._track_last_auto_timestamp:
                        timestamp = self._experiment._track_last_auto_timestamp + 0.000001
                    self._experiment._track_last_auto_timestamp = timestamp
            else:
                timestamp = float(timestamp)

        # Store global last timestamp (for _ts=-1 inheritance across tracks)
        self._experiment._last_timestamp = timestamp

        # Remaining kwargs are data fields
        data = kwargs

        # Write to experiment (will be buffered)
        self._experiment._write_track(self._topic, timestamp, data)

        return self

    def flush(self) -> 'TrackBuilder':
        """
        Flush this topic's buffered entries to storage.

        Example:
            experiment.tracks("robot/position").flush()

        Returns:
            Self for method chaining
        """
        if self._experiment._buffer_manager:
            self._experiment._buffer_manager.flush_track(self._topic)

        return self

    def read(
        self,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        columns: Optional[List[str]] = None,
        format: str = "json"
    ) -> Any:
        """
        Read track data with optional filtering.

        Args:
            start_timestamp: Optional start timestamp filter
            end_timestamp: Optional end timestamp filter
            columns: Optional list of columns to retrieve
            format: Export format ('json', 'jsonl', 'parquet', 'mocap')

        Returns:
            Track data in requested format

        Raises:
            ValueError: If experiment not opened or no client configured

        Example:
            # Get all data as JSON
            data = experiment.tracks("robot/position").read()

            # Get data in time range
            data = experiment.tracks("robot/position").read(
                start_timestamp=0.0,
                end_timestamp=10.0
            )

            # Export as JSONL
            jsonl_bytes = experiment.tracks("robot/position").read(format="jsonl")

            # Export as Parquet
            parquet_bytes = experiment.tracks("robot/position").read(format="parquet")

            # Export as Mocap JSON
            mocap_data = experiment.tracks("robot/position").read(format="mocap")
        """
        # Remote mode
        if self._experiment._client:
            # Need experiment ID for remote mode
            if not self._experiment._experiment_id:
                raise ExperimentError("Experiment must be opened before reading tracks. Use 'with experiment.run:'")

            return self._experiment._client.get_track_data(
                experiment_id=self._experiment._experiment_id,
                topic=self._topic,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                columns=columns,
                format=format
            )

        # Local mode
        if self._experiment._storage:
            return self._experiment._storage.read_track_data(
                owner=self._experiment.run.owner,
                project=self._experiment.run.project,
                prefix=self._experiment.run._folder_path,
                topic=self._topic,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                columns=columns,
                format=format
            )

        raise ExperimentError("No client or storage configured for experiment")

    def list_entries(self) -> List[Dict[str, Any]]:
        """
        List all entries in this track (for remote mode).

        Returns:
            List of entry dicts

        Example:
            entries = experiment.tracks("robot/position").list_entries()
        """
        # Just read with default JSON format
        result = self.read(format="json")

        if isinstance(result, dict) and "entries" in result:
            return result["entries"]

        return []

    def slice(
        self,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ) -> TrackSlice:
        """
        Create an iterable slice of track data with timestamp-based queries.

        The returned TrackSlice object supports:
        - Iteration: for entry in track.slice(...): ...
        - Timestamp queries: entry = track.slice(...).findByTime(timestamp)
        - Length: len(track.slice(...))

        Timestamp queries use floor matching: returns the entry with the
        largest timestamp <= queried timestamp.

        Args:
            start_timestamp: Optional start timestamp (inclusive)
            end_timestamp: Optional end timestamp (inclusive)

        Returns:
            TrackSlice iterator with timestamp query support

        Example:
            # Create slice
            track_slice = experiment.tracks("robot/pose").slice(0.0, 10.0)

            # Iterate through entries
            for entry in track_slice:
                print(entry["timestamp"], entry["position"])

            # Find entry at timestamp 5.5 (floor match)
            # If timestamps are [1.0, 3.0, 5.0, 7.0], returns entry at 5.0
            entry = track_slice.findByTime(5.5)

            # Find entry at exact timestamp
            entry = track_slice.findByTime(5.0)

            # Get number of entries
            count = len(track_slice)
        """
        return TrackSlice(self, start_timestamp, end_timestamp)
