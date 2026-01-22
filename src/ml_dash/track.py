"""
Track API - Timestamped multi-modal data logging for ML experiments.

Tracks are used for storing sparse timestamped data like robot trajectories,
camera poses, sensor readings, etc. Each track has a topic (e.g., "robot/position")
and stores entries with timestamps and arbitrary data fields.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .experiment import Experiment


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

        The _ts parameter is required for the timestamp. All other kwargs are data fields.

        Entries with the same _ts will be merged when flushed.

        Args:
            _ts: Timestamp (required)
            **kwargs: Data fields (e.g., q=[0.1, 0.2], e=[0.5, 0.6])

        Returns:
            Self for method chaining

        Raises:
            ValueError: If _ts is not provided

        Example:
            experiment.tracks("robot/position").append(
                q=[0.1, -0.22, 0.45],
                e=[0.5, 0.0, 0.6],
                a=[1.0, 0.0],
                v=[0.01, 0.02],
                _ts=2.0
            )
        """
        # Extract timestamp
        if '_ts' not in kwargs:
            raise ValueError("Timestamp '_ts' is required for track.append()")

        timestamp = kwargs.pop('_ts')

        # Validate timestamp
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            raise ValueError(f"Timestamp '_ts' must be numeric, got: {type(timestamp)}")

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
        if self._experiment.run._client:
            # Need experiment ID for remote mode
            if not self._experiment._experiment_id:
                raise ValueError("Experiment must be opened before reading tracks. Use 'with experiment.run:'")

            return self._experiment.run._client.get_track_data(
                experiment_id=self._experiment._experiment_id,
                topic=self._topic,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                columns=columns,
                format=format
            )

        # Local mode
        if self._experiment.run._storage:
            return self._experiment.run._storage.read_track_data(
                owner=self._experiment.run.owner,
                project=self._experiment.run.project,
                prefix=self._experiment.run._folder_path,
                topic=self._topic,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                columns=columns,
                format=format
            )

        raise ValueError("No client or storage configured for experiment")

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
