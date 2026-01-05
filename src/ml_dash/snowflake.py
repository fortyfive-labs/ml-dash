"""
Snowflake ID generator for ML-Dash.

Snowflake IDs are 64-bit unique identifiers with the following structure:
- 1 bit: unused (always 0)
- 41 bits: timestamp in milliseconds since custom epoch
- 10 bits: worker/machine ID (0-1023)
- 12 bits: sequence number (0-4095)

This provides:
- Unique IDs across distributed systems
- Time-sortable (newer IDs are larger)
- ~69 years of IDs from custom epoch
- Up to 4096 IDs per millisecond per worker
"""

import time
import threading
import os


class SnowflakeIDGenerator:
    """
    Thread-safe Snowflake ID generator.

    Based on Twitter's Snowflake algorithm.
    """

    # Custom epoch: 2024-01-01 00:00:00 UTC (in milliseconds)
    EPOCH = 1704067200000

    # Bit lengths
    TIMESTAMP_BITS = 41
    WORKER_BITS = 10
    SEQUENCE_BITS = 12

    # Max values
    MAX_WORKER_ID = (1 << WORKER_BITS) - 1  # 1023
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1  # 4095

    # Bit shifts
    TIMESTAMP_SHIFT = WORKER_BITS + SEQUENCE_BITS  # 22
    WORKER_SHIFT = SEQUENCE_BITS  # 12

    def __init__(self, worker_id: int = None):
        """
        Initialize Snowflake ID generator.

        Args:
            worker_id: Worker/machine ID (0-1023). If None, derived from process ID.
        """
        if worker_id is None:
            # Derive from process ID
            worker_id = os.getpid() & self.MAX_WORKER_ID

        if not 0 <= worker_id <= self.MAX_WORKER_ID:
            raise ValueError(f"worker_id must be between 0 and {self.MAX_WORKER_ID}")

        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

    def _current_millis(self) -> int:
        """Get current timestamp in milliseconds since custom epoch."""
        return int(time.time() * 1000) - self.EPOCH

    def _wait_next_millis(self, last_timestamp: int) -> int:
        """Wait until next millisecond."""
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp

    def generate(self) -> int:
        """
        Generate a new Snowflake ID.

        Returns:
            A unique 64-bit integer ID

        Raises:
            RuntimeError: If clock moves backwards
        """
        with self.lock:
            timestamp = self._current_millis()

            # Check for clock moving backwards
            if timestamp < self.last_timestamp:
                raise RuntimeError(
                    f"Clock moved backwards. Refusing to generate ID. "
                    f"Last: {self.last_timestamp}, Current: {timestamp}"
                )

            if timestamp == self.last_timestamp:
                # Same millisecond - increment sequence
                self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE
                if self.sequence == 0:
                    # Sequence overflow - wait for next millisecond
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                # New millisecond - reset sequence
                self.sequence = 0

            self.last_timestamp = timestamp

            # Construct the ID
            snowflake_id = (
                (timestamp << self.TIMESTAMP_SHIFT) |
                (self.worker_id << self.WORKER_SHIFT) |
                self.sequence
            )

            return snowflake_id

    def parse(self, snowflake_id: int) -> dict:
        """
        Parse a Snowflake ID into its components.

        Args:
            snowflake_id: The Snowflake ID to parse

        Returns:
            Dictionary with timestamp, worker_id, and sequence
        """
        timestamp = (snowflake_id >> self.TIMESTAMP_SHIFT) + self.EPOCH
        worker_id = (snowflake_id >> self.WORKER_SHIFT) & self.MAX_WORKER_ID
        sequence = snowflake_id & self.MAX_SEQUENCE

        return {
            "timestamp": timestamp,
            "timestamp_ms": timestamp,
            "worker_id": worker_id,
            "sequence": sequence,
        }


# Global singleton instance
_generator = None
_generator_lock = threading.Lock()


def get_generator() -> SnowflakeIDGenerator:
    """Get or create the global Snowflake ID generator instance."""
    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                _generator = SnowflakeIDGenerator()
    return _generator


def generate_id() -> int:
    """
    Generate a new Snowflake ID using the global generator.

    Returns:
        A unique 64-bit integer ID
    """
    return get_generator().generate()


def parse_id(snowflake_id: int) -> dict:
    """
    Parse a Snowflake ID into its components.

    Args:
        snowflake_id: The Snowflake ID to parse

    Returns:
        Dictionary with timestamp, worker_id, and sequence
    """
    return get_generator().parse(snowflake_id)
