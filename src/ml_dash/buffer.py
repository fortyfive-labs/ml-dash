"""
Background buffering system for ML-Dash time-series resources.

Provides non-blocking writes for logs, metrics, and files by batching
operations in a background thread.
"""

import errno
import logging
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional


from .exceptions import NetworkError, StorageError
from .utils import _serialize_value

_logger = logging.getLogger("ml_dash.buffer")


class BufferConfig:
    """Configuration for buffering behavior."""

    # Internal constants for queue management (not exposed to users)
    _MAX_QUEUE_SIZE = 100_000  # Maximum items before blocking
    _WARNING_THRESHOLD = int(_MAX_QUEUE_SIZE * 0.80)  # Warn at 80% capacity
    _AGGRESSIVE_FLUSH_THRESHOLD = int(_MAX_QUEUE_SIZE * 0.50)  # Trigger immediate flush at 50% capacity

    def __init__(
        self,
        flush_interval: float = 5.0,
        log_batch_size: int = 100,
        metric_batch_size: int = 1000,
        track_batch_size: int = 100,
        file_upload_workers: int = 4,
        buffer_enabled: bool = True,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        """
        Initialize buffer configuration.

        Args:
            flush_interval: Time-based flush interval in seconds (default: 5.0)
            log_batch_size: Max logs per batch (default: 100)
            metric_batch_size: Max metric points per batch (default: 1000)
            track_batch_size: Max track entries per batch (default: 100)
            file_upload_workers: Number of parallel file upload threads (default: 4)
            buffer_enabled: Enable/disable buffering (default: True)
            max_retries: Max retry attempts on transient network errors (default: 3)
            retry_base_delay: Initial retry delay in seconds; doubles each attempt (default: 1.0)
            retry_max_delay: Cap on retry delay in seconds (default: 30.0)
        """
        self.flush_interval = flush_interval
        self.log_batch_size = log_batch_size
        self.metric_batch_size = metric_batch_size
        self.track_batch_size = track_batch_size
        self.file_upload_workers = file_upload_workers
        self.buffer_enabled = buffer_enabled
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    @classmethod
    def from_env(cls) -> "BufferConfig":
        """Create configuration from environment variables."""
        return cls(
            flush_interval=float(os.environ.get("ML_DASH_FLUSH_INTERVAL", "5.0")),
            log_batch_size=int(os.environ.get("ML_DASH_LOG_BATCH_SIZE", "100")),
            metric_batch_size=int(os.environ.get("ML_DASH_METRIC_BATCH_SIZE", "1000")),
            track_batch_size=int(os.environ.get("ML_DASH_TRACK_BATCH_SIZE", "100")),
            file_upload_workers=int(
                os.environ.get("ML_DASH_FILE_UPLOAD_WORKERS", "4")
            ),
            buffer_enabled=os.environ.get("ML_DASH_BUFFER_ENABLED", "true").lower()
            in ("true", "1", "yes"),
            max_retries=int(os.environ.get("ML_DASH_MAX_RETRIES", "3")),
            retry_base_delay=float(os.environ.get("ML_DASH_RETRY_BASE_DELAY", "1.0")),
            retry_max_delay=float(os.environ.get("ML_DASH_RETRY_MAX_DELAY", "30.0")),
        )


class BackgroundBufferManager:
    """Unified buffer manager with background flushing thread."""

    def __init__(self, experiment: "Experiment", config: BufferConfig):
        """
        Initialize background buffer manager.

        Args:
            experiment: Parent experiment instance
            config: Buffer configuration
        """
        self._experiment = experiment
        self._config = config

        # Resource-specific queues with bounded size to prevent OOM
        self._log_queue: Queue = Queue(maxsize=config._MAX_QUEUE_SIZE)
        self._metric_queues: Dict[Optional[str], Queue] = {}  # Per-metric queues
        self._track_buffers: Dict[str, Dict[float, Dict[str, Any]]] = {}  # Per-topic: {timestamp: merged_data}
        self._file_queue: Queue = Queue(maxsize=config._MAX_QUEUE_SIZE)

        # Track last flush times per resource type
        self._last_log_flush = time.time()
        self._last_metric_flush: Dict[Optional[str], float] = {}
        self._last_track_flush: Dict[str, float] = {}  # Per-topic flush times

        # Track warnings to avoid spamming
        self._warned_queues: set = set()

        # Background thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()  # Manual flush trigger

        # One flush at a time: exp.flush() and the background thread share this state
        self._flush_lock = threading.Lock()
        # Guards _track_buffers, which the caller and the flush thread both mutate
        self._track_lock = threading.Lock()

        # A batch whose send failed is kept here, not re-queued, until it is sent.
        # Producers filling the queue therefore cannot push it out.
        self._failed_log_batch: Optional[List[Dict[str, Any]]] = None
        self._failed_metric_batches: Dict[Optional[str], Dict[str, Any]] = {}

        # Failure bookkeeping, keyed by a label such as "logs" or "metric 'train'"
        self._failed_at: Dict[str, float] = {}  # paces background retries
        self._retry_after_until: Dict[str, float] = {}  # server Retry-After deadline
        self._pending_errors: Dict[str, str] = {}  # cleared once the data is sent
        self._lost_errors: Dict[str, str] = {}  # data dropped; reported at next flush/stop

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        """Return True if the exception looks like a transient network failure worth retrying."""
        # OSError / IOError — check errno
        if isinstance(e, OSError):
            transient_errnos = {
                errno.ENETUNREACH,   # 101 — Network unreachable
                errno.ECONNREFUSED,  # 111 — Connection refused
                errno.ETIMEDOUT,     # 110 — Connection timed out
                errno.ECONNRESET,    # 104 — Connection reset by peer
                -3,                  # EAI_AGAIN — Temporary failure in name resolution
            }
            if e.errno in transient_errnos:
                return True
        # httpx errors — transport failures, rate limiting (429) and server-side 5xx responses
        try:
            import httpx
            if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
                return True
            if isinstance(e, httpx.HTTPStatusError) and (
                e.response.status_code >= 500 or e.response.status_code == 429
            ):
                return True
        except ImportError:
            pass
        # Recurse into chained cause (e.g. NetworkError wrapping OSError)
        if e.__cause__ is not None and e.__cause__ is not e:
            return BackgroundBufferManager._is_transient_error(e.__cause__)
        return False

    @staticmethod
    def _retry_after_seconds(e: Exception) -> Optional[float]:
        """Seconds the server asked us to wait (Retry-After), or None if absent or invalid."""
        response = getattr(e, "response", None)
        if response is None:
            if e.__cause__ is not None and e.__cause__ is not e:
                return BackgroundBufferManager._retry_after_seconds(e.__cause__)
            return None
        value = (response.headers.get("Retry-After") or "").strip()
        if not value:
            return None
        if value.isdigit():
            return float(value)
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())

    def _retry_remote_call(self, fn: Callable, description: str, label: str) -> None:
        """
        Call fn() retrying on transient network errors with exponential backoff.

        A server Retry-After is a minimum wait. If it is longer than
        retry_max_delay, this call gives up at once and records the deadline, so
        no later flush sends before it. Logs each retry attempt at WARNING and
        raises on permanent failure.

        Args:
            fn: Zero-argument callable to attempt
            description: Human-readable description for log messages
            label: Resource label, e.g. "logs" or "metric 'train'"
        """
        max_retries = self._config.max_retries
        base_delay = self._config.retry_base_delay
        max_delay = self._config.retry_max_delay

        for attempt in range(max_retries + 1):
            try:
                fn()
                if attempt > 0:
                    _logger.info(
                        "[ML-Dash] %s succeeded on attempt %d/%d",
                        description, attempt + 1, max_retries + 1,
                    )
                return
            except Exception as e:
                is_last = attempt >= max_retries
                is_transient = self._is_transient_error(e)
                retry_after = self._retry_after_seconds(e)
                if retry_after is not None:
                    self._retry_after_until[label] = time.time() + retry_after

                if is_last or not is_transient:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay)
                if retry_after is not None:
                    if retry_after > max_delay:
                        raise  # beyond this flush's retry budget; the caller keeps the batch
                    delay = max(delay, retry_after)
                _logger.warning(
                    "[ML-Dash] %s failed (attempt %d/%d) — %s: %s "
                    "(errno=%s) — retrying in %.1fs",
                    description, attempt + 1, max_retries + 1,
                    type(e).__name__, e,
                    getattr(e.__cause__ or e, "errno", "n/a"),
                    delay,
                )
                time.sleep(delay)

    def _deferred_by_server(self, label: str) -> bool:
        """True while a Retry-After the server gave for this label has not elapsed."""
        until = self._retry_after_until.get(label)
        if until is None or time.time() >= until:
            return False
        self._pending_errors[label] = (
            f"server asked to retry after {until - time.time():.0f}s (Retry-After)"
        )
        return True

    def _background_ready(self, label: str) -> bool:
        """Background retries of a failed batch wait flush_interval and any Retry-After."""
        now = time.time()
        failed_at = self._failed_at.get(label)
        if failed_at is not None and now - failed_at < self._config.flush_interval:
            return False
        return now >= self._retry_after_until.get(label, 0.0)

    def _record_failure(self, label: str, e: Exception) -> None:
        self._failed_at[label] = time.time()
        self._pending_errors[label] = f"{type(e).__name__}: {e}"

    def _record_success(self, label: str) -> None:
        self._failed_at.pop(label, None)
        self._retry_after_until.pop(label, None)
        self._pending_errors.pop(label, None)

    @staticmethod
    def _metric_label(metric_name: Optional[str]) -> str:
        return f"metric '{metric_name}'" if metric_name else "unnamed metric"

    def start(self) -> None:
        """Start background flushing thread."""
        if self._thread is not None:
            return  # Already started

        self._stop_event.clear()
        self._flush_event.clear()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the background thread, then flush everything still buffered.

        Waiting for the background thread's current flush has no timeout. The
        final flush then gives each batch at most ``max_retries`` retries and
        stops draining a queue at its first failed batch, so it ends. How long it
        takes depends on the HTTP timeouts, retry delays and upload sizes.

        Calling it again after a failed stop retries the unsent data and raises
        again if it is still unsent. It returns quietly only when nothing is left.

        Raises:
            NetworkError: if any data could not be sent. The message gives counts
                and errors, never the data itself.
        """
        if self._thread is None and not any(self._pending_counts()) and not self._lost_errors:
            return  # Not started, or already stopped with everything sent

        # Snapshot counts before signalling stop (for the user-facing message)
        log_count, metric_count, track_count, file_count = self._pending_counts()
        total = log_count + metric_count + track_count + file_count

        if total > 0:
            desc = self._describe_pending(log_count, metric_count, track_count, file_count)
            print(f"\n[ML-Dash] Flushing buffered data...", flush=True)
            print(f"[ML-Dash]   - {desc}", flush=True)

        if self._thread is not None:
            # Signal stop and trigger flush
            self._stop_event.set()
            self._flush_event.set()

            # Let the in-progress background flush finish; the final drain runs here
            self._thread.join()
            self._thread = None

        with self._flush_lock:
            errors = self._drain_all_with_progress(total)
        self._raise_if_unsent(errors, "Final flush")

        if total > 0:
            print("[ML-Dash] ✓ All data flushed successfully", flush=True)

    def _check_queue_pressure(self, queue: Queue, queue_name: str) -> None:
        """
        Check queue size and trigger aggressive flushing if needed.

        This prevents OOM by flushing immediately when queue fills up.

        Args:
            queue: The queue to check
            queue_name: Name for warning messages
        """
        qsize = queue.qsize()

        # Trigger immediate flush if queue is getting full
        if qsize >= self._config._AGGRESSIVE_FLUSH_THRESHOLD:
            self._flush_event.set()

        # Warn once if queue is filling up (80% capacity)
        if qsize >= self._config._WARNING_THRESHOLD:
            if queue_name not in self._warned_queues:
                warnings.warn(
                    f"[ML-Dash] {queue_name} queue is {qsize}/{self._config._MAX_QUEUE_SIZE} full. "
                    f"Data is being generated faster than it can be flushed. "
                    f"Consider reducing logging frequency or the background flush will block to prevent OOM.",
                    RuntimeWarning,
                    stacklevel=3
                )
                self._warned_queues.add(queue_name)

    def _pending_counts(self) -> tuple:
        """Return (log_count, metric_count, track_count, file_count) snapshot, failed batches included."""
        with self._track_lock:
            track_count = sum(len(e) for e in self._track_buffers.values())
        return (
            self._log_queue.qsize() + len(self._failed_log_batch or []),
            sum(q.qsize() for q in list(self._metric_queues.values()))
            + sum(len(f["batch"]) for f in list(self._failed_metric_batches.values())),
            track_count,
            self._file_queue.qsize(),
        )

    def _describe_pending(
        self,
        log_count: int,
        metric_count: int,
        track_count: int,
        file_count: int,
    ) -> str:
        """Build a human-readable summary of pending item counts."""
        items = []
        if log_count > 0:
            items.append(f"{log_count} log(s)")
        if metric_count > 0:
            items.append(f"{metric_count} metric point(s)")
        if track_count > 0:
            items.append(f"{track_count} track entry(ies)")
        if file_count > 0:
            items.append(f"{file_count} file(s)")
        return ", ".join(items)

    def _raise_if_unsent(self, errors: List[str], what: str) -> None:
        """Raise NetworkError if data is still unsent or was lost. Counts and errors only."""
        problems = [f"{label}: {msg}" for label, msg in self._lost_errors.items()]
        self._lost_errors.clear()
        problems += errors
        unsent = self._describe_pending(*self._pending_counts())
        if not unsent and not problems:
            return
        lines = [f"[ML-Dash] {what} incomplete."]
        if unsent:
            problems += [f"{label}: {msg}" for label, msg in self._pending_errors.items()]
            lines.append(f"Not sent (held in memory in this process only): {unsent}.")
        if problems:
            lines.append("Errors: " + "; ".join(problems))
        raise NetworkError("\n".join(lines))

    def _drain_all_with_progress(self, total_items: int) -> List[str]:
        """
        Drain all queues to their respective backends.

        Shows an ASCII progress bar when total_items > 200. A queue stops draining at
        its first batch that cannot be sent; that batch stays pending.

        Args:
            total_items: Pre-computed total used to size the progress bar.

        Returns:
            Errors raised while draining (e.g. a failed file upload or local write).
        """
        show_progress = total_items > 200
        errors: List[str] = []

        def update_progress() -> None:
            if show_progress:
                items_flushed = min(total_items, max(0, total_items - sum(self._pending_counts())))
                progress = items_flushed / total_items
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                percent = progress * 100
                print(
                    f'\r[ML-Dash] Flushing: |{bar}| {percent:.1f}%'
                    f' ({items_flushed}/{total_items})',
                    end='',
                    flush=True,
                )

        def drain(label: str, has_pending: Callable[[], bool], flush_once: Callable[[], bool]) -> None:
            try:
                while has_pending():
                    if not flush_once():
                        break
                    update_progress()
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__}: {e}")

        drain(
            "logs",
            lambda: self._failed_log_batch is not None or not self._log_queue.empty(),
            self._flush_logs,
        )

        for metric_name in list(self._metric_queues.keys()):
            drain(
                self._metric_label(metric_name),
                lambda n=metric_name: n in self._failed_metric_batches
                or not self._metric_queues[n].empty(),
                lambda n=metric_name: self._flush_metric(n),
            )

        with self._track_lock:
            topics = list(self._track_buffers.keys())
        for topic in topics:
            drain(
                f"track '{topic}'",
                lambda t=topic: bool(self._track_buffers.get(t)),
                lambda t=topic: self._flush_track(t),
            )

        drain("files", lambda: not self._file_queue.empty(), self._flush_files)

        if show_progress:
            print()  # newline after progress bar

        return errors

    def buffer_log(
        self,
        message: str,
        level: str,
        metadata: Optional[Dict[str, Any]],
        timestamp: Optional[datetime],
    ) -> None:
        """
        Add log to buffer with automatic backpressure.

        If queue is full, this will block until space is available.
        This prevents OOM when logs are generated faster than they can be flushed.

        Args:
            message: Log message
            level: Log level
            metadata: Optional metadata
            timestamp: Optional timestamp
        """
        # Check queue pressure and trigger aggressive flushing if needed
        self._check_queue_pressure(self._log_queue, "Log")

        log_entry = {
            "timestamp": (timestamp or datetime.now(timezone.utc).replace(tzinfo=None)).isoformat() + "Z",
            "level": level,
            "message": message,
        }

        if metadata:
            log_entry["metadata"] = metadata

        # Will block if queue is full (backpressure to prevent OOM)
        self._log_queue.put(log_entry)

    def buffer_metric(
        self,
        metric_name: Optional[str],
        data: Dict[str, Any],
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """
        Add metric datapoint to buffer with automatic backpressure.

        If queue is full, this will block until space is available.
        This prevents OOM when metrics are generated faster than they can be flushed.

        Args:
            metric_name: Metric name (can be None for unnamed metrics)
            data: Data point
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
        """
        # Get or create queue for this metric (with bounded size)
        if metric_name not in self._metric_queues:
            self._metric_queues[metric_name] = Queue(maxsize=self._config._MAX_QUEUE_SIZE)
            self._last_metric_flush[metric_name] = time.time()

        # Check queue pressure and trigger aggressive flushing if needed
        metric_display = f"'{metric_name}'" if metric_name else "unnamed"
        self._check_queue_pressure(
            self._metric_queues[metric_name],
            f"Metric {metric_display}"
        )

        metric_entry = {
            "data": data,
            "description": description,
            "tags": tags,
            "metadata": metadata,
        }

        # Will block if queue is full (backpressure to prevent OOM)
        self._metric_queues[metric_name].put(metric_entry)

    def buffer_track(
        self,
        topic: str,
        timestamp: float,
        data: Dict[str, Any],
    ) -> None:
        """
        Add track entry to buffer (non-blocking) with timestamp-based merging.

        Entries with the same timestamp are automatically merged.

        Args:
            topic: Track topic (e.g., "robot/position")
            timestamp: Entry timestamp
            data: Data fields
        """
        # Serialize data to handle numpy arrays and other non-JSON types
        serialized_data = _serialize_value(data)

        with self._track_lock:
            # Get or create buffer for this topic
            if topic not in self._track_buffers:
                self._track_buffers[topic] = {}
                self._last_track_flush[topic] = time.time()

            # Merge with existing entry at same timestamp
            if timestamp in self._track_buffers[topic]:
                self._track_buffers[topic][timestamp].update(serialized_data)
            else:
                self._track_buffers[topic][timestamp] = serialized_data

    def buffer_file(
        self,
        file_path: str,
        prefix: str,
        filename: str,
        description: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        checksum: str,
        content_type: str,
        size_bytes: int,
        bindrs: Optional[List[str]] = None,
    ) -> None:
        """
        Add file upload to queue with automatic backpressure.

        If queue is full, this will block until space is available.

        Args:
            file_path: Local file path
            prefix: Logical path prefix
            filename: Original filename
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            checksum: SHA256 checksum
            content_type: MIME type
            size_bytes: File size in bytes
            bindrs: Optional list of bindr names this file belongs to
        """
        # Check queue pressure and trigger aggressive flushing if needed
        self._check_queue_pressure(self._file_queue, "File")

        file_entry = {
            "file_path": file_path,
            "prefix": prefix,
            "filename": filename,
            "description": description,
            "tags": tags,
            "bindrs": bindrs,
            "metadata": metadata,
            "checksum": checksum,
            "content_type": content_type,
            "size_bytes": size_bytes,
        }

        # Will block if queue is full (backpressure to prevent OOM)
        self._file_queue.put(file_entry)

    def flush_all(self) -> None:
        """
        Manually flush all buffered data immediately.

        This forces an immediate flush of all queued logs, metrics, tracks, and files
        without waiting for time or size triggers.

        Raises:
            NetworkError: if any data could not be sent. Unsent logs, metrics and
                tracks stay buffered for the next flush; the message gives counts
                and errors, never the data itself.
        """
        with self._flush_lock:
            log_count, metric_count, track_count, file_count = self._pending_counts()
            total = log_count + metric_count + track_count + file_count

            if total > 0:
                desc = self._describe_pending(log_count, metric_count, track_count, file_count)
                print(f"[ML-Dash] Flushing {desc}...", flush=True)

            errors = self._drain_all_with_progress(total)
            self._raise_if_unsent(errors, "Flush")

        if total > 0:
            print("[ML-Dash] ✓ Flush complete", flush=True)

    def flush_tracks(self) -> None:
        """
        Flush all track topics immediately.

        This is called by TracksManager.flush() for global track flush.
        """
        for topic in list(self._track_buffers.keys()):
            self.flush_track(topic)

    def flush_track(self, topic: str) -> None:
        """
        Flush specific track topic immediately.

        Args:
            topic: Track topic to flush
        """
        with self._flush_lock:
            self._flush_track(topic)

    def _flush_loop(self) -> None:
        """Background thread main loop. The final drain runs in stop(), on the caller's thread."""
        while not self._stop_event.is_set():
            # Wait for flush event or timeout (100ms polling interval for faster response)
            triggered = self._flush_event.wait(timeout=0.1)

            with self._flush_lock:
                self._flush_due(triggered)

            # Clear the flush event after processing
            if triggered:
                self._flush_event.clear()

    def _flush_due(self, triggered: bool) -> None:
        """
        One background pass: flush each resource whose time, size or manual trigger fired.

        A resource with a failed batch is retried at most once per flush_interval and
        never before a server Retry-After. Errors that lose data are kept for the
        next exp.flush() or close to report.
        """
        current_time = time.time()

        def background_error(label: str, e: Exception) -> None:
            self._lost_errors[label] = f"{type(e).__name__}: {e}"  # recorded first: warn may raise
            _logger.error("[ML-Dash] Background flush failed for %s: %s: %s",
                          label, type(e).__name__, e, exc_info=True)
            warnings.warn(f"[ML-Dash] Background flush failed for {label}: {e}")

        # Flush logs if time elapsed or queue size exceeded or manual trigger
        has_logs = self._failed_log_batch is not None or not self._log_queue.empty()
        if has_logs and self._background_ready("logs") and (
            triggered
            or current_time - self._last_log_flush >= self._config.flush_interval
            or self._log_queue.qsize() >= self._config.log_batch_size
        ):
            try:
                self._flush_logs()
            except Exception as e:
                background_error("logs", e)

        # Flush metrics (check each metric queue)
        for metric_name, queue in list(self._metric_queues.items()):
            label = self._metric_label(metric_name)
            has_points = metric_name in self._failed_metric_batches or not queue.empty()
            if has_points and self._background_ready(label) and (
                triggered
                or current_time - self._last_metric_flush.get(metric_name, 0)
                >= self._config.flush_interval
                or queue.qsize() >= self._config.metric_batch_size
            ):
                try:
                    self._flush_metric(metric_name)
                except Exception as e:
                    background_error(label, e)

        # Flush tracks (check each topic)
        with self._track_lock:
            topic_sizes = [(topic, len(entries)) for topic, entries in self._track_buffers.items()]
        for topic, size in topic_sizes:
            label = f"track '{topic}'"
            if size and self._background_ready(label) and (
                triggered
                or current_time - self._last_track_flush.get(topic, 0)
                >= self._config.flush_interval
                or size >= self._config.track_batch_size
            ):
                try:
                    self._flush_track(topic)
                except Exception as e:
                    background_error(label, e)

        # Flush files (always process file queue)
        if not self._file_queue.empty():
            try:
                self._flush_files()
            except Exception as e:
                background_error("files", e)

    def _flush_logs(self) -> bool:
        """
        Batch flush logs using client.create_log_entries().

        Returns:
            False if the batch could not be sent; it is kept and sent first next time.
        """
        batch = self._failed_log_batch
        if batch is None:
            # Collect batch
            batch = []
            try:
                while len(batch) < self._config.log_batch_size:
                    log_entry = self._log_queue.get_nowait()
                    batch.append(log_entry)
            except Empty:
                pass  # Queue exhausted

            if not batch:
                return True

        # Write to backends
        if self._experiment._client:
            if self._deferred_by_server("logs"):
                self._failed_log_batch = batch
                return False
            try:
                self._retry_remote_call(
                    lambda: self._experiment._client.create_log_entries(
                        experiment_id=self._experiment._experiment_id,
                        logs=batch,
                    ),
                    f"flush {len(batch)} log(s)",
                    "logs",
                )
            except Exception as e:
                self._failed_log_batch = batch
                self._record_failure("logs", e)
                _logger.warning(
                    "[ML-Dash] Log flush failed (%s: %s) — keeping %d entries for the next flush",
                    type(e).__name__, e, len(batch),
                )
                return False
            self._failed_log_batch = None
            self._record_success("logs")

        if self._experiment._storage:
            # Local storage writes one at a time (no batch API)
            for log_entry in batch:
                try:
                    self._experiment._storage.write_log(
                        owner=self._experiment.owner,
                        project=self._experiment.project,
                        prefix=self._experiment._folder_path,
                        message=log_entry["message"],
                        level=log_entry["level"],
                        metadata=log_entry.get("metadata"),
                        timestamp=log_entry["timestamp"],
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to write log to local storage: {e}\n"
                        f"Check disk space and file permissions."
                    )

        self._last_log_flush = time.time()
        return True

    def _flush_metric(self, metric_name: Optional[str]) -> bool:
        """
        Batch flush metrics using client.append_batch_to_metric().

        Args:
            metric_name: Metric name (can be None for unnamed metrics)

        Returns:
            False if the batch could not be sent; it is kept and sent first next time.
        """
        pending = self._failed_metric_batches.pop(metric_name, None)
        if pending is None:
            queue = self._metric_queues.get(metric_name)
            if queue is None or queue.empty():
                return True

            # Collect batch
            pending = {"batch": [], "description": None, "tags": None, "metadata": None}
            try:
                while len(pending["batch"]) < self._config.metric_batch_size:
                    metric_entry = queue.get_nowait()
                    pending["batch"].append(metric_entry["data"])

                    # Use first non-None description/tags/metadata
                    for field in ("description", "tags", "metadata"):
                        if pending[field] is None and metric_entry[field]:
                            pending[field] = metric_entry[field]
            except Empty:
                pass  # Queue exhausted

            if not pending["batch"]:
                return True

        batch = pending["batch"]
        description = pending["description"]
        tags = pending["tags"]
        metadata = pending["metadata"]
        label = self._metric_label(metric_name)

        # Write to backends
        if self._experiment._client:
            if self._deferred_by_server(label):
                self._failed_metric_batches[metric_name] = pending
                return False
            try:
                self._retry_remote_call(
                    lambda: self._experiment._client.append_batch_to_metric(
                        experiment_id=self._experiment._experiment_id,
                        metric_name=metric_name,
                        data_points=batch,
                        description=description,
                        tags=tags,
                        metadata=metadata,
                    ),
                    f"flush {len(batch)} point(s) to {label}",
                    label,
                )
            except Exception as e:
                self._failed_metric_batches[metric_name] = pending
                self._record_failure(label, e)
                _logger.warning(
                    "[ML-Dash] Metric flush failed (%s: %s) — keeping %d points for %s for the next flush",
                    type(e).__name__, e, len(batch), label,
                )
                return False
            self._record_success(label)

        if self._experiment._storage:
            try:
                self._experiment._storage.append_batch_to_metric(
                    owner=self._experiment.owner,
                    project=self._experiment.project,
                    prefix=self._experiment._folder_path,
                    metric_name=metric_name,
                    data_points=batch,
                    description=description,
                    tags=tags,
                    metadata=metadata,
                )
            except Exception as e:
                metric_display = f"'{metric_name}'" if metric_name else "unnamed metric"
                raise StorageError(
                    f"Failed to flush {len(batch)} points to {metric_display} in local storage: {e}\n"
                    f"Check disk space and file permissions."
                ) from e

        self._last_metric_flush[metric_name] = time.time()
        return True

    def _restore_track_entries(self, topic: str, entries: Dict[float, Dict[str, Any]]) -> None:
        """Put back entries whose send failed; fields added since then win on the same timestamp."""
        with self._track_lock:
            for timestamp, data in self._track_buffers.get(topic, {}).items():
                entries[timestamp] = {**entries.get(timestamp, {}), **data}
            self._track_buffers[topic] = entries

    def _flush_track(self, topic: str) -> bool:
        """
        Batch flush track entries using client.append_batch_to_track().

        Args:
            topic: Track topic

        Returns:
            False if the entries could not be sent; they stay buffered.
        """
        label = f"track '{topic}'"
        with self._track_lock:
            entries_dict = self._track_buffers.get(topic)
            if not entries_dict:
                return True
            if self._experiment._client and self._deferred_by_server(label):
                return False
            # Take the entries out so producers can keep adding while we send
            self._track_buffers[topic] = {}

        # Convert timestamp-indexed dict to batch entries
        batch = []
        for timestamp, data in sorted(entries_dict.items()):
            entry = {"timestamp": timestamp}
            entry.update(data)
            batch.append(entry)

        # Write to remote backend — failed entries go back into the buffer
        if self._experiment._client:
            try:
                self._retry_remote_call(
                    lambda: self._experiment._client.append_batch_to_track(
                        experiment_id=self._experiment._experiment_id,
                        topic=topic,
                        entries=batch,
                    ),
                    f"flush {len(batch)} entry/entries to track '{topic}'",
                    label,
                )
            except Exception as e:
                self._restore_track_entries(topic, entries_dict)
                self._record_failure(label, e)
                _logger.warning(
                    "[ML-Dash] Track flush failed (%s: %s) — "
                    "keeping %d entries in buffer for topic '%s' for the next flush",
                    type(e).__name__, e, len(batch), topic,
                )
                return False
            self._record_success(label)

        # Write to local storage
        if self._experiment._storage:
            try:
                self._experiment._storage.append_batch_to_track(
                    owner=self._experiment.owner,
                    project=self._experiment.project,
                    prefix=self._experiment._folder_path,
                    topic=topic,
                    entries=batch,
                )
            except Exception as e:
                raise StorageError(
                    f"Failed to flush {len(batch)} entries to track '{topic}' in local storage: {e}\n"
                    f"Check disk space and file permissions."
                ) from e

        self._last_track_flush[topic] = time.time()
        return True

    def _flush_files(self) -> bool:
        """Upload files using ThreadPoolExecutor. Raises NetworkError if an upload fails."""
        if self._file_queue.empty():
            return True

        # Collect all pending files
        files_to_upload = []
        try:
            while not self._file_queue.empty():
                file_entry = self._file_queue.get_nowait()
                files_to_upload.append(file_entry)
        except Empty:
            pass  # Queue exhausted

        if not files_to_upload:
            return True

        # Show progress for file uploads
        total_files = len(files_to_upload)
        print(f"[ML-Dash]   Uploading {total_files} file(s)...", flush=True)

        # Upload in parallel using ThreadPoolExecutor
        completed = 0
        with ThreadPoolExecutor(max_workers=self._config.file_upload_workers) as executor:
            # Submit all uploads
            future_to_file = {
                executor.submit(self._upload_single_file, file_entry): file_entry
                for file_entry in files_to_upload
            }

            # Wait for completion and show progress
            for future in as_completed(future_to_file):
                file_entry = future_to_file[future]
                try:
                    future.result()
                    completed += 1
                    if total_files > 1:
                        print(f"[ML-Dash]   [{completed}/{total_files}] Uploaded {file_entry['filename']}", flush=True)
                except Exception as e:
                    raise NetworkError(
                        f"Failed to upload file {file_entry['filename']}: {e}\n"
                        f"File upload failed. Check network connection and file permissions."
                    ) from e
        return True

    def _upload_single_file(self, file_entry: Dict[str, Any]) -> None:
        """
        Upload a single file.

        Args:
            file_entry: File metadata dict
        """
        import os
        import tempfile

        file_path = file_entry["file_path"]
        temp_dir = None

        # Check if file is in a temp directory (created by save methods)
        # If so, we'll need to clean it up after upload
        temp_root = tempfile.gettempdir()
        is_temp_file = file_path.startswith(temp_root)
        if is_temp_file:
            temp_dir = os.path.dirname(file_path)

        try:
            if self._experiment._client:
                self._experiment._client.upload_file(
                    experiment_id=self._experiment._experiment_id,
                    file_path=file_entry["file_path"],
                    prefix=file_entry["prefix"],
                    filename=file_entry["filename"],
                    description=file_entry["description"],
                    tags=file_entry["tags"],
                    metadata=file_entry["metadata"],
                    checksum=file_entry["checksum"],
                    content_type=file_entry["content_type"],
                    size_bytes=file_entry["size_bytes"],
                )

            if self._experiment._storage:
                self._experiment._storage.write_file(
                    owner=self._experiment.owner,
                    project=self._experiment.project,
                    prefix=self._experiment._folder_path,
                    file_path=file_entry["file_path"],
                    path=file_entry["prefix"],
                    filename=file_entry["filename"],
                    description=file_entry["description"],
                    tags=file_entry["tags"],
                    bindrs=file_entry.get("bindrs"),
                    metadata=file_entry["metadata"],
                    checksum=file_entry["checksum"],
                    content_type=file_entry["content_type"],
                    size_bytes=file_entry["size_bytes"],
                )
        finally:
            # Clean up temp file and directory if this was a temp file
            if is_temp_file and temp_dir:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except OSError as e:
                    warnings.warn(
                        f"Failed to clean up temp file {file_path}: {e}\n"
                        f"Check disk space and file permissions."
                    )
