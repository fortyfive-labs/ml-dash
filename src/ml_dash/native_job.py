"""Native run data for a platform job: ``NativeRemoteClient``'s data methods over the job's broker session.

A job's experiment client (``managed.JobManagedRemoteClient``) gets its native session only from the
local job broker, which renews the job attempt's capability and whose pins (namespace, project, run,
job, generation) never change. ``NativeJobDataClient`` reuses ``NativeRemoteClient``'s request,
serialization, paging and error handling unchanged, and replaces only two things:

- the session: always the job client's broker session (``JobManagedRemoteClient._session``). It never
  reads a DreamLake login, a workload credential or an API key, and never mints anything itself;
- the run: every path must name the job's own run id. Any other id is refused before a request.

Run creation, status, metadata, trash/restore/purge and credential management are not data methods
and are not offered here; the server's job capability refuses them as well.
"""
from __future__ import annotations

import threading
from urllib.parse import quote

from .native import NativeRemoteClient

# The NativeRemoteClient methods a job capability may use: exactly the server's job allowlist
# (dreamlake-server `routes/nativeExperiments.ts`, capability kind `job`): parameters, logs,
# metrics (list, points, rows, stats, metadata), tracks (create, append, read, list) and the
# run's own files (list, read, download, update, delete; upload stays on the job client).
NATIVE_JOB_DATA_METHODS = (
    'append_batch_to_metric', 'append_to_metric', 'read_metric_rows', 'read_metric_data',
    'list_metrics', 'get_metric_stats', 'update_metric', 'query_logs', 'delete_parameters',
    'create_track', 'append_to_track', 'append_batch_to_track', 'get_track_data', 'list_tracks',
    'get_file', 'update_file', 'delete_file',
)


class _BrokerDestination:
    """The ``connection`` view ``NativeRemoteClient._send`` needs: the job session's API root."""

    def __init__(self, job):
        self._job = job

    @property
    def _root(self) -> str:
        return self._job._session()['apiUrl']


class NativeJobDataClient(NativeRemoteClient):
    """``NativeRemoteClient``'s data methods, pinned to one job run, over that job's broker session."""

    def __init__(self, job):
        self._job = job
        self.connection = _BrokerDestination(job)
        # The job client's HTTP client: no environment proxy, no redirects.
        self._http = job._http
        self._lock = threading.RLock()
        # '' = no request refused yet; None = the server refused the last session (401): renew it.
        self._token = ''
        # The base `_send` clears its per-mode session cache on a 401 or a retired credential. This client
        # never fills it (its one broker session is `_token`), but it must exist for those paths to reach renewal.
        self._sessions = {}

    def _session_token(self) -> str:
        with self._lock:
            if self._token is None:
                # The base client retries a 401 once with a fresh session: ask the broker for one.
                with self._job._lock:
                    self._job._credential = None
            session = self._job._session()
            if not self._job._native:
                raise ValueError('Job experiment native data requires a native platform run')
            self._token = session['token']
            return self._token

    def _run(self, experiment_id: str) -> str:
        # The run pinned by the validated broker session (`JobManagedRemoteClient._validate_session`), and only it.
        run_id = self._job.connection.run_id
        if not isinstance(experiment_id, str) or experiment_id != run_id:
            raise ValueError('Job experiment request must use the platform run')
        return f'/runs/{quote(run_id, safe="")}'

    def close(self):
        # The HTTP client belongs to the job client, which closes it.
        with self._lock:
            self._token = ''
