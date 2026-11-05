"""Interactive DreamLake-native runs: the ``Experiment`` API over DreamLake's native routes.

``NativeConnection`` signs in with an existing DreamLake login, asks the
project's native ``/session`` route for a short-lived project write session, and
sends every run write to ``/native-experiments/ns/{namespace}/projects/{project}``
on the same DreamLake server. It never talks to a managed Dash service and never
writes anywhere else; where the server stores the run (main or Dash database) is
the server's decision.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit

import httpx

from .exceptions import MetricRowsError, MlDashError
from .managed import _authorization, _login_provider, _url
from .metric_rows import MetricColumn, MetricRowBlock, MetricRowsPage
from .utils import _serialize_value

_KEY = re.compile(r'[A-Za-z0-9_-]{1,128}')
_ID = re.compile(r'[0-9a-f]{24}')
# The server's sessions live 60 s; renew well before so a request never races expiry.
_SESSION_SECONDS = 50
# The native metric rows route returns at most this many rows per request; ml-dash's rows read
# allows a page of up to 10000 (default 1000), which is filled from several requests.
_ROWS_REQUEST_MAX = 1000
_ROWS_DEFAULT, _ROWS_MAX = 1000, 10000


def _metric_name(value) -> bool:
    """A metric name as ml-dash accepts it: one URL path segment (sent percent-encoded) of at most 256
    characters: anything but ``/``, other than ``.``/``..``. Unicode and spaces are fine."""
    return (isinstance(value, str) and 0 < len(value) <= 256 and '/' not in value
            and value not in ('.', '..'))


class NativeExperimentError(MlDashError):
    """A DreamLake-native run request was refused or failed.

    ``status`` is the HTTP status (None for transport failures) and ``code`` the
    server's ``error.code`` (for example ``NATIVE_FORBIDDEN`` or
    ``NATIVE_RUN_WRITE_PROTECTED``). Writes are never replayed after an
    ambiguous failure; the message never echoes a credential or response body.
    """

    def __init__(self, status: int | None, message: str, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class NativeConnection:
    """Select a DreamLake project whose runs use DreamLake-native storage.

    Pass it as ``Experiment(prefix="namespace/project/run", managed=connection)``.
    ``token_provider`` returns the user's current DreamLake login; it is called
    whenever a project session must be minted, so a renewed login is picked up.
    """
    platform_controls_status = False

    def __init__(self, dreamlake_url: str, namespace: str, project: str,
                 token_provider: Callable[[], str]):
        if not namespace or not project or '/' in namespace or '/' in project:
            raise ValueError('Native namespace and project must be individual slugs')
        self.namespace, self.project = namespace, project
        self.dreamlake_url = _url(dreamlake_url)
        self._provider = token_provider
        self._path = f'/native-experiments/ns/{quote(namespace, safe="")}/projects/{quote(project, safe="")}'
        self._root = self.dreamlake_url + self._path
        self._experiment_url = None

    @classmethod
    def from_login(cls, namespace: str, project: str, dreamlake_url=None, config_dir=None):
        """Use the DreamLake login already on this machine (``~/.dreamlake``).

        Same login source and server-pinning rules as
        ``ManagedConnection.from_login``. Does not write files or start a login.
        """
        selected, provider = _login_provider(dreamlake_url, config_dir)
        return cls(selected, namespace, project, provider)

    def client(self):
        return NativeRemoteClient(self)


def _error(response: httpx.Response, action: str) -> NativeExperimentError:
    try:
        body = response.json()
        code = body.get('error', {}).get('code') if isinstance(body, dict) else None
    except ValueError:
        code = None
    code = code if isinstance(code, str) else None
    suffix = f', {code}' if code else ''
    return NativeExperimentError(response.status_code, f'{action} failed (HTTP {response.status_code}{suffix})', code)


class NativeRemoteClient:
    """The subset of ``RemoteClient`` that ``Experiment`` needs, on native routes."""

    def __init__(self, connection: NativeConnection):
        self.connection = connection
        self._http = httpx.Client(follow_redirects=False, timeout=30)
        self._lock = threading.RLock()
        self._token = None
        self._renew_at = 0.0

    # -- session ----------------------------------------------------------------

    def _session_token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._renew_at:
                return self._token
            self._token = None
            try:
                response = self._http.post(self.connection._root + '/session', json={'mode': 'write'},
                    headers={'Authorization': _authorization(self.connection._provider())})
            except httpx.HTTPError:
                raise NativeExperimentError(None, 'DreamLake native session unavailable') from None
            if response.status_code != 200:
                raise _error(response, 'DreamLake native write session')
            try:
                result = response.json()
                token, api_url = result['token'], result['apiUrl']
                # The session is only ever sent to this connection's own server; the
                # advertised destination must name the same project route.
                if urlsplit(_url(api_url)).path != self.connection._path:
                    raise ValueError
                _authorization(token)
            except (ValueError, KeyError, TypeError):
                raise NativeExperimentError(None, 'Invalid DreamLake native session response') from None
            self._token, self._renew_at = token, time.time() + _SESSION_SECONDS
            return token

    def _send(self, method: str, path: str, action: str, **kwargs) -> httpx.Response:
        for attempt in range(2):
            token = self._session_token()
            try:
                response = self._http.request(method, self.connection._root + path,
                    headers={'Authorization': _authorization(token)}, **kwargs)
            except httpx.HTTPError:
                raise NativeExperimentError(None, f'{action} failed: DreamLake unavailable; '
                                                  'reconcile before retrying writes') from None
            if response.status_code == 401 and attempt == 0:
                # 401 is decided before any write, so one retry with a fresh session is safe.
                with self._lock:
                    self._token = None
                continue
            if response.is_success:
                return response
            raise _error(response, action)
        raise AssertionError('unreachable')

    def _json(self, method: str, path: str, action: str, **kwargs) -> dict:
        response = self._send(method, path, action, **kwargs)
        try:
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError
            return result
        except (ValueError, TypeError):
            raise NativeExperimentError(None, f'{action} returned an invalid response') from None

    @staticmethod
    def _run(experiment_id: str) -> str:
        if not isinstance(experiment_id, str) or not _ID.fullmatch(experiment_id):
            raise ValueError('Invalid native run id')
        return f'/runs/{experiment_id}'

    # -- runs -------------------------------------------------------------------

    def create_or_update_experiment(self, project, name, prefix=None, description=None, tags=None,
                                    bindrs=None, write_protected=False, metadata=None, **kwargs):
        """Create the run, or return the existing one with this key.

        ``description``/``tags`` of None leave the stored value unchanged; ``""``
        clears the description and ``[]`` clears the tags (``Experiment(readme=...,
        tags=...)``).
        """
        connection = self.connection
        if (project != connection.project or not isinstance(name, str) or not _KEY.fullmatch(name)
                or prefix != f'{connection.namespace}/{connection.project}/{name}'):
            raise ValueError('Native prefix must be the selected namespace/project/run (no nested folders)')
        if write_protected or bindrs or metadata:
            raise ValueError('Native runs do not support bindrs, run metadata or write protection from the SDK')
        body = {}
        if description is not None:
            body['description'] = description
        if tags is not None:
            body['tags'] = list(tags)
        return self._json('PUT', f'/runs/{quote(name, safe="")}', 'Native run create', json=body)

    def update_experiment_status(self, experiment_id, status):
        return self._json('POST', self._run(experiment_id) + '/status', 'Native run status',
                          json={'status': status})

    # -- parameters, logs, metrics ---------------------------------------------

    def set_parameters(self, experiment_id, data):
        return self._json('POST', self._run(experiment_id) + '/parameters', 'Native parameters write',
                          json={'data': data})

    def get_parameters(self, experiment_id):
        result = self._json('GET', self._run(experiment_id) + '/parameters', 'Native parameters read')
        return (result.get('parameters') or {}).get('data') or None

    def create_log_entries(self, experiment_id, logs):
        return self._json('POST', self._run(experiment_id) + '/logs', 'Native log write',
                          json={'entries': logs})

    def append_batch_to_metric(self, experiment_id, metric_name, data_points, description=None,
                               tags=None, metadata=None, **kwargs):
        """Append points in one request. As in ml-dash, ``description``/``tags``/``metadata`` are the metric's
        descriptor: the server stores them with the points, in the same commit, only when this append creates
        the metric; later appends never change them (use ``update_metric``). An invalid descriptor is refused
        with nothing written."""
        body = {'dataPoints': data_points}
        for key, value in (('description', description), ('tags', list(tags) if tags else None), ('metadata', metadata)):
            if value:
                body[key] = value
        return self._json('POST', self._metric(experiment_id, metric_name), 'Native metric write', json=body)

    def append_to_metric(self, experiment_id, metric_name, data, **kwargs):
        return self.append_batch_to_metric(experiment_id, metric_name, [data], **kwargs)

    def _metric(self, experiment_id, metric_name):
        if not _metric_name(metric_name):
            raise ValueError('Native metric names are 1-256 characters without "/" (not "." or "..")')
        return self._run(experiment_id) + f'/metrics/{quote(metric_name, safe="")}'

    def list_metrics(self, experiment_id):
        return self._json('GET', self._run(experiment_id) + '/metrics', 'Native metric list')['metrics']

    def get_metric_stats(self, experiment_id, metric_name):
        return self._json('GET', self._metric(experiment_id, metric_name) + '/stats', 'Native metric stats')['metric']

    def update_metric(self, experiment_id, metric_name, description=None, tags=None, metadata=None):
        """Set an existing metric's description, tags and/or metadata (None leaves a field unchanged)."""
        body = {key: value for key, value in (('description', description), ('tags', tags), ('metadata', metadata))
                if value is not None}
        return self._json('PATCH', self._metric(experiment_id, metric_name), 'Native metric metadata write', json=body)['metric']

    def read_metric_data(self, experiment_id, metric_name, start_index=0, limit=1000):
        from .client import _index_read_unsupported
        raise _index_read_unsupported('read_metric_data')

    def read_metric_rows(self, experiment_id, metric_name, limit=None, cursor=None):
        """One page of a metric's committed rows as a ``MetricRowsPage``, as ``RemoteClient.read_metric_rows``.

        ``limit``: 1-10000 rows (default 1000); the page is filled from as many native requests as it needs,
        so no row is skipped. ``cursor`` is opaque. ``metric_id`` is the metric's canonical id. Consecutive rows
        with the same fields form a block; numbers are int64 (promoted to float64 when a column mixes them),
        float64, bool or utf8, ``null`` a None cell. A nested object or list has no ml-dash cell type and is
        refused (``MetricRowsError``, code ``unsupported_cell``), never converted.
        """
        if limit is None:
            limit = _ROWS_DEFAULT
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _ROWS_MAX:
            raise MetricRowsError(f'limit must be an integer between 1 and {_ROWS_MAX}', 400, 'invalid_limit')
        if cursor is not None and (not isinstance(cursor, str) or not cursor.isdigit()):
            raise MetricRowsError('Invalid cursor; pass next_cursor from the previous page', 400, 'invalid_cursor')
        rows, metric_id, next_cursor = [], None, cursor
        while True:
            params = {'limit': str(min(limit - len(rows), _ROWS_REQUEST_MAX))}
            if next_cursor is not None:
                params['cursor'] = next_cursor
            try:
                result = self._json('GET', self._metric(experiment_id, metric_name), 'Native metric read', params=params)
            except NativeExperimentError as error:
                raise MetricRowsError(str(error), error.status or 0,
                                      'metric_not_found' if error.status == 404 else error.code) from None
            metric_id = result.get('metricId')
            rows.extend(result.get('rows') or [])
            next_cursor = result.get('nextCursor')
            if next_cursor is None or len(rows) >= limit:
                break
        if not isinstance(metric_id, str):
            raise MetricRowsError('Malformed metric rows response: metricId missing', 200)
        blocks = _blocks(rows)
        return MetricRowsPage(metric_id=metric_id, blocks=blocks, returned=len(rows), has_more=next_cursor is not None,
                              next_cursor=next_cursor, visibility='committed', order='storage')

    # -- logs -------------------------------------------------------------------

    def query_logs(self, experiment_id, limit=None, offset=None, order_by=None, order=None, level=None,
                   start_time=None, end_time=None, search=None):
        """Filtered log page: ``{logs, hasMore, nextCursor}``; ``offset`` is the previous ``nextCursor``."""
        params = {key: str(value) for key, value in (('limit', limit), ('cursor', offset), ('orderBy', order_by),
                  ('order', order), ('startTime', start_time), ('endTime', end_time), ('search', search)) if value is not None}
        if level:
            params['level'] = ','.join([level] if isinstance(level, str) else level)
        return self._json('GET', self._run(experiment_id) + '/logs', 'Native log read', params=params)

    def delete_parameters(self, experiment_id):
        return self._json('DELETE', self._run(experiment_id) + '/parameters', 'Native parameters delete')['parameters']

    # -- tracks -------------------------------------------------------------------
    # ml-dash tracks (``exp.tracks(topic)``) on the run's native track store. Runs homed in main before the
    # Dash store answer 409 ``NATIVE_TRACKS_UNAVAILABLE``.

    def _track(self, experiment_id, topic):
        if not isinstance(topic, str) or not topic or any(not part for part in topic.split('/')):
            raise ValueError('Track topic must be non-empty "/"-separated segments')
        return self._run(experiment_id) + f'/tracks/{quote(topic, safe="")}'

    def create_track(self, experiment_id, topic, description=None, tags=None, metadata=None):
        """Create the topic with its description/tags/metadata, or return the existing track unchanged."""
        self._track(experiment_id, topic)
        body = {'topic': topic}
        for key, value in (('description', description), ('tags', tags), ('metadata', metadata)):
            if value:
                body[key] = value
        return self._json('POST', self._run(experiment_id) + '/tracks', 'Native track create', json=body)

    def append_to_track(self, experiment_id, topic, timestamp, data):
        return self._json('POST', self._track(experiment_id, topic) + '/append', 'Native track write',
                          json={'timestamp': timestamp, 'data': _serialize_value(data)})

    def append_batch_to_track(self, experiment_id, topic, entries):
        """Entries ``{timestamp, ...fields}`` in one commit (at most 10000)."""
        return self._json('POST', self._track(experiment_id, topic) + '/append_batch', 'Native track write',
                          json={'entries': [_serialize_value(entry) for entry in entries]})

    def get_track_data(self, experiment_id, topic, start_timestamp=None, end_timestamp=None, columns=None, format='json'):
        """As ``RemoteClient.get_track_data``: dict for json, bytes for jsonl, parquet (Arrow IPC) and mcap. An mcap
        topic with no exact track exports every track under ``topic/``."""
        params = {'format': format}
        if start_timestamp is not None:
            params['start_timestamp'] = str(start_timestamp)
        if end_timestamp is not None:
            params['end_timestamp'] = str(end_timestamp)
        if columns:
            params['columns'] = ','.join(columns)
        path = self._track(experiment_id, topic) + '/data'
        if format in ('jsonl', 'parquet', 'mcap'):
            return self._send('GET', path, 'Native track read', params=params).content
        return self._json('GET', path, 'Native track read', params=params)

    def list_tracks(self, experiment_id, topic_filter=None):
        """Tracks of the run: all, an exact topic, or ``prefix/*``."""
        params = {'topic': topic_filter} if topic_filter else {}
        return self._json('GET', self._run(experiment_id) + '/tracks', 'Native track list', params=params)['tracks']

    # -- files ------------------------------------------------------------------

    def upload_file(self, experiment_id, file_path, prefix, filename, description=None, tags=None,
                    metadata=None, checksum=None, content_type=None, size_bytes=None, **kwargs):
        digest, size = hashlib.sha256(), 0
        with open(file_path, 'rb') as source:
            for block in iter(lambda: source.read(1 << 20), b''):
                digest.update(block)
                size += len(block)
        if (checksum is not None and checksum != digest.hexdigest()) or (size_bytes is not None and size != size_bytes):
            raise ValueError('File changed after it was queued')
        # Form fields precede the file part: the server reads them from the file part's field map.
        fields = {'prefix': prefix or '/', 'checksum': digest.hexdigest(), 'sizeBytes': str(size)}
        if description:
            fields['description'] = description
        if tags:
            fields['tags'] = ','.join(tags)
        if metadata:
            fields['metadata'] = json.dumps(metadata)
        with open(file_path, 'rb') as source:
            return self._json('POST', self._run(experiment_id) + '/files', 'Native file upload', data=fields,
                              files={'file': (filename, source, content_type or 'application/octet-stream')},
                              timeout=300)

    def list_files(self, experiment_id, limit=500, offset=0):
        files = self._json('GET', self._run(experiment_id) + '/files', 'Native file list')['files']
        return {'files': files[offset:offset + limit], 'hasMore': offset + limit < len(files)}

    def search_files(self, pattern, experiment_id, limit=100, **kwargs):
        matches = []
        for file in self.list_files(experiment_id, limit=100000)['files']:
            folder = (file.get('prefix') or '').strip('/')
            path = (folder + '/' if folder else '') + file['filename']
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(file['filename'], pattern.removeprefix('**/')):
                matches.append({**file, 'path': path})
        return matches[:limit]

    def _file(self, experiment_id, file_id):
        if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
            raise ValueError('Invalid native file id')
        return self._run(experiment_id) + f'/files/{file_id}'

    def get_file(self, experiment_id, file_id):
        return self._json('GET', self._file(experiment_id, file_id), 'Native file read')['file']

    def delete_file(self, experiment_id, file_id):
        """Delete the file, as ml-dash: it leaves every listing and read at once and cannot be restored
        (re-uploading the same path creates it anew). Returns ``{id, nodeId, deleted}``."""
        return self._json('DELETE', self._file(experiment_id, file_id), 'Native file delete')

    def update_file(self, experiment_id, file_id, description=None, tags=None, metadata=None):
        """Set description, tags and/or metadata (None leaves a field unchanged)."""
        body = {key: value for key, value in (('description', description), ('tags', tags), ('metadata', metadata))
                if value is not None}
        return self._json('PATCH', self._file(experiment_id, file_id), 'Native file update', json=body)['file']

    def download_file(self, experiment_id, file_id, dest_path=None):
        path = self._file(experiment_id, file_id)
        info = self._json('GET', path, 'Native file read')['file']
        content = self._send('GET', path + '/download', 'Native file download', timeout=300).content
        if hashlib.sha256(content).hexdigest() != info['checksum']:
            raise NativeExperimentError(None, 'Native file download checksum mismatch')
        target = Path(dest_path or info['filename'])
        target.write_bytes(content)
        return str(target)

    def close(self):
        with self._lock:
            self._token = None
        self._http.close()



def _cell_type(value):
    """The ml-dash wire type of a JSON value, None for a null cell; a nested value has none."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int64'
    if isinstance(value, float):
        return 'float64'
    if isinstance(value, str):
        return 'utf8'
    raise MetricRowsError('Metric row holds a nested value, which has no ml-dash cell type', 200, 'unsupported_cell')


def _blocks(rows):
    """Group rows into blocks of one schema: same field names; a column's type is fixed by its first
    non-null cell (int64 widens to float64); any other type change starts a new block."""
    blocks = []  # [names, types, rows]
    for row in rows:
        names, cells = tuple(row), list(row.values())
        kinds = [_cell_type(value) for value in cells]
        current = blocks[-1] if blocks and blocks[-1][0] == names else None
        if current is not None:
            merged = []
            for have, got in zip(current[1], kinds):
                if have is None or got is None or have == got:
                    merged.append(have or got)
                elif {have, got} == {'int64', 'float64'}:
                    merged.append('float64')
                else:
                    current = None
                    break
            if current is not None:
                current[1] = merged
                current[2].append(cells)
                continue
        blocks.append([names, kinds, [cells]])
    result = []
    for names, kinds, block_rows in blocks:
        kinds = [kind or 'float64' for kind in kinds]
        for row in block_rows:
            for i, kind in enumerate(kinds):
                if kind == 'float64' and isinstance(row[i], int) and not isinstance(row[i], bool):
                    row[i] = float(row[i])
        result.append(MetricRowBlock(tuple(MetricColumn(name, kind) for name, kind in zip(names, kinds)), block_rows))
    return result
