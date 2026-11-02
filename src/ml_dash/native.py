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

from .exceptions import MlDashError
from .managed import _authorization, _login_provider, _url

_KEY = re.compile(r'[A-Za-z0-9_-]{1,128}')
_ID = re.compile(r'[0-9a-f]{24}')
_METRIC = re.compile(r'[A-Za-z0-9_.-]{1,128}')
# The server's sessions live 60 s; renew well before so a request never races expiry.
_SESSION_SECONDS = 50


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
        if not isinstance(metric_name, str) or not _METRIC.fullmatch(metric_name):
            raise ValueError('Native metric names are 1-128 letters, digits, ".", "_" or "-"')
        if description or tags or metadata:
            raise ValueError('Native metrics do not support description, tags or metadata')
        return self._json('POST', self._run(experiment_id) + f'/metrics/{quote(metric_name, safe="")}',
                          'Native metric write', json={'dataPoints': data_points})

    def append_to_metric(self, experiment_id, metric_name, data, **kwargs):
        return self.append_batch_to_metric(experiment_id, metric_name, [data], **kwargs)

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

    def download_file(self, experiment_id, file_id, dest_path=None):
        if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
            raise ValueError('Invalid native file id')
        path = self._run(experiment_id) + f'/files/{file_id}'
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
