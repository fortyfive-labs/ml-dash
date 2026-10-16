"""Opt-in DreamLake managed transport; no stored Dash login or shared writer key.

The provider returns an existing DreamLake session. Short-lived project credentials
stay in memory and are renewed through DreamLake's current project policy.
"""
import threading
import time
import re
import uuid
import base64
import hashlib
import os
import json
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit

import httpx


def _url(value: str) -> str:
    p = urlsplit(value)
    if (p.scheme != 'https' and not (p.scheme == 'http' and p.hostname in ('127.0.0.1', 'localhost'))) or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError('Managed service URL must be HTTPS (HTTP is allowed only for local tests)')
    return value.rstrip('/')


def _authorization(value: str) -> str:
    # Reject control characters before httpx can include a malformed header's
    # value in an exception. Neither credential values nor headers are logged.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9._~+-]{1,16384}', value):
        raise ValueError('Authentication provider returned an invalid credential')
    return f'Bearer {value}'


class ManagedConnection:
    """Select an existing DreamLake organization project, never a Dash namespace.

    Call ``setup()`` explicitly once (safe to retry), then pass the connection as
    ``Experiment(prefix="org/project/run", managed=connection)``. The provider is
    called at credential renewal, so callers can reuse their supported DreamLake
    authentication flow without exporting secrets into a plan or source code.
    """
    def __init__(self, dreamlake_url: str, namespace: str, project: str,
                 token_provider: Callable[[], str]):
        if not namespace or not project or '/' in namespace or '/' in project:
            raise ValueError('Managed namespace and project must be individual slugs')
        self.namespace, self.project = namespace, project
        self._provider = token_provider
        self.dreamlake_url = _url(dreamlake_url)
        self._root = f'{self.dreamlake_url}/managed-experiments/ns/{quote(namespace, safe="")}/projects/{quote(project, safe="")}'
        self._setup_request = None
        self._experiment_url = None

    def client(self):
        return ManagedRemoteClient(self)

    @classmethod
    def from_login(cls, namespace: str, project: str, dreamlake_url=None, config_dir=None):
        """Reuse the supported DreamLake/managed CLI login without a Dash login.

        Reads the current materialized login at each renewal. Credentials for a
        configured server are never forwarded to a different selected endpoint.
        This method does not write files, invoke Keychain or start a login flow.
        """
        directory = Path(config_dir or os.environ.get('ML_DASH_MANAGED_CONFIG_DIR') or Path.home() / '.dreamlake')

        def read(name):
            try:
                value = json.loads((directory / name).read_text())
                return value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                return {}

        configured = _url(read('config.json').get('remote_url') or 'https://api.dreamlake.ai')
        selected = _url(dreamlake_url or os.environ.get('DREAMLAKE_REMOTE') or configured)

        def provider():
            current = _url(read('config.json').get('remote_url') or 'https://api.dreamlake.ai')
            environment_url = _url(os.environ.get('DREAMLAKE_REMOTE') or current)
            override = os.environ.get('DREAMLAKE_API_KEY')
            if override and selected == environment_url:
                return override
            # Re-read config as well as tokens. An environment switch cannot
            # cause an existing connection to send the new environment's token.
            token = read('tokens.json').get('dreamlake-token') if selected == current else None
            if not isinstance(token, str) or not token:
                raise ValueError('No DreamLake login for this server; run ml-dash auth login for the intended environment')
            return token

        return cls(selected, namespace, project, provider)

    def setup(self):
        with httpx.Client(follow_redirects=False, timeout=15) as http:
            headers = {'Authorization': _authorization(self._provider())}
            if self._setup_request is None:
                preview = http.get(f'{self._root}/setup', headers=headers)
                preview.raise_for_status()
                plan = preview.json()
                self._setup_request = {'expectedRevision': plan['revision'], 'planHash': plan['planHash'], 'idempotencyKey': str(uuid.uuid4())}
            response = http.post(f'{self._root}/setup', headers=headers, json=self._setup_request)
            response.raise_for_status()
            return response.json()


class ManagedRemoteClient:
    def __init__(self, connection: ManagedConnection):
        self.connection = connection
        self._http = httpx.Client(follow_redirects=False, timeout=30)
        self._lock = threading.RLock()
        self._credential = None
        self._destination = None
        self._workload_id = None
        self._creation_key = str(uuid.uuid4())

    def _session(self):
        with self._lock:
            if self._credential and self._credential['expiresAt'] > time.time() + 5:
                return self._credential
            response = self._http.post(f'{self.connection._root}/session',
                headers={'Authorization': _authorization(self.connection._provider())}, json={
                    'mode': 'write', **({'workloadId': self._workload_id} if self._workload_id else {'idempotencyKey': self._creation_key})})
            response.raise_for_status()
            result = response.json()
            expires = result.get('expiresAt')
            if not isinstance(expires, (int, float)) or not time.time() < expires <= time.time() + 61 or not isinstance(result.get('token'), str) or not result['token']:
                raise ValueError('Invalid managed credential response')
            destination = (_url(result['apiUrl']), result['instanceId'], result['projectId'])
            if not all(isinstance(v, str) and v for v in destination) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', destination[2]):
                raise ValueError('Invalid managed destination')
            if self._destination and self._destination != destination:
                raise ValueError('Managed destination changed; reconnect explicitly')
            self._destination = destination
            _authorization(result['token'])
            result['apiUrl'] = destination[0]
            if result.get('experimentUrl') is not None:
                view = _url(result['experimentUrl'])
                if self.connection._experiment_url and self.connection._experiment_url != view:
                    raise ValueError('Managed project page changed; reconnect explicitly')
                self.connection._experiment_url = view
            self._workload_id = result.get('workloadId')
            self._credential = result
            return result

    def _request(self, method, path, data):
        session = self._session()
        url = f'{session["apiUrl"]}/projects/{quote(session["projectId"], safe="")}{path}'
        response = self._http.request(method, url, headers={'Authorization': _authorization(session['token'])}, json=data)
        if response.status_code in (401, 403):
            # Let the caller/buffer retain this failed batch; its next explicit
            # retry rechecks current DreamLake policy and workload generation.
            with self._lock:
                self._credential = None
        # No implicit replay of writes after an ambiguous response. The existing
        # buffer retains failures and reports them from Experiment.flush().
        response.raise_for_status()
        return response.json()

    def upload_file(self, experiment_id, file_path, prefix, filename, description=None,
                    tags=None, metadata=None, checksum=None, content_type=None, size_bytes=None, **kwargs):
        if prefix not in ('', '/') or description or tags or metadata:
            raise ValueError('Managed attachment protocol currently accepts root files without metadata')
        with open(file_path, 'rb') as source:
            data = source.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError('Managed attachment exceeds the 8 MiB protocol limit')
        digest = hashlib.sha256(data).hexdigest()
        if (checksum is not None and digest != checksum) or (size_bytes is not None and len(data) != size_bytes):
            raise ValueError('Attachment changed after it was queued')
        return self._request('POST', f'/runs/{quote(experiment_id, safe="")}/files', {
            'filename': filename, 'contentType': content_type or 'application/octet-stream',
            'dataBase64': base64.b64encode(data).decode('ascii')})

    def read_metric_rows(self, experiment_id, metric_name, limit=1000, cursor=None):
        suffix = f'/runs/{quote(experiment_id, safe="")}/metrics/{quote(metric_name, safe="")}?limit={int(limit)}'
        if cursor is not None:
            suffix += f'&cursor={quote(cursor, safe="")}'
        return self._request('GET', suffix, None)

    def get_parameters(self, experiment_id):
        return self._request('GET', f'/runs/{quote(experiment_id, safe="")}/parameters', None)

    def create_or_update_experiment(self, project, name, prefix=None, **kwargs):
        expected = f'{self.connection.namespace}/{self.connection.project}/{name}'
        if project != self.connection.project or prefix != expected or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', name):
            raise ValueError('Managed prefix must be the selected namespace/project/run (no nested folders)')
        if kwargs.get('write_protected') or any(kwargs.get(k) for k in ('description', 'tags', 'bindrs', 'metadata')):
            raise ValueError('Managed minimal protocol does not yet support run metadata or write protection')
        return self._request('PUT', f'/runs/{quote(name, safe="")}', {})

    def set_parameters(self, experiment_id, data):
        return self._request('POST', f'/runs/{quote(experiment_id, safe="")}/parameters', {'data': data})

    def append_batch_to_metric(self, experiment_id, metric_name, data_points, **kwargs):
        if any(kwargs.get(k) for k in ('description', 'tags', 'metadata')):
            raise ValueError('Managed metric metadata is not supported by the minimal protocol')
        return self._request('POST', f'/runs/{quote(experiment_id, safe="")}/metrics/{quote(metric_name, safe="")}', {'dataPoints': data_points})

    def append_to_metric(self, experiment_id, metric_name, data, **kwargs):
        return self.append_batch_to_metric(experiment_id, metric_name, [data], **kwargs)

    def create_log_entries(self, experiment_id, logs):
        return self._request('POST', f'/runs/{quote(experiment_id, safe="")}/logs', {'entries': logs})

    def update_experiment_status(self, experiment_id, status):
        return self._request('POST', f'/runs/{quote(experiment_id, safe="")}/status', {'status': status})

    def close(self):
        self._credential = None
        self._http.close()
