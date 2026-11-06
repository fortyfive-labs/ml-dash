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
import os
import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit

import httpx

from .exceptions import MetricRowsError, MlDashError, NetworkError, StorageError
from .managed import _authorization, _login_provider, _url
from .metric_rows import MetricColumn, MetricRowBlock, MetricRowsPage
from .utils import _serialize_value

_KEY = re.compile(r'[A-Za-z0-9_-]{1,128}')
_ID = re.compile(r'[0-9a-f]{24}')
# A workload credential id: its creating intent's UUID, lower-case.
_WORKLOAD = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}')
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


def _workload_id(value, name):
    if not isinstance(value, str) or not _WORKLOAD.fullmatch(value.lower()):
        raise ValueError(f'{name} must be a UUID')
    return value.lower()


class NativeConnection:
    """Select a DreamLake project whose runs use DreamLake-native storage.

    Pass it as ``Experiment(prefix="namespace/project/run", managed=connection)``.
    ``token_provider`` returns the user's current DreamLake login; it is called
    whenever a session must be minted, so a renewed login is picked up.

    Writes go through a workload credential (a writer), as ``ManagedConnection``'s do:

    - by default each client registers one writer on its first session, under a fresh
      intent UUID, and renews with that writer's id;
    - ``idempotency_key`` registers (or reuses) the writer of that intent instead;
    - ``workload_id`` uses an existing writer, for example in CI with an explicitly
      provided ``DREAMLAKE_API_KEY``. A retired or unknown writer is an error; no
      replacement is ever registered;
    - ``run_id`` binds the writer to that one existing run: the experiment opens that run
      by id, and no other run is written or created.
    - a writer bound to a run is reused with both: ``workload_id=ID, run_id=RUN``. Without
      ``run_id`` its session is rejected as invalid; the SDK never widens or infers the scope.

    The credential is metadata only: sessions come from the login, last 60 seconds, and
    use the person's current project role. Manage writers with ``list_credentials``,
    ``get_credential``, ``create_credential``, ``rotate_credential`` and ``retire_credential``.
    """
    platform_controls_status = False

    def __init__(self, dreamlake_url: str, namespace: str, project: str,
                 token_provider: Callable[[], str], auto_create_project: bool = False, *,
                 workload_id=None, idempotency_key=None, run_id=None):
        if not namespace or not project or '/' in namespace or '/' in project:
            raise ValueError('Native namespace and project must be individual slugs')
        if workload_id is not None and idempotency_key is not None:
            raise ValueError('Pass a workload_id or an idempotency_key, not both')
        if run_id is not None and (not isinstance(run_id, str) or not _ID.fullmatch(run_id)):
            raise ValueError('run_id must be a native run id')
        self.namespace, self.project = namespace, project
        self.dreamlake_url = _url(dreamlake_url)
        self._provider = token_provider
        self._path = f'/native-experiments/ns/{quote(namespace, safe="")}/projects/{quote(project, safe="")}'
        self._root = self.dreamlake_url + self._path
        self._experiment_url = None
        # The official API has a known web origin. Custom deployments keep the
        # existing fallback rather than sending people to an unrelated site.
        if self.dreamlake_url == 'https://api.dreamlake.ai':
            self._experiment_url = (f'https://dreamlake.ai/{quote(namespace, safe="")}/projects/'
                                    f'{quote(project, safe="")}/experiments?provider=dreamlake-native-v1')
        self.workload_id = None if workload_id is None else _workload_id(workload_id, 'workload_id')
        self.idempotency_key = None if idempotency_key is None else _workload_id(idempotency_key, 'idempotency_key')
        self.run_id = run_id

        # As ml-dash's remote experiments: a missing project is created (private) before the first write, through
        # DreamLake's project API and the user's own permission there; an existing one is used as is.
        self.auto_create_project = auto_create_project

    @staticmethod
    def default_namespace(dreamlake_url=None, config_dir=None) -> str:
        """The namespace of the DreamLake login on this machine: ``ML_DASH_NATIVE_NAMESPACE``, else the active
        profile's ``namespace`` (``~/.dreamlake/config.json``, for the login's server), else ``/auth/me``."""
        explicit = os.environ.get('ML_DASH_NATIVE_NAMESPACE')
        if explicit:
            return explicit
        selected, provider = _login_provider(dreamlake_url, config_dir)
        directory = Path(config_dir or os.environ.get('ML_DASH_MANAGED_CONFIG_DIR') or Path.home() / '.dreamlake')
        try:
            config = json.loads((directory / 'config.json').read_text())
        except (OSError, ValueError):
            config = {}
        if isinstance(config, dict) and isinstance(config.get('namespace'), str) and config['namespace'] \
                and _url(config.get('remote_url') or 'https://api.dreamlake.ai') == selected:
            return config['namespace']
        try:
            response = httpx.get(selected + '/auth/me', headers={'Authorization': _authorization(provider())}, timeout=30)
        except httpx.HTTPError:
            raise NativeExperimentError(None, 'DreamLake unavailable while resolving the login namespace') from None
        if response.status_code != 200:
            raise _error(response, 'DreamLake login namespace')
        slug = ((response.json() or {}).get('namespace') or {}).get('slug')
        if not isinstance(slug, str) or not slug:
            raise NativeExperimentError(None, 'The DreamLake login has no namespace; set ML_DASH_NATIVE_NAMESPACE')
        return slug

    @classmethod
    def anonymous(cls, dreamlake_url: str, namespace: str, project: str):
        """Read a public project without a login (as ml-dash's PUBLIC projects): reads only; any write raises."""
        return cls(dreamlake_url, namespace, project, None)

    @classmethod
    def from_login(cls, namespace: str, project: str, dreamlake_url=None, config_dir=None, auto_create_project=False, **writer):
        """Use the DreamLake login already on this machine (``~/.dreamlake``), or an
        explicitly provided ``DREAMLAKE_API_KEY`` for ``DREAMLAKE_REMOTE``.

        Same login source and server-pinning rules as
        ``ManagedConnection.from_login``. Does not write files or start a login.
        ``writer`` takes the ``workload_id``/``idempotency_key``/``run_id`` options.
        """
        selected, provider = _login_provider(dreamlake_url, config_dir)
        return cls(selected, namespace, project, provider, auto_create_project=auto_create_project, **writer)

    def client(self):
        return NativeRemoteClient(self)

    # -- workload credentials (metadata only; the person's login) --------------------

    def _manage(self, method: str, path: str, action: str, **kwargs) -> dict:
        try:
            with httpx.Client(follow_redirects=False, timeout=30) as http:
                response = http.request(method, self._root + path,
                                        headers={'Authorization': _authorization(self._provider())}, **kwargs)
        except httpx.HTTPError:
            raise NativeExperimentError(None, f'{action} failed: DreamLake unavailable') from None
        if response.status_code != 200:
            raise _error(response, action)
        try:
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError
            return result
        except (ValueError, TypeError):
            raise NativeExperimentError(None, f'{action} returned an invalid response') from None

    def list_credentials(self, cursor=None, limit=None) -> dict:
        """Your writers in this project: ``{credentials, hasMore, nextCursor}``."""
        params = {}
        if cursor is not None:
            params['cursor'] = _workload_id(cursor, 'cursor')
        if limit is not None:
            params['limit'] = str(int(limit))
        return self._manage('GET', '/credentials', 'Native credential list', params=params)

    def get_credential(self, credential_id) -> dict:
        return self._manage('GET', f'/credentials/{_workload_id(credential_id, "credential_id")}', 'Native credential inspect')['credential']

    def create_credential(self, idempotency_key, run_id=None) -> dict:
        """Register the writer of this intent (or return the one it registered)."""
        body = {'operation': 'create', 'idempotencyKey': _workload_id(idempotency_key, 'idempotency_key')}
        if run_id is not None:
            body['runId'] = run_id
        return self._manage('POST', '/credentials', 'Native credential create', json=body)['credential']

    def rotate_credential(self, credential_id, expected_generation) -> dict:
        """End the writer's current sessions; its next session uses the new generation."""
        return self._manage('POST', '/credentials', 'Native credential rotate', json={'operation': 'rotate',
            'credentialId': _workload_id(credential_id, 'credential_id'), 'expectedGeneration': expected_generation})['credential']

    def retire_credential(self, credential_id, expected_generation) -> dict:
        """End the writer: its sessions stop and no new one is issued."""
        return self._manage('POST', '/credentials', 'Native credential retire', json={'operation': 'retire',
            'credentialId': _workload_id(credential_id, 'credential_id'), 'expectedGeneration': expected_generation})['credential']


def _segment(value) -> bool:
    """A folder or run name as the server takes it: not blank, not ``.``/``..``, no ``/`` or control character."""
    return (isinstance(value, str) and value.strip() != '' and value not in ('.', '..') and '/' not in value
            and not re.search(r'[\x00-\x1f\x7f]', value))


_ITEM_KINDS = ('metric', 'file', 'parameters')


def _bindr_names(bindrs):
    if bindrs is None:
        return []
    if isinstance(bindrs, str) or not all(isinstance(name, str) and name.strip() for name in bindrs):
        raise ValueError('bindrs must be a list of non-empty bindr names')
    return list(dict.fromkeys(name.strip() for name in bindrs))


def _item_kind(kind):
    if kind not in _ITEM_KINDS:
        raise ValueError('item_kind must be metric, file or parameters')
    return kind


def _native_id(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f'{name} must be a native id')
    return value


def _bindr_target(name, bindr_id, item_kind, item_id):
    if (name is None) == (bindr_id is None):
        raise ValueError('Pass a bindr name or a bindr_id')
    body = {'name': _bindr_names([name])[0]} if name is not None else {'bindrId': _native_id(bindr_id, 'bindr_id')}
    if item_kind is not None or item_id is not None:
        body['item'] = {'kind': _item_kind(item_kind), 'id': _native_id(item_id, 'item_id')}
    return body


def _confirmed_bindrs(result, names):
    """A create that asked for bindrs must answer every one of them attached; anything less is a failed create."""
    if names:
        attached = {bindr.get('name') for bindr in result.get('bindrs') or [] if isinstance(bindr, dict)}
        missing = [name for name in names if name not in attached]
        if missing:
            raise NativeExperimentError(None, f'The server did not confirm the run\'s bindrs {missing}')
    return result


def _error(response: httpx.Response, action: str) -> NativeExperimentError:
    try:
        body = response.json()
        error = body.get('error') if isinstance(body, dict) else None
        # Native routes answer {error: {code}}; DreamLake's other routes (project create) {error: "message"}.
        code = error.get('code') if isinstance(error, dict) else None
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
        # One short-lived session per mode: writes use the writer's session, reads a read session (a READ grant
        # suffices), unless a writer is selected explicitly: then every request uses that writer's scoped session.
        self._sessions = {}
        self._token = None  # a subclass session (`native_job.NativeJobDataClient`) renews on None
        # Creating a missing project is the interactive person's bootstrap only (ml-dash's rdxp): an explicit writer
        # (or a job's broker, or an anonymous connection) never uses the login to create one; its session decides.
        self._project_ready = not connection.auto_create_project or self._explicit_writer() or connection._provider is None
        # As ManagedRemoteClient: one creation intent per client, then the writer it registered.
        self.workload_id = connection.workload_id
        self._creation_key = connection.idempotency_key or str(uuid.uuid4())

    # -- session ----------------------------------------------------------------

    def _ensure_project(self):
        """The interactive person's project bootstrap: an existing project (any role that may read it, e.g. a WRITE
        collaborator without namespace create) is used as it is; only one the server reports missing (404) is
        created (private). Any other answer is the server's refusal and is raised; nothing is created then."""
        connection = self.connection
        namespace = f'{connection.dreamlake_url}/namespaces/{quote(connection.namespace, safe="")}/projects'
        headers = {'Authorization': _authorization(connection._provider())}
        try:
            found = self._http.get(f'{namespace}/{quote(connection.project, safe="")}', headers=headers)
            if found.status_code == 404:
                response = self._http.post(namespace, json={'name': connection.project, 'slug': connection.project,
                                                            'visibility': 'private'}, headers=headers)
        except httpx.HTTPError:
            raise NativeExperimentError(None, 'DreamLake unavailable while creating the project') from None
        if found.status_code == 404:
            # 409: created meanwhile (or not visible to this login): the project session decides the access.
            if response.status_code not in (201, 409):
                raise _error(response, 'DreamLake project create')
        elif found.status_code != 200:
            raise _error(found, 'DreamLake project read')
        self._project_ready = True

    def _session_token(self, mode: str = 'write') -> str:
        with self._lock:
            if mode == 'write' and not self._project_ready:
                self._ensure_project()
            current = self._sessions.get(mode)
            if current and time.time() < current[1]:
                return current[0]
            self._sessions.pop(mode, None)
            try:
                if mode == 'write':
                    body = {'mode': 'write', **({'workloadId': self.workload_id} if self.workload_id
                                                else {'idempotencyKey': self._creation_key})}
                    if self.connection.run_id:
                        body['runId'] = self.connection.run_id
                else:
                    body = {'mode': 'read'}
                response = self._http.post(self.connection._root + '/session', json=body,
                    headers={'Authorization': _authorization(self.connection._provider())})
            except httpx.HTTPError:
                raise NativeExperimentError(None, 'DreamLake native session unavailable') from None
            if response.status_code != 200:
                raise _error(response, f'DreamLake native {mode} session')
            try:
                result = response.json()
                token, api_url = result['token'], result['apiUrl']
                # The session is only ever sent to this connection's own server; the
                # advertised destination must name the same project route.
                if urlsplit(_url(api_url)).path != self.connection._path:
                    raise ValueError
                _authorization(token)
                # A write session must be for exactly the writer (and run) asked for.
                writer = result['workloadId'] if mode == 'write' else None
                if mode == 'write' and (writer != (self.workload_id or self._creation_key) or not isinstance(result['generation'], int)
                        or result.get('runId') != self.connection.run_id):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                raise NativeExperimentError(None, 'Invalid DreamLake native session response') from None
            if mode == 'write':
                self.workload_id = writer
            self._sessions[mode] = (token, time.time() + _SESSION_SECONDS)
            return token

    def _token_for(self, mode: str) -> str:
        # A subclass with its own session (`native_job.NativeJobDataClient`) overrides `_session_token()`
        # without modes: its single session is used for reads and writes alike.
        if type(self)._session_token is not NativeRemoteClient._session_token:
            return self._session_token()
        # An explicitly selected writer scopes every request, reads included, to its session.
        return self._session_token('write' if self._explicit_writer() else mode)

    def _explicit_writer(self) -> bool:
        connection = self.connection
        return bool(getattr(connection, 'workload_id', None) or getattr(connection, 'idempotency_key', None)
                    or getattr(connection, 'run_id', None))

    def _send(self, method: str, path: str, action: str, stream: bool = False, **kwargs) -> httpx.Response:
        """One request with the right session; a 401 renews it once. ``stream``: the body is left unread for the
        caller, who must close the response."""
        read = method == 'GET'
        # A connection without a login provider is anonymous; one without the attribute (a job's broker
        # destination, `native_job._BrokerDestination`) carries its own session.
        anonymous = getattr(self.connection, '_provider', True) is None
        if anonymous and not read:
            raise NativeExperimentError(401, f'{action} needs a DreamLake login (this connection is anonymous)', 'NATIVE_LOGIN_REQUIRED')
        # The session mode actually used (an explicit writer reads with its write session), so a 401 clears that one.
        mode = 'write' if not read or self._explicit_writer() else 'read'
        for attempt in range(2):
            headers = {} if anonymous else {'Authorization': _authorization(self._token_for(mode))}
            try:
                response = self._http.send(self._http.build_request(method, self.connection._root + path, headers=headers, **kwargs),
                                           stream=stream)
            except httpx.HTTPError:
                raise NativeExperimentError(None, f'{action} failed: DreamLake unavailable; '
                                                  'reconcile before retrying writes') from None
            if response.status_code == 401 and attempt == 0 and not anonymous:
                # 401 is decided before any write, so one retry with a fresh session is safe.
                response.close()
                with self._lock:
                    self._sessions.pop(mode, None)
                    self._token = None  # a subclass session (`NativeJobDataClient`) renews on this
                continue
            if response.is_success:
                return response
            if stream:
                # The refusal body carries its code; a body that cannot be read is never parsed.
                try:
                    response.read()
                except httpx.HTTPError:
                    raise NativeExperimentError(None, f'{action} failed: DreamLake unavailable (HTTP {response.status_code}); '
                                                      'reconcile before retrying writes') from None
                finally:
                    response.close()
            error = _error(response, action)
            if error.code == 'NATIVE_CREDENTIAL_RETIRED':
                # Rotated: the next explicit call renews under the new generation. Retired: that
                # renewal is refused. Nothing is replayed here (the failed write stays failed).
                with self._lock:
                    self._sessions.clear()
                    self._token = None
            raise error
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
        """Create the run at ``prefix`` (``namespace/project/[folders/]name``), or return the one there.

        Folders are created as needed and names may be any Unicode (as ml-dash's). ``description``/``tags``
        of None leave the stored value unchanged; ``""`` clears the description and ``[]`` the tags.
        ``metadata`` (a dict) replaces the run's metadata. ``write_protected`` (``Experiment(_write_protected=
        True)``) creates the run protected, with its metadata, in one server transaction, as ml-dash; it is a
        person's action, so only an interactive connection asks for it, over the person's own session.
        """
        connection = self.connection
        head = f'{connection.namespace}/{connection.project}/'
        segments = prefix[len(head):].split('/') if isinstance(prefix, str) and prefix.startswith(head) else []
        if (project != connection.project or not segments or segments[-1] != name
                or not all(_segment(segment) for segment in segments)):
            raise ValueError('Native prefix must be the selected namespace/project, then folders and the run name')
        names = _bindr_names(bindrs)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Native run metadata must be a dict')
        # Interactive means no writer option at all (`_explicit_writer`); the server decides finally.
        if write_protected and self._explicit_writer():
            # Never the person's login behind an explicit writer: protection is the person's own action.
            raise ValueError('Write protection is a person\'s action: use a NativeConnection without '
                             'workload_id, idempotency_key or run_id')
        body = {}
        if description is not None:
            body['description'] = description
        if tags is not None:
            body['tags'] = list(tags)
        if metadata is not None:
            body['metadata'] = metadata
        if connection.run_id:
            # A writer bound to a run opens that run by id, never by key: a renamed run or a reused
            # key cannot redirect it. Its description/tags/metadata are updated in place; its bindrs are attached.
            result = (self._json('PATCH', self._run(connection.run_id), 'Native run update', json=body) if body
                      else self._json('GET', self._run(connection.run_id), 'Native run read'))
            for name in names:
                self.attach_bindr(connection.run_id, name=name)
            return result
        if names:
            # Found, or created at the top level, and attached with the create (before a protected birth), as ml-dash.
            body['bindrs'] = names
        # A root ASCII key keeps today's route (no run lifecycle needed); any other path is a run path.
        path = (f'/runs/{quote(name, safe="")}' if len(segments) == 1 and _KEY.fullmatch(name)
                else '/run-paths/' + '/'.join(quote(segment, safe='') for segment in segments))
        if write_protected:
            body['writeProtected'] = True
            return _confirmed_bindrs(self._person_json('PUT', path, 'Native run create', json=body), names)
        return _confirmed_bindrs(self._json('PUT', path, 'Native run create', json=body), names)

    # -- bindrs (canonical labels of a run or of one of its items) ---------------------------------------------

    def run_bindrs(self, experiment_id, item_kind=None, item_id=None):
        """The bindrs a run (or one item of it: ``item_kind`` metric | file | parameters, ``item_id``) is in:
        ``[{id, name, parent}]``. Read with the connection's read session (a selected writer's own, within its run)."""
        params = {}
        if item_kind is not None or item_id is not None:
            params = {'itemKind': _item_kind(item_kind), 'itemId': _native_id(item_id, 'item_id')}
        return self._json('GET', self._run(experiment_id) + '/bindrs', 'Native run bindrs', params=params)['bindrs']

    def attach_bindr(self, experiment_id, name=None, bindr_id=None, item_kind=None, item_id=None):
        """Attach the run (or one item) to a bindr, by ``name`` (found, or created at the top level) or ``bindr_id``.
        Idempotent: ``attached`` is False when it already was. A writer may, within its project and run."""
        body = _bindr_target(name, bindr_id, item_kind, item_id)
        return self._json('POST', self._run(experiment_id) + '/bindrs', 'Native bindr attach', json=body)

    def detach_bindr(self, experiment_id, name=None, bindr_id=None, item_kind=None, item_id=None):
        """Detach the run (or one item) from a bindr (the bindr stays). ``detached`` is False when it was not attached."""
        body = _bindr_target(name, bindr_id, item_kind, item_id)
        return self._json('DELETE', self._run(experiment_id) + '/bindrs', 'Native bindr detach', json=body)

    def files_in_bindr(self, experiment_id, bindr_name):
        """``experiment.bindrs(name).list()``: the run's files labelled with that bindr (each file with its ``bindrs`` names)."""
        found = []
        for file in self.list_files(experiment_id, limit=100000)['files']:
            names = [bindr['name'] for bindr in self.run_bindrs(experiment_id, 'file', file['id'])]
            if bindr_name in names:
                found.append({**file, 'bindrs': names})
        return found

    def _person_session(self) -> str:
        """A plain project write session of the person (no writer), for one person's action; never cached."""
        with self._lock:
            if not self._project_ready:
                self._ensure_project()
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
            # Only ever sent to this connection's own server and project route, and a person's session only.
            advertised, own = urlsplit(_url(api_url)), urlsplit(self.connection.dreamlake_url)
            if ((advertised.scheme, advertised.netloc, advertised.path) != (own.scheme, own.netloc, self.connection._path)
                    or result.get('workloadId') is not None):
                raise ValueError
            _authorization(token)
        except (ValueError, KeyError, TypeError):
            raise NativeExperimentError(None, 'Invalid DreamLake native session response') from None
        return token

    def _person_json(self, method: str, path: str, action: str, **kwargs) -> dict:
        """One request under a fresh person's session: a 401 (decided before any write) renews it once; any
        other failure is not replayed. A refusal is final: there is no fallback to a writer's session."""
        if getattr(self.connection, '_provider', None) is None:
            # Anonymous (or no login at all): refused before any request, as every native write is.
            raise NativeExperimentError(401, f'{action} needs a DreamLake login (this connection is anonymous)', 'NATIVE_LOGIN_REQUIRED')
        for attempt in range(2):
            token = self._person_session()
            try:
                response = self._http.request(method, self.connection._root + path,
                    headers={'Authorization': _authorization(token)}, **kwargs)
            except httpx.HTTPError:
                raise NativeExperimentError(None, f'{action} failed: DreamLake unavailable; '
                                                  'reconcile before retrying writes') from None
            if response.status_code == 401 and attempt == 0:
                continue
            if not response.is_success:
                raise _error(response, action)
            try:
                result = response.json()
                if not isinstance(result, dict):
                    raise TypeError
                return result
            except (ValueError, TypeError):
                raise NativeExperimentError(None, f'{action} returned an invalid response') from None
        raise AssertionError('unreachable')

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

    def parameters_id(self, experiment_id):
        """The id of the run's current parameters (the ``parameters`` item of its bindr labels), or None before any."""
        result = self._json('GET', self._run(experiment_id) + '/parameters', 'Native parameters read')
        return (result.get('parameters') or {}).get('id')

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

    def get_metric_data(self, experiment_id, metric_name, start_index=None, limit=None, buffer_only=False):
        """As ``RemoteClient.get_metric_data``: unsupported for remote metrics (no server-assigned indices);
        raises ``ConfigurationError`` without a request. Use ``read_metric_rows``."""
        from .client import _index_read_unsupported
        raise _index_read_unsupported('get_metric_data')

    def download_metric_chunk(self, experiment_id, metric_name, chunk_number):
        """As ``RemoteClient.download_metric_chunk``: unsupported for remote metrics (no chunk numbers);
        raises ``ConfigurationError`` without a request. Use ``read_metric_rows``."""
        from .client import _index_read_unsupported
        raise _index_read_unsupported('download_metric_chunk')

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
    # ml-dash tracks (``exp.tracks(topic)``) on the run's native track store, whichever home holds the run.

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

    # -- files: a run's, or the project's (ml-dash's file methods take any file of the project) -------------

    def _pinned_run(self):
        """The one run this client may touch without being told (a job's), or None."""
        return None

    def _file_route(self, experiment_id, file_id):
        """The route of a file of the selected project. ml-dash's ``experiment_id`` is optional (its file methods
        address any file node by id): given, the file is that run's (a job or a run-bound writer still only its own
        run); None, the file is located by id within the selected project, a run's or the project's own (README,
        ``.dashrc``), with this client's scoped session, never the login."""
        if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
            raise ValueError('Invalid native file id')
        if experiment_id is None:
            experiment_id = self._pinned_run()
        if experiment_id is not None:
            return self._file(experiment_id, file_id)
        found = self._locate_file(file_id)
        return self._file(found, file_id) if found else f'/project-files/{file_id}'

    def _locate_file(self, file_id):
        """The run id of a file of the selected project, or None for a project file; NATIVE_FILE_NOT_FOUND otherwise.
        ml-dash's ``node(nodeId)``, sent with the session this client reads with (a project session sees only its
        project; anonymous: a public project), and pinned to the selected namespace and project."""
        connection = self.connection
        anonymous = getattr(connection, '_provider', True) is None
        mode = 'write' if self._explicit_writer() else 'read'
        headers = {} if anonymous else {'Authorization': _authorization(self._token_for(mode))}
        query = 'query($id: String!) { node(nodeId: $id) { id type experimentId project { slug namespace { slug } } } }'
        try:
            response = self._http.post(connection.dreamlake_url + '/native-dash/graphql',
                                       json={'query': query, 'variables': {'id': file_id}}, headers=headers)
        except httpx.HTTPError:
            raise NativeExperimentError(None, 'Native file lookup failed: DreamLake unavailable') from None
        if response.status_code != 200:
            raise _error(response, 'Native file lookup')
        try:
            body = response.json()
            node = (body.get('data') or {}).get('node')
            project = (node or {}).get('project') or {}
        except (ValueError, AttributeError):
            raise NativeExperimentError(None, 'Native file lookup returned an invalid response') from None
        if (not node or body.get('errors') or node.get('type') != 'FILE' or node.get('id') != file_id
                or project.get('slug') != connection.project or (project.get('namespace') or {}).get('slug') != connection.namespace):
            raise NativeExperimentError(404, 'Native file not found in the selected project', 'NATIVE_FILE_NOT_FOUND')
        run_id = node.get('experimentId')
        if run_id is not None and not (isinstance(run_id, str) and _ID.fullmatch(run_id)):
            raise NativeExperimentError(None, 'Native file lookup returned an invalid response')
        return run_id

    @staticmethod
    def _file_view(view):
        """The native file, plus the fields ml-dash's node answer had (``name``, ``type``, ``physicalFile``)."""
        if not isinstance(view, dict):
            return view
        return {**view, 'name': view.get('filename'), 'type': 'FILE', 'physicalFile': {key: view.get(key)
                for key in ('filename', 'checksum', 'sizeBytes', 'contentType')}}

    def get_file(self, experiment_id, file_id):
        """A file of the selected project: a run's (``experiment_id``) or, with None, any of the project's."""
        return self._file_view(self._json('GET', self._file_route(experiment_id, file_id), 'Native file read')['file'])

    def _project_file_change(self, method, route, action, **kwargs):
        """A project file is changed only by a person (the server takes no session for it): explicit management with
        this connection's own login. A selected writer, a job or an anonymous connection is refused before any
        request; none ever falls back to the login."""
        if self._explicit_writer() or getattr(self.connection, '_provider', None) is None:
            raise NativeExperimentError(403, f'{action}: a project file is changed with a DreamLake login, not a writer '
                                             'or job session', 'NATIVE_FORBIDDEN')
        return self.connection._manage(method, route, action, **kwargs)

    def delete_file(self, experiment_id, file_id):
        """Delete the file, as ml-dash: it leaves every listing and read at once and cannot be restored
        (re-uploading the same path creates it anew). Returns ``{id, nodeId, deleted}``."""
        route = self._file_route(experiment_id, file_id)
        if route.startswith('/project-files/'):
            return self._project_file_change('DELETE', route, 'Native project file delete')
        return self._json('DELETE', route, 'Native file delete')

    def update_file(self, experiment_id, file_id, description=None, tags=None, metadata=None):
        """Set description, tags and/or metadata (None leaves a field unchanged)."""
        body = {key: value for key, value in (('description', description), ('tags', tags), ('metadata', metadata))
                if value is not None}
        route = self._file_route(experiment_id, file_id)
        if route.startswith('/project-files/'):
            return self._file_view(self._project_file_change('PATCH', route, 'Native project file update', json=body)['file'])
        return self._file_view(self._json('PATCH', route, 'Native file update', json=body)['file'])

    def download_file_streaming(self, experiment_id, file_id, dest_path):
        """As ``RemoteClient.download_file_streaming``: the file streamed to ``dest_path`` (never held in memory),
        verified against its stored sha256 before it appears there; a mismatch raises ``StorageError`` and leaves
        nothing at ``dest_path``. Returns ``dest_path``. An aborted or failed transfer leaves no partial file."""
        path = self._file_route(experiment_id, file_id)
        checksum = self._json('GET', path, 'Native file read')['file']['checksum']
        target = Path(dest_path)
        temp = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')
        response = self._send('GET', path + '/download', 'Native file download', stream=True, timeout=300)
        digest = hashlib.sha256()
        try:
            with open(temp, 'xb') as out:
                for chunk in response.iter_bytes(chunk_size=65536):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != checksum:
                raise StorageError(f'Checksum verification failed for file {file_id}')
            os.replace(temp, target)
        except httpx.HTTPError:
            raise NativeExperimentError(None, 'Native file download failed: DreamLake unavailable') from None
        finally:
            response.close()
            temp.unlink(missing_ok=True)
        return dest_path

    def get_current_user(self):
        """As ``RemoteClient.get_current_user``: the DreamLake login's profile ``{id, username, email, name, given_name,
        family_name, picture}`` (``username``: the login's namespace), from DreamLake's ``/auth/me``. Explicit
        management with the person's own login: an anonymous connection, or a job's broker client, has none and
        raises ``AuthenticationError``; a selected writer's session is never used for it."""
        from .auth.exceptions import AuthenticationError
        provider = getattr(self.connection, '_provider', None)
        if provider is None:
            raise AuthenticationError('Not authenticated: get_current_user needs a DreamLake login')
        try:
            response = self._http.get(self.connection.dreamlake_url + '/auth/me', headers={'Authorization': _authorization(provider())})
        except httpx.HTTPError as error:
            raise NetworkError(f'Failed to fetch current user from server: {type(error).__name__}') from None
        if response.status_code == 401:
            raise AuthenticationError('Not authenticated: the DreamLake login was refused')
        if response.status_code != 200:
            raise NetworkError(f'Failed to fetch current user from server: HTTP {response.status_code}')
        try:
            me = response.json()
            return {'id': me['id'], 'username': (me.get('namespace') or {}).get('slug'), 'email': me.get('email'),
                    'name': me.get('name'), 'given_name': me.get('given_name'), 'family_name': me.get('family_name'),
                    'picture': me.get('picture')}
        except (ValueError, KeyError, TypeError, AttributeError):
            raise NetworkError('Failed to fetch current user from server: invalid response') from None

    def download_file(self, experiment_id, file_id, dest_path=None):
        path = self._file_route(experiment_id, file_id)
        info = self._json('GET', path, 'Native file read')['file']
        content = self._send('GET', path + '/download', 'Native file download', timeout=300).content
        if hashlib.sha256(content).hexdigest() != info['checksum']:
            raise NativeExperimentError(None, 'Native file download checksum mismatch')
        target = Path(dest_path or info['filename'])
        target.write_bytes(content)
        return str(target)

    # -- GraphQL (ml-dash's API, served natively at /native-dash/graphql) -------------

    @property
    def namespace(self):
        return self.connection.namespace

    def _graphql_headers(self):
        # Anonymous: public projects only. An explicit writer: its scoped session, never the user's login (the
        # server narrows a writer to its project and run). Otherwise the user's login, as ml-dash's raw queries.
        if self.connection._provider is None:
            return {}
        if self._explicit_writer():
            return {'Authorization': _authorization(self._session_token('write'))}
        return {'Authorization': _authorization(self.connection._provider())}

    def graphql_query(self, query, variables=None):
        """Run an ml-dash GraphQL query or mutation against DreamLake's native compatibility endpoint.

        Sent with the user's DreamLake login (not the project session): ml-dash queries may span projects,
        and every project is authorized by the server. Errors raise ``NetworkError`` as ``RemoteClient`` does.
        """
        try:
            response = self._http.post(self.connection.dreamlake_url + '/native-dash/graphql',
                                       json={'query': query, 'variables': variables or {}},
                                       headers=self._graphql_headers())
        except httpx.HTTPError:
            raise NativeExperimentError(None, 'DreamLake native GraphQL unavailable') from None
        response.raise_for_status()
        result = response.json()
        if result.get('errors'):
            raise NetworkError('; '.join(error.get('message', str(error)) for error in result['errors']))
        return result.get('data') or {}

    def delete_project(self, project_slug=None):
        """As ``RemoteClient.delete_project``: the project with all its runs, metrics, files and logs. Natively this is
        DreamLake's project delete (the person's login and ``project:delete`` role) with the native experiment data
        accepted for permanent deletion in the same transaction (``nativeExperiments=purge``), purged in the background
        as ml-dash's delete was; ``expectedProjectId`` makes it refuse (409) if the slug meanwhile names another project.

        Returns ``{projectId, deleted, message, ...}`` with DreamLake's reply (``project``, ``nativeCleanup``, counts).
        A selected writer, a job session or an anonymous connection never deletes a project.
        """
        connection = self.connection
        provider = getattr(connection, '_provider', None)
        if provider is None or self._explicit_writer():
            raise NativeExperimentError(403, 'Deleting a project needs the DreamLake login, not a writer or job session', 'NATIVE_FORBIDDEN')
        slug = project_slug or connection.project
        url = f'{connection.dreamlake_url}/namespaces/{quote(connection.namespace, safe="")}/projects/{quote(slug, safe="")}'
        try:
            found = self._http.get(url, headers={'Authorization': _authorization(provider())})
            if found.status_code == 404:
                raise NativeExperimentError(404, f"Project '{slug}' not found in namespace '{connection.namespace}'", 'NATIVE_PROJECT_NOT_FOUND')
            if found.status_code != 200:
                raise _error(found, 'DreamLake project read')
            project_id = (found.json() or {}).get('id')
            if not isinstance(project_id, str) or not _ID.fullmatch(project_id):
                raise NativeExperimentError(None, 'DreamLake project read returned an invalid response')
            response = self._http.delete(url, params={'expectedProjectId': project_id, 'nativeExperiments': 'purge'},
                                         headers={'Authorization': _authorization(provider())})
        except httpx.HTTPError:
            raise NativeExperimentError(None, 'DreamLake unavailable while deleting the project') from None
        if response.status_code != 200:
            raise _error(response, 'DreamLake project delete')
        body = response.json()
        if not isinstance(body, dict):
            raise NativeExperimentError(None, 'DreamLake project delete returned an invalid response')
        return {**body, 'projectId': project_id, 'deleted': body.get('deleted', True),
                'message': f"Project '{slug}' deleted; its experiment data is purged in the background"}

    def close(self):
        with self._lock:
            self._sessions.clear()
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


# ml-dash's GraphQL helpers run unchanged on the native endpoint: they only call graphql_query and namespace.
from .client import RemoteClient as _RemoteClient  # noqa: E402

for _name in ('list_projects_graphql', 'list_experiments_graphql', 'get_experiment_graphql',
              'get_experiment_by_path_graphql', 'search_experiments_graphql'):
    setattr(NativeRemoteClient, _name, getattr(_RemoteClient, _name))
del _name
