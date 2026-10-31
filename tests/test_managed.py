"""Managed credentials never consult the independent Dash login."""
import json
import httpx
import pytest

from ml_dash.managed import ManagedConnection
from ml_dash import Experiment


def test_managed_sdk_flush_and_renewal(monkeypatch):
    now = [1000]
    calls = []
    issued = [0]
    def handle(request):
        calls.append((request.method, request.url.path, request.headers.get('authorization'), request.content))
        if request.url.path.endswith('/session'):
            issued[0] += 1
            return httpx.Response(200, json=dict(apiUrl='https://dash.test/api/managed', experimentUrl='https://web.test/only-dl-org/projects/research/experiments', instanceId='managed-test', projectId='p1', workloadId='workload-one', generation=1, token=f'ephemeral-{issued[0]}', expiresAt=now[0]+60))
        if request.method == 'PUT':
            return httpx.Response(200, json={'experiment': {'id': 'run1'}})
        return httpx.Response(200, json={'count': 1})
    monkeypatch.setattr('ml_dash.managed.time.time', lambda: now[0])
    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.managed.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    connection = ManagedConnection('https://dreamlake.test', 'only-dl-org', 'research', lambda: 'synthetic-dreamlake-session')
    with Experiment(prefix='only-dl-org/research/probe', managed=connection).run as exp:
        exp.params.set(lr=0.001)
        assert exp._view_url() == 'https://web.test/only-dl-org/projects/research/experiments?run=run1'
        exp.metrics('train').log(step=0, loss=0.125)
        exp.flush()
        now[0] += 56
        exp.metrics('train').log(step=1, loss=0.1)
        exp.flush()
    assert issued[0] == 2
    sessions = [json.loads(c[3]) for c in calls if c[1].endswith('/session')]
    assert 'idempotencyKey' in sessions[0]
    assert sessions[1]['workloadId'] == 'workload-one'
    metrics = [c for c in calls if '/metrics/' in c[1]]
    assert len(metrics) == 2
    assert metrics[0][2] == 'Bearer ephemeral-1'
    assert metrics[1][2] == 'Bearer ephemeral-2'
    assert json.loads(metrics[1][3])['dataPoints'][0]['loss'] == 0.1
    assert 'ephemeral' not in repr(connection)


def test_session_cannot_redirect_token_or_change_destination(monkeypatch):
    now = [1000]
    issued = [0]
    def handle(request):
        issued[0] += 1
        return httpx.Response(200, json=dict(apiUrl=f'https://dash{issued[0]}.test/api/managed', instanceId='one', projectId='p1', token='secret-test', expiresAt=now[0]+60))
    monkeypatch.setattr('ml_dash.managed.time.time', lambda: now[0])
    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.managed.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    connection = ManagedConnection('https://dreamlake.test', 'org', 'project', lambda: 'dl-session')
    client = connection.client()
    client._session()
    now[0] += 56
    with pytest.raises(ValueError, match='destination changed'):
        client._session()


def test_managed_does_not_accept_mismatched_project():
    connection = ManagedConnection('https://dreamlake.test', 'org', 'project', lambda: 'unused')
    with pytest.raises(ValueError, match='prefix'):
        connection.client().create_or_update_experiment('other', 'run', prefix='org/other/run')


def test_interrupted_setup_reuses_the_reviewed_operation(monkeypatch):
    requests = []
    def handle(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'revision': 0, 'planHash': 'reviewed-plan'})
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            raise httpx.ReadError('synthetic interrupted response', request=request)
        return httpx.Response(200, json={'status': 'ready'})
    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.managed.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    connection = ManagedConnection('https://dreamlake.test', 'org', 'project', lambda: 'synthetic-session')
    with pytest.raises(httpx.ReadError):
        connection.setup()
    assert connection.setup()['status'] == 'ready'
    assert requests[0] == requests[1]


def test_malformed_credential_is_rejected_without_echo():
    connection = ManagedConnection('https://dreamlake.test', 'org', 'project', lambda: 'secret-value\nunsafe')
    with pytest.raises(ValueError) as failure:
        connection.client()._session()
    assert 'secret-value' not in str(failure.value)


def test_from_login_reuses_current_server_only_and_reloads_tokens(tmp_path, monkeypatch):
    monkeypatch.delenv('DREAMLAKE_API_KEY', raising=False)
    monkeypatch.delenv('DREAMLAKE_REMOTE', raising=False)
    (tmp_path / 'config.json').write_text(json.dumps({'remote_url': 'https://dreamlake.test'}))
    (tmp_path / 'tokens.json').write_text(json.dumps({'dreamlake-token': 'synthetic-login-one'}))
    connection = ManagedConnection.from_login('org', 'project', config_dir=tmp_path)
    assert connection._provider() == 'synthetic-login-one'
    (tmp_path / 'tokens.json').write_text(json.dumps({'dreamlake-token': 'synthetic-login-two'}))
    assert connection._provider() == 'synthetic-login-two'
    other = ManagedConnection.from_login('org', 'project', 'https://other.test', config_dir=tmp_path)
    with pytest.raises(ValueError, match='No DreamLake login'):
        other._provider()
    (tmp_path / 'config.json').write_text(json.dumps({'remote_url': 'https://other.test'}))
    with pytest.raises(ValueError, match='No DreamLake login'):
        connection._provider()
    monkeypatch.setenv('DREAMLAKE_API_KEY', 'synthetic-other-environment')
    with pytest.raises(ValueError, match='No DreamLake login'):
        connection._provider()
