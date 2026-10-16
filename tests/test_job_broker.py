"""Real loopback sockets, synthetic credentials; no user login or Keychain."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ml_dash import Experiment
from ml_dash.managed import ManagedConnection


@pytest.fixture
def broker(monkeypatch):
    state = {'calls': [], 'override': {}, 'status': 200}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers.get('content-length', '0'))) or b'{}')
            state['calls'].append((self.path, data, self.headers.get('authorization')))
            if self.path == '/session':
                assert data == {}
                assert self.headers.get('authorization') == 'Bearer synthetic-broker'
                result = dict(apiUrl=state['url'] + '/api/managed', experimentUrl=state['url'] + '/lab/projects/research/experiments',
                    namespace='lab', project='research', runKey='job-attempt', runId='fixed-run',
                    instanceId='instance-one', projectId='project-one', workloadId='workload-one', generation=1,
                    token='synthetic-lease', expiresAt=time.time()+60)
                result.update(state['override'])
                status = state['status']
            else:
                assert self.headers.get('authorization') == 'Bearer synthetic-lease'
                result, status = {'count': 1}, 200
            payload = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    state['url'] = f'http://127.0.0.1:{server.server_port}'
    monkeypatch.setenv('DREAMLAKE_EXPERIMENT_BROKER', state['url'])
    monkeypatch.setenv('DREAMLAKE_EXPERIMENT_BROKER_TOKEN', 'synthetic-broker')
    monkeypatch.setenv('ML_DASH_NO_VERSION_CHECK', '1')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_default_experiment_reuses_platform_run_and_writes(broker, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    with Experiment().run as exp:
        assert exp.run.prefix == 'lab/research/job-attempt'
        exp.params.set(lr=0.001)
        exp.metrics('loss').log(step=0, loss=0.1)
        exp.flush()
    assert any('/runs/fixed-run/metrics/loss' == p.removeprefix('/api/managed/projects/project-one') for p, _, _ in broker['calls'])
    assert all('synthetic' not in str(p) for p, _, _ in broker['calls'])
    assert not (tmp_path / '.dash').exists()
    output = capsys.readouterr()
    assert 'synthetic' not in output.out + output.err


@pytest.mark.parametrize('denied', [401, 403])
def test_multiple_renewals_then_revocation_never_fall_back(broker, denied):
    c = ManagedConnection.from_job()
    client = c.client()
    for _ in range(3):
        client._credential['expiresAt'] = 0
        client.set_parameters('fixed-run', {'x': 1})
    assert sum(p == '/session' for p, _, _ in broker['calls']) == 4
    client._credential['expiresAt'] = 0
    broker['status'] = denied
    broker['override']['message'] = 'never-print-synthetic-secret'
    with pytest.raises(ValueError) as error:
        client.set_parameters('fixed-run', {})
    assert 'synthetic' not in str(error.value)
    assert all(p == '/session' or p.startswith('/api/managed/projects/project-one/runs/fixed-run/') for p, _, _ in broker['calls'])
    client.close()


def test_broker_redirect_is_rejected_and_proxy_environment_ignored(broker, monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('ALL_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')
    c = ManagedConnection.from_job()
    c.client()._credential['expiresAt'] = 0
    broker['status'] = 307
    with pytest.raises(ValueError):
        c.client().set_parameters('fixed-run', {})
    assert len(broker['calls']) == 2
    c.client().close()


@pytest.mark.parametrize('field,value', [('runId','another-run'), ('runKey','another-key'), ('namespace','elsewhere'), ('project','other'), ('workloadId','other-workload'), ('generation',2), ('apiUrl','https://other.test/api/managed'), ('experimentUrl','https://other.test/view')])
def test_changed_pins_fail_before_write(broker, field, value):
    c = ManagedConnection.from_job()
    c.client()._credential['expiresAt'] = 0
    broker['override'][field] = value
    with pytest.raises(ValueError, match='job experiment session'):
        c.client().set_parameters('fixed-run', {})
    assert all(p == '/session' for p, _, _ in broker['calls'])
    c.client().close()


@pytest.mark.parametrize('endpoint', ['https://127.0.0.1:1234', 'http://localhost:1234', 'http://127.0.0.1:1234/path', 'http://user:secret@127.0.0.1:1234', 'http://127.0.0.1:1234?token=secret', 'http://127.0.0.1:1234#fragment', 'http://example.test:1234', 'http://127.0.0.1', ''])
def test_invalid_endpoint_rejected_without_echo(monkeypatch, endpoint):
    monkeypatch.setenv('DREAMLAKE_EXPERIMENT_BROKER', endpoint)
    monkeypatch.setenv('DREAMLAKE_EXPERIMENT_BROKER_TOKEN', 'synthetic-broker')
    with pytest.raises(ValueError) as error:
        Experiment()
    assert 'secret' not in str(error.value)


def test_missing_secret_and_missing_endpoint_fail_closed(monkeypatch):
    monkeypatch.setenv('DREAMLAKE_EXPERIMENT_BROKER', 'http://127.0.0.1:1234')
    monkeypatch.delenv('DREAMLAKE_EXPERIMENT_BROKER_TOKEN', raising=False)
    with pytest.raises(ValueError):
        Experiment()
    monkeypatch.delenv('DREAMLAKE_EXPERIMENT_BROKER')
    monkeypatch.setenv('DREAMLAKE_EXPERIMENT_BROKER_TOKEN', 'synthetic-broker')
    with pytest.raises(ValueError):
        Experiment()


def test_explicit_scope_and_standalone_conflicts_rejected(broker):
    for options in [dict(prefix='lab/research/other'), dict(dash_url=True), dict(dash_root='/tmp/other'), dict(managed=ManagedConnection('https://example.test','lab','research',lambda:'unused'))]:
        with pytest.raises(ValueError):
            Experiment(**options)
    c = ManagedConnection.from_job()
    with pytest.raises(ValueError):
        c.client().set_parameters('other-run', {})
    with pytest.raises(ValueError):
        c.client().create_or_update_experiment('research','other',prefix='lab/research/other')
    c.client().close()


@pytest.mark.parametrize('override', [{'expiresAt':0}, {'expiresAt':time.time()+600}, {'runId':None}, {'generation':0}, {'token':'secret\nunsafe'}])
def test_invalid_lease_is_sanitized(broker, override):
    broker['override'].update(override)
    with pytest.raises(ValueError) as error:
        ManagedConnection.from_job()
    assert 'secret' not in str(error.value)
