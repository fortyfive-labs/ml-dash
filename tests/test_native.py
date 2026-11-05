"""Interactive DreamLake-native runs through Experiment(managed=NativeConnection(...))."""
import json
import os

import httpx
import pytest

from ml_dash import Experiment, NativeConnection

ROOT = '/native-experiments/ns/lab/projects/research'
RUN_ID = 'a' * 24


def test_native_experiment_writes_only_native_routes_with_project_sessions(monkeypatch):
    calls, issued, rejected = [], [0], []

    def handle(request):
        path, auth = request.url.path, request.headers.get('authorization')
        body = json.loads(request.content) if request.headers.get('content-type') == 'application/json' else None
        calls.append((request.method, request.url.host, path, auth, body))
        if path == ROOT + '/session':
            issued[0] += 1
            return httpx.Response(200, json={'token': f'session-{issued[0]}', 'expiresAt': 0,
                                             'apiUrl': 'https://dreamlake.test' + ROOT})
        if path.endswith('/parameters') and not rejected:
            rejected.append(auth)  # an expired session: refused before any write
            return httpx.Response(401, json={'error': {'code': 'NATIVE_LOGIN_REQUIRED'}})
        if request.method == 'PUT':
            return httpx.Response(200, json={'experiment': {'id': RUN_ID, **(body or {})}})
        return httpx.Response(200, json={'count': 1})

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client',
                        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    connection = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'user-login')
    with Experiment(prefix='lab/research/probe', managed=connection, readme='', tags=['lr', 'v2']).run as exp:
        exp.params.set(lr=0.001)
        exp.log('hello')
        exp.metrics('train').log(step=0, loss=0.5)

    assert all(host == 'dreamlake.test' and path.startswith(ROOT + '/') for _, host, path, _, _ in calls)
    # The user's login only mints project sessions; data routes carry a session.
    assert {auth for _, _, path, auth, _ in calls if path.endswith('/session')} == {'Bearer user-login'}
    assert all(auth.startswith('Bearer session-') for _, _, path, auth, _ in calls if not path.endswith('/session'))
    put = next(c for c in calls if c[0] == 'PUT')
    assert put[2] == ROOT + '/runs/probe' and put[4] == {'description': '', 'tags': ['lr', 'v2']}
    # The 401 parameters write was retried once with a fresh session and then succeeded.
    params = [c for c in calls if c[2].endswith('/parameters')]
    assert [c[3] for c in params] == ['Bearer session-1', 'Bearer session-2']
    assert params[1][4] == {'data': {'lr': 0.001}}
    written = {c[2].rsplit('/', 1)[-1]: c[4] for c in calls if c[0] == 'POST' and RUN_ID in c[2]}
    assert written['logs']['entries'][0]['message'] == 'hello'
    assert written['train']['dataPoints'][0]['loss'] == 0.5
    assert written['status'] == {'status': 'COMPLETED'}


def _local_server():
    path = os.environ.get('ML_DASH_NATIVE_TEST_ENV')
    if not path:
        pytest.skip('set ML_DASH_NATIVE_TEST_ENV to a local dreamlake-server env.json')
    return json.loads(open(path).read())


def test_interactive_native_run_against_local_server(tmp_path):
    """Against an isolated dreamlake-server (DREAMLAKE_NATIVE_DASH_STORE=write): ml-dash public API only."""
    import subprocess
    import uuid

    env = _local_server()
    ns, project, key = env['namespace'], env['project'], f'interactive-{uuid.uuid4().hex[:8]}'
    connect = lambda: NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])  # noqa: E731
    source = tmp_path / 'weights.bin'
    source.write_bytes(os.urandom(4096))

    with Experiment(prefix=f'{ns}/{project}/{key}', managed=connect(), readme='lr sweep', tags=['a', 'b']).run as exp:
        run_id = exp.id
        assert exp._experiment_data['experiment']['description'] == 'lr sweep'
        exp.params.set(lr=0.001, model={'depth': 4})
        exp.log('hello native')
        exp.metrics('train').log(step=0, loss=0.5)
        exp.files('ckpt').upload(str(source))
        exp.flush()
        downloaded = exp.files('ckpt/weights.bin').download(to=str(tmp_path / 'back.bin'))
    assert open(downloaded, 'rb').read() == source.read_bytes()

    # Reopening the same key gets the same run; "" and [] clear the human metadata.
    with Experiment(prefix=f'{ns}/{project}/{key}', managed=connect(), readme='', tags=[]).run as exp:
        assert exp.id == run_id
        assert exp._experiment_data['experiment']['description'] is None
        assert exp._experiment_data['experiment']['tags'] == []
        assert exp.params.get()['lr'] == 0.001

    with httpx.Client(base_url=env['base_url'] + f'/native-experiments/ns/{ns}/projects/{project}',
                      headers={'Authorization': 'Bearer ' + env['writer_token']}) as http:
        run = http.get(f'/runs/{run_id}').json()['experiment']
        logs = http.get(f'/runs/{run_id}/logs').json()['logs']
        points = http.get(f'/runs/{run_id}/metrics/train').json()['rows']
    assert run['status'] == 'COMPLETED' and run['description'] is None and run['tags'] == []
    assert [entry['message'] for entry in logs] == ['hello native']
    assert points[0]['loss'] == 0.5

    if env.get('mongo_container'):
        # Homed in the Dash database; no run Node in the main database.
        script = (f"print(db.getSiblingDB('{env['dash_db']}').runs.countDocuments({{_id: ObjectId('{run_id}')}}),"
                  f" db.getSiblingDB('{env['main_db']}').Node.countDocuments({{name: '{key}'}}))")
        counts = subprocess.run(['docker', 'exec', env['mongo_container'], 'mongosh', '--quiet', '--eval', script],
                                capture_output=True, text=True, check=True).stdout.split()
        assert counts == ['1', '0']


def test_native_run_data_edits_against_local_server(tmp_path):
    """Metric metadata/stats/rows, file update/delete and filtered logs through the public API."""
    import uuid

    env = _local_server()
    ns, project, key = env['namespace'], env['project'], f'edits-{uuid.uuid4().hex[:8]}'
    connection = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])
    source = tmp_path / 'notes.txt'
    source.write_text('draft')

    with Experiment(prefix=f'{ns}/{project}/{key}', managed=connection).run as exp:
        loss = exp.metrics('train', description='training loss', tags=['train'], metadata={'unit': 'nats'})
        loss.log(step=0, loss=0.5, tag='warmup')
        loss.log(step=1, loss=0.25, tag='main')
        exp.log('epoch 1 started')
        exp.log('nan detected', level='error')
        exp.files('reports').upload(str(source))
        exp.flush()

        stats = loss.stats()
        assert stats['description'] == 'training loss' and stats['tags'] == ['train']
        assert stats['metadata'] == {'unit': 'nats'} and stats['totalDataPoints'] == 2
        assert [m['name'] for m in loss.list_all()] == ['train']
        page = loss.read_rows()
        assert page.returned == 2 and not page.has_more
        # Rows as stored: the SDK's points also carry their `_ts` timestamp.
        assert [(c.name, c.type) for c in page.blocks[0].columns] == [
            ('step', 'int64'), ('loss', 'float64'), ('tag', 'utf8'), ('_ts', 'float64')]
        assert [row[:3] for row in page.blocks[0].rows] == [[0, 0.5, 'warmup'], [1, 0.25, 'main']]

        listed = exp.files('reports').list()
        assert [f['filename'] for f in listed] == ['notes.txt']
        updated = exp.files(file_id=listed[0]['id'], description='final', tags=['report'], metadata={'pages': 1}).update()
        assert updated['description'] == 'final' and updated['tags'] == ['report'] and updated['metadata'] == {'pages': 1}
        assert exp.files('reports/notes.txt').delete()['deleted'] is True
        assert exp.files('reports').list() == []

        client = exp._client
        errors = client.query_logs(exp.id, level=['error'])['logs']
        assert [entry['message'] for entry in errors] == ['nan detected']
        assert [e['message'] for e in client.query_logs(exp.id, search='epoch', order='asc')['logs']] == ['epoch 1 started']
