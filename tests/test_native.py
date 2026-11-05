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
    """Metric descriptor/stats/rows, file update/delete and filtered logs through the public API."""
    import uuid

    from ml_dash import MetricRowsError
    from ml_dash.native import NativeExperimentError

    env = _local_server()
    ns, project, key = env['namespace'], env['project'], f'edits-{uuid.uuid4().hex[:8]}'
    connection = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])
    source = tmp_path / 'notes.txt'
    source.write_text('draft')

    with Experiment(prefix=f'{ns}/{project}/{key}', managed=connection).run as exp:
        client = exp._client
        # An invalid descriptor is refused with nothing written: the metric does not exist afterwards.
        with pytest.raises(NativeExperimentError) as refused:
            client.append_batch_to_metric(exp.id, 'bad', [{'x': 1}], metadata=['not', 'an', 'object'])
        assert refused.value.status == 400
        with pytest.raises(NativeExperimentError) as missing:
            client.get_metric_stats(exp.id, 'bad')
        assert missing.value.status == 404

        loss = exp.metrics('train', description='training loss', tags=['train'], metadata={'unit': 'nats'})
        loss.log(step=0, loss=1, tag='warmup')
        loss.log(step=1, loss=0.25, tag='main')
        exp.flush()
        # The descriptor came with the first points; a later append's descriptor never changes it.
        client.append_batch_to_metric(exp.id, 'train', [{'step': 2, 'loss': 0.2, 'tag': 'main'}], description='ignored')
        for step in range(1205):
            exp.metrics('big').log(step=step)
        exp.metrics('nested').log(step=0, config={'depth': 4})
        exp.log('epoch 1 started')
        exp.log('nan detected', level='error')
        exp.files('reports').upload(str(source))
        exp.flush()

        stats = loss.stats()
        assert stats['description'] == 'training loss' and stats['tags'] == ['train']
        assert stats['metadata'] == {'unit': 'nats'} and stats['totalDataPoints'] == 3
        assert stats['schema']['fields'][:3] == ['step', 'loss', 'tag']
        assert {m['name']: m['metricId'] for m in loss.list_all()}['train'] == stats['metricId']

        page = loss.read_rows()
        assert page.metric_id == stats['metricId'] and page.returned == 3 and not page.has_more
        # int and float cells of one column share a float64 column; `_ts` is the SDK's timestamp field.
        assert [(c.name, c.type) for c in page.blocks[0].columns][:3] == [('step', 'int64'), ('loss', 'float64'), ('tag', 'utf8')]
        assert [row[:3] for row in page.blocks[0].rows][:2] == [[0, 1.0, 'warmup'], [1, 0.25, 'main']]

        # A page larger than one native request is filled completely, and the cursor resumes exactly.
        big = exp.metrics('big')
        first = big.read_rows(limit=1100)
        assert first.returned == 1100 and first.has_more
        rest = big.read_rows(limit=1100, cursor=first.next_cursor)
        steps = [row[0] for page_ in (first, rest) for block in page_.blocks for row in block.rows]
        assert steps == list(range(1205)) and not rest.has_more
        assert len(list(big.iter_row_blocks(limit=500))) >= 1
        with pytest.raises(MetricRowsError) as invalid:
            big.read_rows(limit=10001)
        assert invalid.value.code == 'invalid_limit'
        # A nested value has no ml-dash cell type: refused, not stringified.
        with pytest.raises(MetricRowsError) as nested:
            exp.metrics('nested').read_rows()
        assert nested.value.code == 'unsupported_cell'

        listed = exp.files('reports').list()
        assert [f['filename'] for f in listed] == ['notes.txt']
        updated = exp.files(file_id=listed[0]['id'], description='final', tags=['report'], metadata={'pages': 1}).update()
        assert updated['description'] == 'final' and updated['tags'] == ['report'] and updated['metadata'] == {'pages': 1}
        assert exp.files('reports/notes.txt').delete()['deleted'] is True
        assert exp.files('reports').list() == []

        errors = client.query_logs(exp.id, level=['error'])['logs']
        assert [entry['message'] for entry in errors] == ['nan detected']
        assert [e['message'] for e in client.query_logs(exp.id, search='epoch', order='asc')['logs']] == ['epoch 1 started']


def test_native_graphql_helpers_against_local_server(monkeypatch):
    """ml-dash's GraphQL helpers and raw queries reach DreamLake's native compatibility endpoint."""
    import uuid

    from ml_dash.exceptions import NetworkError

    env = _local_server()
    ns, project, key = env['namespace'], env['project'], f'gql-{uuid.uuid4().hex[:8]}'
    connection = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])
    with Experiment(prefix=f'{ns}/{project}/{key}', managed=connection, readme='graphql', tags=['g']).run as exp:
        exp.params.set(lr=0.1)
        exp.metrics('loss').log(step=0, loss=1.0)
        exp.flush()
        client = exp._client
        projects = client.list_projects_graphql(ns)['projects']
        assert any(p['slug'] == project for p in projects)
        runs = client.list_experiments_graphql(project, namespace_slug=ns)['experiments']
        assert any(r['name'] == key and r['displayPath'] == f'{ns}/{project}/{key}' for r in runs)
        detail = client.get_experiment_graphql(project, key, namespace_slug=ns)
        assert detail['id'] == exp.id and detail['parameters']['data'] == {'lr': 0.1} and detail['metrics'] == [{'name': 'loss'}]
        found = client.search_experiments_graphql(f'{ns}/{project}/{key}')
        assert [e['id'] for e in found['experiments']] == [exp.id]
        raw = client.graphql_query('query ($id: ID!) { experimentById(id: $id) { name tags project { slug } } }', {'id': exp.id})
        assert raw['experimentById'] == {'name': key, 'tags': ['g'], 'project': {'slug': project}}
        with pytest.raises(NetworkError):
            client.graphql_query('mutation { deleteNode(nodeId: "x") { id } }')

    # rdxp, opted into the native backend.
    monkeypatch.setenv('ML_DASH_BACKEND', 'native')
    monkeypatch.setenv('ML_DASH_NATIVE_NAMESPACE', ns)
    scratch = f'scratch-{uuid.uuid4().hex[:6]}'
    monkeypatch.setenv('ML_DASH_NATIVE_PROJECT', scratch)  # absent: rdxp creates it (private), as ml-dash's does
    import importlib
    import sys

    import ml_dash.native as native
    monkeypatch.setattr(native, '_login_provider', lambda url, config: (env['base_url'], lambda: env['writer_token']))
    sys.modules.pop('ml_dash.remote_auto_start', None)
    rdxp = importlib.import_module('ml_dash.remote_auto_start').rdxp
    with rdxp.run:
        rdxp.log('native rdxp')
    assert rdxp.id
    with httpx.Client(base_url=env['base_url'], headers={'Authorization': 'Bearer ' + env['writer_token']}) as http:
        created = http.get(f'/native-experiments/ns/{ns}/projects/{scratch}/runs/{rdxp.id}')
    assert created.status_code == 200 and created.json()['experiment']['name'] == 'rdxp'


def test_native_default_namespace_and_project_create_refusal(tmp_path, monkeypatch):
    """The login's namespace comes from the active profile (not the OS user); a refused create raises."""
    from ml_dash.native import NativeExperimentError
    monkeypatch.delenv('ML_DASH_NATIVE_NAMESPACE', raising=False)
    (tmp_path / 'config.json').write_text(json.dumps({'remote_url': 'https://dreamlake.test', 'namespace': 'team-lab'}))
    (tmp_path / 'tokens.json').write_text(json.dumps({'dreamlake-token': 'user-login'}))
    assert NativeConnection.default_namespace(config_dir=str(tmp_path)) == 'team-lab'

    def handle(request):
        if request.url.path == '/namespaces/team-lab/projects':
            return httpx.Response(403, json={'error': 'forbidden'})
        raise AssertionError('nothing else may be called after a refused create: ' + request.url.path)

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    connection = NativeConnection('https://dreamlake.test', 'team-lab', 'scratch', lambda: 'user-login', auto_create_project=True)
    with pytest.raises(NativeExperimentError) as refused:
        connection.client().set_parameters('a' * 24, {'x': 1})
    assert refused.value.status == 403
