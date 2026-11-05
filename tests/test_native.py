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
                                             'apiUrl': 'https://dreamlake.test' + ROOT,
                                             'workloadId': body.get('workloadId') or body['idempotencyKey'], 'generation': 1})
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
    # One writer per client, as ManagedConnection: registered once, then renewed by its id.
    sessions = [c[4] for c in calls if c[2].endswith('/session')]
    assert set(sessions[0]) == {'mode', 'idempotencyKey'} and sessions[1] == {'mode': 'write', 'workloadId': sessions[0]['idempotencyKey']}


def _writer_server(monkeypatch, state):
    """A native server double for writer sessions: generation, retirement and the bound run."""
    calls = []

    def handle(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, path, request.headers.get('authorization'), body))
        if path == ROOT + '/session':
            writer = body.get('workloadId') or body.get('idempotencyKey')
            if not state['active']:
                return httpx.Response(403, json={'error': {'code': 'NATIVE_CREDENTIAL_RETIRED'}})
            return httpx.Response(200, json={'token': f'gen-{state["generation"]}', 'expiresAt': 0,
                                             'apiUrl': 'https://dreamlake.test' + ROOT, 'workloadId': writer,
                                             'generation': state['generation'], **({'runId': body['runId']} if 'runId' in body else {})})
        if request.headers.get('authorization') == state.get('expired'):
            return httpx.Response(401, json={'error': {'code': 'NATIVE_SESSION_EXPIRED'}})
        if request.headers.get('authorization') != f'Bearer gen-{state["generation"]}' or not state['active']:
            return httpx.Response(403, json={'error': {'code': 'NATIVE_CREDENTIAL_RETIRED'}})
        if request.method in ('GET', 'PATCH') and path == f'{ROOT}/runs/{RUN_ID}':
            return httpx.Response(200, json={'experiment': {'id': RUN_ID, 'name': 'renamed'}})
        if request.method == 'GET' and path == f'{ROOT}/runs/{RUN_ID}/metrics':
            return httpx.Response(200, json={'metrics': []})
        return httpx.Response(200, json={'count': 1})

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    return calls


def test_native_writer_renews_after_rotation_and_is_never_replaced_once_retired(monkeypatch):
    from ml_dash.native import NativeExperimentError
    state = {'generation': 1, 'active': True}
    calls = _writer_server(monkeypatch, state)
    writer = '0b0b0b0b-0000-4000-8000-000000000001'
    client = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'ci-login', workload_id=writer).client()
    client.set_parameters(RUN_ID, {'lr': 1})
    state['generation'] = 2  # rotated: the old session is refused once, the next call renews under generation 2
    with pytest.raises(NativeExperimentError) as refused:
        client.set_parameters(RUN_ID, {'lr': 2})
    assert refused.value.code == 'NATIVE_CREDENTIAL_RETIRED'
    client.set_parameters(RUN_ID, {'lr': 3})
    state['active'] = False  # retired
    with pytest.raises(NativeExperimentError):
        client.set_parameters(RUN_ID, {'lr': 4})
    with pytest.raises(NativeExperimentError) as ended:
        client.set_parameters(RUN_ID, {'lr': 5})
    assert ended.value.code == 'NATIVE_CREDENTIAL_RETIRED'
    sessions = [c[3] for c in calls if c[1].endswith('/session')]
    # Every session names the given writer; none registers a replacement.
    assert sessions and all(s == {'mode': 'write', 'workloadId': writer} for s in sessions)
    assert {'data': {'lr': 3}} in [c[3] for c in calls if c[1].endswith('/parameters')]


def test_native_explicit_writer_read_renews_its_writer_session_once_on_401(monkeypatch):
    state = {'generation': 1, 'active': True}
    calls = _writer_server(monkeypatch, state)
    writer = '0b0b0b0b-0000-4000-8000-000000000001'
    client = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'ci-login', workload_id=writer).client()
    client.set_parameters(RUN_ID, {'lr': 1})
    # The writer's cached session expires: its read is refused (401) once, renews that writer's session, then succeeds.
    state.update(expired='Bearer gen-1', generation=2)
    del calls[:]
    assert client.list_metrics(RUN_ID) == []
    metrics = f'{ROOT}/runs/{RUN_ID}/metrics'
    assert [(c[0], c[1], c[2]) for c in calls] == [('GET', metrics, 'Bearer gen-1'), ('POST', ROOT + '/session', 'Bearer ci-login'),
                                                   ('GET', metrics, 'Bearer gen-2')]
    assert calls[1][3] == {'mode': 'write', 'workloadId': writer}


def test_native_run_bound_writer_opens_its_run_by_id(monkeypatch):
    state = {'generation': 1, 'active': True}
    calls = _writer_server(monkeypatch, state)
    connection = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'ci-login', run_id=RUN_ID)
    client = connection.client()
    assert client.create_or_update_experiment('research', 'probe', prefix='lab/research/probe')['experiment']['id'] == RUN_ID
    assert not any(c[0] == 'PUT' for c in calls)
    session = next(c[3] for c in calls if c[1].endswith('/session'))
    assert session['runId'] == RUN_ID and set(session) == {'mode', 'idempotencyKey', 'runId'}
    with pytest.raises(ValueError):
        NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'x', workload_id='not-a-uuid')


def test_native_explicit_writer_never_asks_for_protection(monkeypatch):
    """An explicit writer's write_protected is refused before any request: never the person's login behind it."""
    state = {'generation': 1, 'active': True}
    calls = _writer_server(monkeypatch, state)
    writer = '0b0b0b0b-0000-4000-8000-000000000002'
    for options in ({'workload_id': writer}, {'idempotency_key': writer}, {'run_id': RUN_ID}):
        connection = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'ci-login', **options)
        with pytest.raises(ValueError, match="person's action"):
            Experiment(prefix='lab/research/folder/probe', managed=connection, _write_protected=True).run.start()
    # Anonymous: no login to act as a person; refused cleanly, also before any request.
    from ml_dash.native import NativeExperimentError
    anonymous = NativeConnection.anonymous('https://dreamlake.test', 'lab', 'research').client()
    with pytest.raises(NativeExperimentError) as refused:
        anonymous.create_or_update_experiment('research', 'probe', prefix='lab/research/folder/probe', write_protected=True)
    assert refused.value.code == 'NATIVE_LOGIN_REQUIRED'
    assert calls == []


def test_native_protected_create_uses_one_person_session_pinned_to_its_server(monkeypatch):
    """Interactive write_protected: one plain person's session, minted for the create only, pinned to the project route."""
    calls, advertised = [], ['https://dreamlake.test' + ROOT]

    def handle(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.raw_path.decode(), request.headers.get('authorization'), body))
        if request.url.path == ROOT + '/session':
            return httpx.Response(200, json={'token': 'person-session', 'expiresAt': 0, 'apiUrl': advertised[0]})
        return httpx.Response(200, json={'experiment': {'id': RUN_ID, **body}})

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    client = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'user-login').client()
    result = client.create_or_update_experiment('research', '实验 1', prefix='lab/research/2026-11-05/实验 1',
                                                write_protected=True, metadata={'seed': 1})
    assert result['experiment']['writeProtected'] is True
    assert calls[0] == ('POST', ROOT + '/session', 'Bearer user-login', {'mode': 'write'})
    assert calls[1][:3] == ('PUT', ROOT + '/run-paths/2026-11-05/%E5%AE%9E%E9%AA%8C%201', 'Bearer person-session')
    assert calls[1][3] == {'metadata': {'seed': 1}, 'writeProtected': True}
    # A session advertised for another server is never used.
    from ml_dash.native import NativeExperimentError
    advertised[0] = 'https://elsewhere.test' + ROOT
    with pytest.raises(NativeExperimentError):
        client.create_or_update_experiment('research', 'p', prefix='lab/research/p', write_protected=True)
    assert [c[0] for c in calls[2:]] == ['POST']


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


def test_native_tracks_against_local_server():
    """ml-dash tracks through the public API: append/flush, ranged reads, slice and findByTime floor match."""
    import uuid

    from ml_dash.exceptions import StorageError

    env = _local_server()
    ns, project, key = env['namespace'], env['project'], f'tracks-{uuid.uuid4().hex[:8]}'
    connection = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])
    with Experiment(prefix=f'{ns}/{project}/{key}', managed=connection).run as exp:
        pose = exp.tracks('robot/pose')
        for ts in (1.0, 3.0, 5.0, 7.0):
            pose.append(q=[ts, ts + 0.5], ee={'x': ts}, _ts=ts)
        exp.tracks('camera/left').append(width=640, _ts=-1)  # inherits the last timestamp, 7.0
        exp.tracks.flush()

        entries = pose.list_entries()
        assert [e['timestamp'] for e in entries] == [1.0, 3.0, 5.0, 7.0]
        assert entries[0] == {'timestamp': 1.0, 'q': [1.0, 1.5], 'ee.x': 1.0}
        ranged = pose.read(start_timestamp=3.0, end_timestamp=5.0, columns=['q'])
        assert ranged['count'] == 2 and ranged['entries'][0] == {'timestamp': 3.0, 'q': [3.0, 3.5]}
        assert pose.read(format='jsonl').count(b'\n') == 3

        window = pose.slice(start_timestamp=1.0, end_timestamp=7.0)
        assert len(window) == 4 and [e['timestamp'] for e in window] == [1.0, 3.0, 5.0, 7.0]
        assert window.findByTime(5.5)['timestamp'] == 5.0
        assert [e['timestamp'] for e in window.findByTime([1.0, 3.5, 9.0])] == [1.0, 3.0, 7.0]
        with pytest.raises(StorageError):
            window.findByTime(0.5)

        # Sparse and nested fields round-trip flattened; time selection through slice.
        imu = exp.tracks('robot/imu')
        imu.append(acc={'x': 0.1, 'y': 0.2}, _ts=10.0)
        imu.append(gyro=[1, 2, 3], _ts=11.0)
        imu.append(acc={'x': 0.3}, temp=36.5, _ts=12.0)
        exp.tracks.flush()
        assert imu.list_entries() == [{'timestamp': 10.0, 'acc.x': 0.1, 'acc.y': 0.2}, {'timestamp': 11.0, 'gyro': [1, 2, 3]},
                                      {'timestamp': 12.0, 'acc.x': 0.3, 'temp': 36.5}]
        assert [e['timestamp'] for e in imu.slice(start_timestamp=11.0, end_timestamp=12.0)] == [11.0, 12.0]

        client = exp._client
        assert [t['topic'] for t in client.list_tracks(exp.id, 'robot/*')] == ['robot/pose', 'robot/imu']
        listed = {t['topic']: t for t in client.list_tracks(exp.id)}
        assert listed['robot/imu']['columns'] == ['acc.x', 'acc.y', 'gyro', 'temp'] and listed['robot/imu']['totalEntries'] == 3
        camera = client.get_track_data(exp.id, 'camera/left')
        assert camera == {'entries': [{'timestamp': 7.0, 'width': 640}], 'count': 1}
        created = client.create_track(exp.id, 'robot/pose', description='ignored')
        assert created['topic'] == 'robot/pose' and created['experimentId'] == exp.id
        # Binary exports, as ml-dash: bytes. MCAP starts with its magic; a prefix exports every topic under it.
        assert pose.read(format='mcap').startswith(b'\x89MCAP0\r\n')
        assert client.get_track_data(exp.id, 'robot', format='mcap').startswith(b'\x89MCAP0\r\n')
        assert pose.read(format='parquet')[:6] == b'ARROW1'


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
        if request.url.path == '/namespaces/team-lab/projects/scratch':
            return httpx.Response(404, json={'error': 'not found'})  # missing: only then is a create sent
        if request.url.path == '/namespaces/team-lab/projects':
            return httpx.Response(403, json={'error': 'forbidden'})
        raise AssertionError('nothing else may be called after a refused create: ' + request.url.path)

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    connection = NativeConnection('https://dreamlake.test', 'team-lab', 'scratch', lambda: 'user-login', auto_create_project=True)
    with pytest.raises(NativeExperimentError) as refused:
        connection.client().set_parameters('a' * 24, {'x': 1})
    assert refused.value.status == 403



def test_native_project_bootstrap_uses_an_existing_project_and_never_runs_for_an_explicit_writer(monkeypatch):
    """A WRITE collaborator (no namespace create) writes to the existing project with no create; an explicit writer
    never uses the login to look up or create a project, even with auto_create_project."""
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path, request.headers.get('authorization')))
        if request.url.path == '/namespaces/team-lab/projects/scratch':
            return httpx.Response(200, json={'id': 'f' * 24, 'name': 'scratch'})
        if request.url.path.startswith('/namespaces/'):
            return httpx.Response(403, json={'error': 'no namespace create'})
        if request.url.path.endswith('/session'):
            body = json.loads(request.content)
            return httpx.Response(200, json={'token': 'w', 'expiresAt': 0, 'apiUrl': 'https://dreamlake.test/native-experiments/ns/team-lab/projects/scratch',
                                             'workloadId': body.get('workloadId') or body['idempotencyKey'], 'generation': 1})
        return httpx.Response(200, json={'parameters': {}})

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    NativeConnection('https://dreamlake.test', 'team-lab', 'scratch', lambda: 'user-login', auto_create_project=True) \
        .client().set_parameters('a' * 24, {'x': 1})
    assert [c[:2] for c in calls if c[1].startswith('/namespaces/')] == [('GET', '/namespaces/team-lab/projects/scratch')]
    assert calls[-1][:2] == ('POST', '/native-experiments/ns/team-lab/projects/scratch/runs/' + 'a' * 24 + '/parameters')

    calls.clear()
    writer = '0b0b0b0b-0000-4000-8000-000000000003'
    NativeConnection('https://dreamlake.test', 'team-lab', 'scratch', lambda: 'user-login', auto_create_project=True,
                     workload_id=writer).client().set_parameters('a' * 24, {'x': 2})
    assert not [c for c in calls if c[1].startswith('/namespaces/')]
    assert [c[1].rsplit('/', 1)[-1] for c in calls] == ['session', 'parameters']

def test_native_read_session_and_anonymous_public_reads(tmp_path):
    """A READ grant reads through a read session; an anonymous connection reads a public project, never writes."""
def test_native_workload_credentials_against_local_server(tmp_path):
    """Writers against an isolated dreamlake-server: register, write, rotate, a fresh CI process, retire, bound run."""
    import subprocess
    import sys
    import uuid

    from ml_dash.native import NativeExperimentError

    env = _local_server()
    if not env.get('reader_token'):
        pytest.skip('harness without a READ-grant user')
    ns, project, key = env['namespace'], env['project'], f'reads-{uuid.uuid4().hex[:8]}'
    with Experiment(prefix=f'{ns}/{project}/{key}', managed=NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])).run as exp:
        exp.params.set(lr=0.1)
        exp.metrics('loss').log(step=0, loss=1.0)
        exp.flush()
        run_id = exp.id
    reader = NativeConnection(env['base_url'], ns, project, lambda: env['reader_token']).client()
    assert reader.get_parameters(run_id) == {'lr': 0.1}
    assert [m['name'] for m in reader.list_metrics(run_id)] == ['loss']
    with pytest.raises(NativeExperimentError) as refused:
        reader.set_parameters(run_id, {'lr': 0.2})
    assert refused.value.status == 403
    anonymous = NativeConnection.anonymous(env['base_url'], ns, env['public_project']).client()
    assert isinstance(anonymous.list_files(env['public_run'])['files'], list)
    assert anonymous.graphql_query('query ($id: ID!) { experimentById(id: $id) { id } }', {'id': env['public_run']})['experimentById'] == {'id': env['public_run']}
    with pytest.raises(NativeExperimentError) as local:
        anonymous.set_parameters(env['public_run'], {'x': 1})
    assert local.value.code == 'NATIVE_LOGIN_REQUIRED'
    ns, project, base = env['namespace'], env['project'], env['base_url']
    connection = NativeConnection(base, ns, project, lambda: env['writer_token'])
    known = {c['id'] for c in connection.list_credentials(limit=100)['credentials']}
    source = tmp_path / 'weights.bin'
    source.write_bytes(os.urandom(2048))

    # Default: the client registers one writer on its first session and writes through it.
    with Experiment(prefix=f'{ns}/{project}/writer-{uuid.uuid4().hex[:8]}', managed=connection).run as exp:
        run_id = exp.id
        exp.params.set(lr=0.1)
        exp.metrics('train').log(step=0, loss=1.0)
        exp.files('ckpt').upload(str(source))
        exp.flush()
    (writer,) = {c['id'] for c in connection.list_credentials(limit=100)['credentials']} - known
    credential = connection.get_credential(writer)
    assert sorted(credential) == ['active', 'createdAt', 'generation', 'id', 'namespaceId', 'projectId', 'runId', 'updatedAt']
    assert (credential['runId'], credential['generation'], credential['active']) == (None, 1, True)

    client = NativeConnection(base, ns, project, lambda: env['writer_token'], workload_id=writer).client()
    client.set_parameters(run_id, {'seed': 1})
    assert connection.rotate_credential(writer, 1)['generation'] == 2
    assert connection.rotate_credential(writer, 1)['generation'] == 2  # the exact retry is unchanged
    with pytest.raises(NativeExperimentError) as stale:
        client.set_parameters(run_id, {'seed': 2})
    assert stale.value.code == 'NATIVE_CREDENTIAL_RETIRED'
    client.set_parameters(run_id, {'seed': 3})  # renewed under generation 2

    # CI: a new process with no saved login, only an explicit DREAMLAKE_API_KEY and the writer id.
    script = ('import sys; from ml_dash import NativeConnection; '
              'c = NativeConnection.from_login(sys.argv[1], sys.argv[2], workload_id=sys.argv[3]).client(); '
              'c.append_batch_to_metric(sys.argv[4], "ci", [{"v": 1}]); print(c.workload_id)')
    ci = subprocess.run([sys.executable, '-c', script, ns, project, writer, run_id], capture_output=True, text=True,
                        env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'ML_DASH_MANAGED_CONFIG_DIR': str(tmp_path / 'none'),
                             'DREAMLAKE_REMOTE': base, 'DREAMLAKE_API_KEY': env['writer_token']})
    assert ci.returncode == 0, ci.stderr[-2000:]
    assert ci.stdout.strip() == writer

    retired = connection.retire_credential(writer, 2)
    assert (retired['generation'], retired['active']) == (3, False)
    with pytest.raises(NativeExperimentError):
        client.set_parameters(run_id, {'seed': 4})
    with pytest.raises(NativeExperimentError) as ended:
        NativeConnection(base, ns, project, lambda: env['writer_token'], workload_id=writer).client().set_parameters(run_id, {'seed': 5})
    assert ended.value.code == 'NATIVE_CREDENTIAL_RETIRED'
    assert {c['id'] for c in connection.list_credentials(limit=100)['credentials']} - known == {writer}  # no replacement

    with httpx.Client(base_url=base + f'/native-experiments/ns/{ns}/projects/{project}',
                      headers={'Authorization': 'Bearer ' + env['writer_token']}) as http:
        assert http.get(f'/runs/{run_id}/parameters').json()['parameters']['data'] == {'lr': 0.1, 'seed': 3}
        assert http.get(f'/runs/{run_id}/metrics/ci').json()['rows'] == [{'v': 1}]
        assert len(http.get(f'/runs/{run_id}/files').json()['files']) == 1

    # A writer bound to a run opens it by id whatever key the experiment names, and writes no other run.
    with Experiment(prefix=f'{ns}/{project}/other-{uuid.uuid4().hex[:8]}', managed=connection).run as other:
        other_id = other.id
    bound = NativeConnection(base, ns, project, lambda: env['writer_token'], run_id=run_id)
    with Experiment(prefix=f'{ns}/{project}/any-name', managed=bound).run as exp:
        assert exp.id == run_id
        exp.metrics('bound').log(step=0, v=1)
    with pytest.raises(NativeExperimentError) as mismatch:
        bound.client().set_parameters(other_id, {'seed': 6})
    assert mismatch.value.code == 'NATIVE_RUN_SCOPE_MISMATCH'


def test_native_nested_paths_metadata_and_protection_against_local_server():
    """Against an isolated dreamlake-server with the run lifecycle: nested Unicode (and date) prefixes, metadata, protection."""
    import uuid

    from ml_dash.native import NativeExperimentError

    env = _local_server()
    ns, project, base = env['namespace'], env['project'], env['base_url']
    connection = NativeConnection(base, ns, project, lambda: env['writer_token'])
    prefix = f'{ns}/{project}/2026-11-05/扫描 {uuid.uuid4().hex[:6]}/训练 α'
    with Experiment(prefix=prefix, managed=connection, metadata={'seed': 7}).run as exp:
        run_id = exp.id
        exp.params.set(lr=0.1)
    with Experiment(prefix=prefix, managed=connection).run as again:
        assert again.id == run_id
    client = connection.client()
    view = client._json('GET', f'/runs/{run_id}', 'read')['experiment']
    assert view['path'] == prefix.split('/', 2)[2] and view['metadata'] == {'seed': 7}

    # A person's protected create: born protected with its metadata; the next data write is refused.
    protected = client.create_or_update_experiment(project, '冻结', prefix=f'{ns}/{project}/final {uuid.uuid4().hex[:6]}/冻结',
                                                   metadata={'tag': 'final'}, write_protected=True)['experiment']
    assert (protected['writeProtected'], protected['metadata']) == (True, {'tag': 'final'})
    with pytest.raises(NativeExperimentError) as refused:
        client.set_parameters(protected['id'], {'lr': 1})
    assert refused.value.code == 'NATIVE_RUN_WRITE_PROTECTED'

    # A writer bound to the first run updates its metadata in place, by id.
    bound = NativeConnection(base, ns, project, lambda: env['writer_token'], run_id=run_id)
    with Experiment(prefix=f'{ns}/{project}/any', managed=bound, metadata={'seed': 8}).run as exp:
        assert exp.id == run_id
    assert client._json('GET', f'/runs/{run_id}', 'read')['experiment']['metadata'] == {'seed': 8}


def test_native_metric_names_follow_ml_dash_single_segment_rule(monkeypatch):
    """ml-dash 0.13.0 sent any name as one percent-encoded path segment; `/` and dot segments never routed."""
    seen = []

    def handle(request):
        seen.append(request.url.raw_path.decode())
        if request.url.path.endswith('/session'):
            body = json.loads(request.content)  # a read session, or the writer session contract: echo the writer asked for
            writer = {} if body['mode'] == 'read' else {'workloadId': body.get('workloadId') or body['idempotencyKey'], 'generation': 1}
            return httpx.Response(200, json={'token': 's', 'expiresAt': 0, 'apiUrl': 'https://dreamlake.test' + ROOT, **writer})
        return httpx.Response(200, json={'metric': {'name': 'x'}})

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    client = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'login').client()
    for name in ('训练 loss', 'a' * 256, 'acc?x#y'):
        client.get_metric_stats(RUN_ID, name)
    assert seen[-3:] == [f'{ROOT}/runs/{RUN_ID}/metrics/%E8%AE%AD%E7%BB%83%20loss/stats',
                         f'{ROOT}/runs/{RUN_ID}/metrics/{"a" * 256}/stats', f'{ROOT}/runs/{RUN_ID}/metrics/acc%3Fx%23y/stats']
    for bad in ('', 'a/b', '.', '..', 'a' * 257, None):
        with pytest.raises(ValueError):
            client.get_metric_stats(RUN_ID, bad)


def test_native_graphql_uses_an_explicit_writers_session(monkeypatch):
    """With an explicit writer, raw GraphQL is sent with that writer's scoped session, never the user's login."""
    seen = []

    def handle(request):
        if request.url.path.endswith('/session'):
            body = json.loads(request.content)
            return httpx.Response(200, json={'token': 'writer-session', 'expiresAt': 0, 'apiUrl': 'https://dreamlake.test' + ROOT,
                                             'workloadId': body.get('workloadId'), 'generation': 1})
        seen.append((request.url.path, request.headers.get('authorization')))
        return httpx.Response(200, json={'data': {'me': None}})

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    writer = '0b7f3a9e-1c2d-4e5f-8a9b-0c1d2e3f4a5b'
    scoped = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'user-login', workload_id=writer).client()
    scoped.graphql_query('query { me { id } }')
    human = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'user-login').client()
    human.graphql_query('query { me { id } }')
    assert seen == [('/native-dash/graphql', 'Bearer writer-session'), ('/native-dash/graphql', 'Bearer user-login')]


def test_native_delete_project_uses_dreamlakes_project_delete_with_native_purge(monkeypatch):
    """delete_project: DreamLake's project delete, pinned to the project id read first, with the native data purged."""
    from ml_dash.native import NativeExperimentError
    project_id = 'b' * 24
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path, dict(request.url.params), request.headers.get('authorization')))
        if request.method == 'GET' and request.url.path == '/namespaces/lab/projects/research':
            return httpx.Response(200, json={'id': project_id, 'slug': 'research'})
        if request.method == 'GET' and request.url.path == '/namespaces/lab/projects/missing':
            return httpx.Response(404, json={'error': 'not found'})
        if request.method == 'DELETE' and request.url.path == '/namespaces/lab/projects/research':
            return httpx.Response(200, json={'deleted': True, 'project': {'id': project_id}, 'nativeCleanup': {'accepted': True, 'homes': []}})
        raise AssertionError(f'unexpected {request.method} {request.url.path}')

    real_client = httpx.Client
    monkeypatch.setattr('ml_dash.native.httpx.Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    client = NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'user-login').client()
    result = client.delete_project('research')
    assert result['projectId'] == project_id and result['deleted'] is True and result['nativeCleanup'] == {'accepted': True, 'homes': []}
    assert calls[-1][:3] == ('DELETE', '/namespaces/lab/projects/research', {'expectedProjectId': project_id, 'nativeExperiments': 'purge'})
    assert calls[-1][3] == 'Bearer user-login'
    with pytest.raises(NativeExperimentError) as missing:
        client.delete_project('missing')
    assert missing.value.status == 404
    # A selected writer or an anonymous connection never deletes a project, and calls nothing.
    calls.clear()
    for connection in (NativeConnection('https://dreamlake.test', 'lab', 'research', lambda: 'user-login', workload_id='00000000-0000-4000-8000-000000000001'),
                       NativeConnection.anonymous('https://dreamlake.test', 'lab', 'research')):
        with pytest.raises(NativeExperimentError) as refused:
            connection.client().delete_project()
        assert refused.value.status == 403
    assert calls == []


def test_native_explicit_writer_graphql_against_local_server():
    """Real SDK -> gateway: a run-bound writer's raw GraphQL uses its scoped session (never the login), reads its own
    run only, and is refused (403) once the writer is retired; a new session is refused too."""
    import uuid

    from ml_dash.native import NativeExperimentError

    env = _local_server()
    ns, project = env['namespace'], env['project']
    human = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])
    keys = [f'writer-own-{uuid.uuid4().hex[:6]}', f'writer-other-{uuid.uuid4().hex[:6]}']
    ids = []
    for key in keys:
        with Experiment(prefix=f'{ns}/{project}/{key}', managed=human).run as exp:
            ids.append(exp.id)
    writer = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'],
                              idempotency_key=str(uuid.uuid4()), run_id=ids[0]).client()
    query = 'query ($id: ID!) { experimentById(id: $id) { id name } }'
    assert writer.graphql_query(query, {'id': ids[0]})['experimentById'] == {'id': ids[0], 'name': keys[0]}
    assert writer.graphql_query(query, {'id': ids[1]})['experimentById'] is None
    assert writer.graphql_query('query { me { id } }')['me'] is None
    listed = writer.graphql_query('query ($p: String!, $n: String) { experimentsPaginated(namespaceSlug: $n, projectSlug: $p) { totalCount experiments { id } } }',
                                  {'p': project, 'n': ns})['experimentsPaginated']
    assert listed == {'totalCount': 1, 'experiments': [{'id': ids[0]}]}

    credential = human.get_credential(writer.workload_id)
    human.retire_credential(credential['id'], credential['generation'])
    with pytest.raises(httpx.HTTPStatusError) as retired:
        writer.graphql_query(query, {'id': ids[0]})
    assert retired.value.response.status_code == 403
    writer._sessions.clear()
    with pytest.raises(NativeExperimentError) as refused:
        writer.graphql_query(query, {'id': ids[0]})
    assert refused.value.status == 403


def test_native_bindrs_against_local_server(tmp_path):
    """Real SDK -> gateway (DREAMLAKE_NATIVE_RUN_BINDRS on): `bindrs=` at create and its retry (no duplicates),
    `experiment.bindrs(name).list()`, item labels (a file, the parameters), and a run-bound writer's scope."""
    import uuid

    from ml_dash.native import NativeExperimentError

    env = _local_server()
    ns, project, key = env['namespace'], env['project'], f'labels-{uuid.uuid4().hex[:6]}'
    human = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'])
    source = tmp_path / 'curve.txt'
    source.write_text('curve')
    with Experiment(prefix=f'{ns}/{project}/{key}', managed=human, bindrs=['best']).run as exp:
        exp.params.set(lr=0.1)
        exp.files().upload(str(source))
        exp.flush()
        client, run_id = exp._client, exp.id
        assert [b['name'] for b in client.run_bindrs(run_id)] == ['best']
        file_id = client.list_files(run_id)['files'][0]['id']
        figures = client.attach_bindr(run_id, name='figures', item_kind='file', item_id=file_id)['bindr']
        # A retry by name keeps one membership; by id the answer says whether it changed anything.
        client.attach_bindr(run_id, name='figures', item_kind='file', item_id=file_id)
        assert client.attach_bindr(run_id, bindr_id=figures['id'], item_kind='file', item_id=file_id)['attached'] is False
        assert [b['name'] for b in client.run_bindrs(run_id, 'file', file_id)] == ['figures']
        assert [f['id'] for f in exp.bindrs('figures').list()] == [file_id]
        assert exp.bindrs('best').list() == []
        params_id = client.parameters_id(run_id)
        client.attach_bindr(run_id, name='sweep', item_kind='parameters', item_id=params_id)
        assert [b['name'] for b in client.run_bindrs(run_id, 'parameters', params_id)] == ['sweep']
    # The same create again (an SDK retry): the run keeps exactly one membership of `best`.
    with Experiment(prefix=f'{ns}/{project}/{key}', managed=human, bindrs=['best']).run as again:
        assert again.id == run_id
        assert [b['name'] for b in again._client.run_bindrs(run_id)] == ['best']
    # A run-bound writer labels its own run only, with its scoped session.
    other = human.client().create_or_update_experiment(project, f'{key}-other', prefix=f'{ns}/{project}/{key}-other')['experiment']['id']
    writer = NativeConnection(env['base_url'], ns, project, lambda: env['writer_token'], idempotency_key=str(uuid.uuid4()), run_id=run_id).client()
    assert writer.attach_bindr(run_id, name='writer-made')['bindr']['name'] == 'writer-made'
    with pytest.raises(NativeExperimentError) as refused:
        writer.attach_bindr(other, name='writer-made')
    assert refused.value.status == 403
    assert {b['name'] for b in human.client().run_bindrs(run_id)} == {'best', 'writer-made'}
    assert human.client().detach_bindr(run_id, name='writer-made')['detached'] is True
