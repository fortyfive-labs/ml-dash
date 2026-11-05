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
        if request.headers.get('authorization') != f'Bearer gen-{state["generation"]}' or not state['active']:
            return httpx.Response(403, json={'error': {'code': 'NATIVE_CREDENTIAL_RETIRED'}})
        if request.method in ('GET', 'PATCH') and path == f'{ROOT}/runs/{RUN_ID}':
            return httpx.Response(200, json={'experiment': {'id': RUN_ID, 'name': 'renamed'}})
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


def test_native_workload_credentials_against_local_server(tmp_path):
    """Writers against an isolated dreamlake-server: register, write, rotate, a fresh CI process, retire, bound run."""
    import subprocess
    import sys
    import uuid

    from ml_dash.native import NativeExperimentError

    env = _local_server()
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


def test_native_metric_names_follow_ml_dash_single_segment_rule(monkeypatch):
    """ml-dash 0.13.0 sent any name as one percent-encoded path segment; `/` and dot segments never routed."""
    seen = []

    def handle(request):
        seen.append(request.url.raw_path.decode())
        if request.url.path.endswith('/session'):
            body = json.loads(request.content)  # the writer session contract: echo the writer asked for
            return httpx.Response(200, json={'token': 's', 'expiresAt': 0, 'apiUrl': 'https://dreamlake.test' + ROOT,
                                             'workloadId': body.get('workloadId') or body['idempotencyKey'], 'generation': 1})
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
