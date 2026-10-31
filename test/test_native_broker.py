"""Exercise the public Experiment API through a real loopback broker and API."""

import json
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ml_dash import Experiment
from ml_dash.exceptions import ConfigurationError


def test_job_scoped_experiment_renews_and_never_uses_dash(monkeypatch, tmp_path):
    calls = []
    state = {"api": "", "renewals": 0, "revoked": False}
    run_id = "a" * 24
    file_id = "b" * 24
    file_info = {"id": file_id, "filename": "model.txt", "prefix": "/models",
                 "checksum": hashlib.sha256(b"model state").hexdigest()}

    class API(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            raw = self.rfile.read(length)
            body = raw if self.path.endswith("/files") else json.loads(raw)
            calls.append((self.path, self.headers.get("Authorization"), body))
            if state["revoked"]:
                self.send_response(403)
                self.end_headers()
                return
            assert self.path.startswith(f"/native-experiments/ns/team/projects/train/runs/{run_id}/")
            assert self.headers["Authorization"].startswith("Bearer lease-")
            if self.path.endswith("/files"):
                assert self.headers["Content-Type"].startswith("multipart/form-data;")
                assert b"model state" in raw
            result = file_info if self.path.endswith("/files") else {"count": 1}
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            assert self.headers["Authorization"].startswith("Bearer lease-")
            if self.path.endswith("/files"):
                payload = json.dumps({"files": [file_info], "hasMore": False}).encode()
            elif self.path.endswith("/files/" + file_id):
                payload = json.dumps({"file": file_info}).encode()
            elif self.path.endswith("/files/" + file_id + "/download"):
                payload = b"model state"
            else:
                raise AssertionError(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    api = ThreadingHTTPServer(("127.0.0.1", 0), API)
    state["api"] = f"http://127.0.0.1:{api.server_port}"

    class Broker(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/session"
            assert self.headers["Authorization"] == "Bearer broker-secret"
            assert self.rfile.read(int(self.headers["Content-Length"])) == b"{}"
            state["renewals"] += 1
            if state["revoked"]:
                self.send_response(403)
                self.end_headers()
                return
            lease = {"namespace": "team", "project": "train", "runKey": "job-" + run_id,
                     "runId": run_id, "projectId": "p1", "instanceId": "dreamlake-native-v1",
                     "workloadId": "w1", "generation": 1,
                     "apiUrl": state["api"] + "/native-experiments/ns/team/projects/train",
                     "experimentUrl": "https://dreamlake.ai/team/train?run=" + run_id,
                     "token": "lease-" + str(state["renewals"]), "expiresAt": time.time() + 60}
            payload = json.dumps(lease).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    broker = ThreadingHTTPServer(("127.0.0.1", 0), Broker)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (api, broker)]
    for thread in threads:
        thread.start()
    monkeypatch.setenv("DREAMLAKE_EXPERIMENT_BROKER", f"http://127.0.0.1:{broker.server_port}")
    monkeypatch.setenv("DREAMLAKE_EXPERIMENT_BROKER_TOKEN", "broker-secret")
    monkeypatch.setenv("ML_DASH_BUFFER_ENABLED", "false")
    try:
        experiment = Experiment(dash_url="https://api.dash.ml")
        assert experiment._storage is None
        with experiment.run as exp:
            exp.params.set(lr=0.1)
            exp.log("training")
            exp.metrics("loss").log(value=0.5)
            exp.files("models").save_text("model state", to="model.txt")
            assert [item["id"] for item in exp.files("models").list()] == [file_id]
            output = exp.files("models/model.txt").download(to=str(tmp_path / "model.txt"))
            assert (tmp_path / "model.txt").read_text() == "model state"
            assert output == str(tmp_path / "model.txt")
        assert len(calls) == 4
        assert calls[0][0].endswith("/parameters")
        assert calls[1][0].endswith("/logs")
        assert calls[2][0].endswith("/metrics/loss")
        assert calls[3][0].endswith("/files")
        assert state["renewals"] >= 8
        assert all(token != "Bearer broker-secret" for _, token, _ in calls)
        with pytest.raises(ConfigurationError):
            Experiment(prefix="other/project/run")
        state["revoked"] = True
        with pytest.raises(ConfigurationError):
            experiment._client.set_parameters(experiment_id=run_id, data={"x": 1})
    finally:
        broker.shutdown()
        api.shutdown()
        broker.server_close()
        api.server_close()
