"""Isolated lifecycle tests: no Dash, DreamLake, or production database access."""

import json
import os
import subprocess
import sys
from unittest.mock import Mock

import httpx
import pytest

from ml_dash import Experiment, JobLifecycle
from ml_dash.exceptions import NetworkError


@pytest.fixture
def delivery(monkeypatch):
  sent = []
  handler = {"fail": None, "mutate": None}

  def post(client, url, *, json, headers):
    sent.append({"url": url, "payload": dict(json), "headers": dict(headers)})
    if handler["fail"]:
      raise handler["fail"]
    receipt = dict(json, jobId="job-1", acceptedAt="2026-10-11T12:00:00.000Z")
    if handler["mutate"]:
      handler["mutate"](receipt)
    return httpx.Response(200, json={"receipt": receipt, "applicationStatus": json["state"]},
                          request=httpx.Request("POST", url))

  monkeypatch.setattr(httpx.Client, "post", post)
  return sent, handler


def lifecycle(tmp_path, **changes):
  args = dict(api_url="http://127.0.0.1:9999/api", namespace="test-namespace",
              job_id="job-1", attempt_id="exec-1", token="private-test-token",
              journal_dir=tmp_path / "journal")
  args.update(changes)
  return JobLifecycle(**args)


def journal(tmp_path):
  return json.loads((tmp_path / "journal" / "lifecycle.json").read_text())


def test_lifecycle_is_opt_in_and_keeps_exact_payload(tmp_path, delivery):
  sent, _ = delivery
  report = lifecycle(tmp_path)
  assert sent == []
  report.start()
  report.complete()
  report.complete()
  report.flush()
  assert [r["payload"]["state"] for r in sent] == ["running", "succeeded"]
  assert all(set(r["payload"]) == {"version", "attemptId", "eventId", "state"} for r in sent)
  assert all(r["payload"]["attemptId"] == "exec-1" for r in sent)
  assert all(r["headers"] == {"Authorization": "Bearer private-test-token"} for r in sent)
  assert all(r["url"] == "http://127.0.0.1:9999/api/namespaces/test-namespace/runs/job-1/lifecycle" for r in sent)
  assert "private-test-token" not in (tmp_path / "journal" / "lifecycle.json").read_text()
  assert (tmp_path / "journal").stat().st_mode & 0o777 == 0o700
  assert (tmp_path / "journal" / "lifecycle.json").stat().st_mode & 0o777 == 0o600
  with pytest.raises(RuntimeError, match="terminal"):
    report.start()
  with pytest.raises(RuntimeError, match="different terminal"):
    report.fail()


def test_ack_loss_restarts_and_replays_identical_terminal(tmp_path, delivery):
  sent, handler = delivery
  report = lifecycle(tmp_path)
  report.start()
  handler["fail"] = httpx.ReadError("response lost after acceptance")
  with pytest.raises(NetworkError):
    report.complete()
  pending = journal(tmp_path)["events"][1]
  assert pending["receipt"] is None
  assert pending["payload"] == sent[-1]["payload"]
  handler["fail"] = None
  resumed = lifecycle(tmp_path, token="refreshed-token")
  resumed.flush()
  assert sent[-1]["payload"] == pending["payload"]
  assert sent[-1]["headers"]["Authorization"] == "Bearer refreshed-token"
  assert journal(tmp_path)["events"][1]["receipt"] is not None


@pytest.mark.parametrize("field,value", [("api_url", "https://another.test/api"),
  ("namespace", "another"), ("job_id", "job-2"), ("attempt_id", "exec-2")])
def test_exact_binding_fenced_before_request(tmp_path, delivery, field, value):
  report = lifecycle(tmp_path)
  report.start()
  count = len(delivery[0])
  with pytest.raises(ValueError, match="binding"):
    lifecycle(tmp_path, **{field: value})
  assert len(delivery[0]) == count


@pytest.mark.parametrize("field,value", [("jobId", "job-2"), ("attemptId", "exec-2"),
  ("eventId", "other-event"), ("state", "failed"), ("version", 2), ("acceptedAt", None)])
def test_mismatched_receipt_stays_pending(tmp_path, delivery, field, value):
  delivery[1]["mutate"] = lambda receipt: receipt.update({field: value})
  with pytest.raises(NetworkError, match="acknowledgment"):
    lifecycle(tmp_path).start()
  assert journal(tmp_path)["events"][0]["receipt"] is None


def test_start_failure_retains_id_and_cannot_invent_terminal(tmp_path, delivery):
  delivery[1]["fail"] = httpx.ConnectError("offline")
  report = lifecycle(tmp_path)
  with pytest.raises(NetworkError):
    report.start()
  delivery[1]["fail"] = None
  lifecycle(tmp_path).flush()
  assert delivery[0][0]["payload"] == delivery[0][1]["payload"]
  assert len(journal(tmp_path)["events"]) == 1
  assert journal(tmp_path)["events"][0]["payload"]["state"] == "running"


def test_rejects_unstarted_terminal_and_corruption(tmp_path, delivery):
  report = lifecycle(tmp_path)
  with pytest.raises(RuntimeError, match="start"):
    report.complete()
  (tmp_path / "journal" / "lifecycle.json").write_text('{"version":1}')
  with pytest.raises(ValueError):
    report.flush()
  assert delivery[0] == []


def test_rejects_insecure_or_symlink_directory(tmp_path):
  insecure = tmp_path / "public"
  insecure.mkdir(mode=0o755)
  with pytest.raises(ValueError, match="private"):
    lifecycle(tmp_path, journal_dir=insecure)
  private = tmp_path / "private"
  private.mkdir(mode=0o700)
  symlink = tmp_path / "link"
  symlink.symlink_to(private)
  with pytest.raises(ValueError, match="symlink"):
    lifecycle(tmp_path, journal_dir=symlink)


def test_concurrent_process_cannot_write_locked_journal(tmp_path, delivery):
  report = lifecycle(tmp_path)
  code = '''
import sys
from ml_dash import JobLifecycle
try:
    JobLifecycle(api_url="http://127.0.0.1:9999/api", namespace="test-namespace",
                 job_id="job-1", attempt_id="exec-1", token="child-token", journal_dir=sys.argv[1])
except RuntimeError as error:
    assert "already in use" in str(error)
else:
    raise AssertionError("concurrent journal writer was allowed")
'''
  with report._locked():
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path / "journal")],
                               env=dict(os.environ, ML_DASH_NO_VERSION_CHECK="1"),
                               capture_output=True, text=True, timeout=20)
  assert completed.returncode == 0, completed.stderr
  assert journal(tmp_path)["events"] == []


def test_crash_after_persist_before_send_survives_new_process(tmp_path, delivery):
  code = '''
import os, sys
from ml_dash import JobLifecycle
report = JobLifecycle(api_url="http://127.0.0.1:9999/api", namespace="test-namespace",
                      job_id="job-1", attempt_id="exec-1", token="child-token", journal_dir=sys.argv[1])
report._send_pending = lambda data: os._exit(17)
report.start()
'''
  completed = subprocess.run([sys.executable, "-c", code, str(tmp_path / "journal")],
                             env=dict(os.environ, ML_DASH_NO_VERSION_CHECK="1"),
                             capture_output=True, text=True, timeout=20)
  assert completed.returncode == 17, completed.stderr
  payload = journal(tmp_path)["events"][0]["payload"]
  lifecycle(tmp_path).flush()
  assert delivery[0][0]["payload"] == payload
  assert len(journal(tmp_path)["events"]) == 1


def test_experiment_context_reports_application_contract(tmp_path, delivery, monkeypatch):
  monkeypatch.setenv("ML_DASH_BUFFER_ENABLED", "false")
  with Experiment(prefix="test/project/run", dash_root=str(tmp_path / "experiment"),
                  job_lifecycle=lifecycle(tmp_path)).run:
    pass
  assert [r["payload"]["state"] for r in delivery[0]] == ["running", "succeeded"]


def test_body_exception_remains_primary_on_delivery_failure(tmp_path, delivery):
  report = lifecycle(tmp_path)
  exp = Experiment(prefix="test/project/run", dash_root=str(tmp_path / "experiment"), job_lifecycle=report)
  with pytest.raises(ValueError, match="application error"):
    with exp.run:
      delivery[1]["fail"] = httpx.ReadError("lost ack")
      raise ValueError("application error")
  assert journal(tmp_path)["events"][1]["payload"]["state"] == "failed"


def test_terminal_recorded_before_dash_patch_and_flush_failure_means_failed(tmp_path, delivery):
  report = lifecycle(tmp_path)
  exp = Experiment(prefix="test/project/run", dash_root=str(tmp_path / "experiment"), job_lifecycle=report)
  exp.run.start()
  if exp._buffer_manager:
    exp._buffer_manager.stop()
  buffer = Mock()
  buffer.stop.side_effect = NetworkError("buffer flush failure")
  exp._buffer_manager = buffer
  remote = Mock()

  def update(**kwargs):
    assert journal(tmp_path)["events"][1]["payload"]["state"] == "failed"
    assert kwargs["status"] == "FAILED"
    raise NetworkError("dash API unavailable")

  remote.update_experiment_status.side_effect = update
  exp._client = remote
  exp._experiment_id = "experiment-1"
  with pytest.raises(NetworkError, match="buffer flush failure"):
    exp.run.complete()
  assert delivery[0][-1]["payload"]["state"] == "failed"
  assert journal(tmp_path)["events"][1]["receipt"] is not None


def test_durable_write_failure_prevents_send(tmp_path, delivery, monkeypatch):
  report = lifecycle(tmp_path)
  def no_disk_space(data):
    raise OSError("no disk space")
  monkeypatch.setattr(report, "_write", no_disk_space)
  with pytest.raises(OSError, match="disk"):
    report.start()
  assert delivery[0] == []
  assert journal(tmp_path)["events"] == []


def test_event_is_already_persisted_at_send(tmp_path, delivery, monkeypatch):
  original = httpx.Client.post
  def inspect_persisted(client, url, *, json, headers):
    events = journal(tmp_path)["events"]
    assert any(event["payload"] == json and event["receipt"] is None for event in events)
    return original(client, url, json=json, headers=headers)
  monkeypatch.setattr(httpx.Client, "post", inspect_persisted)
  report = lifecycle(tmp_path)
  report.start()
  report.cancel()
  assert delivery[0][-1]["payload"]["state"] == "cancelled"


def test_terminal_ack_loss_process_death_keeps_exact_evidence(tmp_path, delivery):
  code = '''
import os, sys
import httpx
from ml_dash import JobLifecycle

def post(client, url, *, json, headers):
    if json["state"] == "succeeded":
        # Simulate server acceptance followed by lost ACK and process death.
        os._exit(18)
    return httpx.Response(200, json={"receipt": dict(json, jobId="job-1", acceptedAt="2026-10-11T12:00:00Z")},
                          request=httpx.Request("POST", url))
httpx.Client.post = post
report = JobLifecycle(api_url="http://127.0.0.1:9999/api", namespace="test-namespace",
                      job_id="job-1", attempt_id="exec-1", token="child-token", journal_dir=sys.argv[1])
report.start()
report.complete()
'''
  completed = subprocess.run([sys.executable, "-c", code, str(tmp_path / "journal")],
                             env=dict(os.environ, ML_DASH_NO_VERSION_CHECK="1"),
                             capture_output=True, text=True, timeout=20)
  assert completed.returncode == 18, completed.stderr
  pending = journal(tmp_path)["events"][1]["payload"]
  lifecycle(tmp_path).flush()
  assert len(delivery[0]) == 1  # acknowledged start is not replayed
  assert delivery[0][0]["payload"] == pending


@pytest.mark.parametrize("status", [401, 403, 404, 409, 500, 302])
def test_denial_and_conflict_keep_evidence_pending(tmp_path, delivery, monkeypatch, status):
  def post(client, url, **kwargs):
    return httpx.Response(status, request=httpx.Request("POST", url))
  monkeypatch.setattr(httpx.Client, "post", post)
  with pytest.raises(NetworkError, match=f"HTTP {status}"):
    lifecycle(tmp_path).start()
  assert journal(tmp_path)["events"][0]["receipt"] is None


@pytest.mark.parametrize("failure", ["terminal_ack", "dash_patch"])
def test_same_experiment_cannot_rerun_after_terminal_delivery_failure(tmp_path, delivery, failure):
  report = lifecycle(tmp_path)
  exp = Experiment(prefix="test/project/run", dash_root=str(tmp_path / "experiment"), job_lifecycle=report)
  exp.run.start()
  if failure == "terminal_ack":
    delivery[1]["fail"] = httpx.ReadError("lost terminal ack")
  else:
    exp._client = Mock()
    exp._experiment_id = "experiment-1"
    exp._client.update_experiment_status.side_effect = NetworkError("dash API unavailable")
  with pytest.raises(NetworkError):
    exp.run.complete()
  assert not exp._is_open
  delivery[1]["fail"] = None
  with pytest.raises(RuntimeError, match="terminal"):
    with exp.run:
      pytest.fail("application body executed again under an existing terminal outcome")
  report.flush()
  assert journal(tmp_path)["events"][1]["payload"]["state"] == "succeeded"
