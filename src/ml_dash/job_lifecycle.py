"""Explicit application lifecycle evidence for an existing DreamLake job attempt.

The private journal is an outbox, not a second authority. The authenticated job
API owns outcomes. No process/experiment status is read and no exit hook is used.
"""

import json
import os
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from .exceptions import NetworkError


class JobLifecycle:
  """Durable delivery of one start and one terminal event for an exact attempt.

  ``api_url`` is the existing DreamLake API base, separate from Dash's API.
  ``journal_dir`` must be dedicated private durable POSIX storage. Reconstruct
  with the same binding and call :meth:`flush` after interruption; no event is
  created merely by constructing or flushing this object.
  """

  _STATES = frozenset(("running", "succeeded", "failed", "cancelled"))

  def __init__(self, *, api_url: str, namespace: str, job_id: str,
               attempt_id: str, token: str, journal_dir):
    parsed = urlsplit(api_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment):
      raise ValueError("api_url must be an HTTP(S) API base without credentials, query or fragment")
    for name, value in (("namespace", namespace), ("job_id", job_id),
                        ("attempt_id", attempt_id), ("token", token)):
      if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    self._binding = {"apiUrl": api_url.rstrip("/"), "namespace": namespace,
                     "jobId": job_id, "attemptId": attempt_id}
    self._token = token
    self._directory = Path(journal_dir).absolute()
    self._mutex = threading.Lock()
    # Persist every newly created directory entry as well as the journal file.
    missing = []
    directory = self._directory
    while not directory.exists():
      missing.append(directory)
      directory = directory.parent
    for directory in reversed(missing):
      directory.mkdir(mode=0o700, exist_ok=True)
      parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
      try:
        os.fsync(parent_fd)
      finally:
        os.close(parent_fd)
    info = self._directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077
        or info.st_uid != os.getuid()):
      raise ValueError("journal_dir must be an owned private directory (mode 0700), not a symlink")
    self._path = self._directory / "lifecycle.json"
    self._url = (self._binding["apiUrl"] + "/namespaces/" + quote(namespace, safe="")
                 + "/runs/" + quote(job_id, safe="") + "/lifecycle")
    with self._locked():
      if self._path.exists():
        self._read()
      else:
        self._write({"version": 1, "binding": self._binding, "events": []})

  @contextmanager
  def _locked(self):
    # flock is per open file description, so independent objects/processes also
    # contend. The thread guard prevents concurrent calls on the same object.
    import fcntl
    if not self._mutex.acquire(blocking=False):
      raise RuntimeError("Job lifecycle journal is already in use")
    fd = None
    try:
      fd = os.open(self._directory / "lifecycle.lock",
                   os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
      info = os.fstat(fd)
      if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("Lifecycle lock must be an owned private regular file")
      try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError as exc:
        raise RuntimeError("Job lifecycle journal is already in use") from exc
      yield
    finally:
      if fd is not None:
        os.close(fd)
      self._mutex.release()

  def _read(self):
    fd = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r") as file:
      info = os.fstat(file.fileno())
      if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("Lifecycle journal must be an owned private regular file")
      data = json.load(file)
    if (not isinstance(data, dict) or data.get("version") != 1
        or data.get("binding") != self._binding):
      raise ValueError("Lifecycle journal binding or version does not match")
    events = data.get("events")
    if not isinstance(events, list) or len(events) > 2:
      raise ValueError("Invalid lifecycle journal events")
    seen = set()
    for i, event in enumerate(events):
      if not isinstance(event, dict) or set(event) != {"payload", "receipt"}:
        raise ValueError("Invalid lifecycle journal event")
      payload = event["payload"]
      if (not isinstance(payload, dict)
          or set(payload) != {"version", "attemptId", "eventId", "state"}
          or payload["version"] != 1
          or payload["attemptId"] != self._binding["attemptId"]
          or payload["state"] not in self._STATES
          or (i == 0) != (payload["state"] == "running")):
        raise ValueError("Invalid lifecycle journal payload")
      try:
        event_id = str(uuid.UUID(payload["eventId"]))
      except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid lifecycle journal event ID") from exc
      if event_id != payload["eventId"] or event_id in seen:
        raise ValueError("Invalid lifecycle journal event ID")
      seen.add(event_id)
      if event["receipt"] is not None:
        self._validate_receipt(event["receipt"], payload)
    return data

  def _write(self, data):
    fd, temporary = tempfile.mkstemp(prefix=".lifecycle-", dir=self._directory)
    try:
      with os.fdopen(fd, "w") as file:
        json.dump(data, file, sort_keys=True, separators=(",", ":"))
        file.flush()
        os.fsync(file.fileno())
      os.replace(temporary, self._path)
      directory_fd = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY)
      try:
        os.fsync(directory_fd)
      finally:
        os.close(directory_fd)
    finally:
      if os.path.exists(temporary):
        os.unlink(temporary)

  def _validate_receipt(self, receipt, payload):
    expected = dict(payload, jobId=self._binding["jobId"])
    if (not isinstance(receipt, dict)
        or any(receipt.get(key) != value for key, value in expected.items())
        or not isinstance(receipt.get("acceptedAt"), str) or not receipt["acceptedAt"]):
      raise NetworkError("Job lifecycle acknowledgment does not match the recorded event")

  def _send_pending(self, data):
    # A retry is deliberate (flush/start/same terminal method). Transient failures
    # never discard evidence or replace an event ID. Redirects must not move auth.
    with httpx.Client(timeout=30.0, follow_redirects=False) as client:
      for event in data["events"]:
        if event["receipt"] is not None:
          continue
        try:
          response = client.post(self._url, json=event["payload"],
                                 headers={"Authorization": f"Bearer {self._token}"})
          response.raise_for_status()
          receipt = response.json().get("receipt")
          self._validate_receipt(receipt, event["payload"])
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
          # Do not include request/response bodies or credentials in errors.
          status = getattr(getattr(exc, "response", None), "status_code", None)
          detail = f"HTTP {status}" if status else type(exc).__name__
          raise NetworkError(f"Job lifecycle delivery failed ({detail}); evidence remains pending") from exc
        event["receipt"] = receipt
        self._write(data)

  def flush(self):
    """Deliver pending immutable evidence; never infer or create an outcome."""
    with self._locked():
      self._send_pending(self._read())

  def _report(self, state):
    with self._locked():
      data = self._read()
      events = data["events"]
      if state == "running":
        if len(events) > 1:
          raise RuntimeError("This attempt already has terminal evidence; flush delivery instead of restarting it")
      elif not events:
        raise RuntimeError("Call start before reporting a terminal application outcome")
      if state != "running" and len(events) > 1 and events[1]["payload"]["state"] != state:
        raise RuntimeError("This attempt already has a different terminal application outcome")
      if not events or (state != "running" and len(events) == 1):
        events.append({"payload": {"version": 1, "attemptId": self._binding["attemptId"],
                                   "eventId": str(uuid.uuid4()), "state": state}, "receipt": None})
        self._write(data)  # Evidence must survive BEFORE any HTTP request.
      self._send_pending(data)

  def start(self):
    """Record application start and deliver pending evidence."""
    self._report("running")

  def complete(self):
    """Record explicit application success and deliver pending evidence."""
    self._report("succeeded")

  def fail(self):
    """Record explicit application failure and deliver pending evidence."""
    self._report("failed")

  def cancel(self):
    """Record explicit application cancellation and deliver pending evidence."""
    self._report("cancelled")
