"""Nymph job-scoped DreamLake experiment transport.

The broker secret never leaves loopback. Every API call renews a short-lived,
run-bound lease, so cancelled or replaced attempts lose write access promptly.
"""

import os
import re
import time
from pathlib import Path
from fnmatch import fnmatch
from urllib.parse import quote, urlparse

import httpx

from .exceptions import ConfigurationError


_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_METRIC = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class NativeBrokerClient:
    def __init__(self):
        endpoint = os.environ.get("DREAMLAKE_EXPERIMENT_BROKER")
        secret = os.environ.get("DREAMLAKE_EXPERIMENT_BROKER_TOKEN")
        if not endpoint or not secret:
            raise ConfigurationError("DreamLake experiment broker address and token are both required")
        url = urlparse(endpoint)
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1")
                or url.username or url.password or url.path or url.query or url.fragment):
            raise ConfigurationError("DreamLake experiment broker must be a loopback HTTP origin")
        self._endpoint = endpoint
        self._secret = secret
        self._pins = None
        self._lease = self._renew()
        self.experiment_url = self._lease["experimentUrl"]

    @property
    def namespace(self):
        return self._lease["namespace"]

    @property
    def project(self):
        return self._lease["project"]

    @property
    def run_key(self):
        return self._lease["runKey"]

    @property
    def run_id(self):
        return self._lease["runId"]

    def _renew(self):
        with httpx.Client(timeout=20, trust_env=False, follow_redirects=False) as client:
            response = client.post(self._endpoint + "/session", content=b"{}", headers={
                "Authorization": "Bearer " + self._secret, "Content-Type": "application/json"})
            if response.status_code != 200:
                raise ConfigurationError("DreamLake experiment broker denied the session")
            response.raise_for_status()
            lease = response.json()
        keys = ("namespace", "project", "runKey", "runId", "projectId",
                "instanceId", "workloadId", "generation", "apiUrl", "experimentUrl")
        if not isinstance(lease, dict) or any(key not in lease for key in keys):
            raise ConfigurationError("Invalid DreamLake experiment lease")
        if any(not isinstance(lease[key], str) or not _ID.fullmatch(lease[key])
               for key in keys[:7]) or not isinstance(lease["generation"], int) or lease["generation"] < 1:
            raise ConfigurationError("Invalid DreamLake experiment lease identity")
        if (not isinstance(lease.get("token"), str) or not lease["token"]
                or not isinstance(lease.get("expiresAt"), (int, float))
                or not time.time() < lease["expiresAt"] <= time.time() + 61):
            raise ConfigurationError("Invalid DreamLake experiment lease lifetime")
        api = urlparse(lease["apiUrl"])
        expected_path = ("/native-experiments/ns/" + quote(lease["namespace"], safe="")
                         + "/projects/" + quote(lease["project"], safe=""))
        if (api.scheme != "https" and not (api.scheme == "http" and api.hostname in ("127.0.0.1", "::1"))
                or api.username or api.password or api.query or api.fragment or api.path != expected_path):
            raise ConfigurationError("Invalid DreamLake experiment API destination")
        view = urlparse(lease["experimentUrl"])
        if (view.scheme != "https" and not (view.scheme == "http" and view.hostname in ("127.0.0.1", "::1"))
                or view.username or view.password or view.fragment):
            raise ConfigurationError("Invalid DreamLake experiment view destination")
        if view.query != "run=" + lease["runId"]:
            raise ConfigurationError("DreamLake experiment view does not match the job run")
        pins = tuple(lease[key] for key in keys)
        if self._pins is not None and pins != self._pins:
            raise ConfigurationError("DreamLake experiment lease changed run identity")
        self._pins = pins
        return lease

    def _request(self, method, path, *, payload=None, params=None):
        lease = self._renew()
        with httpx.Client(timeout=30, trust_env=False, follow_redirects=False) as client:
            response = client.request(method, lease["apiUrl"] + "/runs/" + lease["runId"] + path,
                                      headers={"Authorization": "Bearer " + lease["token"]},
                                      json=payload, params=params)
            if not 200 <= response.status_code < 300:
                response.raise_for_status()
                raise ConfigurationError("DreamLake experiment API redirected the request")
            response.raise_for_status()
            return response.json()

    def _bound(self, experiment_id):
        if experiment_id != self.run_id:
            raise ConfigurationError("Experiment does not match the DreamLake job run")

    def create_or_update_experiment(self, *, project, **_kwargs):
        if project != self.project:
            raise ConfigurationError("Experiment project does not match the DreamLake job")
        return {"experiment": {"id": self.run_id}}

    def update_experiment_status(self, *, experiment_id, status):
        self._bound(experiment_id)
        # The compute service owns the terminal result, including process failures.
        return {"status": status}

    def set_parameters(self, *, experiment_id, data):
        self._bound(experiment_id)
        return self._request("POST", "/parameters", payload={"data": data})

    def get_parameters(self, *, experiment_id):
        self._bound(experiment_id)
        return (self._request("GET", "/parameters").get("parameters") or {}).get("data", {})

    def create_log_entries(self, *, experiment_id, logs):
        self._bound(experiment_id)
        return self._request("POST", "/logs", payload={"entries": logs})

    def append_to_metric(self, *, experiment_id, metric_name, data, **_kwargs):
        return self.append_batch_to_metric(experiment_id=experiment_id, metric_name=metric_name,
                                           data_points=[data])

    def append_batch_to_metric(self, *, experiment_id, metric_name, data_points, **_kwargs):
        self._bound(experiment_id)
        if not isinstance(metric_name, str) or not _METRIC.fullmatch(metric_name):
            raise ConfigurationError("DreamLake metric name must use letters, numbers, _, . or -")
        return self._request("POST", "/metrics/" + quote(metric_name, safe=""),
                             payload={"dataPoints": data_points})

    def append_batch_to_track(self, *, experiment_id, topic, entries):
        return self.append_batch_to_metric(experiment_id=experiment_id, metric_name=topic,
                                           data_points=entries)

    def list_metrics(self, *, experiment_id):
        self._bound(experiment_id)
        return self._request("GET", "/metrics")["metrics"]

    def upload_file(self, *, experiment_id, file_path, prefix, filename, description,
                    tags, metadata, checksum, content_type, size_bytes, **_kwargs):
        self._bound(experiment_id)
        lease = self._renew()
        fields = {"prefix": prefix or "/", "checksum": checksum,
                  "sizeBytes": str(size_bytes)}
        if description:
            fields["description"] = description
        if tags:
            fields["tags"] = ",".join(tags)
        if metadata:
            import json
            fields["metadata"] = json.dumps(metadata)
        with open(file_path, "rb") as source:
            with httpx.Client(timeout=120, trust_env=False, follow_redirects=False) as client:
                response = client.post(lease["apiUrl"] + "/runs/" + lease["runId"] + "/files",
                                       headers={"Authorization": "Bearer " + lease["token"]},
                                       data=fields, files={"file": (filename, source, content_type or "application/octet-stream")})
                if response.status_code != 200 and response.status_code != 201:
                    response.raise_for_status()
                    raise ConfigurationError("DreamLake experiment API redirected the upload")
                return response.json()

    def list_files(self, *, experiment_id, limit=500, offset=0):
        self._bound(experiment_id)
        files = self._request("GET", "/files")["files"]
        return {"files": files[offset:offset + limit], "hasMore": offset + limit < len(files)}

    def search_files(self, *, pattern, experiment_id, limit=100, **_kwargs):
        self._bound(experiment_id)
        files = self._request("GET", "/files")["files"]
        matches = []
        for file in files:
            path = (file.get("prefix") or "").strip("/")
            full_path = (path + "/" if path else "") + file["filename"]
            if fnmatch(full_path, pattern) or fnmatch(file["filename"], pattern.removeprefix("**/")):
                matches.append({**file, "path": full_path})
        return matches[:limit]

    def download_file(self, *, experiment_id, file_id, dest_path=None):
        self._bound(experiment_id)
        if not re.fullmatch(r"[a-f0-9]{24}", file_id):
            raise ConfigurationError("Invalid DreamLake artifact ID")
        info = self._request("GET", "/files/" + file_id)["file"]
        lease = self._renew()
        with httpx.Client(timeout=120, trust_env=False, follow_redirects=False) as client:
            response = client.get(lease["apiUrl"] + "/runs/" + lease["runId"] +
                                  "/files/" + file_id + "/download",
                                  headers={"Authorization": "Bearer " + lease["token"]})
            if response.status_code != 200:
                response.raise_for_status()
                raise ConfigurationError("DreamLake experiment API redirected the download")
            data = response.content
        import hashlib
        if hashlib.sha256(data).hexdigest() != info["checksum"]:
            raise ConfigurationError("DreamLake artifact checksum mismatch")
        target = Path(dest_path or info["filename"])
        target.write_bytes(data)
        return str(target)

    def unsupported(self, *_args, **_kwargs):
        raise ConfigurationError("This ML-Dash operation is not yet available for DreamLake jobs")

    delete_file = update_file = unsupported
    read_metric_data = read_metric_rows = get_metric_stats = unsupported
