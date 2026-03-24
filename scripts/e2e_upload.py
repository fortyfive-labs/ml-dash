#!/usr/bin/env python3
"""
E2E tests for the ml-dash upload command.

Creates local experiments, runs cmd_upload, then verifies data on the server
via RemoteClient.

Usage:
    cd /Users/57block/fortyfive/ml-dash
    uv run scripts/e2e_upload.py

Prerequisites:
    ml-dash login   (stores token in system keychain)
    ml-dash server running at http://localhost:3000
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# Add src to path when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_dash import Experiment
from ml_dash.auth.token_storage import get_token_storage
from ml_dash.cli_commands.upload import UploadState, cmd_upload
from ml_dash.client import RemoteClient

SERVER_URL = "http://localhost:3000"
NAMESPACE = "tom"

failures = 0


def check(label: str, passed: bool, detail: str = "") -> bool:
    global failures
    status = "✓" if passed else "✗"
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not passed:
        failures += 1
    return passed


def make_args(dash_root: Path, **overrides) -> argparse.Namespace:
    """Build a minimal cmd_upload Namespace with sensible defaults."""
    defaults = dict(
        path=str(dash_root),
        dash_url=SERVER_URL,
        tracks=None,
        remote_path=None,
        project=None,
        target=None,
        dry_run=False,
        strict=False,
        verbose=False,
        batch_size=100,
        skip_logs=False,
        skip_metrics=False,
        skip_files=False,
        skip_params=False,
        resume=False,
        state_file=str(dash_root / ".upload-state.json"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def ts() -> str:
    """Microsecond timestamp suffix for unique names."""
    return str(int(time.time() * 1_000_000))


# ============================================================================
# Step 1: Basic upload — logs, params, metrics
# ============================================================================

def test_basic_upload(client: RemoteClient):
    print("\nStep 1: Basic upload (logs + params + metrics)...")

    suffix = ts()
    project = f"e2e-upload-basic-{suffix}"
    exp_name = "run-001"
    prefix = f"{NAMESPACE}/{project}/{exp_name}"

    with tempfile.TemporaryDirectory() as tmp:
        dash_root = Path(tmp)

        # Create local experiment
        exp = Experiment(prefix=prefix, dash_root=str(dash_root))
        with exp.run as e:
            e.log("Starting training")
            e.log("Epoch 1 complete")
            e.params.set(lr=0.001, batch_size=32, epochs=10)
            for i in range(5):
                e.metrics("train").log(loss=1.0 - i * 0.1, step=i)
            for i in range(5):
                e.metrics("eval").log(loss=1.1 - i * 0.1, step=i)

        # Upload
        args = make_args(dash_root)
        rc = cmd_upload(args)
        check("cmd_upload returns 0", rc == 0, f"got {rc}")

        # Verify on server
        exp_data = client.get_experiment_graphql(project, exp_name)
        check("experiment created on server", exp_data is not None)
        if not exp_data:
            return

        exp_id = exp_data["id"]

        log_total = (exp_data.get("logMetadata") or {}).get("totalLogs", 0)
        check("logs uploaded", log_total >= 2, f"totalLogs={log_total}")

        params_data = exp_data.get("parameters")
        check("parameters uploaded", params_data is not None)

        metric_names = {m["name"] for m in exp_data.get("metrics") or []}
        check("train metric uploaded", "train" in metric_names)
        check("eval metric uploaded", "eval" in metric_names)

        train_pts = next(
            (m["metricMetadata"]["totalDataPoints"]
             for m in exp_data["metrics"] if m["name"] == "train"),
            0,
        )
        check("train has 5 data points", train_pts == 5, f"got {train_pts}")


# ============================================================================
# Step 2: Skip flags — skip-logs, skip-metrics
# ============================================================================

def test_skip_flags(client: RemoteClient):
    print("\nStep 2: Skip flags (skip-logs + skip-metrics, params only)...")

    suffix = ts()
    project = f"e2e-upload-skip-{suffix}"
    exp_name = "run-001"
    prefix = f"{NAMESPACE}/{project}/{exp_name}"

    with tempfile.TemporaryDirectory() as tmp:
        dash_root = Path(tmp)

        exp = Experiment(prefix=prefix, dash_root=str(dash_root))
        with exp.run as e:
            e.log("Log entry")
            e.params.set(lr=0.01)
            e.metrics("train").log(loss=0.5, step=0)

        args = make_args(dash_root, skip_logs=True, skip_metrics=True)
        rc = cmd_upload(args)
        check("cmd_upload returns 0", rc == 0, f"got {rc}")

        exp_data = client.get_experiment_graphql(project, exp_name)
        check("experiment created", exp_data is not None)
        if not exp_data:
            return

        log_total = (exp_data.get("logMetadata") or {}).get("totalLogs", 0)
        check("logs skipped (totalLogs=0)", log_total == 0, f"totalLogs={log_total}")

        metric_names = [m["name"] for m in exp_data.get("metrics") or []]
        check("metrics skipped", len(metric_names) == 0, f"metrics={metric_names}")

        check("params uploaded", exp_data.get("parameters") is not None)


# ============================================================================
# Step 3: Project filter — only matching experiments uploaded
# ============================================================================

def test_project_filter(client: RemoteClient):
    print("\nStep 3: Project filter...")

    suffix = ts()
    project_a = f"e2e-filter-a-{suffix}"
    project_b = f"e2e-filter-b-{suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        dash_root = Path(tmp)

        for proj in (project_a, project_b):
            exp = Experiment(
                prefix=f"{NAMESPACE}/{proj}/run-001",
                dash_root=str(dash_root),
            )
            with exp.run as e:
                e.log(f"Log from {proj}")

        # Upload only project_a
        args = make_args(dash_root, project=project_a)
        rc = cmd_upload(args)
        check("cmd_upload returns 0", rc == 0, f"got {rc}")

        exp_a = client.get_experiment_graphql(project_a, "run-001")
        check("project_a uploaded", exp_a is not None)

        exp_b = client.get_experiment_graphql(project_b, "run-001")
        check("project_b NOT uploaded", exp_b is None)


# ============================================================================
# Step 4: Dry run — nothing uploaded
# ============================================================================

def test_dry_run(client: RemoteClient):
    print("\nStep 4: Dry run (no data should be uploaded)...")

    suffix = ts()
    project = f"e2e-dryrun-{suffix}"
    exp_name = "run-001"
    prefix = f"{NAMESPACE}/{project}/{exp_name}"

    with tempfile.TemporaryDirectory() as tmp:
        dash_root = Path(tmp)

        exp = Experiment(prefix=prefix, dash_root=str(dash_root))
        with exp.run as e:
            e.log("Should not be uploaded")
            e.params.set(lr=0.001)

        args = make_args(dash_root, dry_run=True)
        rc = cmd_upload(args)
        check("cmd_upload (dry-run) returns 0", rc == 0, f"got {rc}")

        # Experiment must NOT exist on server
        exp_data = client.get_experiment_graphql(project, exp_name)
        check("experiment NOT created in dry run", exp_data is None)


# ============================================================================
# Step 5: Resume — skip already-completed experiments
# ============================================================================

def test_resume(client: RemoteClient):
    print("\nStep 5: Resume (skip already-completed)...")

    suffix = ts()
    project = f"e2e-resume-{suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        dash_root = Path(tmp)

        for i in range(1, 4):
            exp = Experiment(
                prefix=f"{NAMESPACE}/{project}/run-00{i}",
                dash_root=str(dash_root),
            )
            with exp.run as e:
                e.log(f"Run {i} log")

        state_file = dash_root / ".upload-state.json"

        # Simulate: run-001 and run-002 already completed
        state = UploadState(
            dash_root=str(dash_root.absolute()),
            remote_url=SERVER_URL,
            completed_experiments=[f"{project}/run-001", f"{project}/run-002"],
        )
        state.save(state_file)

        args = make_args(
            dash_root,
            resume=True,
            state_file=str(state_file),
        )
        rc = cmd_upload(args)
        check("cmd_upload (resume) returns 0", rc == 0, f"got {rc}")

        # Only run-003 should exist on server (001 and 002 were "already done")
        exp1 = client.get_experiment_graphql(project, "run-001")
        exp2 = client.get_experiment_graphql(project, "run-002")
        exp3 = client.get_experiment_graphql(project, "run-003")

        check("run-001 skipped (not on server)", exp1 is None)
        check("run-002 skipped (not on server)", exp2 is None)
        check("run-003 uploaded", exp3 is not None)


# ============================================================================
# Step 6: Track upload (--tracks + --remote-path)
# ============================================================================

def test_track_upload(client: RemoteClient):
    print("\nStep 6: Track upload (--tracks + --remote-path)...")

    suffix = ts()
    project = f"e2e-track-{suffix}"
    exp_name = "run-001"
    prefix = f"{NAMESPACE}/{project}/{exp_name}"

    # Create the experiment on the server first
    exp = Experiment(prefix=prefix, dash_url=SERVER_URL)
    with exp.run as e:
        e.log("Experiment for track upload")
        exp_id = e._experiment_id

    check("remote experiment created", exp_id is not None, f"id={exp_id}")
    if not exp_id:
        return

    with tempfile.TemporaryDirectory() as tmp:
        # Build a JSONL track file
        track_file = Path(tmp) / "robot_position.jsonl"
        entries = [
            {"timestamp": 1000 + i * 100, "x": float(i), "y": float(i) * 0.5}
            for i in range(10)
        ]
        with open(track_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        remote_path = f"{NAMESPACE}/{project}/{exp_name}/robot/position"

        args = make_args(
            Path(tmp),
            tracks=str(track_file),
            remote_path=remote_path,
        )
        rc = cmd_upload(args)
        check("cmd_upload (track) returns 0", rc == 0, f"got {rc}")


# ============================================================================
# Step 7: Target prefix (--target remaps namespace/project)
# ============================================================================

def test_target_prefix(client: RemoteClient):
    print("\nStep 7: Target prefix (--target)...")

    suffix = ts()
    local_project = f"e2e-local-{suffix}"
    target_project = f"e2e-target-{suffix}"
    exp_name = "run-001"

    with tempfile.TemporaryDirectory() as tmp:
        dash_root = Path(tmp)

        # Create local experiment under local_project
        exp = Experiment(
            prefix=f"{NAMESPACE}/{local_project}/{exp_name}",
            dash_root=str(dash_root),
        )
        with exp.run as e:
            e.log("Remapped experiment")
            e.params.set(lr=0.005)

        # Upload to a different target project
        target = f"{NAMESPACE}/{target_project}"
        args = make_args(dash_root, target=target)
        rc = cmd_upload(args)
        check("cmd_upload (target) returns 0", rc == 0, f"got {rc}")

        # Experiment should appear under target_project, not local_project
        in_target = client.get_experiment_graphql(target_project, exp_name)
        in_local = client.get_experiment_graphql(local_project, exp_name)

        check("experiment under target project", in_target is not None)
        check("experiment NOT under local project", in_local is None)


# ============================================================================
# Main
# ============================================================================

def main():
    global failures
    print("\n=== Upload E2E ===\n")

    token = get_token_storage().load("ml-dash-token")
    if not token:
        print("ERROR: Not logged in. Run `ml-dash login` first.")
        sys.exit(1)

    print(f"Namespace: {NAMESPACE}")
    print(f"Server:    {SERVER_URL}")

    client = RemoteClient(base_url=SERVER_URL, namespace=NAMESPACE)

    test_basic_upload(client)
    test_skip_flags(client)
    test_project_filter(client)
    test_dry_run(client)
    test_resume(client)
    test_track_upload(client)
    test_target_prefix(client)

    print(f"\n{'=' * 40}")
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK(S) FAILED")
    print("=" * 40 + "\n")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
