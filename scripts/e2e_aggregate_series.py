#!/usr/bin/env python3
"""
E2E test for POST /api/experiments/aggregate/series.

Creates 3 experiments with different simulated learning curves,
then calls the aggregate series endpoint and prints results.

Usage:
    cd /Users/57block/fortyfive/ml-dash
    uv run scripts/e2e_aggregate_series.py
"""

import math
import os
import random
import sys
import time

import httpx

# Larger batch → fewer HTTP round-trips for large datasets (50k–70k points)
os.environ.setdefault("ML_DASH_METRIC_BATCH_SIZE", "1000")

from ml_dash import Experiment
from ml_dash.auth.token_storage import get_token_storage

SERVER_URL = "http://localhost:3000"
PROJECT = "test-aggregate-series"
NAMESPACE = "tom"


def make_learning_curve(n_steps: int, noise: float, seed: int):
    rng = random.Random(seed)
    return [
        max(0.001, math.exp(-0.03 * i) + rng.gauss(0, noise))
        for i in range(n_steps)
    ]


def create_experiment(prefix: str, n_steps: int, noise: float, seed: int) -> str:
    print(f"  Creating experiment: {prefix.split('/')[-1]} ({n_steps} steps, noise={noise})")
    exp = Experiment(prefix=prefix, dash_url=SERVER_URL, dash_root=None)
    with exp.run as e:
        for step, loss in enumerate(make_learning_curve(n_steps, noise, seed)):
            # Use natural metric name "train"; xKey="step" and yKeys=["loss"]
            # are column names within the metric data — not metric names.
            e.metrics("train").log(step=step, loss=loss)
        exp_id = e._experiment_id
    # Buffer is flushed synchronously when the context exits, so data is in DB now.
    print(f"  → ID: {exp_id}")
    return exp_id


def call_aggregate_series(experiment_ids: list, token: str, **kwargs) -> dict:
    body = {
        "experimentIds": experiment_ids,
        "xKey": "step",
        "yKeys": ["loss"],
        **kwargs,
    }
    resp = httpx.post(
        f"{SERVER_URL}/api/experiments/aggregate/series",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    )
    if not resp.is_success:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return {}
    return resp.json()


def check(label: str, passed: bool, detail: str = ""):
    status = "✓" if passed else "✗"
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n=== Aggregate Series E2E ===\n")

    # Setup
    token = get_token_storage().load("ml-dash-token")
    if not token:
        print("ERROR: Not logged in. Run `ml-dash login` first.")
        sys.exit(1)

    print(f"Namespace: {NAMESPACE}")
    print(f"Project:   {PROJECT}\n")

    # --- Step 1: Create experiments ---
    print("Step 1: Creating experiments...")
    ts = int(time.time())
    configs = [
        ("fast-convergence", 50_000, 0.05, 10),
        ("slow-convergence", 60_000, 0.05, 20),
        ("noisy",            70_000, 0.20, 30),
    ]
    experiment_ids = []
    for suffix, n_steps, noise, seed in configs:
        prefix = f"{NAMESPACE}/{PROJECT}/run-{ts}/{suffix}"
        exp_id = create_experiment(prefix, n_steps, noise, seed)
        experiment_ids.append(exp_id)

    print(f"\nCreated {len(experiment_ids)} experiments: {experiment_ids}\n")

    # --- Step 1b: Verify S3 chunks were created ---
    # Chunking is triggered asynchronously; poll until all experiments have >=1 chunk (max 90s).
    print("Step 1b: Polling for S3 chunks (threshold=50k, max 90s)...")
    deadline = time.time() + 90
    while time.time() < deadline:
        all_chunked = all(
            httpx.get(
                f"{SERVER_URL}/api/experiments/{exp_id}/metrics/train/stats",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
            ).json().get("totalChunks", 0) >= 1
            for exp_id in experiment_ids
        )
        if all_chunked:
            break
        time.sleep(5)
    failures = 0
    for exp_id in experiment_ids:
        resp = httpx.get(
            f"{SERVER_URL}/api/experiments/{exp_id}/metrics/train/stats",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        if resp.is_success:
            stats = resp.json()
            total = int(stats.get("totalDataPoints", 0))
            chunked = int(stats.get("chunkedDataPoints", 0))
            buffered = int(stats.get("bufferedDataPoints", 0))
            chunks = int(stats.get("totalChunks", 0))
            print(f"  exp {exp_id[-6:]}: total={total}, chunked={chunked}, buffered={buffered}, chunks={chunks}")
            failures += not check(f"  exp {exp_id[-6:]} has >=1 S3 chunk", chunks >= 1, f"totalChunks={chunks}")
        else:
            print(f"  ERROR fetching stats for {exp_id}: {resp.status_code}")
            failures += 1

    # --- Step 2: Call aggregate series ---
    print("Step 2: Calling aggregate series (interpolate, bins=50)...")
    data = call_aggregate_series(
        experiment_ids, token,
        alignmentMethod="interpolate",
        bins=50,
        statistics=["MEAN", "STD", "PERCENTILE_25", "PERCENTILE_75"],
    )

    if not data:
        print("\nFAILED — no response data")
        sys.exit(1)

    # --- Step 3: Validate and print results ---
    print("\nStep 3: Validating response...\n")

    # The ml-dash-server returns a ComputeResponse wrapper: {status, result: {series, metadata}, metadata: ComputeMetadata}
    result_body = data.get("result", {})
    series = result_body.get("series", [])
    meta = result_body.get("metadata", {})

    failures += not check("series present", len(series) > 0, f"got {len(series)}")
    failures += not check("metadata present", bool(meta))

    if series:
        s = series[0]
        d = s.get("data", {})
        x = d.get("x", [])
        mean = d.get("mean", [])
        std = d.get("std", [])
        p25 = d.get("percentile_25", [])
        p75 = d.get("percentile_75", [])
        alignment = s.get("alignment", {})

        failures += not check("metricName == 'loss'", s.get("metricName") == "loss", repr(s.get("metricName")))
        failures += not check("xKey == 'step'", s.get("xKey") == "step")
        failures += not check("x non-empty", len(x) > 0, f"{len(x)} points")
        failures += not check("x <= bins (50)", len(x) <= 50, f"{len(x)} points")
        failures += not check("x starts near 0", x[0] <= 1, f"x[0]={x[0]}")
        failures += not check("x ends >= 10000", x[-1] >= 10_000, f"x[-1]={x[-1]:.1f}")
        failures += not check("mean same len as x", len(mean) == len(x))
        failures += not check("std same len as x", len(std) == len(x))
        failures += not check("p25 same len as x", len(p25) == len(x))
        failures += not check("p75 same len as x", len(p75) == len(x))
        failures += not check("mean values finite", all(math.isfinite(v) for v in mean))
        failures += not check("mean values in range (0, 10)", all(0 < v < 10 for v in mean))
        failures += not check("alignment method == interpolate", alignment.get("method") == "interpolate")
        failures += not check("alignedSteps > 0", alignment.get("alignedSteps", 0) > 0, str(alignment.get("alignedSteps")))

        print(f"\n  x range: [{x[0]:.2f}, {x[-1]:.2f}] ({len(x)} points)")
        print(f"  mean[0]={mean[0]:.4f}  mean[-1]={mean[-1]:.4f}")
        print(f"  std[0]={std[0]:.4f}  alignedSteps={alignment.get('alignedSteps')}  interpolated={alignment.get('interpolatedPoints')}")

    failures += not check("experimentsProcessed == 3", meta.get("experimentsProcessed") == 3, str(meta.get("experimentsProcessed")))
    failures += not check("totalDataPoints > 0", meta.get("totalDataPoints", 0) > 0, str(meta.get("totalDataPoints")))

    # --- Step 4: bins=20 limit test ---
    print("\nStep 4: Testing bins=20 limit...")
    data2 = call_aggregate_series(experiment_ids, token, bins=20)
    if data2:
        x2 = data2["result"]["series"][0]["data"]["x"]
        failures += not check("bins=20 → len(x) <= 20", len(x2) <= 20, f"got {len(x2)}")

    # --- Step 5: common_steps alignment ---
    print("\nStep 5: Testing common_steps alignment...")
    data3 = call_aggregate_series(experiment_ids, token, alignmentMethod="common_steps")
    if data3:
        method = data3["result"]["series"][0]["alignment"]["method"]
        failures += not check("common_steps alignment returned", method == "common_steps", repr(method))

    # --- Step 6: Error cases ---
    print("\nStep 6: Testing error cases...")
    r = call_aggregate_series([], token)
    failures += not check("empty experimentIds → 400", not r)

    resp_bad = httpx.post(
        f"{SERVER_URL}/api/experiments/aggregate/series",
        json={"experimentIds": ["bad-id-1"], "xKey": "step", "yKeys": ["loss"]},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    failures += not check("invalid IDs → 404", resp_bad.status_code == 404, str(resp_bad.status_code))

    resp_no_xkey = httpx.post(
        f"{SERVER_URL}/api/experiments/aggregate/series",
        json={"experimentIds": experiment_ids, "yKeys": ["loss"]},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    failures += not check("missing xKey → 400", resp_no_xkey.status_code == 400, str(resp_no_xkey.status_code))

    # --- Summary ---
    print(f"\n{'='*40}")
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK(S) FAILED")
    print('='*40 + "\n")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
