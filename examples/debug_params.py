#!/usr/bin/env python3
"""Debug script to see what server returns for parameters"""

import sys
import requests
sys.path.insert(0, '/Users/57block/PycharmProjects/vuer-dashboard/ml-logger/src')

from ml_dash import Experiment

# Create experiment with remote backend
exp = Experiment(
    namespace="test",
    workspace="debug",
    prefix="params-debug",
    remote="http://localhost:4000"
)

with exp.run():
    # Set parameters
    exp.params.set(model="resnet50", lr=0.001)
    exp.params.extend(epochs=100)
    exp.params.update("lr", 0.0001)

    # Check what server has
    url = f"{exp.backend.server_url}/api/v1/parameters"
    params = {"runId": exp.backend.run_id}
    response = requests.get(url, params=params)
    result = response.json()

    print("Server response:")
    print("=" * 60)
    import json
    print(json.dumps(result, indent=2))
