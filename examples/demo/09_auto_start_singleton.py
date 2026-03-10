"""
09 - Auto-start singleton (dxp): Remote experiment with minimal setup.

`dxp` is a pre-configured Experiment singleton in remote mode.
It reads the authenticated user from the server and connects to
https://api.dash.ml automatically.

Requirements:
  - Run `ml-dash login` once to store your token.
  - A network connection to the ml-dash server.

Covers:
  - Importing dxp
  - Using dxp as a context manager
  - Using dxp with manual start / complete
  - RUN global config object
"""

# ---------------------------------------------------------------------------
# NOTE: This example requires authentication (`ml-dash login`).
#       Uncomment the blocks below after logging in.
# ---------------------------------------------------------------------------

# from ml_dash.auto_start import dxp
#
# # 1. Context manager (recommended)
# with dxp.run:
#     dxp.log("Hello from dxp!")
#     dxp.params.set(lr=1e-3, epochs=10)
#     for epoch in range(10):
#         dxp.metrics("train").log(loss=1.0 / (epoch + 1), epoch=epoch)
#     dxp.flush()
#
#
# # 2. Manual start / complete
# dxp.run.start()
# dxp.params.set(model="resnet50")
# dxp.metrics("train").log(loss=0.5, step=0)
# dxp.flush()
# dxp.run.complete()


# ---------------------------------------------------------------------------
# RUN — global experiment config object
#   Populated via params-proto with environment-variable overrides.
# ---------------------------------------------------------------------------
from ml_dash.run import RUN

# Inspect RUN defaults
run = RUN()   # create an instance to access properties correctly
print("user       :", run.user)
print("project    :", run.project)
print("prefix     :", run.prefix)
print("api_url    :", run.api_url)
print("now        :", run.now)
print("date       :", run.date)
print("datetime   :", run.datetime_str)

# Override via environment variables before importing ml_dash:
#   export ML_DASH_PREFIX="alice/nlp/my-run"
#   export ML_DASH_URL="http://localhost:3000"
#   export ML_DASH_USER="alice"
#   python train.py
