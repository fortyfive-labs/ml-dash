"""
07 - Experiment Modes: Local, Remote, and Hybrid.

Covers:
  - Local-only mode      (default, writes to .dash/)
  - Custom local path    (dash_root=)
  - Remote-only mode     (dash_url=, dash_root=None)
  - Hybrid mode          (dash_url= + dash_root=  — writes to both)
  - Default remote       (dash_url=True uses https://api.dash.ml)
  - userinfo singleton   (query authenticated user from server)
  - Tags, readme, metadata on the experiment
  - exp.id and exp.data  (experiment ID and server response)
  - Environment variables that control behaviour
"""

from ml_dash import Experiment

# ---------------------------------------------------------------------------
# 1. Local-only mode  (default)
#    Writes data to ".dash/" directory relative to CWD.
# ---------------------------------------------------------------------------
with Experiment(prefix="alice/nlp/local-exp").run as exp:
    exp.params.set(lr=1e-3)
    exp.metrics("train").log(loss=0.5)


# ---------------------------------------------------------------------------
# 2. Custom local storage directory
# ---------------------------------------------------------------------------
with Experiment(
    prefix="alice/nlp/local-custom-path",
    dash_root="/tmp/my-experiments",
).run as exp:
    exp.metrics("train").log(loss=0.4)


# ---------------------------------------------------------------------------
# 3. Remote-only mode
#    dash_url=  points to your ml-dash server.
#    dash_root=None  disables local storage.
#    Run `ml-dash login` first to store your token.
# ---------------------------------------------------------------------------
# with Experiment(
#     prefix="alice/nlp/remote-only",
#     dash_url="http://localhost:3000",
#     dash_root=None,
# ).run as exp:
#     exp.params.set(lr=1e-3)
#     exp.metrics("train").log(loss=0.5)
#     exp.flush()   # send buffered data to the server


# ---------------------------------------------------------------------------
# 4. Hybrid mode  (local backup + remote sync)
#    Writes to ".dash/" AND sends to the remote server.
# ---------------------------------------------------------------------------
# with Experiment(
#     prefix="alice/nlp/hybrid",
#     dash_url="http://localhost:3000",
#     dash_root=".dash",
# ).run as exp:
#     exp.metrics("train").log(loss=0.5)
#     exp.flush()


# ---------------------------------------------------------------------------
# 5. Default remote server  (dash_url=True → https://api.dash.ml)
# ---------------------------------------------------------------------------
# with Experiment(
#     prefix="alice/nlp/production-exp",
#     dash_url=True,
#     dash_root=None,
# ).run as exp:
#     exp.metrics("train").log(loss=0.3)
#     exp.flush()


# ---------------------------------------------------------------------------
# 6. Query authenticated user from server
# ---------------------------------------------------------------------------
# from ml_dash import userinfo
# print("Logged-in user:", userinfo.username)
# print("Email:", userinfo.email)
#
# with Experiment(
#     prefix=f"{userinfo.username}/my-project/exp-1",
#     dash_url=True,
#     dash_root=None,
# ).run as exp:
#     exp.metrics("train").log(loss=0.3)


# ---------------------------------------------------------------------------
# 7. Experiment-level metadata: tags, readme, metadata
# ---------------------------------------------------------------------------
with Experiment(
    prefix="alice/nlp/annotated-exp",
    dash_root="/tmp/ml-dash-demo",
    tags=["baseline", "bert", "squad"],
    readme="BERT fine-tuning on SQuAD 2.0 with AdamW.",
    metadata={"gpu": "A100", "framework": "pytorch", "version": "2.1"},
).run as exp:
    exp.params.set(lr=3e-5)
    exp.metrics("train").log(loss=0.4, step=0)


# ---------------------------------------------------------------------------
# 8. Accessing experiment ID and server response data
# ---------------------------------------------------------------------------
exp2 = Experiment(prefix="alice/nlp/id-demo", dash_root="/tmp/ml-dash-demo")
exp2.run.start()
# exp2.id and exp2.data are populated only in remote mode (dash_url= provided)
# In local-only mode they are None.
print("Experiment ID (local mode):", exp2.id)    # None in local mode
print("Experiment data (local mode):", exp2.data) # None in local mode
exp2.run.complete()


# ---------------------------------------------------------------------------
# 9. Environment variables
#
#   ML_DASH_PREFIX    – default prefix (equivalent to passing prefix=)
#   ML_DASH_ROOT      – local storage root (equivalent to dash_root=)
#   ML_DASH_URL       – remote server URL (equivalent to dash_url=)
#   ML_DASH_USER      – default user/owner
#   ML_DASH_BUFFER_ENABLED  – "true"/"false", controls background buffering
#
#   Example:
#       export ML_DASH_PREFIX="alice/nlp/my-run"
#       export ML_DASH_URL="http://localhost:3000"
#       python train.py    # Experiment() auto-reads these vars
# ---------------------------------------------------------------------------

print("Done.")
