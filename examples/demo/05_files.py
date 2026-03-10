"""
05 - Files: Uploading, saving, listing, downloading, and deleting files.

Covers:
  - files("dir").upload()     – upload an existing file from disk
  - files.save_text()         – write a string
  - files.save_json()         – write any JSON-serialisable object
  - files.save_blob()         – write raw bytes
  - files.save_image()        – save a PIL Image or numpy array
  - files.save_fig()          – save a matplotlib figure
  - files.save_torch()        – save a torch.Tensor or state dict
  - files.save_pkl()          – save any pickle-able object
  - files.save_video()        – save a video (list of frames)
  - files.list()              – list files, with optional glob pattern
  - files.download()          – download to disk
  - files("path").exists()    – check existence
  - files("path").read_text() – read text content back
  - files.delete()            – delete by path or pattern
"""

import json
import os
import tempfile
import numpy as np
from ml_dash import Experiment

DASH_ROOT = "/tmp/ml-dash-demo"

exp = Experiment(prefix="alice/vision/files-demo", dash_root=DASH_ROOT)
exp.run.start()

# ---------------------------------------------------------------------------
# 1. Upload an existing file from disk
# ---------------------------------------------------------------------------
# Write to a non-temp path so the buffer won't auto-delete it
model_path = "/tmp/ml-dash-demo-fake-model.pt"
with open(model_path, "wb") as f:
    f.write(b"fake-model-weights")

exp.files("models").upload(model_path, to="checkpoint_epoch10.pt")
exp.files("models").upload(
    model_path,
    to="best_model.pt",
    description="Best validation checkpoint",
    tags=["best", "checkpoint"],
    metadata={"epoch": 10, "val_acc": 0.847},
)
exp.flush()          # ensure all buffered file writes finish
os.unlink(model_path)

# ---------------------------------------------------------------------------
# 2. Save text content directly
# ---------------------------------------------------------------------------
exp.files.save_text("epoch,loss,acc\n1,0.8,0.62\n2,0.6,0.74\n", to="results/metrics.csv")
exp.files.save_text("<html><body>Training report</body></html>", to="report.html")

# ---------------------------------------------------------------------------
# 3. Save JSON
# ---------------------------------------------------------------------------
config = {"model": "resnet50", "lr": 3e-4, "epochs": 100}
exp.files.save_json(config, to="config.json")
exp.files.save_json({"train_acc": 0.91, "val_acc": 0.85}, to="results/final_scores.json")

# ---------------------------------------------------------------------------
# 4. Save raw bytes
# ---------------------------------------------------------------------------
exp.files.save_blob(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, to="assets/placeholder.png")

# ---------------------------------------------------------------------------
# 5. Save a numpy array / PIL image
# ---------------------------------------------------------------------------
image_array = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
exp.files("images").save_image(image_array, to="sample_output.png")

# ---------------------------------------------------------------------------
# 6. Save a matplotlib figure
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0.9, 0.7, 0.5, 0.3])
    ax.set_title("Training Loss")
    exp.files("plots").save_fig(fig, to="loss_curve.png")
    plt.close(fig)
except ImportError:
    pass  # matplotlib is optional

# ---------------------------------------------------------------------------
# 7. Save a PyTorch tensor / state dict
# ---------------------------------------------------------------------------
try:
    import torch
    state_dict = {"weight": torch.randn(10, 10), "bias": torch.zeros(10)}
    exp.files("models").save_torch(state_dict, to="model_state.pt")
except ImportError:
    pass  # torch is optional

# ---------------------------------------------------------------------------
# 8. Save with pickle
# ---------------------------------------------------------------------------
sklearn_model = {"type": "fake-sklearn-model", "coef": [0.1, 0.2, 0.3]}
exp.files("models").save_pkl(sklearn_model, to="sklearn_model.pkl")

# ---------------------------------------------------------------------------
# 9. Save a video (list of numpy frames)
# ---------------------------------------------------------------------------
frames = [(np.random.rand(64, 64, 3) * 255).astype(np.uint8) for _ in range(10)]
exp.files("videos").save_video(frames, to="rollout.mp4", fps=10)

# ---------------------------------------------------------------------------
# 10. List files
# ---------------------------------------------------------------------------
all_files = exp.files.list()
print(f"Total files: {len(all_files)}")

model_files = exp.files.list("models/*")
print("Model files:", [f["path"] for f in model_files])

# ---------------------------------------------------------------------------
# 11. Check existence and read text
# ---------------------------------------------------------------------------
if exp.files("results/metrics.csv").exists():
    content = exp.files("results/metrics.csv").read_text()
    print("metrics.csv:\n", content)

# ---------------------------------------------------------------------------
# 12. Download a file
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    dest = exp.files("config.json").download(to=os.path.join(tmpdir, "config.json"))
    print("Downloaded to:", dest)

# ---------------------------------------------------------------------------
# 13. Delete files
# ---------------------------------------------------------------------------
exp.files.delete("assets/placeholder.png")      # single file by path
exp.files.delete("images/*")                    # all files matching a pattern

exp.run.complete()
