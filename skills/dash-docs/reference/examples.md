# Examples

Complete scripts you can copy and run. Each one writes to `.dash/` in local
mode, so you don't need an account. Add `dash_url="https://api.dash.ml"` to
the `Experiment` to send the run to dash.ml as well.

- [Simple Training](reference/examples-simple-training.md) — A minimal loop with parameters, per-epoch metrics, and logs.
- [PyTorch MNIST](reference/examples-pytorch-mnist.md) — A full PyTorch run, from config to the best and final checkpoints.
- [Hyperparameter Search](reference/examples-hyperparameter-search.md) — A grid sweep, one experiment per configuration under one project.
- [Comparing Experiments](reference/examples-experiment-comparison.md) — Train several architectures and rank their results.
- [Logging & Debugging](reference/examples-logging-debugging.md) — Use log levels and metadata to find out what went wrong in a run.

## Tutorial scripts

The ml-dash repository has a numbered set of runnable tutorials in
[`examples/tutorials/`](https://github.com/fortyfive-labs/ml-dash/tree/main/examples/tutorials):

```bash
git clone https://github.com/fortyfive-labs/ml-dash
cd ml-dash/examples/tutorials

python 01_basic_experiment.py
python 02_logging_example.py
python 03_parameters_example.py
python 04_metrics_example.py
python 05_files_example.py
python 06_complete_training.py
python 07_project_root_example.py
python three_usage_styles.py

# Robotics: MuJoCo pose tracking
python robotics/mujoco_tracking.py
```

After a run, browse what it wrote under `.dash/`, or upload it with
`ml-dash upload` (see the [CLI Reference](reference/reference-cli.md#ml-dash-upload)).
