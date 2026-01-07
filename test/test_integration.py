"""Comprehensive integration tests for complete workflows in both local and remote modes."""

import getpass
import json
import random

import pytest


class TestCompleteWorkflows:
  """Integration tests for complete ML experiment workflows."""

  def test_complete_ml_workflow_local(self, local_experiment, tmp_proj, sample_files):
    """Test complete ML experiment workflow in local mode."""
    with local_experiment(
      prefix="experiments/2024/ml-experiment",
      project="experiments",
      description="Complete ML training experiment",
      tags=["ml", "training", "test"],
    ).run as experiment:
      # 1. Set hyperparameters
      experiment.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        model="resnet50",
        optimizer="adam",
      )

      experiment.log("Experiment started", level="info")

      # 2. Metric training metrics over epochs
      for epoch in range(10):
        train_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
        val_loss = 1.2 / (epoch + 1) + random.uniform(-0.05, 0.05)
        accuracy = min(0.95, 0.5 + epoch * 0.05)

        experiment.metrics("train").log(loss=train_loss)
        experiment.metrics("eval").log(loss=val_loss)
        experiment.metrics("eval").log(loss=accuracy)

        experiment.log(
          f"Epoch {epoch + 1}/10 complete",
          level="info",
          metadata={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
          },
        )

      # 3. Upload artifacts
      experiment.files("models").upload(
        sample_files["model"], tags=["final", "best"], description="Final trained model"
      )

      experiment.files("configs").upload(
        sample_files["config"], tags=["config"], description="Training configuration"
      )

      experiment.log("Experiment completed successfully", level="info")

    # Verify everything was created
    # Storage structure: root_path / owner / project / prefix
    owner = getpass.getuser()
    experiment_dir = tmp_proj / owner / "experiments/experiments/2024/ml-experiment"
    assert (experiment_dir / "experiment.json").exists()
    assert (experiment_dir / "parameters.json").exists()
    assert (experiment_dir / "logs/logs.jsonl").exists()
    assert (experiment_dir / "metrics/train/data.jsonl").exists()
    assert (experiment_dir / "metrics/eval/data.jsonl").exists()
    assert (experiment_dir / "metrics/eval/data.jsonl").exists()

  @pytest.mark.remote
  def test_complete_ml_workflow_remote(self, remote_experiment, sample_files):
    """Test complete ML workflow in remote mode."""
    with remote_experiment(
      name="ml-experiment-remote",
      project="experiments",
      description="Remote ML training experiment",
      tags=["ml", "remote"],
    ).run as experiment:
      experiment.params.set(
        learning_rate=0.001, batch_size=64, epochs=5, model="transformer"
      )

      experiment.log("Remote experiment started", level="info")

      for epoch in range(5):
        loss = 1.0 / (epoch + 1)
        experiment.metrics("train").log(loss=loss, epoch=epoch)
        experiment.log(f"Epoch {epoch + 1}/5", metadata={"loss": loss})

      experiment.files("models").upload(sample_files["model"])
      experiment.log("Remote experiment completed", level="info")


class TestHyperparameterSearch:
  """Integration tests for hyperparameter search workflows."""

  def test_grid_search_local(self, local_experiment, tmp_proj):
    """Test grid search hyperparameter optimization."""
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [16, 32, 64]

    for lr in learning_rates:
      for bs in batch_sizes:
        experiment_name = f"grid-lr{lr}-bs{bs}".replace(".", "_")

        with local_experiment(
          name=experiment_name,
          project="hyperparam-search",
          description=f"Grid search: lr={lr}, bs={bs}",
          tags=["grid-search", f"lr-{lr}", f"bs-{bs}"],
        ).run as experiment:
          # Metric hyperparameters
          experiment.params.set(learning_rate=lr, batch_size=bs, epochs=10)

          # Simulate training
          final_acc = 0.5 + random.random() * 0.4
          final_loss = 0.5 - random.random() * 0.3

          experiment.metrics("eval").log(loss=final_acc, epoch=9)
          experiment.metrics("train").log(loss=final_loss, epoch=9)

          experiment.log(f"Grid search run complete: acc={final_acc:.4f}")

    # Verify all experiments were created
    project_dir = tmp_proj / getpass.getuser() / "hyperparam-search"
    experiments = [d for d in project_dir.iterdir() if d.is_dir()]
    assert len(experiments) == 9  # 3 LRs × 3 batch sizes

  @pytest.mark.remote
  def test_random_search_remote(self, remote_experiment):
    """Test random search in remote mode."""
    for run in range(5):
      lr = random.choice([0.1, 0.01, 0.001, 0.0001])
      bs = random.choice([16, 32, 64])

      with remote_experiment(
        name=f"random-search-run-{run}", project="random-search", tags=["random-search"]
      ).run as experiment:
        experiment.params.set(learning_rate=lr, batch_size=bs)
        acc = 0.6 + random.random() * 0.3
        experiment.metrics("eval").log(loss=acc, run=run)
        experiment.log(f"Run {run} complete")


class TestIterativeExperimentation:
  """Integration tests for iterative experimentation."""

  def test_iterative_improvements_local(self, local_experiment, tmp_proj):
    """Test iterative model improvements."""
    experiments = [
      {
        "name": "baseline",
        "description": "Baseline model",
        "params": {"lr": 0.01, "layers": 3},
        "expected_acc": 0.75,
      },
      {
        "name": "deeper",
        "description": "Deeper network",
        "params": {"lr": 0.01, "layers": 5},
        "expected_acc": 0.82,
      },
      {
        "name": "lower-lr",
        "description": "Lower learning rate",
        "params": {"lr": 0.001, "layers": 5},
        "expected_acc": 0.85,
      },
      {
        "name": "best",
        "description": "Best configuration",
        "params": {"lr": 0.001, "layers": 7},
        "expected_acc": 0.90,
      },
    ]

    for exp in experiments:
      with local_experiment(
        name=f"exp-{exp['name']}",
        project="iterative",
        description=exp["description"],
        tags=["iterative", exp["name"]],
      ).run as experiment:
        experiment.params.set(**exp["params"])
        experiment.metrics("val_accuracy").log(loss=exp["expected_acc"], step=0)
        experiment.log(f"{exp['name']} experiment complete", level="info")

    # Verify progression
    project_dir = tmp_proj / getpass.getuser() / "iterative"
    assert (project_dir / "exp-baseline").exists()
    assert (project_dir / "exp-deeper").exists()
    assert (project_dir / "exp-lower-lr").exists()
    assert (project_dir / "exp-best").exists()


class TestMultiExperimentPipeline:
  """Integration tests for multi-experiment pipelines."""

  def test_ml_pipeline_local(self, local_experiment, tmp_proj, sample_files):
    """Test complete ML pipeline with multiple stages."""
    # Stage 1: Data preprocessing
    with local_experiment(
      prefix="pipeline/stage-1/01-preprocessing",
      project="pipeline",
      tags=["pipeline", "preprocessing"],
    ).run as experiment:
      experiment.log("Starting data preprocessing", level="info")
      experiment.params.set(
        data_source="raw_data.csv",
        preprocessing_steps=["normalize", "augment", "split"],
      )
      experiment.metrics("samples_processed").log(loss=10000, step=0)
      experiment.files("data").upload(sample_files["results"])
      experiment.log("Preprocessing complete", level="info")

    # Stage 2: Training
    with local_experiment(
      prefix="pipeline/stage-2/02-training",
      project="pipeline",
      tags=["pipeline", "training"],
    ).run as experiment:
      experiment.log("Starting model training", level="info")
      experiment.params.set(model="resnet50", epochs=10, batch_size=32)
      for i in range(10):
        experiment.metrics("train").log(loss=1.0 / (i + 1), epoch=i)
      experiment.files("models").upload(sample_files["model"])
      experiment.log("Training complete", level="info")

    # Stage 3: Evaluation
    with local_experiment(
      prefix="pipeline/stage-3/03-evaluation",
      project="pipeline",
      tags=["pipeline", "evaluation"],
    ).run as experiment:
      experiment.log("Starting model evaluation", level="info")
      experiment.params.set(test_set="test.csv")
      experiment.metrics("test_accuracy").log(loss=0.95, step=0)
      experiment.metrics("test_loss").log(loss=0.15, step=0)
      experiment.log("Evaluation complete", level="info")

    # Verify all stages
    # Storage structure: root_path / owner / project / prefix
    owner = getpass.getuser()
    assert (tmp_proj / owner / "pipeline/pipeline/stage-1/01-preprocessing").exists()
    assert (tmp_proj / owner / "pipeline/pipeline/stage-2/02-training").exists()
    assert (tmp_proj / owner / "pipeline/pipeline/stage-3/03-evaluation").exists()

  @pytest.mark.remote
  def test_pipeline_remote(self, remote_experiment, sample_files):
    """Test pipeline in remote mode."""
    stages = ["preprocessing", "training", "evaluation"]

    for i, stage in enumerate(stages):
      with remote_experiment(
        name=f"stage-{i + 1}-{stage}",
        project="pipeline-remote",
        tags=["pipeline", stage],
      ).run as experiment:
        experiment.log(f"Starting {stage}")
        experiment.params.set(stage=stage, order=i + 1)
        experiment.metrics("progress").log(loss=(i + 1) / len(stages) * 100, step=i)
        experiment.log(f"{stage} complete")


class TestDebuggingWorkflow:
  """Integration tests for debugging workflows."""

  def test_experiment_local(self, local_experiment, tmp_proj):
    """Test comprehensive debugging workflow."""
    with local_experiment(
      name="debug-training",
      project="debugging",
      description="Training with debug logging",
      tags=["debug", "verbose"],
    ).run as experiment:
      experiment.params.set(learning_rate=0.001, batch_size=32, debug_mode=True)

      experiment.log("Debug experiment started", level="debug")
      experiment.log("Initializing model", level="debug")

      for epoch in range(5):
        experiment.log(f"Starting epoch {epoch + 1}", level="debug")

        loss = 1.0 / (epoch + 1)

        if epoch == 2:
          experiment.log(
            "Learning rate may be too high",
            level="warn",
            metadata={"current_lr": 0.001, "suggested_lr": 0.0001},
          )

        if random.random() < 0.5:
          experiment.log(
            "Gradient clipping applied",
            level="warn",
            metadata={"gradient_norm": 15.5, "max_norm": 10.0},
          )

        experiment.metrics("train").log(loss=loss, epoch=epoch)
        experiment.log(
          f"Epoch {epoch + 1} complete", level="info", metadata={"loss": loss}
        )

      experiment.log("Debug experiment complete", level="info")

    # Verify comprehensive logs
    logs_file = (
      tmp_proj / getpass.getuser() / "debugging/debug-training/logs/logs.jsonl"
    )
    with open(logs_file) as f:
      logs = [json.loads(line) for line in f]

    assert len(logs) >= 10  # Multiple log levels


class TestAllFeaturesCombined:
  """Integration test using all features together."""

  def test_kitchen_sink_local(self, local_experiment, tmp_proj, sample_files):
    """Test experiment using every available feature."""
    with local_experiment(
      prefix="tests/comprehensive/kitchen-sink",
      project="full-test",
      description="Test of all features combined",
      tags=["test", "comprehensive", "all-features"],
    ).run as experiment:
      # Parameters (simple and nested)
      experiment.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=5,
        **{
          "model": {"architecture": "resnet50", "pretrained": True, "layers": 50},
          "optimizer": {"type": "adam", "beta1": 0.9, "beta2": 0.999},
        },
      )

      # Logging at multiple levels
      experiment.log("Starting comprehensive test", level="info")
      experiment.log("Debug info: all systems go", level="debug")

      # Metric multiple metrics
      for i in range(5):
        experiment.metrics("train").log(loss=0.5 - i * 0.1, epoch=i)
        experiment.metrics("eval").log(loss=0.6 - i * 0.1, epoch=i)
        experiment.metrics("eval").log(loss=0.7 + i * 0.05, epoch=i)
        experiment.metrics("train").log(lr=0.001 * (0.9**i), epoch=i)

        experiment.log(f"Epoch {i + 1} metrics metriced", level="info")

      # Upload multiple files
      experiment.files("models").upload(sample_files["model"])
      experiment.files("configs").upload(sample_files["config"])
      experiment.files("results").upload(sample_files["results"])

      # Warnings and errors
      experiment.log("Simulated warning", level="warn")
      experiment.log("Test error handling", level="error", metadata={"error": "test"})

      experiment.log("Comprehensive test complete", level="info")

    # Verify everything exists
    # Storage structure: root_path / owner / project / prefix
    owner = getpass.getuser()
    experiment_dir = tmp_proj / owner / "full-test/tests/comprehensive/kitchen-sink"
    assert (experiment_dir / "experiment.json").exists()
    assert (experiment_dir / "parameters.json").exists()
    assert (experiment_dir / "logs/logs.jsonl").exists()
    assert (experiment_dir / "metrics/train/data.jsonl").exists()
    assert (experiment_dir / "metrics/eval/data.jsonl").exists()
    assert (experiment_dir / "metrics/eval/data.jsonl").exists()
    assert (experiment_dir / "metrics/train/data.jsonl").exists()

    # Verify parameters
    with open(experiment_dir / "parameters.json") as f:
      params = json.load(f)["data"]
      assert params["learning_rate"] == 0.001
      assert params["model.architecture"] == "resnet50"
      assert params["optimizer.type"] == "adam"

  @pytest.mark.remote
  def test_kitchen_sink_remote(self, remote_experiment, sample_files):
    """Test all features combined in remote mode."""
    with remote_experiment(
      name="kitchen-sink-remote",
      project="full-test-remote",
      description="Remote test of all features",
      tags=["test", "remote", "comprehensive"],
    ).run as experiment:
      # Parameters
      experiment.params.set(
        learning_rate=0.001,
        batch_size=64,
        **{"model": {"type": "transformer", "layers": 12}},
      )

      # Logging
      experiment.log("Starting remote comprehensive test", level="info")

      # Metrics
      for i in range(3):
        experiment.metrics("train").log(loss=0.5 - i * 0.1, epoch=i)
        experiment.metrics("eval").log(loss=0.8 + i * 0.05, epoch=i)

      # Files
      experiment.files("models").upload(sample_files["model"])
      experiment.files("configs").upload(sample_files["config"])

      experiment.log("Remote comprehensive test complete", level="info")


class TestRealWorldScenarios:
  """Integration tests for real-world scenarios."""

  def test_failed_experiment_recovery_local(self, local_experiment, tmp_proj):
    """Test recovering from failed experiment."""
    # First attempt (fails)
    try:
      with local_experiment(name="recovery-test", project="recovery").run as experiment:
        experiment.params.set(attempt=1)
        experiment.log("Starting experiment attempt 1")
        experiment.metrics("train").log(loss=0.5, epoch=0)
        raise RuntimeError("Simulated failure")
    except RuntimeError:
      pass

    # Recovery attempt
    with local_experiment(name="recovery-test", project="recovery").run as experiment:
      experiment.params.set(attempt=2, recovered=True)
      experiment.log("Recovered and restarting")
      experiment.metrics("train").log(loss=0.4, epoch=1)
      experiment.metrics("train").log(loss=0.3, epoch=2)
      experiment.log("Recovery successful")

    # Verify both attempts are recorded
    experiment_dir = tmp_proj / getpass.getuser() / "recovery/recovery-test"
    assert experiment_dir.exists()

  def test_comparison_experiments_local(self, local_experiment, tmp_proj):
    """Test running comparison experiments."""
    models = ["resnet18", "resnet50", "vit-base"]

    for model_name in models:
      with local_experiment(
        name=f"comparison-{model_name}",
        project="comparisons",
        tags=["comparison", model_name],
      ).run as experiment:
        experiment.params.set(model=model_name, epochs=10)

        # Simulate different performance
        base_acc = {"resnet18": 0.75, "resnet50": 0.85, "vit-base": 0.90}
        final_acc = base_acc[model_name] + random.uniform(-0.02, 0.02)

        experiment.metrics("eval").log(loss=final_acc, epoch=9)
        experiment.log(
          f"{model_name} training complete", metadata={"final_acc": final_acc}
        )

    # Verify all comparison runs
    project_dir = tmp_proj / getpass.getuser() / "comparisons"
    assert len([d for d in project_dir.iterdir() if d.is_dir()]) == 3

  @pytest.mark.slow
  def test_long_running_experiment_local(self, local_experiment):
    """Test long-running experiment with many data points."""
    with local_experiment(name="long-run", project="longtest").run as experiment:
      experiment.params.set(total_steps=1000)

      # Metric many data points
      for step in range(100):
        experiment.metrics("train").log(loss=1.0 / (step + 1), step=step)

        if step % 10 == 0:
          experiment.log(f"Progress: {step}/100 steps", metadata={"step": step})

      experiment.log("Long-running experiment complete")


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
