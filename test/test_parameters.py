"""Comprehensive tests for parameter metricing in both local and remote modes."""

import getpass
import json

import pytest


class TestBasicParameters:
  """Tests for basic parameter operations."""

  def test_simple_parameters_local(self, local_experiment, tmp_proj, sample_data):
    """Test setting simple parameters in local mode."""
    with local_experiment(name="params-test", project="test").run as experiment:
      experiment.params.set(**sample_data["simple_params"])

    params_file = tmp_proj / getpass.getuser() / "test/params-test/parameters.json"
    assert params_file.exists()

    with open(params_file) as f:
      params_data = json.load(f)

    params = params_data["data"]
    assert params["learning_rate"] == 0.001
    assert params["batch_size"] == 32
    assert params["epochs"] == 100

  @pytest.mark.remote
  def test_simple_parameters_remote(self, remote_experiment, sample_data):
    """Test setting simple parameters in remote mode."""
    with remote_experiment(name="params-test-remote", project="test").run as experiment:
      experiment.params.set(**sample_data["simple_params"])

  def test_parameter_types_local(self, local_experiment, tmp_proj):
    """Test different parameter data types."""
    with local_experiment(name="param-types", project="test").run as experiment:
      experiment.params.set(
        int_param=42,
        float_param=3.14159,
        str_param="hello world",
        bool_param_true=True,
        bool_param_false=False,
        none_param=None,
        list_param=[1, 2, 3, 4, 5],
        dict_param={"nested": "value"},
      )

    params_file = tmp_proj / getpass.getuser() / "test/param-types/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["int_param"] == 42
    assert params["float_param"] == 3.14159
    assert params["str_param"] == "hello world"
    assert params["bool_param_true"] is True
    assert params["bool_param_false"] is False
    assert params["none_param"] is None
    assert params["list_param"] == [1, 2, 3, 4, 5]

  @pytest.mark.remote
  def test_parameter_types_remote(self, remote_experiment):
    """Test different parameter types in remote mode."""
    with remote_experiment(name="param-types-remote", project="test").run as experiment:
      experiment.params.set(
        int_param=100, float_param=2.71828, str_param="remote test", bool_param=True
      )


class TestNestedParameters:
  """Tests for nested parameter structures and flattening."""

  def test_nested_parameters_flattening_local(
    self, local_experiment, tmp_proj, sample_data
  ):
    """Test that nested parameters are automatically flattened."""
    with local_experiment(name="nested-params", project="test").run as experiment:
      experiment.params.set(**sample_data["nested_params"])

    params_file = tmp_proj / getpass.getuser() / "test/nested-params/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    # Check flattened keys
    assert params["model.architecture"] == "resnet50"
    assert params["model.pretrained"] is True
    assert params["model.layers.conv1.filters"] == 64
    assert params["model.layers.conv1.kernel"] == 3
    assert params["model.layers.conv2.filters"] == 128
    assert params["optimizer.type"] == "adam"
    assert params["optimizer.beta1"] == 0.9
    assert params["optimizer.lr"] == 0.001

  @pytest.mark.remote
  def test_nested_parameters_remote(self, remote_experiment, sample_data):
    """Test nested parameters in remote mode."""
    with remote_experiment(
      name="nested-params-remote", project="test"
    ).run as experiment:
      experiment.params.set(**sample_data["nested_params"])

  def test_deeply_nested_parameters_local(self, local_experiment, tmp_proj):
    """Test deeply nested parameter structures."""
    with local_experiment(name="deep-nested", project="test").run as experiment:
      experiment.params.set(
        **{
          "config": {
            "model": {
              "encoder": {"layers": 12, "heads": 8, "hidden_size": 768},
              "decoder": {"layers": 6, "heads": 8},
            },
            "training": {
              "optimizer": {"name": "adam", "lr": 0.001, "betas": [0.9, 0.999]}
            },
          }
        }
      )

    params_file = tmp_proj / getpass.getuser() / "test/deep-nested/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["config.model.encoder.layers"] == 12
    assert params["config.model.encoder.heads"] == 8
    assert params["config.training.optimizer.name"] == "adam"

  def test_mixed_flat_and_nested_local(self, local_experiment, tmp_proj):
    """Test mixing flat and nested parameters."""
    with local_experiment(name="mixed-params", project="test").run as experiment:
      experiment.params.set(
        learning_rate=0.001,
        batch_size=32,
        **{
          "model": {"name": "resnet", "layers": 50},
          "optimizer": {"type": "adam", "momentum": 0.9},
        },
      )

    params_file = tmp_proj / getpass.getuser() / "test/mixed-params/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["learning_rate"] == 0.001
    assert params["batch_size"] == 32
    assert params["model.name"] == "resnet"
    assert params["optimizer.type"] == "adam"


class TestParameterUpdates:
  """Tests for updating existing parameters."""

  def test_parameter_update_local(self, local_experiment, tmp_proj):
    """Test updating existing parameters."""
    with local_experiment(name="param-update", project="test").run as experiment:
      # Set initial parameters
      experiment.params.set(learning_rate=0.01, batch_size=32)

      # Update learning rate
      experiment.params.set(learning_rate=0.001)

      # Add new parameter
      experiment.params.set(epochs=100)

    params_file = tmp_proj / getpass.getuser() / "test/param-update/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["learning_rate"] == 0.001  # Updated
    assert params["batch_size"] == 32  # Unchanged
    assert params["epochs"] == 100  # New

  @pytest.mark.remote
  def test_parameter_update_remote(self, remote_experiment):
    """Test updating parameters in remote mode."""
    with remote_experiment(
      name="param-update-remote", project="test"
    ).run as experiment:
      experiment.params.set(version=1)
      experiment.params.set(version=2)
      experiment.params.set(final=True)

  def test_multiple_parameter_updates_local(self, local_experiment, tmp_proj):
    """Test multiple parameter update operations."""
    with local_experiment(name="multi-update", project="test").run as experiment:
      experiment.params.set(lr=0.1, batch_size=32)
      experiment.params.set(lr=0.01, ent_coef=0.01)
      experiment.params.set(lr=0.001, ent_coef=0.001)

    params_file = tmp_proj / getpass.getuser() / "test/multi-update/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["lr"] == 0.001
    assert params["ent_coef"] == 0.001

  def test_overwrite_nested_parameter_local(self, local_experiment, tmp_proj):
    """Test overwriting nested parameters."""
    with local_experiment(name="overwrite-nested", project="test").run as experiment:
      experiment.params.set(**{"model": {"name": "vgg", "layers": 16}})
      experiment.params.set(**{"model": {"name": "resnet", "layers": 50}})

    params_file = tmp_proj / getpass.getuser() / "test/overwrite-nested/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["model.name"] == "resnet"
    assert params["model.layers"] == 50


class TestParameterEdgeCases:
  """Tests for edge cases in parameter handling."""

  def test_empty_parameters_local(self, local_experiment, tmp_proj):
    """Test experiment with no parameters set."""
    with local_experiment(name="no-params", project="test").run as experiment:
      experiment.log("No parameters")

    params_file = tmp_proj / getpass.getuser() / "test/no-params/parameters.json"
    if params_file.exists():
      with open(params_file) as f:
        params_data = json.load(f)
        assert params_data.get("data", {}) == {}

  def test_parameters_with_special_keys_local(self, local_experiment, tmp_proj):
    """Test parameters with special characters in keys."""
    with local_experiment(name="special-keys", project="test").run as experiment:
      experiment.params.set(
        **{
          "key_with_underscore": 1,
          "key-with-dash": 2,
          "key.with.dot": 3,
          "key with space": 4,
        }
      )

    params_file = tmp_proj / getpass.getuser() / "test/special-keys/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["key_with_underscore"] == 1
    assert params["key-with-dash"] == 2

  def test_large_parameter_set_local(self, local_experiment, tmp_proj):
    """Test setting a large number of parameters."""
    large_params = {f"param_{i}": i * 0.001 for i in range(1000)}

    with local_experiment(name="large-params", project="test").run as experiment:
      experiment.params.set(**large_params)

    params_file = tmp_proj / getpass.getuser() / "test/large-params/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert len(params) == 1000
    assert params["param_0"] == 0.0
    assert params["param_999"] == 0.999

  def test_very_long_parameter_value_local(self, local_experiment, tmp_proj):
    """Test parameter with very long string value."""
    long_value = "A" * 10000

    with local_experiment(name="long-value", project="test").run as experiment:
      experiment.params.set(long_param=long_value)

    params_file = tmp_proj / getpass.getuser() / "test/long-value/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert len(params["long_param"]) == 10000

  def test_unicode_parameter_values_local(self, local_experiment, tmp_proj):
    """Test parameters with unicode values."""
    with local_experiment(name="unicode-params", project="test").run as experiment:
      experiment.params.set(
        japanese="こんにちは", chinese="你好", emoji="🚀 🎉 💯", arabic="مرحبا"
      )

    params_file = tmp_proj / getpass.getuser() / "test/unicode-params/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["japanese"] == "こんにちは"
    assert params["emoji"] == "🚀 🎉 💯"

  def test_numeric_edge_values_local(self, local_experiment, tmp_proj):
    """Test parameters with edge case numeric values."""
    with local_experiment(name="numeric-edge", project="test").run as experiment:
      experiment.params.set(
        zero=0,
        negative=-42,
        large_int=999999999999,
        small_float=0.0000001,
        large_float=999999.999999,
        scientific=1.23e-10,
      )

    params_file = tmp_proj / getpass.getuser() / "test/numeric-edge/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["zero"] == 0
    assert params["negative"] == -42
    assert params["large_int"] == 999999999999

  @pytest.mark.remote
  def test_large_parameter_set_remote(self, remote_experiment):
    """Test setting many parameters in remote mode."""
    large_params = {f"param_{i}": i for i in range(100)}

    with remote_experiment(
      name="large-params-remote", project="test"
    ).run as experiment:
      experiment.params.set(**large_params)


class TestParameterCombinations:
  """Tests for complex parameter combinations."""

  def test_ml_training_parameters_local(self, local_experiment, tmp_proj):
    """Test typical ML training parameter structure."""
    with local_experiment(name="ml-params", project="test").run as experiment:
      experiment.params.set(
        **{
          "model": {
            "architecture": "transformer",
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "dropout": 0.1,
          },
          "training": {
            "batch_size": 32,
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "warmup_steps": 500,
            "weight_decay": 0.01,
          },
          "data": {
            "train_size": 10000,
            "val_size": 1000,
            "test_size": 1000,
            "max_length": 512,
          },
        }
      )

    params_file = tmp_proj / getpass.getuser() / "test/ml-params/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["model.architecture"] == "transformer"
    assert params["training.batch_size"] == 32
    assert params["data.train_size"] == 10000

  def test_config_file_like_parameters_local(self, local_experiment, tmp_proj):
    """Test parameters structured like a config file."""
    with local_experiment(name="config-params", project="test").run as experiment:
      experiment.params.set(
        **{
          "experiment": {
            "name": "baseline",
            "version": "1.0",
            "tags": ["baseline", "test"],
          },
          "paths": {
            "data": "/data/train",
            "output": "/output/models",
            "logs": "/output/logs",
          },
          "hyperparameters": {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001},
        }
      )

    params_file = tmp_proj / getpass.getuser() / "test/config-params/parameters.json"
    with open(params_file) as f:
      params = json.load(f)["data"]

    assert params["experiment.name"] == "baseline"
    assert params["paths.data"] == "/data/train"
    assert params["hyperparameters.lr"] == 0.001


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
