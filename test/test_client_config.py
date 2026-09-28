"""Tests for client configuration: API URL, token lookup, version check, GraphQL queries (no network)."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

import ml_dash
from ml_dash.auth.token_storage import load_token
from ml_dash.auth.exceptions import StorageError

FAKE_TOKEN = "fake-token-for-tests"


def test_userinfo_uses_run_api_url(monkeypatch):
  """F6: userinfo asks the same server as Experiment(dash_url=True)."""
  from ml_dash.client import UserInfo
  from ml_dash.run import RUN

  monkeypatch.setattr(RUN, "api_url", "http://stub.invalid")
  remote_client = MagicMock()
  monkeypatch.setattr("ml_dash.client.RemoteClient", remote_client)

  UserInfo().username

  remote_client.assert_called_once_with("http://stub.invalid")


class TestVersionCheck:
  """F7: opt-out, and failures never break the import."""

  def test_opt_out_skips_request(self, monkeypatch):
    monkeypatch.setenv("ML_DASH_NO_VERSION_CHECK", "1")
    get = MagicMock()
    monkeypatch.setattr(httpx, "get", get)
    ml_dash._check_version_compatibility()
    get.assert_not_called()

  @pytest.mark.parametrize("outcome", [
    httpx.ReadError("connection dropped"),
    httpx.Response(200, text="<html>not json</html>"),
  ])
  def test_network_and_json_failures_are_swallowed(self, monkeypatch, outcome):
    monkeypatch.delenv("ML_DASH_NO_VERSION_CHECK", raising=False)
    if isinstance(outcome, Exception):
      get = MagicMock(side_effect=outcome)
    else:
      get = MagicMock(return_value=outcome)
    monkeypatch.setattr(httpx, "get", get)
    ml_dash._check_version_compatibility()
    get.assert_called_once()


class NoKeyringError(Exception):
  """Stand-in for keyring.errors.NoKeyringError (no backend available)."""


def _fake_keyring(monkeypatch, get_password):
  """Replace the keyring module; get_password=None makes `import keyring` fail."""
  if get_password is None:
    monkeypatch.setitem(sys.modules, "keyring", None)
  else:
    fake = SimpleNamespace(get_password=get_password,
                           errors=SimpleNamespace(NoKeyringError=NoKeyringError))
    monkeypatch.setitem(sys.modules, "keyring", fake)


def _raise(error):
  def fail(*args):
    raise error
  return fail


def _write_cli_encrypted(config_dir, tokens):
  from cryptography.fernet import Fernet

  key = Fernet.generate_key()
  (config_dir / "encryption.key").write_text(key.decode())  # the CLI writes base64 text
  (config_dir / "tokens.encrypted").write_bytes(Fernet(key).encrypt(json.dumps(tokens).encode()))


def _files(path):
  return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


class TestTokenFallback:
  """F8: read the CLI's stores in order, read-only; denied or corrupt stores never fall through."""

  def test_encrypted_file_used_when_keyring_has_no_entry(self, monkeypatch, tmp_path):
    _fake_keyring(monkeypatch, lambda *a: None)
    _write_cli_encrypted(tmp_path, {"ml-dash-token": FAKE_TOKEN})
    assert load_token("ml-dash-token", tmp_path) == FAKE_TOKEN

  def test_plaintext_file_used_when_keyring_has_no_backend(self, monkeypatch, tmp_path):
    _fake_keyring(monkeypatch, _raise(NoKeyringError("no backend")))
    (tmp_path / "tokens.json").write_text(json.dumps({"ml-dash-token": FAKE_TOKEN}))
    assert load_token("ml-dash-token", tmp_path) == FAKE_TOKEN

  def test_keyring_value_wins(self, monkeypatch, tmp_path):
    _fake_keyring(monkeypatch, lambda *a: "keyring-value")
    _write_cli_encrypted(tmp_path, {"ml-dash-token": FAKE_TOKEN})
    assert load_token("ml-dash-token", tmp_path) == "keyring-value"

  def test_config_dir_from_environment(self, monkeypatch, tmp_path):
    _fake_keyring(monkeypatch, None)
    monkeypatch.setenv("ML_DASH_CONFIG_DIR", str(tmp_path))
    _write_cli_encrypted(tmp_path, {"ml-dash-token": FAKE_TOKEN})
    assert load_token("ml-dash-token") == FAKE_TOKEN

  def test_denied_keyring_does_not_fall_back(self, monkeypatch, tmp_path):
    _fake_keyring(monkeypatch, _raise(RuntimeError(f"keychain locked: {FAKE_TOKEN}")))
    _write_cli_encrypted(tmp_path, {"ml-dash-token": FAKE_TOKEN})
    with pytest.raises(StorageError, match="keyring: RuntimeError$") as excinfo:
      load_token("ml-dash-token", tmp_path)
    assert FAKE_TOKEN not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__

  @pytest.mark.parametrize("damage", ["wrong_key", "missing_key", "not_a_dict", "not_a_string"])
  def test_corrupt_encrypted_store_raises_without_content(self, monkeypatch, tmp_path, damage):
    from cryptography.fernet import Fernet

    _fake_keyring(monkeypatch, lambda *a: None)
    if damage == "not_a_dict":
      _write_cli_encrypted(tmp_path, [FAKE_TOKEN])
    elif damage == "not_a_string":
      _write_cli_encrypted(tmp_path, {"ml-dash-token": [FAKE_TOKEN]})
    else:
      _write_cli_encrypted(tmp_path, {"ml-dash-token": FAKE_TOKEN})
    if damage == "wrong_key":
      (tmp_path / "encryption.key").write_text(Fernet.generate_key().decode())
    elif damage == "missing_key":
      (tmp_path / "encryption.key").unlink()
    (tmp_path / "tokens.json").write_text(json.dumps({"ml-dash-token": "plain"}))

    with pytest.raises(StorageError) as excinfo:
      load_token("ml-dash-token", tmp_path)
    assert FAKE_TOKEN not in str(excinfo.value)
    error = excinfo.value
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__

  def test_corrupt_plaintext_store_raises(self, monkeypatch, tmp_path):
    _fake_keyring(monkeypatch, lambda *a: None)
    (tmp_path / "tokens.json").write_text('{"ml-dash-token": ')
    with pytest.raises(StorageError):
      load_token("ml-dash-token", tmp_path)

  @pytest.mark.parametrize("keyring_present", [True, False])
  def test_lookup_writes_nothing(self, monkeypatch, tmp_path, keyring_present):
    _fake_keyring(monkeypatch, (lambda *a: None) if keyring_present else None)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ML_DASH_CONFIG_DIR", raising=False)

    assert load_token("ml-dash-token") is None  # default ~/.dash does not exist
    assert load_token("ml-dash-token", tmp_path / "missing") is None
    assert _files(tmp_path) == ["home"]


@pytest.mark.parametrize("method, args", [
  ("list_experiments_graphql", {"project_slug": "p", "namespace_slug": "ns"}),
  ("get_experiment_graphql", {"project_slug": "p", "experiment_name": "e", "namespace_slug": "ns"}),
  ("search_experiments_graphql", {"pattern": "ns/p/*"}),
])
def test_graphql_queries_omit_removed_count_fields(method, args):
  """F10: the server schema no longer has logMetadata or metricMetadata."""
  from ml_dash.client import RemoteClient

  client = RemoteClient("http://stub.invalid", api_key="fake", namespace="ns")
  captured = []

  def capture(query, variables=None):
    captured.append(query)
    raise LookupError("stop after capturing the query")

  client.graphql_query = capture
  with pytest.raises(LookupError):
    getattr(client, method)(**args)

  assert captured
  assert "logMetadata" not in captured[0] and "metricMetadata" not in captured[0]
