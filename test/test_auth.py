"""Tests for OAuth2 authentication system."""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ml_dash.config import ConfigManager
from ml_dash.auth_server import AuthCallbackServer, CallbackHandler
from ml_dash.auth import OAuth2AuthFlow
from ml_dash import Experiment


# ===== ConfigManager Tests =====

def test_config_manager_initialization():
    """Test ConfigManager initialization with custom directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "test-config"
        manager = ConfigManager(config_dir=config_dir)

        assert manager.config_dir == config_dir
        assert manager.config_path == config_dir / "config.json"


def test_config_manager_ensure_dir():
    """Test that config directory is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "test-config"
        manager = ConfigManager(config_dir=config_dir)

        assert not config_dir.exists()

        manager.ensure_config_dir()

        assert config_dir.exists()


def test_config_manager_save_and_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        config = {"key": "value", "number": 42}
        manager.save_config(config)

        loaded = manager.load_config()
        assert loaded == config


def test_config_manager_load_nonexistent():
    """Test loading config when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        config = manager.load_config()
        assert config == {}


def test_config_manager_save_token():
    """Test saving authentication token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        manager.save_token("test-token-123", "https://auth.example.com")

        config = manager.load_config()
        assert config["token"] == "test-token-123"
        assert config["auth_server"] == "https://auth.example.com"
        assert "updated_at" in config


def test_config_manager_get_token():
    """Test retrieving saved token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        manager.save_token("test-token-456")
        token = manager.get_token()

        assert token == "test-token-456"


def test_config_manager_get_token_with_server_match():
    """Test retrieving token with matching auth server."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        manager.save_token("test-token", "https://auth.example.com")
        token = manager.get_token(auth_server="https://auth.example.com")

        assert token == "test-token"


def test_config_manager_get_token_with_server_mismatch():
    """Test retrieving token with mismatched auth server."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        manager.save_token("test-token", "https://auth.example.com")
        token = manager.get_token(auth_server="https://other.example.com")

        assert token is None


def test_config_manager_clear_token():
    """Test clearing saved token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        manager.save_token("test-token")
        assert manager.get_token() == "test-token"

        manager.clear_token()
        assert manager.get_token() is None


def test_config_manager_is_authenticated():
    """Test checking authentication status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=Path(tmpdir))

        assert not manager.is_authenticated()

        manager.save_token("test-token")
        assert manager.is_authenticated()

        manager.clear_token()
        assert not manager.is_authenticated()


# ===== AuthCallbackServer Tests =====

def test_callback_server_initialization():
    """Test callback server initialization."""
    server = AuthCallbackServer(port=52845)

    assert server.port == 52845
    assert server.server is None
    assert server.thread is None


def test_callback_server_get_callback_url():
    """Test getting callback URL."""
    server = AuthCallbackServer(port=52845)
    url = server.get_callback_url()

    assert url == "http://localhost:52845"


def test_callback_server_start_stop():
    """Test starting and stopping server."""
    server = AuthCallbackServer()

    server.start()
    assert server.server is not None
    assert server.thread is not None
    assert server.thread.is_alive()

    server.stop()
    time.sleep(0.2)  # Give thread time to stop
    assert server.server is None


def test_callback_server_context_manager():
    """Test using server as context manager."""
    server = AuthCallbackServer()

    with server:
        assert server.server is not None
        assert server.thread is not None
        assert server.thread.is_alive()

    time.sleep(0.2)
    assert server.server is None


def test_callback_server_wait_for_token_timeout():
    """Test waiting for token with timeout."""
    server = AuthCallbackServer()

    # Reset token state
    with CallbackHandler.token_lock:
        CallbackHandler.received_token = None

    server.start()

    try:
        # Short timeout
        token = server.wait_for_token(timeout=0.5)
        assert token is None
    finally:
        server.stop()


def test_callback_server_wait_for_token_success():
    """Test waiting for token successfully."""
    server = AuthCallbackServer()

    # Reset token state
    with CallbackHandler.token_lock:
        CallbackHandler.received_token = None

    server.start()

    try:
        # Simulate token arrival
        with CallbackHandler.token_lock:
            CallbackHandler.received_token = "test-token-789"

        token = server.wait_for_token(timeout=2.0)
        assert token == "test-token-789"
    finally:
        server.stop()


def test_callback_server_port_fallback():
    """Test port fallback when port is in use."""
    # Start server on port 52845
    server1 = AuthCallbackServer(port=52845)
    server1.start()

    try:
        # Try to start another server on same port - should fall back
        server2 = AuthCallbackServer(port=52845)
        server2.start()

        try:
            # Should use different port
            assert server2.port != server1.port
            assert server2.port in [52846, 52847]
        finally:
            server2.stop()
    finally:
        server1.stop()


# ===== OAuth2AuthFlow Tests =====

@patch('ml_dash.auth.webbrowser.open')
@patch('ml_dash.auth.AuthCallbackServer')
def test_oauth2_flow_authenticate_success(mock_server_class, mock_browser):
    """Test successful authentication flow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock server
        mock_server = Mock()
        mock_server.get_callback_url.return_value = "http://localhost:52845"
        mock_server.wait_for_token.return_value = "test-token-success"
        mock_server_class.return_value = mock_server

        # Mock browser
        mock_browser.return_value = True

        # Create flow with temp config
        config_manager = ConfigManager(config_dir=Path(tmpdir))
        flow = OAuth2AuthFlow(config_manager=config_manager)

        success = flow.authenticate()

        assert success
        assert config_manager.get_token() == "test-token-success"


@patch('ml_dash.auth.webbrowser.open')
@patch('ml_dash.auth.AuthCallbackServer')
def test_oauth2_flow_authenticate_timeout(mock_server_class, mock_browser):
    """Test authentication flow timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock server
        mock_server = Mock()
        mock_server.get_callback_url.return_value = "http://localhost:52845"
        mock_server.wait_for_token.return_value = None  # Timeout
        mock_server_class.return_value = mock_server

        # Mock browser
        mock_browser.return_value = True

        # Create flow with temp config
        config_manager = ConfigManager(config_dir=Path(tmpdir))
        flow = OAuth2AuthFlow(config_manager=config_manager, timeout=0.5)

        success = flow.authenticate()

        assert not success
        assert config_manager.get_token() is None


@patch('ml_dash.auth.webbrowser.open')
@patch('ml_dash.auth.AuthCallbackServer')
def test_oauth2_flow_authenticate_browser_fail(mock_server_class, mock_browser):
    """Test authentication flow when browser fails to open."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock server
        mock_server = Mock()
        mock_server.get_callback_url.return_value = "http://localhost:52845"
        mock_server.wait_for_token.return_value = "test-token"
        mock_server_class.return_value = mock_server

        # Mock browser failure
        mock_browser.return_value = False

        # Create flow with temp config
        config_manager = ConfigManager(config_dir=Path(tmpdir))
        flow = OAuth2AuthFlow(config_manager=config_manager)

        # Should still work (URL printed for manual opening)
        success = flow.authenticate()

        assert success
        assert config_manager.get_token() == "test-token"


def test_oauth2_flow_check_status_authenticated():
    """Test checking status when authenticated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_manager = ConfigManager(config_dir=Path(tmpdir))
        config_manager.save_token("test-token")

        flow = OAuth2AuthFlow(config_manager=config_manager)
        authenticated = flow.check_status()

        assert authenticated


def test_oauth2_flow_check_status_not_authenticated():
    """Test checking status when not authenticated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_manager = ConfigManager(config_dir=Path(tmpdir))

        flow = OAuth2AuthFlow(config_manager=config_manager)
        authenticated = flow.check_status()

        assert not authenticated


def test_oauth2_flow_logout():
    """Test logout functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_manager = ConfigManager(config_dir=Path(tmpdir))
        config_manager.save_token("test-token")

        flow = OAuth2AuthFlow(config_manager=config_manager)
        success = flow.logout()

        assert success
        assert config_manager.get_token() is None


# ===== Experiment Integration Tests =====

def test_experiment_uses_saved_token():
    """Test that Experiment automatically uses saved token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save token
        config_manager = ConfigManager(config_dir=Path(tmpdir))
        config_manager.save_token("test-token-experiment")

        # Patch ConfigManager class to return our instance
        with patch('ml_dash.config.ConfigManager', return_value=config_manager):
            # Create experiment in remote mode without api_key or user_name
            # This should automatically load the saved token
            try:
                exp = Experiment(
                    name="test",
                    project="test",
                    remote="http://localhost:3000"
                )

                # Should have initialized with saved token
                assert exp._client is not None
            except Exception as e:
                # The experiment might fail to connect to server, but token should be loaded
                # Check that it's not the auth error
                assert "Authentication required" not in str(e)


def test_experiment_error_without_auth():
    """Test that Experiment raises helpful error without authentication."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No saved token
        config_manager = ConfigManager(config_dir=Path(tmpdir))

        # Patch ConfigManager class to return our instance
        with patch('ml_dash.config.ConfigManager', return_value=config_manager):
            # Try to create experiment without auth
            with pytest.raises(ValueError) as exc_info:
                Experiment(
                    name="test",
                    project="test",
                    remote="http://localhost:3000"
                )

            # Should have helpful error message
            error_msg = str(exc_info.value)
            assert "Authentication required" in error_msg
            assert "ml-dash setup" in error_msg


# ===== CLI Tests (Simple smoke tests) =====

def test_cli_creates_parser():
    """Test that CLI parser can be created."""
    from ml_dash.cli import create_parser

    parser = create_parser()
    assert parser is not None


def test_cli_main_no_command():
    """Test CLI main with no command shows help."""
    from ml_dash.cli import main

    exit_code = main([])
    assert exit_code == 0


def test_cli_main_unknown_command():
    """Test CLI main with unknown command."""
    from ml_dash.cli import main

    # Unknown command should raise SystemExit
    with pytest.raises(SystemExit):
        main(["unknown"])
