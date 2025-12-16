"""Tests for authentication module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import time

from ml_dash.auth import device_secret, token_storage, device_flow, exceptions


class TestDeviceSecret:
    """Tests for device secret generation and hashing."""

    def test_generate_device_secret_length(self):
        """Test that generated device secret is correct length."""
        secret = device_secret.generate_device_secret()
        # 256 bits = 32 bytes = 64 hex characters
        assert len(secret) == 64
        assert all(c in "0123456789abcdef" for c in secret)

    def test_generate_device_secret_randomness(self):
        """Test that generated secrets are different."""
        secret1 = device_secret.generate_device_secret()
        secret2 = device_secret.generate_device_secret()
        assert secret1 != secret2

    def test_hash_device_secret_deterministic(self):
        """Test that hashing is deterministic."""
        secret = "test-secret-123"
        hash1 = device_secret.hash_device_secret(secret)
        hash2 = device_secret.hash_device_secret(secret)
        assert hash1 == hash2

    def test_hash_device_secret_format(self):
        """Test that hash is SHA256 format."""
        secret = "test-secret"
        hashed = device_secret.hash_device_secret(secret)
        # SHA256 produces 64 hex characters
        assert len(hashed) == 64
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_hash_device_secret_different_inputs(self):
        """Test that different secrets produce different hashes."""
        hash1 = device_secret.hash_device_secret("secret1")
        hash2 = device_secret.hash_device_secret("secret2")
        assert hash1 != hash2

    def test_get_or_create_device_secret_creates_new(self):
        """Test that get_or_create creates new secret when none exists."""
        from ml_dash.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))

            # Should create new secret
            secret1 = device_secret.get_or_create_device_secret(config)
            assert secret1 is not None
            assert len(secret1) == 64

            # Should return same secret on second call
            secret2 = device_secret.get_or_create_device_secret(config)
            assert secret1 == secret2

    def test_get_or_create_device_secret_uses_existing(self):
        """Test that get_or_create uses existing secret from config."""
        from ml_dash.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))
            existing_secret = "existing-secret-123"
            config.set("device_secret", existing_secret)
            config.save()

            secret = device_secret.get_or_create_device_secret(config)
            assert secret == existing_secret


class TestTokenStorage:
    """Tests for token storage backends."""

    def test_plaintext_storage_save_and_load(self):
        """Test plaintext storage save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.PlaintextFileStorage(Path(tmpdir))

            # Store token
            test_token = "test-token-123"
            storage.store("test-key", test_token)

            # Load token
            loaded = storage.load("test-key")
            assert loaded == test_token

    def test_plaintext_storage_delete(self):
        """Test plaintext storage deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.PlaintextFileStorage(Path(tmpdir))

            # Store and delete
            storage.store("test-key", "test-token")
            storage.delete("test-key")

            # Should return None after deletion
            loaded = storage.load("test-key")
            assert loaded is None

    def test_plaintext_storage_nonexistent_key(self):
        """Test loading nonexistent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.PlaintextFileStorage(Path(tmpdir))
            loaded = storage.load("nonexistent")
            assert loaded is None

    def test_plaintext_storage_file_format(self):
        """Test that plaintext storage creates valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.PlaintextFileStorage(Path(tmpdir))
            storage.store("key1", "token1")
            storage.store("key2", "token2")

            # Check file exists and is valid JSON
            tokens_file = Path(tmpdir) / "tokens.json"
            assert tokens_file.exists()

            with open(tokens_file) as f:
                data = json.load(f)

            assert data["key1"] == "token1"
            assert data["key2"] == "token2"

    def test_encrypted_storage_save_and_load(self):
        """Test encrypted storage save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.EncryptedFileStorage(Path(tmpdir))

            # Store token
            test_token = "test-token-123"
            storage.store("test-key", test_token)

            # Load token
            loaded = storage.load("test-key")
            assert loaded == test_token

    def test_encrypted_storage_encryption(self):
        """Test that encrypted storage actually encrypts data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.EncryptedFileStorage(Path(tmpdir))

            secret_token = "super-secret-token"
            storage.store("test-key", secret_token)

            # Read raw file - should be encrypted (not plaintext)
            tokens_file = Path(tmpdir) / "tokens.encrypted"
            raw_data = tokens_file.read_bytes()

            # Token should not appear in plaintext
            assert secret_token.encode() not in raw_data

    def test_encrypted_storage_delete(self):
        """Test encrypted storage deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = token_storage.EncryptedFileStorage(Path(tmpdir))

            storage.store("test-key", "test-token")
            storage.delete("test-key")

            loaded = storage.load("test-key")
            assert loaded is None

    def test_encrypted_storage_persistence_across_instances(self):
        """Test that encrypted storage works across multiple instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance - store
            storage1 = token_storage.EncryptedFileStorage(Path(tmpdir))
            storage1.store("test-key", "test-token")

            # Second instance - load (should use same encryption key)
            storage2 = token_storage.EncryptedFileStorage(Path(tmpdir))
            loaded = storage2.load("test-key")
            assert loaded == "test-token"

    def test_keyring_storage_save_and_load(self):
        """Test keyring storage save and load."""
        try:
            import keyring
            # Only run if keyring is actually available
            storage = token_storage.KeyringStorage()

            # Store token
            storage.store("test-key-auth", "test-token-123")

            # Load token
            loaded = storage.load("test-key-auth")
            assert loaded == "test-token-123"

            # Cleanup
            storage.delete("test-key-auth")
        except ImportError:
            pytest.skip("keyring not available")

    def test_keyring_storage_delete(self):
        """Test keyring storage deletion."""
        try:
            import keyring
            storage = token_storage.KeyringStorage()

            # Store then delete
            storage.store("test-key-auth-delete", "test-token")
            storage.delete("test-key-auth-delete")

            # Should return None after deletion
            loaded = storage.load("test-key-auth-delete")
            assert loaded is None
        except ImportError:
            pytest.skip("keyring not available")

    def test_get_token_storage_priority(self):
        """Test that get_token_storage falls back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should return some storage backend
            storage = token_storage.get_token_storage(Path(tmpdir))
            assert storage is not None

            # Should be able to store and retrieve
            storage.store("test-key", "test-value")
            assert storage.load("test-key") == "test-value"
            storage.delete("test-key")


class TestDeviceFlow:
    """Tests for device authorization flow."""

    def test_device_flow_response_parsing(self):
        """Test DeviceFlowResponse dataclass parsing."""
        data = {
            "device_code": "test-device-code",
            "user_code": "ABC-123",
            "verification_uri": "https://auth.example.com/device",
            "verification_uri_complete": "https://auth.example.com/device?code=ABC-123",
            "expires_in": 600,
            "interval": 5
        }

        response = device_flow.DeviceFlowResponse(**data)
        assert response.device_code == "test-device-code"
        assert response.user_code == "ABC-123"
        assert response.expires_in == 600

    @patch('ml_dash.auth.device_flow.httpx.post')
    def test_start_device_flow_success(self, mock_post):
        """Test starting device flow successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "device_code": "test-device-code",
            "user_code": "ABC-123",
            "verification_uri": "https://auth.ml-dash.com/device",
            "verification_uri_complete": "https://auth.ml-dash.com/device?code=ABC-123",
            "expires_in": 600,
            "interval": 5
        }
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )

        result = client.start_device_flow()

        assert result.user_code == "ABC-123"
        assert result.device_code == "test-device-code"
        mock_post.assert_called_once()

    @patch('ml_dash.auth.device_flow.httpx.post')
    @patch('ml_dash.auth.device_flow.time.sleep')
    def test_poll_for_token_success(self, mock_sleep, mock_post):
        """Test polling for token until authorization."""
        # First 2 calls return authorization_pending (non-200), 3rd returns token (200)
        response1 = Mock()
        response1.status_code = 400
        response1.json.return_value = {"error": "authorization_pending"}

        response2 = Mock()
        response2.status_code = 400
        response2.json.return_value = {"error": "authorization_pending"}

        response3 = Mock()
        response3.status_code = 200
        response3.json.return_value = {"access_token": "test-access-token"}

        mock_post.side_effect = [response1, response2, response3]

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )
        client._device_code = "test-device-code"
        client._interval = 1  # Short interval for testing

        token = client.poll_for_token(max_attempts=10)

        assert token == "test-access-token"
        assert mock_post.call_count == 3

    @patch('ml_dash.auth.device_flow.httpx.post')
    def test_poll_for_token_expired(self, mock_post):
        """Test polling when device code expires."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "expired_token"}
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )
        client._device_code = "test-device-code"
        client._interval = 1

        with pytest.raises(exceptions.DeviceCodeExpiredError):
            client.poll_for_token(max_attempts=5)

    @patch('ml_dash.auth.device_flow.httpx.post')
    def test_poll_for_token_denied(self, mock_post):
        """Test polling when user denies authorization."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "access_denied"}
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )
        client._device_code = "test-device-code"
        client._interval = 1

        with pytest.raises(exceptions.AuthorizationDeniedError):
            client.poll_for_token(max_attempts=5)

    @patch('ml_dash.auth.device_flow.httpx.post')
    @patch('ml_dash.auth.device_flow.time.sleep')
    def test_poll_for_token_timeout(self, mock_sleep, mock_post):
        """Test polling timeout when max attempts reached."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "authorization_pending"}
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )
        client._device_code = "test-device-code"
        client._interval = 1

        with pytest.raises(TimeoutError):
            client.poll_for_token(max_attempts=3)

        assert mock_post.call_count == 3

    @patch('ml_dash.auth.device_flow.httpx.post')
    def test_exchange_token_success(self, mock_post):
        """Test exchanging vuer-auth token for ml-dash token."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "ml_dash_token": "permanent-ml-dash-token"
        }
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )

        ml_dash_token = client.exchange_token("vuer-auth-token")

        assert ml_dash_token == "permanent-ml-dash-token"
        mock_post.assert_called_once()

        # Verify Authorization header was set
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer vuer-auth-token"

    @patch('ml_dash.auth.device_flow.httpx.post')
    def test_exchange_token_failure(self, mock_post):
        """Test token exchange failure."""
        import httpx

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Invalid token", request=Mock(), response=Mock()
        )
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )

        with pytest.raises(exceptions.TokenExchangeError):
            client.exchange_token("invalid-token")

    def test_device_flow_client_initialization(self):
        """Test DeviceFlowClient initialization."""
        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com/"
        )

        # Should strip trailing slash
        assert client.ml_dash_server_url == "https://api.ml-dash.com"
        assert client.device_secret == "test-secret"

    @patch('ml_dash.auth.device_flow.httpx.post')
    def test_poll_for_token_progress_callback(self, mock_post):
        """Test that progress callback is called during polling."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "authorization_pending"}
        mock_post.return_value = mock_response

        client = device_flow.DeviceFlowClient(
            device_secret="test-secret",
            ml_dash_server_url="https://api.ml-dash.com"
        )
        client._device_code = "test-device-code"
        client._interval = 0.1

        callback_calls = []
        def progress_callback(elapsed):
            callback_calls.append(elapsed)

        with pytest.raises(TimeoutError):
            client.poll_for_token(max_attempts=3, progress_callback=progress_callback)

        # Callback should have been called
        assert len(callback_calls) > 0


class TestAuthExceptions:
    """Tests for authentication exceptions."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        with pytest.raises(exceptions.AuthenticationError):
            raise exceptions.AuthenticationError("Test error")

    def test_not_authenticated_error(self):
        """Test NotAuthenticatedError exception."""
        with pytest.raises(exceptions.NotAuthenticatedError):
            raise exceptions.NotAuthenticatedError("Not logged in")

    def test_device_code_expired_error(self):
        """Test DeviceCodeExpiredError exception."""
        with pytest.raises(exceptions.DeviceCodeExpiredError):
            raise exceptions.DeviceCodeExpiredError("Code expired")

    def test_authorization_denied_error(self):
        """Test AuthorizationDeniedError exception."""
        with pytest.raises(exceptions.AuthorizationDeniedError):
            raise exceptions.AuthorizationDeniedError("User denied")

    def test_token_exchange_error(self):
        """Test TokenExchangeError exception."""
        with pytest.raises(exceptions.TokenExchangeError):
            raise exceptions.TokenExchangeError("Exchange failed")

    def test_storage_error(self):
        """Test StorageError exception."""
        with pytest.raises(exceptions.StorageError):
            raise exceptions.StorageError("Storage failed")
