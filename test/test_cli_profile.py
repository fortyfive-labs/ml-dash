"""Tests for cli_commands/profile.py — cmd_profile flows."""

import argparse
import time
from unittest.mock import MagicMock, patch

import pytest

from ml_dash.cli_commands.profile import cmd_profile


def _args(dash_url=None, json=False, cached=False):
    return argparse.Namespace(dash_url=dash_url, json=json, cached=cached)


def _valid_payload(days_ahead=30):
    return {
        "exp": int(time.time()) + days_ahead * 86400,
        "sub": "user-abc123",
        "username": "tom",
        "name": "Tom Tao",
        "email": "tom@example.com",
    }


def _expired_payload():
    return {"exp": int(time.time()) - 3600, "username": "tom", "sub": "user-abc123"}


def _mock_storage(token):
    storage = MagicMock()
    storage.load.return_value = token
    return storage


class TestNoToken:
    def test_no_token_returns_0(self):
        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage(None)):
            result = cmd_profile(_args())
        assert result == 0

    def test_no_token_json_returns_0(self):
        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage(None)):
            result = cmd_profile(_args(json=True))
        assert result == 0


class TestExpiredToken:
    def test_expired_token_returns_0(self):
        payload = _expired_payload()
        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload):
            result = cmd_profile(_args())
        assert result == 0

    def test_expired_token_json_shows_unauthenticated(self):
        payload = _expired_payload()
        captured = {}

        def capture_json(data):
            import json
            captured["data"] = json.loads(data)

        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload), \
             patch("rich.console.Console.print_json", side_effect=capture_json):
            cmd_profile(_args(json=True))

        assert captured["data"]["authenticated"] is False


class TestValidTokenCachedMode:
    def test_cached_mode_returns_0(self):
        payload = _valid_payload()
        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload):
            result = cmd_profile(_args(cached=True))
        assert result == 0

    def test_cached_mode_json_shows_authenticated(self):
        payload = _valid_payload()
        captured = {}

        def capture_json(data):
            import json
            captured["data"] = json.loads(data)

        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload), \
             patch("rich.console.Console.print_json", side_effect=capture_json):
            cmd_profile(_args(json=True, cached=True))

        assert captured["data"]["authenticated"] is True
        assert captured["data"]["source"] == "token"


class TestValidTokenServerMode:
    def test_fresh_fetch_succeeds(self):
        payload = _valid_payload()
        fresh = {"username": "tom", "name": "Tom Tao", "email": "tom@example.com", "sub": "abc"}
        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload), \
             patch("ml_dash.cli_commands.profile._fetch_fresh_profile", return_value=fresh):
            result = cmd_profile(_args())
        assert result == 0

    def test_fresh_fetch_fails_falls_back_to_token(self):
        payload = _valid_payload()
        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload), \
             patch("ml_dash.cli_commands.profile._fetch_fresh_profile", return_value=None):
            result = cmd_profile(_args())
        assert result == 0

    def test_server_mode_json_source_is_server(self):
        payload = _valid_payload()
        fresh = {"username": "tom", "name": "Tom Tao", "email": "tom@example.com", "sub": "abc"}
        captured = {}

        def capture_json(data):
            import json
            captured["data"] = json.loads(data)

        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload), \
             patch("ml_dash.cli_commands.profile._fetch_fresh_profile", return_value=fresh), \
             patch("rich.console.Console.print_json", side_effect=capture_json):
            cmd_profile(_args(json=True))

        assert captured["data"]["source"] == "server"

    def test_fallback_json_source_is_token(self):
        payload = _valid_payload()
        captured = {}

        def capture_json(data):
            import json
            captured["data"] = json.loads(data)

        with patch("ml_dash.cli_commands.profile.get_token_storage",
                   return_value=_mock_storage("fake.jwt.token")), \
             patch("ml_dash.cli_commands.profile.decode_jwt_payload", return_value=payload), \
             patch("ml_dash.cli_commands.profile._fetch_fresh_profile", return_value=None), \
             patch("rich.console.Console.print_json", side_effect=capture_json):
            cmd_profile(_args(json=True))

        assert captured["data"]["source"] == "token"
        assert "warning" in captured["data"]
