"""Tests for _check_token_expiration in profile command."""

import time

import pytest

from ml_dash.cli_commands.profile import _check_token_expiration


class TestCheckTokenExpiration:
    def test_no_exp_field(self):
        is_expired, msg = _check_token_expiration({})
        assert is_expired is False
        assert msg is None

    def test_exp_field_none(self):
        is_expired, msg = _check_token_expiration({"exp": None})
        assert is_expired is False
        assert msg is None

    def test_expired_token(self):
        past = int(time.time()) - 3600  # 1 hour ago
        is_expired, msg = _check_token_expiration({"exp": past})
        assert is_expired is True
        assert "expired" in msg.lower()

    def test_just_expired(self):
        just_past = int(time.time()) - 1
        is_expired, msg = _check_token_expiration({"exp": just_past})
        assert is_expired is True

    def test_expires_soon_within_one_day(self):
        soon = int(time.time()) + 3600  # 1 hour from now
        is_expired, msg = _check_token_expiration({"exp": soon})
        assert is_expired is False
        assert msg is not None
        assert "hours" in msg

    def test_expires_in_several_days(self):
        future = int(time.time()) + 5 * 86400  # 5 days from now
        is_expired, msg = _check_token_expiration({"exp": future})
        assert is_expired is False
        assert msg is not None
        assert "days" in msg
        assert "5" in msg

    def test_expires_exactly_at_one_day_boundary(self):
        """Exactly 24 hours left — should report hours, not days."""
        exactly_one_day = int(time.time()) + 86400 - 1
        is_expired, msg = _check_token_expiration({"exp": exactly_one_day})
        assert is_expired is False
        assert "hours" in msg

    def test_expires_just_over_one_day(self):
        """Just over 24 hours — should report days."""
        just_over = int(time.time()) + 86400 + 60
        is_expired, msg = _check_token_expiration({"exp": just_over})
        assert is_expired is False
        assert "days" in msg
