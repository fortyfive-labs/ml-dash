"""Tests for cli_commands/api.py — pure helpers and cmd_api with mocked client."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from ml_dash.cli_commands.api import build_query, cmd_api, extract_path, fix_quotes


class TestExtractPath:
    def test_single_key(self):
        assert extract_path({"me": {"username": "ge"}}, ".me") == {"username": "ge"}

    def test_nested_keys(self):
        data = {"data": {"me": {"username": "tom"}}}
        assert extract_path(data, ".data.me.username") == "tom"

    def test_leading_dot_optional(self):
        assert extract_path({"a": 1}, "a") == 1

    def test_list_index(self):
        assert extract_path({"items": ["x", "y", "z"]}, ".items.1") == "y"

    def test_missing_key_raises_key_error(self):
        with pytest.raises(KeyError):
            extract_path({"a": 1}, ".b")

    def test_empty_path_returns_data(self):
        data = {"a": 1}
        assert extract_path(data, "") is data

    def test_wrong_type_raises(self):
        with pytest.raises((KeyError, TypeError)):
            extract_path("not-a-dict", ".key")


class TestFixQuotes:
    def test_single_to_double(self):
        assert fix_quotes("user(name: 'hello')") == 'user(name: "hello")'

    def test_no_quotes_unchanged(self):
        assert fix_quotes("me { username }") == "me { username }"

    def test_already_double_unchanged(self):
        result = fix_quotes('user(name: "hello")')
        assert result == 'user(name: "hello")'

    def test_multiple_occurrences(self):
        result = fix_quotes("a('x') b('y')")
        assert result == 'a("x") b("y")'

    def test_empty_string(self):
        assert fix_quotes("") == ""


class TestBuildQuery:
    def test_bare_query_wrapped_in_braces(self):
        result = build_query("me { username }", is_mutation=False)
        assert result == "{ me { username } }"

    def test_bare_mutation_wrapped(self):
        result = build_query("updateUser { id }", is_mutation=True)
        assert result.startswith("mutation {")
        assert "updateUser" in result

    def test_already_brace_wrapped_unchanged(self):
        q = "{ me { username } }"
        assert build_query(q, is_mutation=False) == q

    def test_already_query_keyword_unchanged(self):
        q = "query { me { username } }"
        assert build_query(q, is_mutation=False) == q

    def test_already_mutation_keyword_unchanged(self):
        q = "mutation { updateUser { id } }"
        assert build_query(q, is_mutation=True) == q

    def test_single_quotes_converted(self):
        result = build_query("user(name: 'test') { id }", is_mutation=False)
        assert "'" not in result
        assert '"test"' in result

    def test_whitespace_stripped(self):
        result = build_query("  me { username }  ", is_mutation=False)
        assert not result.startswith("  ")


class TestCmdApi:
    def _args(self, query=None, mutation=None, jq=None, dash_url=None):
        return argparse.Namespace(query=query, mutation=mutation, jq=jq, dash_url=dash_url)

    def test_successful_query_returns_0(self):
        mock_client = MagicMock()
        mock_client.graphql_query.return_value = {"data": {"me": {"username": "tom"}}}

        with patch("ml_dash.cli_commands.api.RemoteClient", return_value=mock_client):
            result = cmd_api(self._args(query="me { username }"))

        assert result == 0

    def test_successful_mutation_returns_0(self):
        mock_client = MagicMock()
        mock_client.graphql_query.return_value = {"data": {"updateUser": {"id": "123"}}}

        with patch("ml_dash.cli_commands.api.RemoteClient", return_value=mock_client):
            result = cmd_api(self._args(mutation="updateUser(username: 'new') { id }"))

        assert result == 0

    def test_jq_extraction_applied(self):
        mock_client = MagicMock()
        mock_client.graphql_query.return_value = {"data": {"me": {"username": "tom"}}}

        with patch("ml_dash.cli_commands.api.RemoteClient", return_value=mock_client):
            result = cmd_api(self._args(query="me { username }", jq=".data.me.username"))

        assert result == 0

    def test_invalid_jq_path_returns_1(self):
        mock_client = MagicMock()
        mock_client.graphql_query.return_value = {"data": {}}

        with patch("ml_dash.cli_commands.api.RemoteClient", return_value=mock_client):
            result = cmd_api(self._args(query="me { username }", jq=".data.nonexistent.deep"))

        assert result == 1

    def test_client_exception_returns_1(self):
        mock_client = MagicMock()
        mock_client.graphql_query.side_effect = Exception("Connection refused")

        with patch("ml_dash.cli_commands.api.RemoteClient", return_value=mock_client):
            result = cmd_api(self._args(query="me { username }"))

        assert result == 1

    def test_scalar_result_printed(self):
        mock_client = MagicMock()
        mock_client.graphql_query.return_value = "tom"

        with patch("ml_dash.cli_commands.api.RemoteClient", return_value=mock_client):
            result = cmd_api(self._args(query="me { username }"))

        assert result == 0
