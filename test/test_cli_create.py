"""Tests for cli_commands/create.py — argument parsing and project creation."""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from ml_dash.cli_commands.create import _create_project, cmd_create


def _args(project, description=None, dash_url=None):
    return argparse.Namespace(project=project, description=description, dash_url=dash_url)


def _quiet_console():
    return Console(quiet=True)


class TestCmdCreateArgumentParsing:
    def test_three_part_project_returns_1(self):
        result = cmd_create(_args("ns/proj/extra"))
        assert result == 1

    def test_one_part_uses_null_namespace(self):
        with patch("ml_dash.cli_commands.create._create_project") as mock_fn:
            mock_fn.return_value = 0
            cmd_create(_args("my-project"))

        assert mock_fn.call_args.kwargs["namespace"] is None
        assert mock_fn.call_args.kwargs["project_name"] == "my-project"

    def test_two_part_splits_namespace_and_name(self):
        with patch("ml_dash.cli_commands.create._create_project") as mock_fn:
            mock_fn.return_value = 0
            cmd_create(_args("tom/my-project"))

        assert mock_fn.call_args.kwargs["namespace"] == "tom"
        assert mock_fn.call_args.kwargs["project_name"] == "my-project"

    def test_leading_trailing_slashes_stripped(self):
        with patch("ml_dash.cli_commands.create._create_project") as mock_fn:
            mock_fn.return_value = 0
            cmd_create(_args("/my-project/"))

        assert mock_fn.call_args.kwargs["project_name"] == "my-project"

    def test_description_forwarded(self):
        with patch("ml_dash.cli_commands.create._create_project") as mock_fn:
            mock_fn.return_value = 0
            cmd_create(_args("tom/proj", description="A test project"))

        assert mock_fn.call_args.kwargs["description"] == "A test project"


class TestCreateProject:
    def _mock_client(self, namespace="tom", response_json=None):
        mock_response = MagicMock()
        mock_response.json.return_value = response_json or {"id": "abc123", "slug": "my-project"}
        mock_response.raise_for_status.return_value = None

        mock_http = MagicMock()
        mock_http.post.return_value = mock_response

        mock_client = MagicMock()
        mock_client._client = mock_http
        mock_client.namespace = namespace

        return mock_client

    def test_success_returns_0(self):
        mock_client = self._mock_client()

        with patch("ml_dash.cli_commands.create.RemoteClient", return_value=mock_client):
            result = _create_project(
                namespace="tom", project_name="my-project",
                description=None, dash_url="http://localhost:3000",
                console=_quiet_console(),
            )

        assert result == 0

    def test_no_namespace_returns_1(self):
        mock_client = self._mock_client(namespace=None)

        with patch("ml_dash.cli_commands.create.RemoteClient", return_value=mock_client):
            result = _create_project(
                namespace=None, project_name="my-project",
                description=None, dash_url="http://localhost:3000",
                console=_quiet_console(),
            )

        assert result == 1

    def test_conflict_409_returns_0(self):
        conflict_exc = Exception("Conflict")
        conflict_response = MagicMock()
        conflict_response.status_code = 409
        conflict_exc.response = conflict_response

        mock_http = MagicMock()
        mock_http.post.return_value.raise_for_status.side_effect = conflict_exc

        mock_client = MagicMock()
        mock_client._client = mock_http
        mock_client.namespace = "tom"

        with patch("ml_dash.cli_commands.create.RemoteClient", return_value=mock_client):
            result = _create_project(
                namespace="tom", project_name="my-project",
                description=None, dash_url="http://localhost:3000",
                console=_quiet_console(),
            )

        assert result == 0

    def test_other_exception_returns_1(self):
        mock_client = self._mock_client()
        mock_client._client.post.side_effect = Exception("Network error")

        with patch("ml_dash.cli_commands.create.RemoteClient", return_value=mock_client):
            result = _create_project(
                namespace="tom", project_name="my-project",
                description=None, dash_url="http://localhost:3000",
                console=_quiet_console(),
            )

        assert result == 1

    def test_description_included_in_request(self):
        mock_client = self._mock_client()

        with patch("ml_dash.cli_commands.create.RemoteClient", return_value=mock_client):
            _create_project(
                namespace="tom", project_name="my-project",
                description="Test description",
                dash_url="http://localhost:3000",
                console=_quiet_console(),
            )

        post_kwargs = mock_client._client.post.call_args
        body = post_kwargs.kwargs.get("json") or post_kwargs[1].get("json") or post_kwargs[0][1]
        assert body["description"] == "Test description"
