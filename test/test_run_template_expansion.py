"""Tests for RUN template expansion in __post_init__ and __setattr__."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ml_dash.run import RUN


def _reset_counter(value=0):
    RUN.job_counter = value


class TestJobCounterExpansion:
    def test_job_counter_expands_in_prefix(self, tmpdir):
        """job_counter placeholder is replaced by the instance's counter value."""
        _reset_counter(1)
        run = RUN(
            prefix="tom/proj/{job_counter:03d}",
            entry=str(tmpdir),
            project_root=str(tmpdir),
        )
        # Instance receives counter=1 from class default; __post_init__ later increments class
        assert "{job_counter" not in run.prefix
        assert "001" in run.prefix

    def test_job_counter_increments_across_instances(self, tmpdir):
        """Each RUN instance gets a distinct counter value."""
        _reset_counter(0)
        run1 = RUN(entry=str(tmpdir), project_root=str(tmpdir),
                   prefix="tom/p/{job_counter:03d}")
        run2 = RUN(entry=str(tmpdir), project_root=str(tmpdir),
                   prefix="tom/p/{job_counter:03d}")
        assert run1.prefix != run2.prefix

    def test_prefix_without_braces_unchanged(self, tmpdir):
        """A prefix with no template syntax is left as-is."""
        _reset_counter(0)
        run = RUN(
            prefix="tom/project/baseline",
            entry=str(tmpdir),
            project_root=str(tmpdir),
        )
        assert run.prefix == "tom/project/baseline"

    def test_now_format_expands(self, tmpdir):
        """now datetime placeholder is expanded."""
        _reset_counter(0)
        fixed = datetime(2026, 3, 1, 12, 0, 0)
        run = RUN(
            prefix="tom/proj/{now:%Y-%m-%d}",
            entry=str(tmpdir),
            project_root=str(tmpdir),
            now=fixed,
        )
        assert "{now" not in run.prefix
        assert "2026-03-01" in run.prefix

    def test_multiple_placeholders_expand(self, tmpdir):
        """Multiple placeholders all expand."""
        _reset_counter(5)
        fixed = datetime(2026, 1, 1, 0, 0, 0)
        run = RUN(
            prefix="tom/proj/{now:%Y}/{job_counter:03d}",
            entry=str(tmpdir),
            project_root=str(tmpdir),
            now=fixed,
        )
        assert "{" not in run.prefix
        assert "2026" in run.prefix
        assert "005" in run.prefix


class TestExpTemplateExpansion:
    def test_exp_id_generates_id(self, tmpdir):
        """{EXP.id} in prefix triggers snowflake ID generation."""
        _reset_counter(0)
        run = RUN(
            prefix="tom/proj/{EXP.id}",
            entry=str(tmpdir),
            project_root=str(tmpdir),
        )
        assert "{EXP.id}" not in run.prefix
        assert len(run.prefix.split("/")[-1]) > 0

    def test_exp_id_is_non_empty_string_in_prefix(self, tmpdir):
        """{EXP.id} is replaced with a non-empty generated ID."""
        _reset_counter(0)
        run = RUN(
            prefix="tom/proj/{EXP.id}",
            entry=str(tmpdir),
            project_root=str(tmpdir),
        )
        id_in_prefix = run.prefix.split("/")[-1]
        assert len(id_in_prefix) > 0
        assert id_in_prefix.isdigit()  # snowflake IDs are numeric strings

    def test_exp_unknown_attr_raises(self, tmpdir):
        """{EXP.nonexistent} raises AttributeError."""
        _reset_counter(0)
        with pytest.raises(AttributeError):
            RUN(
                prefix="tom/proj/{EXP.nonexistent_attr}",
                entry=str(tmpdir),
                project_root=str(tmpdir),
            )
