"""Tests for BackgroundBufferManager queue pressure and warning behavior."""

import threading
import warnings
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from ml_dash.buffer import BackgroundBufferManager, BufferConfig


def _make_manager():
    """Create a BackgroundBufferManager with a mock experiment (not started)."""
    mock_exp = MagicMock()
    config = BufferConfig()
    mgr = BackgroundBufferManager(experiment=mock_exp, config=config)
    return mgr


class TestCheckQueuePressure:
    def test_below_threshold_no_warning(self):
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = 0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr._check_queue_pressure(q, "test-queue")

        assert len(caught) == 0

    def test_at_warning_threshold_emits_warning(self):
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = BufferConfig._WARNING_THRESHOLD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr._check_queue_pressure(q, "train-metrics")

        assert len(caught) == 1
        assert caught[0].category == RuntimeWarning
        assert "train-metrics" in str(caught[0].message)

    def test_above_warning_threshold_emits_warning(self):
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = BufferConfig._WARNING_THRESHOLD + 100

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr._check_queue_pressure(q, "logs")

        assert len(caught) == 1
        assert caught[0].category == RuntimeWarning

    def test_warning_emitted_only_once_per_queue(self):
        """Warning is suppressed after the first alert per queue name."""
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = BufferConfig._WARNING_THRESHOLD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr._check_queue_pressure(q, "train-metrics")
            mgr._check_queue_pressure(q, "train-metrics")
            mgr._check_queue_pressure(q, "train-metrics")

        assert len(caught) == 1

    def test_different_queues_each_get_one_warning(self):
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = BufferConfig._WARNING_THRESHOLD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr._check_queue_pressure(q, "train-metrics")
            mgr._check_queue_pressure(q, "eval-metrics")

        assert len(caught) == 2

    def test_at_aggressive_threshold_sets_flush_event(self):
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = BufferConfig._AGGRESSIVE_FLUSH_THRESHOLD

        assert not mgr._flush_event.is_set()
        mgr._check_queue_pressure(q, "train-metrics")
        assert mgr._flush_event.is_set()

    def test_below_aggressive_threshold_no_flush_event(self):
        mgr = _make_manager()
        q = MagicMock(spec=Queue)
        q.qsize.return_value = BufferConfig._AGGRESSIVE_FLUSH_THRESHOLD - 1

        mgr._check_queue_pressure(q, "train-metrics")
        assert not mgr._flush_event.is_set()
