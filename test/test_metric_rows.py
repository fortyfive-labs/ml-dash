"""Tests for raw metric rows reads (F11), against a fake HTTPX transport (no network)."""

import json
import math
from unittest.mock import MagicMock

import httpx
import pytest

from ml_dash import ConfigurationError, MetricRowsError
from ml_dash.client import RemoteClient
from ml_dash.metric import MetricBuilder
from ml_dash.metric_rows import MetricColumn


def _page(blocks, has_more=False, next_cursor=None, **overrides):
  body = {
    "format": "ml-dash.metric-rows.v1",
    "metricId": "42",
    "visibility": "committed",
    "order": "storage",
    "blocks": blocks,
    "returned": sum(len(b["rows"]) for b in blocks),
    "hasMore": has_more,
    "nextCursor": next_cursor,
  }
  body.update(overrides)
  return body


def _client(responses, requests=None):
  """A RemoteClient whose REST calls get the given (status, body) responses in turn."""
  responses = list(responses)

  def handler(request):
    if requests is not None:
      requests.append(request)
    status, body = responses.pop(0)
    if isinstance(body, str):
      return httpx.Response(status, text=body)
    return httpx.Response(status, json=body)

  client = RemoteClient("http://stub.invalid", namespace="tom", api_key="fake-token-for-tests")
  client._rest_client = httpx.Client(
    base_url=client.base_url, transport=httpx.MockTransport(handler)
  )
  return client


class TestReadMetricRows:

  def test_decodes_cells_and_keeps_block_schemas_and_duplicate_columns(self):
    requests = []
    blocks = [
      {
        "columns": [{"name": "step", "type": "int64"}, {"name": "v", "type": "float64"},
                    {"name": "v", "type": "float32"}, {"name": "big", "type": "uint64"}],
        "rows": [
          ["9007199254740993", "NaN", "-0", "18446744073709551615"],
          ["-5", 1, "Infinity", None],
          ["-5", 1, "Infinity", None],  # duplicate rows are kept
        ],
      },
      {
        "columns": [{"name": "ok", "type": "bool"}, {"name": "tag", "type": "utf8"},
                    {"name": "n", "type": "int32"}, {"name": "h", "type": "float16"}],
        "rows": [[True, "a", 3, "-Infinity"]],
      },
    ]
    client = _client([(200, _page(blocks, True, "c1"))], requests)

    page = client.read_metric_rows("1", "train/loss", limit=4, cursor="c0")

    assert requests[0].url.raw_path == b"/api/experiments/1/metrics/train%2Floss/rows?limit=4&cursor=c0"
    assert (page.metric_id, page.returned, page.has_more, page.next_cursor) == ("42", 4, True, "c1")
    first, second = page.blocks
    assert first.columns == (MetricColumn("step", "int64"), MetricColumn("v", "float64"),
                             MetricColumn("v", "float32"), MetricColumn("big", "uint64"))
    step, nan, neg_zero, big = first.rows[0]
    assert step == 2**53 + 1 and big == 2**64 - 1
    assert math.isnan(nan) and neg_zero == 0.0 and math.copysign(1.0, neg_zero) == -1.0
    assert first.rows[1] == first.rows[2] == [-5, 1.0, math.inf, None]
    assert isinstance(first.rows[1][1], float)
    assert [c.name for c in second.columns] == ["ok", "tag", "n", "h"]
    assert second.rows == [[True, "a", 3, -math.inf]]

  def test_empty_metric(self):
    page = _client([(200, _page([]))]).read_metric_rows("1", "train")
    assert page.blocks == [] and page.returned == 0 and page.next_cursor is None

  @pytest.mark.parametrize("status, body, code", [
    (409, {"error": "Conflict", "code": "snapshot_changed", "message": "restart the read"}, "snapshot_changed"),
    (404, {"error": "Not Found", "code": "metric_not_found", "message": "Metric not found"}, "metric_not_found"),
    (400, {"error": "Bad Request", "code": "invalid_cursor", "message": "bad cursor"}, "invalid_cursor"),
    (404, {"message": "Route GET:/api/x not found", "statusCode": 404}, None),  # no rows route
    (502, "<html>bad gateway</html>", None),
  ])
  def test_http_errors_keep_status_and_code(self, status, body, code):
    requests = []
    client = _client([(status, body)], requests)
    with pytest.raises(MetricRowsError) as excinfo:
      client.read_metric_rows("1", "train", cursor="c1")
    assert (excinfo.value.status_code, excinfo.value.code) == (status, code)
    assert len(requests) == 1  # never restarted without the cursor

  @pytest.mark.parametrize("damage", [
    {"format": "ml-dash.metric-rows.v2"},
    {"returned": 1},  # fewer rows than claimed: a partial page
    {"hasMore": True},  # no cursor to continue with
    {"nextCursor": "c1"},  # cursor on the last page
    {"blocks": [{"columns": [{"name": "x", "type": "float64"}], "rows": [[1.0, 2.0]]}], "returned": 1},
    {"blocks": [{"columns": [{"name": "x", "type": "decimal"}], "rows": []}]},
    {"blocks": [{"columns": [{"name": "x", "type": "int64"}], "rows": [[5]]}], "returned": 1},
    {"blocks": [{"columns": [{"name": "x", "type": "float64"}], "rows": [["nan"]]}], "returned": 1},
    {"blocks": [{"columns": [{"name": "x", "type": "int32"}], "rows": [[True]]}], "returned": 1},
  ])
  def test_malformed_response_is_rejected(self, damage):
    body = _page([])
    body.update(damage)
    client = _client([(200, body)])
    with pytest.raises(MetricRowsError, match="Malformed") as excinfo:
      client.read_metric_rows("1", "train")
    assert (excinfo.value.status_code, excinfo.value.code) == (200, None)


class TestLegacyIndexReads:

  @pytest.mark.parametrize("call", [
    lambda c: c.read_metric_data("1", "train", start_index=5),
    lambda c: c.get_metric_data("1", "train", start_index=5),
    lambda c: c.download_metric_chunk("1", "train", 0),
  ])
  def test_remote_index_reads_raise_without_request(self, call):
    requests = []
    with pytest.raises(ConfigurationError, match="read_rows"):
      call(_client([], requests))
    assert requests == []

  def test_remote_only_read_raises(self, remote_experiment):
    exp = remote_experiment()
    exp._experiment_id = "1"
    with pytest.raises(ConfigurationError, match="read_rows"):
      MetricBuilder(exp, "train").read(start_index=0, limit=10)

  def test_hybrid_read_uses_local_storage(self, local_experiment):
    with local_experiment("tom/test/rows-hybrid").run as exp:
      exp.metrics("m").log(step=0)
      exp.metrics("m").log(step=1)
      exp.flush()
      exp._client = MagicMock()
      result = exp.metrics("m").read(start_index=1, limit=10)
      exp._client = None
    assert [p["data"]["step"] for p in result["data"]] == [1]


class TestMetricBuilderRows:

  def test_iter_row_blocks_follows_cursors(self, remote_experiment):
    requests = []
    block = lambda i: {"columns": [{"name": "step", "type": "int64"}], "rows": [[str(i)]]}
    exp = remote_experiment()
    exp._experiment_id = "1"
    exp._client = _client([
      (200, _page([block(0), block(1)], True, "c1")),
      (200, _page([], True, "c2")),  # a page may hold no rows and still continue
      (200, _page([block(2)])),
    ], requests)

    blocks = list(MetricBuilder(exp, "train").iter_row_blocks(limit=2))

    assert [b.rows for b in blocks] == [[[0]], [[1]], [[2]]]
    assert [r.url.params.get("cursor") for r in requests] == [None, "c1", "c2"]
    assert {r.url.params["limit"] for r in requests} == {"2"}

  def test_iter_row_blocks_raises_on_snapshot_change_without_restart(self, remote_experiment):
    requests = []
    exp = remote_experiment()
    exp._experiment_id = "1"
    exp._client = _client([
      (200, _page([{"columns": [], "rows": [[]]}], True, "c1")),
      (409, {"code": "snapshot_changed", "message": "restart the read without a cursor"}),
    ], requests)

    blocks = MetricBuilder(exp, "train").iter_row_blocks()
    assert next(blocks).rows == [[]]
    with pytest.raises(MetricRowsError) as excinfo:
      next(blocks)
    assert excinfo.value.code == "snapshot_changed"
    assert len(requests) == 2

  def test_read_rows_needs_a_server(self, local_experiment):
    with local_experiment("tom/test/rows-local").run as exp:
      with pytest.raises(ConfigurationError, match="local-only"):
        exp.metrics("m").read_rows()
