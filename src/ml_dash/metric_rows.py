"""
Raw metric rows: committed metric data read back as the server stores it.

Response format ``ml-dash.metric-rows.v1`` of
``GET /api/experiments/:expId/metrics/:metricName/rows``. A page holds blocks;
each block has its own columns, which may differ between blocks and may repeat a
name. Rows are therefore kept as lists addressed by column position, never as
dicts keyed by name. Rows are in storage order (neither step nor arrival order)
and include only committed data, not points still in the server's hot tail.

Cells decode to Python values:

- float16/32/64: float. The strings "NaN", "Infinity", "-Infinity" and "-0"
  become the matching floats.
- int8..int64, uint8..uint64: int (64-bit values arrive as decimal strings).
- bool: bool. utf8: str.
- null: None. A column absent from a block is simply not in its columns.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .exceptions import MetricRowsError

METRIC_ROWS_FORMAT = "ml-dash.metric-rows.v1"

_FLOAT_TYPES = {"float16", "float32", "float64"}
_INT_TYPES = {"int8", "int16", "int32", "uint8", "uint16", "uint32"}
_INT64_PATTERNS = {
    "int64": re.compile(r"-?(0|[1-9][0-9]*)"),
    "uint64": re.compile(r"0|[1-9][0-9]*"),
}
_FLOAT_SENTINELS = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf, "-0": -0.0}
COLUMN_TYPES = _FLOAT_TYPES | _INT_TYPES | set(_INT64_PATTERNS) | {"bool", "utf8"}


@dataclass(frozen=True)
class MetricColumn:
    """One column of a block: its name (not unique) and wire type, e.g. "float64"."""

    name: str
    type: str


@dataclass(frozen=True)
class MetricRowBlock:
    """Rows sharing one schema. ``rows[i][j]`` is the cell of ``columns[j]``."""

    columns: Tuple[MetricColumn, ...]
    rows: List[List[Any]]


@dataclass(frozen=True)
class MetricRowsPage:
    """
    One page of a raw rows read.

    ``returned`` is the row count of this page only; no total is known. Pass
    ``next_cursor`` to read the next page; it is None when ``has_more`` is False.
    """

    metric_id: str
    blocks: List[MetricRowBlock]
    returned: int
    has_more: bool
    next_cursor: Optional[str]
    visibility: str  # always "committed" in v1
    order: str  # always "storage" in v1


def _decode_cell(type_: str, value: Any) -> Any:
    if value is None:
        return None
    if type_ in _FLOAT_TYPES:
        if isinstance(value, str) and value in _FLOAT_SENTINELS:
            return _FLOAT_SENTINELS[value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif type_ in _INT64_PATTERNS:
        if isinstance(value, str) and _INT64_PATTERNS[type_].fullmatch(value):
            return int(value)
    elif type_ in _INT_TYPES:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif type_ == "bool":
        if isinstance(value, bool):
            return value
    elif isinstance(value, str):  # utf8
        return value
    raise ValueError(f"cell does not match column type {type_}")


def parse_metric_rows_page(body: Any, status_code: int) -> MetricRowsPage:
    """
    Check a rows response and decode its cells.

    Only what could otherwise yield wrong or missing rows is checked: the format,
    column types, row widths, cells against their types, ``returned`` against the
    rows present, and the cursor against ``hasMore``.

    Raises:
        MetricRowsError: with ``code=None`` if the response breaks the contract.
    """

    def malformed(why: str) -> MetricRowsError:
        return MetricRowsError(f"Malformed metric rows response: {why}", status_code)

    if not isinstance(body, dict):
        raise malformed("not a JSON object")
    if body.get("format") != METRIC_ROWS_FORMAT:
        raise malformed(f"format is {body.get('format')!r}, expected {METRIC_ROWS_FORMAT!r}")
    if body.get("visibility") != "committed" or body.get("order") != "storage":
        raise malformed("unexpected visibility or order")
    if not isinstance(body.get("metricId"), str) or not isinstance(body.get("blocks"), list):
        raise malformed("metricId or blocks missing")

    blocks = []
    for b, block in enumerate(body["blocks"]):
        if not isinstance(block, dict) or not isinstance(block.get("columns"), list) \
                or not isinstance(block.get("rows"), list):
            raise malformed(f"block {b} has no columns or rows")
        columns = []
        for column in block["columns"]:
            if not isinstance(column, dict) or not isinstance(column.get("name"), str) \
                    or column.get("type") not in COLUMN_TYPES:
                raise malformed(f"block {b} has a column with no name or an unknown type")
            columns.append(MetricColumn(column["name"], column["type"]))
        rows = []
        for r, row in enumerate(block["rows"]):
            if not isinstance(row, list) or len(row) != len(columns):
                raise malformed(f"block {b} row {r} does not match its {len(columns)} columns")
            try:
                rows.append([_decode_cell(c.type, v) for c, v in zip(columns, row)])
            except ValueError as e:
                raise malformed(f"block {b} row {r}: {e}") from None
        blocks.append(MetricRowBlock(tuple(columns), rows))

    returned = body.get("returned")
    has_more = body.get("hasMore")
    next_cursor = body.get("nextCursor")
    if type(returned) is not int or returned != sum(len(block.rows) for block in blocks):
        raise malformed("returned does not match the rows in the page")
    if not isinstance(has_more, bool) or (
        (not isinstance(next_cursor, str) or not next_cursor) if has_more else next_cursor is not None
    ):
        raise malformed("hasMore and nextCursor disagree")

    return MetricRowsPage(
        metric_id=body["metricId"],
        blocks=blocks,
        returned=returned,
        has_more=has_more,
        next_cursor=next_cursor,
        visibility=body["visibility"],
        order=body["order"],
    )


def error_from_response(response: Any) -> MetricRowsError:
    """A MetricRowsError for a non-2xx rows response, keeping its status and code."""
    code = message = None
    try:
        body: Dict[str, Any] = response.json()
        if isinstance(body, dict):
            code = body.get("code") if isinstance(body.get("code"), str) else None
            message = body.get("message") if isinstance(body.get("message"), str) else None
    except ValueError:
        pass
    text = f"Metric rows read failed: HTTP {response.status_code}"
    if code:
        text += f" ({code})"
    if message:
        text += f": {message}"
    if response.status_code == 404 and code is None:
        text += " (this server may not support raw metric row reads)"
    return MetricRowsError(text, response.status_code, code)
