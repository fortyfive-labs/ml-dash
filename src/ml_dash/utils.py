"""
Shared utility functions for ml-dash SDK.
"""

from typing import Any


def _serialize_value(value: Any) -> Any:
    """
    Convert value to JSON-serializable format.

    Handles numpy arrays, numpy scalars, nested dicts/lists, etc.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable value
    """
    # numpy array → list
    if hasattr(value, '__array__') or (hasattr(value, 'tolist') and hasattr(value, 'dtype')):
        try:
            return value.tolist()
        except AttributeError:
            pass

    # numpy scalar → Python scalar
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass

    # Recursively handle dicts
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Recursively handle lists/tuples
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    # int, float, str, bool, None — pass through
    return value
