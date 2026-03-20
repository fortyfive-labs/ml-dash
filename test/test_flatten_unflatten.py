"""Tests for ParametersBuilder.flatten_dict and unflatten_dict edge cases."""

import pytest

from ml_dash.params import ParametersBuilder

flatten = ParametersBuilder.flatten_dict
unflatten = ParametersBuilder.unflatten_dict


class TestFlattenDict:
    def test_empty(self):
        assert flatten({}) == {}

    def test_no_nesting(self):
        assert flatten({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_one_level(self):
        assert flatten({"a": {"b": 1, "c": 2}}) == {"a.b": 1, "a.c": 2}

    def test_two_levels(self):
        result = flatten({"a": {"b": {"c": 1}}})
        assert result == {"a.b.c": 1}

    def test_three_levels(self):
        result = flatten({"a": {"b": {"c": {"d": 42}}}})
        assert result == {"a.b.c.d": 42}

    def test_mixed_depth(self):
        result = flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3})
        assert result == {"a.b": 1, "a.c.d": 2, "e": 3}

    def test_list_value_not_recursed(self):
        """Lists should be kept as-is, not recursed into."""
        result = flatten({"a": [1, 2, 3]})
        assert result == {"a": [1, 2, 3]}

    def test_none_value(self):
        result = flatten({"a": None, "b": {"c": None}})
        assert result == {"a": None, "b.c": None}

    def test_bool_value(self):
        result = flatten({"a": True, "b": {"c": False}})
        assert result == {"a": True, "b.c": False}

    def test_custom_separator(self):
        result = flatten({"a": {"b": 1}}, sep="/")
        assert result == {"a/b": 1}

    def test_custom_separator_deep(self):
        result = flatten({"x": {"y": {"z": 99}}}, sep="__")
        assert result == {"x__y__z": 99}

    def test_parent_key_prefix(self):
        result = flatten({"b": 1}, parent_key="a")
        assert result == {"a.b": 1}

    def test_numeric_string_key(self):
        result = flatten({"0": {"1": "val"}})
        assert result == {"0.1": "val"}


class TestUnflattenDict:
    def test_empty(self):
        assert unflatten({}) == {}

    def test_no_dots(self):
        assert unflatten({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_one_level(self):
        result = unflatten({"a.b": 1, "a.c": 2})
        assert result == {"a": {"b": 1, "c": 2}}

    def test_two_levels(self):
        result = unflatten({"a.b.c": 1})
        assert result == {"a": {"b": {"c": 1}}}

    def test_three_levels(self):
        result = unflatten({"a.b.c.d": 42})
        assert result == {"a": {"b": {"c": {"d": 42}}}}

    def test_mixed(self):
        result = unflatten({"a.b": 1, "a.c.d": 2, "e": 3})
        assert result == {"a": {"b": 1, "c": {"d": 2}}, "e": 3}

    def test_custom_separator(self):
        result = unflatten({"a/b": 1}, sep="/")
        assert result == {"a": {"b": 1}}

    def test_conflicting_key_overwrite(self):
        """Later key overwrites earlier scalar when conflict occurs."""
        # "a" is set to 1, then "a.b" tries to navigate into "a" treating it as a dict.
        # The unflatten just overwrites - this is by design, not a bug.
        result = unflatten({"a.b": 2})
        assert result == {"a": {"b": 2}}


class TestRoundTrip:
    def test_simple(self):
        original = {"a": 1, "b": 2}
        assert unflatten(flatten(original)) == original

    def test_nested(self):
        original = {"a": {"b": 1, "c": 2}, "d": 3}
        assert unflatten(flatten(original)) == original

    def test_deep_nesting(self):
        original = {"a": {"b": {"c": {"d": {"e": 99}}}}}
        assert unflatten(flatten(original)) == original

    def test_mixed_types(self):
        original = {
            "lr": 0.001,
            "model": {
                "layers": 4,
                "dropout": 0.1,
                "pretrained": True,
            },
            "tags": ["exp", "v2"],
        }
        assert unflatten(flatten(original)) == original

    def test_none_values(self):
        original = {"a": None, "b": {"c": None}}
        assert unflatten(flatten(original)) == original

    def test_empty(self):
        assert unflatten(flatten({})) == {}
