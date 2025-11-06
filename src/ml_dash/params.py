"""
Parameters module for ML-Dash SDK.

Provides fluent API for parameter management with automatic dict flattening.
Nested dicts are flattened to dot-notation: {"model": {"lr": 0.001}} → {"model.lr": 0.001}
"""

from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import Experiment


class ParametersBuilder:
    """
    Fluent interface for parameter operations.

    Usage:
        experiment.parameters().set(model={"lr": 0.001}, optimizer="adam")
        params = experiment.parameters().get()
        params_nested = experiment.parameters().get(flatten=False)
    """

    def __init__(self, experiment: 'Experiment'):
        """
        Initialize parameters builder.

        Args:
            experiment: Parent experiment instance
        """
        self._experiment = experiment

    def set(self, **kwargs) -> 'ParametersBuilder':
        """
        Set/merge parameters. Always merges with existing parameters (upsert behavior).

        Nested dicts are automatically flattened:
            set(model={"lr": 0.001, "batch_size": 32})
            → {"model.lr": 0.001, "model.batch_size": 32}

        Args:
            **kwargs: Parameters to set (can be nested dicts)

        Returns:
            Self for potential chaining

        Raises:
            RuntimeError: If experiment is not open
            RuntimeError: If experiment is write-protected

        Examples:
            # Set nested parameters
            experiment.parameters().set(
                model={"lr": 0.001, "batch_size": 32},
                optimizer="adam"
            )

            # Merge/update specific parameters
            experiment.parameters().set(model={"lr": 0.0001})  # Only updates model.lr

            # Set flat parameters with dot notation
            experiment.parameters().set(**{"model.lr": 0.001, "model.batch_size": 32})
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.run.start() or context manager.")

        if self._experiment._write_protected:
            raise RuntimeError("Experiment is write-protected and cannot be modified.")

        # Flatten the kwargs
        flattened = self.flatten_dict(kwargs)

        if not flattened:
            # No parameters to set, just return
            return self

        # Write parameters through experiment
        self._experiment._write_params(flattened)

        return self

    def get(self, flatten: bool = True) -> Dict[str, Any]:
        """
        Get parameters from the experiment.

        Args:
            flatten: If True, returns flattened dict with dot notation.
                    If False, returns nested dict structure.

        Returns:
            Parameters dict (flattened or nested based on flatten arg)

        Raises:
            RuntimeError: If experiment is not open

        Examples:
            # Get flattened parameters
            params = experiment.parameters().get()
            # → {"model.lr": 0.001, "model.batch_size": 32, "optimizer": "adam"}

            # Get nested parameters
            params = experiment.parameters().get(flatten=False)
            # → {"model": {"lr": 0.001, "batch_size": 32}, "optimizer": "adam"}
        """
        if not self._experiment._is_open:
            raise RuntimeError("Experiment not open. Use experiment.open() or context manager.")

        # Read parameters through experiment
        params = self._experiment._read_params()

        if params is None:
            return {}

        # Return as-is if flatten=True (stored flattened), or unflatten if needed
        if flatten:
            return params
        else:
            return self.unflatten_dict(params)

    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary into dot-notation keys.

        Args:
            d: Dictionary to flatten (can contain nested dicts)
            parent_key: Prefix for keys (used in recursion)
            sep: Separator character (default: '.')

        Returns:
            Flattened dictionary with dot-notation keys

        Examples:
            >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
            {"a.b": 1, "a.c": 2, "d": 3}

            >>> flatten_dict({"model": {"lr": 0.001, "layers": {"hidden": 128}}})
            {"model.lr": 0.001, "model.layers.hidden": 128}
        """
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                # Recursively flatten nested dicts
                items.extend(ParametersBuilder.flatten_dict(v, new_key, sep=sep).items())
            else:
                # Keep non-dict values as-is
                items.append((new_key, v))

        return dict(items)

    @staticmethod
    def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """
        Unflatten a dot-notation dictionary into nested structure.

        Args:
            d: Flattened dictionary with dot-notation keys
            sep: Separator character (default: '.')

        Returns:
            Nested dictionary structure

        Examples:
            >>> unflatten_dict({"a.b": 1, "a.c": 2, "d": 3})
            {"a": {"b": 1, "c": 2}, "d": 3}

            >>> unflatten_dict({"model.lr": 0.001, "model.layers.hidden": 128})
            {"model": {"lr": 0.001, "layers": {"hidden": 128}}}
        """
        result = {}

        for key, value in d.items():
            parts = key.split(sep)
            current = result

            # Navigate/create nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the final value
            current[parts[-1]] = value

        return result
