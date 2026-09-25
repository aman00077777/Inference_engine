"""YAML-based configuration with dot-notation access.

Provides ``Config``, a recursive dictionary wrapper that lets callers
navigate nested configuration trees with attribute syntax
(e.g. ``config.model.hidden_dim``) instead of bracket notation.

Expected config params: Any — this class *is* the configuration container.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml

from fusion.exceptions import ConfigError


# Sentinel used by Config.get() to distinguish a missing key from a key
# explicitly set to None in the YAML.
_MISSING = object()


class Config:
    """Recursive dictionary wrapper with dot-notation attribute access.

    All nested ``dict`` values are automatically wrapped in ``Config``
    instances so that arbitrary depth is navigable with plain attribute
    syntax.

    Args:
        data (Dict[str, Any]): Raw configuration dictionary.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = data or {}
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(value: Any) -> Any:
        """Recursively wrap dicts in ``Config`` instances."""
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(item) for item in value]
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        """Recursively unwrap ``Config`` instances back to plain dicts."""
        if isinstance(value, Config):
            return value.to_dict()
        if isinstance(value, list):
            return [Config._unwrap(item) for item in value]
        return value

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Raise a helpful ``ConfigError`` for missing keys.

        Args:
            name (str): Attribute name that was not found.

        Raises:
            ConfigError: Lists the available keys in the current node.
        """
        available = list(self.__dict__.keys())
        raise ConfigError(
            f"Configuration key '{name}' not found. "
            f"Available keys: {available}",
            details={"missing_key": name, "available_keys": available},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation lookup with a fallback default.

        Supports nested keys separated by dots
        (e.g. ``config.get("model.hidden_dim", 256)``).

        Args:
            key (str): Dot-separated key path.
            default (Any): Value returned when the key is absent.

        Returns:
            Any: The resolved value or *default*.
        """
        keys = key.split(".")
        current: Any = self
        for k in keys:
            if isinstance(current, Config):
                current = current.__dict__.get(k, _MISSING)
            elif isinstance(current, dict):
                current = current.get(k, _MISSING)
            else:
                return default
            if current is _MISSING:
                return default
        return current

    def update(self, overrides: Dict[str, Any]) -> None:
        """Mutate the config in place with *overrides*.

        Nested dicts are merged recursively; scalar values are replaced.

        Args:
            overrides (Dict[str, Any]): Key-value pairs to apply.
        """
        for key, value in overrides.items():
            existing = self.__dict__.get(key)
            if isinstance(existing, Config) and isinstance(value, dict):
                existing.update(value)
            else:
                setattr(self, key, self._wrap(value))

    def merge(self, other: "Config") -> "Config":
        """Deep-merge *other* into a **copy** of this config.

        Values from *other* win on conflict.

        Args:
            other (Config): Another ``Config`` to merge.

        Returns:
            Config: A new ``Config`` containing the merged result.
        """
        merged = copy.deepcopy(self.to_dict())
        self._deep_merge(merged, other.to_dict())
        return Config(merged)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a YAML file and return a ``Config``.

        Args:
            path (str): Filesystem path to the YAML file.

        Returns:
            Config: Parsed configuration.

        Raises:
            ConfigError: If the file cannot be read or parsed.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:
            raise ConfigError(
                f"Failed to load config from '{path}': {exc}",
                details={"path": path},
            ) from exc

        if not isinstance(data, dict):
            raise ConfigError(
                f"Expected a YAML mapping at top level, got {type(data).__name__}",
                details={"path": path, "type": type(data).__name__},
            )
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a ``Config`` from a plain dictionary.

        Args:
            data (Dict[str, Any]): Configuration dictionary.

        Returns:
            Config: Wrapped configuration.
        """
        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain nested dictionary.

        Returns:
            Dict[str, Any]: Unwrapped dictionary.
        """
        return {
            key: self._unwrap(value)
            for key, value in self.__dict__.items()
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Recursively merge *override* into *base* (in place)."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def keys(self):
        """Return config keys."""
        return self.__dict__.keys()

    def values(self):
        """Return config values."""
        return self.__dict__.values()

    def items(self):
        """Return config key-value pairs."""
        return self.__dict__.items()
