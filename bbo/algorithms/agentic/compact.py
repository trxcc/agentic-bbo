"""Compact candidate formats for agentic optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...core import SearchSpace

COMPACT_COORD_DECIMALS = 4


def bboplace_xy_names(search_space: SearchSpace) -> tuple[list[str], list[str]] | None:
    """Return paired BBOPlace x/y parameter names when the space has that shape."""

    names = set(search_space.names())
    n_macro = 0
    while f"x_{n_macro}" in names and f"y_{n_macro}" in names:
        n_macro += 1
    if n_macro <= 0:
        return None
    x_names = [f"x_{index}" for index in range(n_macro)]
    y_names = [f"y_{index}" for index in range(n_macro)]
    if names != set(x_names + y_names):
        return None
    return x_names, y_names


def bboplace_macro_count(search_space: SearchSpace) -> int | None:
    xy_names = bboplace_xy_names(search_space)
    if xy_names is None:
        return None
    x_names, _ = xy_names
    return len(x_names)


def is_compact_xy_space(search_space: SearchSpace) -> bool:
    return bboplace_xy_names(search_space) is not None


def compact_coord(value: Any, *, decimals: int = COMPACT_COORD_DECIMALS) -> float:
    return round(float(value), decimals)


def compact_xy_config(
    search_space: SearchSpace,
    config: Mapping[str, Any],
    *,
    decimals: int = COMPACT_COORD_DECIMALS,
) -> dict[str, list[float]]:
    xy_names = bboplace_xy_names(search_space)
    if xy_names is None:
        raise ValueError("Search space is not a paired BBOPlace x/y space.")
    normalized = search_space.coerce_config(dict(config), use_defaults=False)
    x_names, y_names = xy_names
    return {
        "x": [compact_coord(normalized[name], decimals=decimals) for name in x_names],
        "y": [compact_coord(normalized[name], decimals=decimals) for name in y_names],
    }


def expand_compact_xy_config(search_space: SearchSpace, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    xy_names = bboplace_xy_names(search_space)
    if xy_names is None or "x" not in candidate or "y" not in candidate:
        return None
    raw_x = candidate["x"]
    raw_y = candidate["y"]
    if not _is_json_array(raw_x) or not _is_json_array(raw_y):
        return None
    x_values = list(raw_x)
    y_values = list(raw_y)
    x_names, y_names = xy_names
    if len(x_values) != len(x_names) or len(y_values) != len(y_names):
        return None
    expanded: dict[str, Any] = {}
    for name, value in zip(x_names, x_values, strict=True):
        expanded[name] = float(value)
    for name, value in zip(y_names, y_values, strict=True):
        expanded[name] = float(value)
    return expanded


def _is_json_array(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = [
    "COMPACT_COORD_DECIMALS",
    "bboplace_macro_count",
    "bboplace_xy_names",
    "compact_coord",
    "compact_xy_config",
    "expand_compact_xy_config",
    "is_compact_xy_space",
]
