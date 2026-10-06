"""Netlist-blind deterministic placement repair, in grid coordinates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class NoLegalPlacement(ValueError):
    """A greedy repair failed; this does not establish geometric infeasibility."""


@dataclass(frozen=True)
class Macro:
    name: str
    width: float
    height: float
    index: int


@dataclass(frozen=True)
class RepairResult:
    positions: dict[str, tuple[float, float]]
    used_fallback: bool
    moved_macros: int


def validate_layout(positions: Mapping[str, Sequence[float]], macros: Sequence[Macro],
                    *, grid_width: int, grid_height: int) -> None:
    """Check completeness, grid alignment, conservative footprints and bounds."""
    if not macros or len({m.name for m in macros}) != len(macros):
        raise ValueError("Macros must be nonempty and uniquely named")
    if set(positions) != {m.name for m in macros}:
        raise ValueError("Layout must contain every macro exactly once")
    rectangles = []
    for m in macros:
        if not all(math.isfinite(v) and v > 0 for v in (m.width, m.height)):
            raise ValueError("Macro sizes must be finite and positive")
        x, y = positions[m.name]
        if not all(math.isfinite(v) and float(v).is_integer() for v in (x, y)):
            raise ValueError("Layout must contain finite integer grid positions")
        rect = (x, y, x + math.ceil(m.width), y + math.ceil(m.height))
        if x < 0 or y < 0 or rect[2] > grid_width or rect[3] > grid_height:
            raise ValueError("Layout lies outside the canvas")
        for other in rectangles:
            if rect[0] < other[2] and other[0] < rect[2] and rect[1] < other[3] and other[1] < rect[3]:
                raise ValueError("Layout contains overlapping macros")
        rectangles.append(rect)


def repair_geometry(coordinates: Mapping[str, Sequence[float]], macros: Sequence[Macro], *,
                    grid_width: int, grid_height: int,
                    distance_scale: tuple[float, float] = (1.0, 1.0),
                    fallback: RepairResult | None = None) -> RepairResult:
    """Floor inputs and minimize physical L1 displacement; break ties by x/y.

    With a validated fallback every accepted input yields a complete legal
    layout. Without one, failure raises during initialization preparation only.
    No connectivity or objective data can enter this function.
    """
    if grid_width <= 0 or grid_height <= 0 or any(not math.isfinite(v) or v <= 0 for v in distance_scale):
        raise ValueError("Invalid canvas or distance scale")
    names = {m.name for m in macros}
    if not names or len(names) != len(macros) or set(coordinates) != names:
        raise ValueError("Candidate must contain every macro exactly once")
    if any(not math.isfinite(v) or v <= 0 for m in macros for v in (m.width, m.height)):
        raise ValueError("Macro sizes must be finite and positive")
    for xy in coordinates.values():
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError("Coordinates must be finite pairs")
        if not (0 <= xy[0] <= grid_width and 0 <= xy[1] <= grid_height):
            raise ValueError("Candidate coordinates outside declared bounds")
    if fallback is not None:
        validate_layout(fallback.positions, macros, grid_width=grid_width, grid_height=grid_height)
    placed = {}
    rectangles = []
    moved = 0
    for m in sorted(macros, key=lambda item: (-(item.width * item.height), item.index)):
        width, height = math.ceil(m.width), math.ceil(m.height)
        nx, ny = grid_width - width + 1, grid_height - height + 1
        legal = np.ones((max(0, nx), max(0, ny)), dtype=bool)
        for x, y, right, top in rectangles:
            legal[max(0, x - width + 1):min(nx, right), max(0, y - height + 1):min(ny, top)] = False
        xs, ys = np.nonzero(legal)
        if not len(xs):
            if fallback is None:
                raise NoLegalPlacement("Greedy geometry repair has no legal position")
            return RepairResult(dict(fallback.positions), True, len(macros))
        tx, ty = (math.floor(v) for v in coordinates[m.name])
        distances = np.abs(xs - tx) * distance_scale[0] + np.abs(ys - ty) * distance_scale[1]
        choice = int(np.argmin(distances))
        x, y = int(xs[choice]), int(ys[choice])
        placed[m.name] = (float(x), float(y))
        rectangles.append((x, y, x + width, y + height))
        moved += int((x, y) != (tx, ty))
    validate_layout(placed, macros, grid_width=grid_width, grid_height=grid_height)
    return RepairResult(placed, False, moved)


def freeze_worst_initialization(results: Sequence[tuple[float, RepairResult]]) -> RepairResult:
    """Choose the legal initialization with the largest finite HPWL."""
    legal = [(float(score), result) for score, result in results
             if math.isfinite(score) and not result.used_fallback and result.positions]
    if not legal:
        raise ValueError("BBOPlace initialization contains no legal repaired layout")
    return max(legal, key=lambda item: item[0])[1]
