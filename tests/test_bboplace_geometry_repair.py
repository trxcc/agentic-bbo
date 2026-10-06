import math

import pytest

from bbo.tasks.bboplace.geometry_repair import Macro, RepairResult, NoLegalPlacement, freeze_worst_initialization, repair_geometry, validate_layout


def test_repair_returns_complete_nonoverlapping_layout():
    macros = (Macro("a", 2, 2, 0), Macro("b", 2, 2, 1))
    result = repair_geometry({"a": (0, 0), "b": (0, 0)}, macros, grid_width=4, grid_height=2)
    assert not result.used_fallback
    assert set(result.positions) == {"a", "b"}
    assert result.positions["a"] != result.positions["b"]


def test_repair_uses_frozen_fallback_when_no_cell_exists():
    fallback = RepairResult({"a": (0, 0), "b": (2, 0)}, False, 0)
    macros = (Macro("a", 2, 2, 0), Macro("b", 2, 2, 1))
    result = repair_geometry({"a": (1, 0), "b": (0, 0)}, macros, grid_width=4, grid_height=2, fallback=fallback)
    assert result.positions == fallback.positions
    assert result.used_fallback
    assert not fallback.used_fallback


def test_fractional_sizes_round_up_and_floor_coordinates():
    macros = (Macro("a", 1.1, 1.2, 0), Macro("b", 1.1, 1.2, 1))
    result = repair_geometry({"a": (0.9, 0.9), "b": (0.9, 0.9)}, macros, grid_width=4, grid_height=2)
    assert result.positions == {"a": (0, 0), "b": (2, 0)}
    again = repair_geometry(result.positions, macros, grid_width=4, grid_height=2)
    assert again.positions == result.positions
    assert again.moved_macros == 0


def test_physical_manhattan_distance_and_boundary_contact():
    macros = (Macro("a", 1, 1, 0), Macro("b", 1, 1, 1))
    result = repair_geometry({"a": (0, 0), "b": (0, 0)}, macros, grid_width=2, grid_height=2, distance_scale=(1, 10))
    assert result.positions["b"] == (1, 0)


def test_bad_fallback_is_not_accepted():
    with pytest.raises(ValueError, match="every macro"):
        repair_geometry({"a": (1, 0), "b": (0, 0)}, [Macro("a", 2, 2, 0), Macro("b", 2, 2, 1)],
                        grid_width=4, grid_height=2, fallback=RepairResult({"a": (0, 0)}, False, 0))


def test_preparation_failure_raises_without_returning_empty_layout():
    with pytest.raises(NoLegalPlacement):
        repair_geometry({"a": (1, 0), "b": (0, 0)}, [Macro("a", 2, 2, 0), Macro("b", 2, 2, 1)],
                        grid_width=4, grid_height=2)


def test_worst_initialization_is_maximum_legal_score():
    good = RepairResult({"a": (0, 0)}, False, 0)
    bad = RepairResult({"a": (1, 1)}, False, 1)
    invalid = RepairResult({}, True, 0)
    assert freeze_worst_initialization([(1.0, good), (3.0, bad), (99.0, invalid), (math.inf, good)]) == bad
