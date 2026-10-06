"""Host-only geometry evaluator for the public no-MGO release.

Bundles contain attested geometry/netlists and frozen, re-evaluated initial
observations. They must never be copied into an agent workspace.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .geometry_repair import Macro, NoLegalPlacement, RepairResult, repair_geometry, validate_layout

PROTOCOL = "geometry_repair_worst_initial_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class PlacementData:
    def __init__(self, data: dict):
        self.data = copy.deepcopy(data)
        self.names = list(data["macro_names"])
        self.nodes = data["nodes"]
        self.nets = data["nets"]
        self.grid = tuple(data["grid"])
        self.canvas = tuple(data["canvas"])
        self.scale = (self.canvas[0] / self.grid[0], self.canvas[1] / self.grid[1])
        self.macros = [Macro(name, self.nodes[name]["size_x"] / self.scale[0],
                             self.nodes[name]["size_y"] / self.scale[1], i)
                       for i, name in enumerate(self.names)]
        self.config_names = [f"x_{i}" for i in range(len(self.names))] + [f"y_{i}" for i in range(len(self.names))]

    def coordinates(self, row) -> dict:
        a = np.asarray(row, dtype=float)
        n = len(self.names)
        if a.shape != (2 * n,) or not np.isfinite(a).all():
            raise ValueError("Expected one complete finite placement vector")
        if np.any(a[:n] < 0) or np.any(a[:n] > self.grid[0]) or np.any(a[n:] < 0) or np.any(a[n:] > self.grid[1]):
            raise ValueError("Placement vector outside declared bounds")
        return {name: (float(a[i]), float(a[i + n])) for i, name in enumerate(self.names)}

    def validate(self, layout) -> None:
        validate_layout(layout, self.macros, grid_width=self.grid[0], grid_height=self.grid[1])

    def vector(self, layout) -> list[float]:
        self.validate(layout)
        return [float(layout[name][axis]) for axis in (0, 1) for name in self.names]

    def config(self, layout) -> dict:
        return dict(zip(self.config_names, self.vector(layout), strict=True))

    def repair(self, row, fallback: RepairResult | None = None) -> RepairResult:
        return repair_geometry(self.coordinates(row), self.macros, grid_width=self.grid[0],
                               grid_height=self.grid[1], distance_scale=self.scale, fallback=fallback)

    def hpwl(self, layout) -> float:
        """Same weighted macro-pin HPWL as frozen upstream comp_res."""
        self.validate(layout)
        total = 0.0
        for net in self.nets.values():
            # Retain upstream accumulator initialization, including pin offsets.
            xmax = ymax = 0.0
            xmin, ymin = self.canvas[0] * 1.1, self.canvas[1] * 1.1
            for name, pin in net["nodes"].items():
                node = self.nodes[name]
                x = layout[name][0] * self.scale[0] + node["size_x"] / 2 + pin["x_offset"]
                y = layout[name][1] * self.scale[1] + node["size_y"] / 2 + pin["y_offset"]
                xmax, xmin, ymax, ymin = max(xmax, x), min(xmin, x), max(ymax, y), min(ymin, y)
            total += ((xmax - xmin) + (ymax - ymin)) * net.get("weight", 1.0)
        if not math.isfinite(total) or total < 0:
            raise ValueError("Invalid HPWL from placement assets")
        return float(total)



def prepare_bundle(data: dict, configs: list[dict], *, seed: int) -> dict:
    """Re-evaluate a shared initialization once, then freeze its worst legal point."""
    task = PlacementData(data)
    observations = []
    for index, config in enumerate(configs):
        if set(config) != set(task.config_names):
            raise ValueError("Initialization configuration does not match placement coordinates")
        row = [config[name] for name in task.config_names]
        try:
            result = task.repair(row)
            observations.append(dict(trial_id=index, config=config, layout=result.positions,
                                     hpwl=task.hpwl(result.positions), repair_fallback=False))
        except NoLegalPlacement:
            observations.append(dict(trial_id=index, config=config, repair_fallback=True))
    legal = [o for o in observations if not o["repair_fallback"]]
    if not legal:
        raise ValueError("No legal shared initialization; refusing to prepare task")
    worst = max(legal, key=lambda o: o["hpwl"])
    reference = dict(trial_id=worst["trial_id"], hpwl=worst["hpwl"], layout=copy.deepcopy(worst["layout"]))
    for o in observations:
        if o["repair_fallback"]:
            o.update(layout=copy.deepcopy(reference["layout"]), hpwl=reference["hpwl"])
    bundle = dict(protocol=PROTOCOL, benchmark=data["benchmark"], seed=seed, data=data,
                  data_sha256=digest(data), initializations=observations, fallback=reference)
    bundle["sha256"] = digest(bundle)
    return bundle


class RepairEvaluator:
    def __init__(self, bundle: dict):
        packet = dict(bundle)
        checksum = packet.pop("sha256")
        if packet["protocol"] != PROTOCOL or digest(packet) != checksum or digest(packet["data"]) != packet["data_sha256"]:
            raise ValueError("Invalid or modified repair bundle")
        self.bundle = packet
        self.sha256 = checksum
        self.data = PlacementData(packet["data"])
        reference = packet["fallback"]
        self.data.validate(reference["layout"])
        if self.data.hpwl(reference["layout"]) != reference["hpwl"]:
            raise ValueError("Fallback HPWL does not match its layout")
        initial = packet["initializations"]
        legal = [o for o in initial if not o["repair_fallback"]]
        if not legal or reference["trial_id"] != max(legal, key=lambda o: o["hpwl"])["trial_id"]:
            raise ValueError("Fallback must be the worst legal shared initialization")
        self.reference = RepairResult(dict(reference["layout"]), False, 0)

    def evaluate(self, row) -> dict:
        result = self.data.repair(row, self.reference)
        hpwl = self.bundle["fallback"]["hpwl"] if result.used_fallback else self.data.hpwl(result.positions)
        return dict(hpwl=hpwl, layout=result.positions, repair_fallback=result.used_fallback,
                    moved_macros=result.moved_macros, fallback_initial_trial_id=self.bundle["fallback"]["trial_id"])



class RepairService:
    """Load precomputed per-task/seed bundles; never initialize lazily on a request."""
    def __init__(self, directory: Path):
        self.evaluators = {}
        for path in sorted(directory.glob("*__s*.json")):
            evaluator = RepairEvaluator(json.loads(path.read_text()))
            key = (evaluator.bundle["benchmark"], evaluator.bundle["seed"])
            if key in self.evaluators:
                raise ValueError("Duplicate task/seed repair bundle")
            self.evaluators[key] = evaluator
        if not self.evaluators:
            raise ValueError("No prepared repair bundles found")

    def _get(self, payload):
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("Explicit geometry repair protocol required")
        key = (str(payload.get("benchmark")), int(payload.get("seed", -1)))
        if key not in self.evaluators:
            raise ValueError("No prepared bundle for requested task/seed")
        evaluator = self.evaluators[key]
        if payload.get("bundle_sha256") != evaluator.sha256:
            raise ValueError("Requested bundle fingerprint differs from service")
        if payload.get("n_macro") != len(evaluator.data.names):
            raise ValueError("Macro count differs from prepared task")
        return evaluator

    def evaluate_payload(self, payload):
        evaluator = self._get(payload)
        rows = payload.get("x")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("One candidate per evaluation request is required")
        result = evaluator.evaluate(rows[0])
        return dict(status="success", protocol=PROTOCOL, bundle_sha256=evaluator.sha256,
                    hpwl=[result["hpwl"]], repair=[result])
