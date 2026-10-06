"""Host-only unit-cube codec, without original names/types/defaults in its schema."""
from __future__ import annotations

import hashlib
import math
import random

from ...core import CategoricalParam, FloatParam, IntParam, SearchSpace


class AnonymousUnitCodec:
    def __init__(self, space: SearchSpace, *, transforms: dict[str, str] | None = None, salt: str):
        self.raw = space
        self.parameters = list(space)
        self.names = {p.name: f"x{i + 1}" for i, p in enumerate(self.parameters)}
        self.space = SearchSpace([FloatParam(self.names[p.name], low=0, high=1, default=0.5) for p in self.parameters])
        self.transforms = dict(transforms or {})
        self.choices = {}
        for p in self.parameters:
            if isinstance(p, CategoricalParam):
                values = list(p.choices)
                seed = int.from_bytes(hashlib.sha256(f"{salt}:{p.name}".encode()).digest(), "big")
                random.Random(seed).shuffle(values)
                self.choices[p.name] = values
            elif not isinstance(p, (FloatParam, IntParam)):
                raise ValueError("Unsupported anonymous parameter type")

    def _scale(self, p):
        mode = self.transforms.get(p.name, "log" if p.log else "linear")
        if mode == "linear":
            return float, float
        if mode == "log":
            return math.log, math.exp
        if mode == "logit":
            return lambda x: math.log(x / (1 - x)), lambda x: 1 / (1 + math.exp(-x))
        raise ValueError("Unsupported anonymous numeric transform")

    def encode(self, config: dict) -> dict:
        self.raw.validate_config(config)
        result = {}
        for p in self.parameters:
            value = config[p.name]
            if isinstance(p, CategoricalParam):
                choices = self.choices[p.name]
                u = (choices.index(value) + 0.5) / len(choices)
            elif isinstance(p, IntParam):
                u = (value - p.low + 0.5) / (p.high - p.low + 1)
            else:
                transform, _ = self._scale(p)
                low, high = transform(p.low), transform(p.high)
                u = (transform(value) - low) / (high - low) if low != high else 0.5
            result[self.names[p.name]] = min(1.0, max(0.0, float(u)))
        return result

    def decode(self, config: dict) -> dict:
        # Reject out-of-bounds/NaN/incomplete inputs; do not silently clip them.
        self.space.validate_config(config)
        result = {}
        for p in self.parameters:
            u = float(config[self.names[p.name]])
            if isinstance(p, CategoricalParam):
                choices = self.choices[p.name]
                value = choices[min(len(choices) - 1, int(u * len(choices)))]
            elif isinstance(p, IntParam):
                value = p.low + min(p.high - p.low, int(u * (p.high - p.low + 1)))
            elif u == 0 or p.low == p.high:
                value = p.low
            elif u == 1:
                value = p.high
            else:
                transform, inverse = self._scale(p)
                value = inverse(transform(p.low) + u * (transform(p.high) - transform(p.low)))
            result[p.name] = value
        self.raw.validate_config(result)
        return result
