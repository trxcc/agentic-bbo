"""Categorical-to-continuous search-space conversion helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .space import CategoricalParam, FloatParam, IntParam, ParameterSpec, SearchSpace


@dataclass(frozen=True)
class ContinuousFeatureSpec:
    """One continuous feature emitted by a search-space converter."""

    name: str
    low: float
    high: float


class ContinuousSearchSpaceConverter(ABC):
    """Convert a structured search space into a continuous feature interface."""

    strategy_name: str = "unknown"

    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space

    @property
    @abstractmethod
    def feature_specs(self) -> tuple[ContinuousFeatureSpec, ...]:
        """Ordered continuous feature specifications."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.feature_specs)

    def continuous_bounds(self) -> np.ndarray:
        return np.asarray([(feature.low, feature.high) for feature in self.feature_specs], dtype=float)

    @abstractmethod
    def encode_vector(self, config: dict[str, Any]) -> np.ndarray:
        """Encode one structured config into a continuous feature vector."""

    @abstractmethod
    def decode_vector(self, values: Iterable[float], *, clip: bool = True) -> dict[str, Any]:
        """Decode one continuous feature vector back into a structured config."""

    def encode_feature_config(self, config: dict[str, Any]) -> dict[str, float]:
        vector = self.encode_vector(config)
        return {name: float(value) for name, value in zip(self.feature_names, vector, strict=True)}

    def decode_feature_config(self, feature_config: dict[str, Any], *, clip: bool = True) -> dict[str, Any]:
        vector = [float(feature_config[name]) for name in self.feature_names]
        return self.decode_vector(vector, clip=clip)

    def continuous_api_config(self) -> dict[str, dict[str, Any]]:
        return {
            feature.name: {
                "type": "real",
                "space": "linear",
                "range": (float(feature.low), float(feature.high)),
            }
            for feature in self.feature_specs
        }


@dataclass(frozen=True)
class _EncodedParameterSpec:
    param: ParameterSpec
    feature_slice: slice


class OneHotCategoricalConverter(ContinuousSearchSpaceConverter):
    """Encode categorical parameters as one-hot blocks and numeric parameters as identity axes."""

    strategy_name = "onehot"

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__(search_space)
        feature_specs: list[ContinuousFeatureSpec] = []
        param_specs: list[_EncodedParameterSpec] = []
        offset = 0
        for param in search_space:
            start = offset
            if isinstance(param, FloatParam):
                feature_specs.append(ContinuousFeatureSpec(name=param.name, low=float(param.low), high=float(param.high)))
                offset += 1
            elif isinstance(param, IntParam):
                feature_specs.append(ContinuousFeatureSpec(name=param.name, low=float(param.low), high=float(param.high)))
                offset += 1
            elif isinstance(param, CategoricalParam):
                for choice in param.choices:
                    feature_specs.append(ContinuousFeatureSpec(name=f"{param.name}::{choice}", low=0.0, high=1.0))
                    offset += 1
            else:  # pragma: no cover - defensive guard against future subclasses.
                raise TypeError(f"Unsupported parameter type `{type(param).__name__}` for one-hot conversion.")
            param_specs.append(_EncodedParameterSpec(param=param, feature_slice=slice(start, offset)))
        self._feature_specs = tuple(feature_specs)
        self._param_specs = tuple(param_specs)

    @property
    def feature_specs(self) -> tuple[ContinuousFeatureSpec, ...]:
        return self._feature_specs

    def encode_vector(self, config: dict[str, Any]) -> np.ndarray:
        normalized = self.search_space.coerce_config(config, use_defaults=False)
        values: list[float] = []
        for encoded in self._param_specs:
            param = encoded.param
            value = normalized[param.name]
            if isinstance(param, FloatParam):
                values.append(float(value))
            elif isinstance(param, IntParam):
                values.append(float(value))
            else:
                assert isinstance(param, CategoricalParam)
                values.extend(1.0 if value == choice else 0.0 for choice in param.choices)
        return np.asarray(values, dtype=float)

    def decode_vector(self, values: Iterable[float], *, clip: bool = True) -> dict[str, Any]:
        vector = np.asarray([float(value) for value in values], dtype=float)
        if len(vector) != len(self._feature_specs):
            raise ValueError(f"Expected {len(self._feature_specs)} continuous values, got {len(vector)}.")

        config: dict[str, Any] = {}
        for encoded in self._param_specs:
            param = encoded.param
            block = vector[encoded.feature_slice]
            if isinstance(param, FloatParam):
                numeric_value = float(block[0])
                if clip:
                    numeric_value = min(max(numeric_value, float(param.low)), float(param.high))
                config[param.name] = param.coerce(numeric_value)
            elif isinstance(param, IntParam):
                numeric_value = float(block[0])
                if clip:
                    numeric_value = min(max(numeric_value, float(param.low)), float(param.high))
                config[param.name] = param.coerce(int(round(numeric_value)))
            else:
                assert isinstance(param, CategoricalParam)
                if clip:
                    block = np.clip(block, 0.0, 1.0)
                choice_index = int(np.argmax(block))
                config[param.name] = param.choices[choice_index]
        return config


def build_continuous_converter(
    search_space: SearchSpace,
    *,
    strategy: str = "onehot",
) -> ContinuousSearchSpaceConverter:
    """Build a categorical-to-continuous converter for a mixed search space."""

    if strategy != "onehot":
        raise ValueError(f"Unknown categorical-to-continuous strategy `{strategy}`.")
    return OneHotCategoricalConverter(search_space)


__all__ = [
    "ContinuousFeatureSpec",
    "ContinuousSearchSpaceConverter",
    "OneHotCategoricalConverter",
    "build_continuous_converter",
]
