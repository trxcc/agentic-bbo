"""Search-space primitives for benchmark-oriented BBO."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - exercised only when the optional extra is installed.
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
except ImportError:  # pragma: no cover - default path for this repo.
    Categorical = ConfigurationSpace = Float = Integer = None
    CategoricalHyperparameter = UniformFloatHyperparameter = UniformIntegerHyperparameter = None


@dataclass(frozen=True)
class ParameterSpec(ABC):
    """Base class for a structured search-space parameter."""

    name: str
    default: Any | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter name must be non-empty.")

    @abstractmethod
    def coerce(self, value: Any) -> Any:
        """Convert a raw value into the parameter's canonical type."""

    @abstractmethod
    def validate(self, value: Any) -> None:
        """Validate a canonicalized value."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Any:
        """Sample one value from the parameter domain."""

    def effective_default(self) -> Any:
        if self.default is None:
            raise ValueError(f"Parameter `{self.name}` does not define a default value.")
        return self.coerce(self.default)


@dataclass(frozen=True)
class FloatParam(ParameterSpec):
    """Continuous floating-point parameter."""

    low: float = 0.0
    high: float = 1.0
    log: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low > self.high:
            raise ValueError(f"FloatParam `{self.name}` has low > high.")
        if self.log and self.low <= 0:
            raise ValueError(f"FloatParam `{self.name}` must have low > 0 when log=True.")
        if self.default is not None:
            self.validate(self.coerce(self.default))

    def coerce(self, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"FloatParam `{self.name}` does not accept boolean values.")
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FloatParam `{self.name}` could not coerce value {value!r}.") from exc
        self.validate(coerced)
        return coerced

    def validate(self, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"FloatParam `{self.name}` requires a numeric value.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"FloatParam `{self.name}` requires a finite value.")
        if numeric < self.low or numeric > self.high:
            raise ValueError(
                f"FloatParam `{self.name}` expects {self.low} <= value <= {self.high}, got {numeric}."
            )

    def sample(self, rng: random.Random) -> float:
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def effective_default(self) -> float:
        return self.coerce(self.default if self.default is not None else (self.low + self.high) / 2.0)


@dataclass(frozen=True)
class IntParam(ParameterSpec):
    """Discrete integer parameter."""

    low: int = 0
    high: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low > self.high:
            raise ValueError(f"IntParam `{self.name}` has low > high.")
        if self.log and self.low <= 0:
            raise ValueError(f"IntParam `{self.name}` must have low > 0 when log=True.")
        if self.default is not None:
            self.validate(self.coerce(self.default))

    def coerce(self, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"IntParam `{self.name}` does not accept boolean values.")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"IntParam `{self.name}` received non-integral value {value!r}.")
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IntParam `{self.name}` could not coerce value {value!r}.") from exc
        self.validate(coerced)
        return coerced

    def validate(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"IntParam `{self.name}` requires an integer value.")
        if value < self.low or value > self.high:
            raise ValueError(f"IntParam `{self.name}` expects {self.low} <= value <= {self.high}, got {value}.")

    def sample(self, rng: random.Random) -> int:
        if self.log:
            sampled = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
            return int(min(max(round(sampled), self.low), self.high))
        return rng.randint(self.low, self.high)

    def effective_default(self) -> int:
        midpoint = int(round((self.low + self.high) / 2.0))
        return self.coerce(self.default if self.default is not None else midpoint)


@dataclass(frozen=True)
class CategoricalParam(ParameterSpec):
    """Finite categorical parameter."""

    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.choices:
            raise ValueError(f"CategoricalParam `{self.name}` must define at least one choice.")
        if self.default is not None and self.default not in self.choices:
            raise ValueError(
                f"CategoricalParam `{self.name}` default {self.default!r} is not in {self.choices!r}."
            )

    def coerce(self, value: Any) -> Any:
        if value not in self.choices:
            raise ValueError(
                f"CategoricalParam `{self.name}` expects one of {self.choices!r}, got {value!r}."
            )
        return value

    def validate(self, value: Any) -> None:
        self.coerce(value)

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def effective_default(self) -> Any:
        return self.default if self.default is not None else self.choices[0]


class SearchSpace:
    """Ordered collection of search-space parameters."""

    def __init__(self, parameters: Iterable[ParameterSpec]):
        self._parameters = tuple(parameters)
        if not self._parameters:
            raise ValueError("SearchSpace must define at least one parameter.")
        names = [param.name for param in self._parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"SearchSpace contains duplicate parameter names: {names!r}")
        self._by_name = {param.name: param for param in self._parameters}

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> ParameterSpec:
        return self._by_name[name]

    def names(self) -> list[str]:
        return [param.name for param in self._parameters]

    def defaults(self) -> dict[str, Any]:
        return {param.name: param.effective_default() for param in self._parameters}

    def contains(self, name: str) -> bool:
        return name in self._by_name

    def sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        local_rng = rng or random.Random()
        return {param.name: param.sample(local_rng) for param in self._parameters}

    def validate_config(
        self,
        config: dict[str, Any],
        *,
        allow_partial: bool = False,
        reject_extra: bool = True,
    ) -> None:
        self.coerce_config(config, allow_partial=allow_partial, reject_extra=reject_extra)

    def coerce_config(
        self,
        config: dict[str, Any],
        *,
        allow_partial: bool = False,
        reject_extra: bool = True,
        use_defaults: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise TypeError("config must be a dict mapping parameter names to values.")
        if reject_extra:
            unexpected = sorted(set(config) - set(self._by_name))
            if unexpected:
                raise ValueError(f"Unexpected parameters: {unexpected!r}")

        normalized: dict[str, Any] = {}
        missing: list[str] = []
        for param in self._parameters:
            if param.name in config:
                normalized[param.name] = param.coerce(config[param.name])
            elif not allow_partial:
                if use_defaults:
                    normalized[param.name] = param.effective_default()
                else:
                    missing.append(param.name)
        if missing:
            raise ValueError(f"Missing required parameters: {missing!r}")
        return normalized

    def numeric_parameters(self) -> tuple[FloatParam | IntParam, ...]:
        numeric: list[FloatParam | IntParam] = []
        for param in self._parameters:
            if isinstance(param, (FloatParam, IntParam)):
                numeric.append(param)
                continue
            raise TypeError(f"SearchSpace contains non-numeric parameter `{param.name}`.")
        return tuple(numeric)

    def numeric_bounds(self) -> np.ndarray:
        numeric = self.numeric_parameters()
        return np.asarray([(float(param.low), float(param.high)) for param in numeric], dtype=float)

    def to_numeric_vector(self, config: dict[str, Any]) -> np.ndarray:
        normalized = self.coerce_config(config, use_defaults=False)
        return np.asarray([float(normalized[param.name]) for param in self.numeric_parameters()], dtype=float)

    def from_numeric_vector(self, values: Iterable[float], *, clip: bool = True) -> dict[str, Any]:
        numeric = self.numeric_parameters()
        vector = list(values)
        if len(vector) != len(numeric):
            raise ValueError(f"Expected {len(numeric)} numeric values, got {len(vector)}.")

        config: dict[str, Any] = {}
        for param, value in zip(numeric, vector, strict=True):
            numeric_value = float(value)
            if clip:
                numeric_value = min(max(numeric_value, float(param.low)), float(param.high))
            if isinstance(param, IntParam):
                config[param.name] = param.coerce(int(round(numeric_value)))
            else:
                config[param.name] = param.coerce(numeric_value)
        return config


def _require_configspace() -> None:
    if ConfigurationSpace is None:
        raise ImportError(
            "ConfigSpace is not installed. Install the optional interop dependencies with `uv sync --extra interop`."
        )


def from_configspace(space: Any) -> SearchSpace:
    """Convert a ConfigSpace instance into a `SearchSpace`."""

    _require_configspace()
    assert UniformIntegerHyperparameter is not None
    params: list[ParameterSpec] = []
    for hp in space.values():
        if isinstance(hp, UniformIntegerHyperparameter):
            params.append(
                IntParam(
                    name=hp.name,
                    low=hp.lower,
                    high=hp.upper,
                    log=hp.log,
                    default=hp.default_value,
                )
            )
        elif isinstance(hp, UniformFloatHyperparameter):
            params.append(
                FloatParam(
                    name=hp.name,
                    low=hp.lower,
                    high=hp.upper,
                    log=hp.log,
                    default=hp.default_value,
                )
            )
        elif isinstance(hp, CategoricalHyperparameter):
            params.append(
                CategoricalParam(
                    name=hp.name,
                    choices=tuple(hp.choices),
                    default=hp.default_value,
                )
            )
        else:
            raise TypeError(f"Unsupported ConfigSpace hyperparameter type: {type(hp).__name__}")
    return SearchSpace(params)


def to_configspace(space: SearchSpace, seed: int | None = None) -> Any:
    """Convert a `SearchSpace` into a ConfigSpace instance."""

    _require_configspace()
    assert ConfigurationSpace is not None
    cs = ConfigurationSpace(seed=seed or 0)
    for param in space:
        if isinstance(param, IntParam):
            cs.add(
                Integer(
                    param.name,
                    bounds=(param.low, param.high),
                    default=param.effective_default(),
                    log=param.log,
                )
            )
        elif isinstance(param, FloatParam):
            cs.add(
                Float(
                    param.name,
                    bounds=(param.low, param.high),
                    default=param.effective_default(),
                    log=param.log,
                )
            )
        elif isinstance(param, CategoricalParam):
            cs.add(
                Categorical(
                    param.name,
                    items=list(param.choices),
                    default=param.effective_default(),
                )
            )
        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")
    return cs


__all__ = [
    "CategoricalParam",
    "FloatParam",
    "IntParam",
    "ParameterSpec",
    "SearchSpace",
    "from_configspace",
    "to_configspace",
]
