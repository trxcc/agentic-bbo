"""Scrambled Sobol sequence baseline over transformed numeric spaces."""

from __future__ import annotations

import json
from typing import Any

from ...core import ExternalOptimizerAdapter, TrialObservation, TrialSuggestion, UnitCubeSearchSpaceConverter
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization


def require_sobol_engine() -> Any:
    try:
        from torch.quasirandom import SobolEngine
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise ImportError(
            "`sobol_search` requires torch. Install the HPO dependencies with `uv sync --extra hpo`."
        ) from exc
    return SobolEngine


class SobolSearchAlgorithm(ExternalOptimizerAdapter):
    """Deterministic scrambled Sobol search with ask/tell replay support."""

    def __init__(self, *, scramble: bool = True, max_duplicate_attempts: int = 128) -> None:
        super().__init__()
        if max_duplicate_attempts <= 0:
            raise ValueError("max_duplicate_attempts must be positive.")
        self.scramble = bool(scramble)
        self.max_duplicate_attempts = int(max_duplicate_attempts)
        self._seed = 0
        self._converter: UnitCubeSearchSpaceConverter | None = None
        self._engine: Any | None = None
        self._seen: set[str] = set()
        self._draw_count = 0
        self._history_count = 0
        self._fixed_initialization: FixedInitializationProtocol | None = None

    @property
    def name(self) -> str:
        return "sobol_search"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("SobolSearchAlgorithm supports exactly one objective.")
        self.bind_task_spec(task_spec)
        transforms = task_spec.metadata.get("parameter_transforms")
        if transforms is not None and not isinstance(transforms, dict):
            raise TypeError("Task metadata `parameter_transforms` must be a mapping.")
        self._converter = UnitCubeSearchSpaceConverter(
            task_spec.search_space,
            transforms=transforms,
        )
        self._seed = int(seed)
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=self._seed)
        self._reset_sequence()

    def ask(self) -> TrialSuggestion:
        if self._fixed_initialization is not None and self._history_count < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(self._history_count, algorithm=self.name)
            suggestion.metadata.update(
                {
                    "sobol_phase": "benchmark_initialization",
                    "sobol_seed": self._seed,
                }
            )
            self._seen.add(_config_identity(suggestion.config))
            return suggestion
        converter = self._require_converter()
        engine = self._require_engine()
        last_config: dict[str, Any] | None = None
        for attempt in range(self.max_duplicate_attempts):
            unit = engine.draw(1).detach().cpu().numpy().reshape(-1)
            draw_index = self._draw_count
            self._draw_count += 1
            config = converter.decode_vector(unit, clip=True)
            last_config = config
            identity = _config_identity(config)
            if identity not in self._seen:
                self._seen.add(identity)
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "sobol_backend": "torch.quasirandom.SobolEngine",
                        "sobol_scramble": self.scramble,
                        "sobol_seed": self._seed,
                        "sobol_draw_index": draw_index,
                        "sobol_duplicate_attempts": attempt,
                        "feature_transform": "unit_cube",
                        "parameter_transforms": dict(converter.transforms),
                    },
                )
        assert last_config is not None
        self._seen.add(_config_identity(last_config))
        return TrialSuggestion(
            config=last_config,
            metadata={
                "sobol_backend": "torch.quasirandom.SobolEngine",
                "sobol_scramble": self.scramble,
                "sobol_seed": self._seed,
                "sobol_draw_index": self._draw_count - 1,
                "sobol_duplicate_exhausted": True,
                "feature_transform": "unit_cube",
                "parameter_transforms": dict(converter.transforms),
            },
        )

    def tell(self, observation: TrialObservation) -> None:
        self._seen.add(_config_identity(observation.suggestion.config))
        self._history_count += 1
        self.update_best_incumbent(observation)

    def replay(self, history: list[TrialObservation]) -> None:
        self._best = None
        self._reset_sequence()
        for observation in history:
            expected = self.ask()
            self.assert_matching_config(expected.config, observation.suggestion.config)
            self.tell(self.make_replay_observation(expected, observation))

    def _reset_sequence(self) -> None:
        SobolEngine = require_sobol_engine()
        converter = self._require_converter()
        self._engine = SobolEngine(
            dimension=len(converter.feature_specs),
            scramble=self.scramble,
            seed=self._seed,
        )
        self._seen = set()
        self._draw_count = 0
        self._history_count = 0

    def _require_converter(self) -> UnitCubeSearchSpaceConverter:
        if self._converter is None:
            raise RuntimeError("SobolSearchAlgorithm.setup() must be called before ask/tell.")
        return self._converter

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("SobolSearchAlgorithm.setup() must be called before ask/tell.")
        return self._engine


def _config_identity(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["SobolSearchAlgorithm", "require_sobol_engine"]
