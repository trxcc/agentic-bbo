"""Optuna TPE algorithm wired to the benchmark ask/tell protocol."""

from __future__ import annotations

from typing import Any

from ...core import ExternalOptimizerAdapter, TrialObservation, TrialSuggestion
from .optuna_utils import build_distributions, objective_direction_to_optuna, require_optuna, suggest_from_param
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization


class OptunaTpeAlgorithm(ExternalOptimizerAdapter):
    """Single-entry Optuna TPE optimizer with deterministic replay via logger history."""

    def __init__(self) -> None:
        super().__init__()
        self._study: Any | None = None
        self._trial_state_enum: Any | None = None
        self._pending_trials: dict[int, Any] = {}
        self._distributions: dict[str, Any] = {}
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._ask_count = 0

    @property
    def name(self) -> str:
        return "optuna_tpe"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        optuna = require_optuna()
        if len(task_spec.objectives) != 1:
            raise ValueError("OptunaTpeAlgorithm currently supports exactly one objective.")

        self.bind_task_spec(task_spec)
        self._distributions = build_distributions(task_spec.search_space)
        self._trial_state_enum = optuna.trial.TrialState
        self._study = optuna.create_study(
            direction=objective_direction_to_optuna(task_spec.primary_objective.direction),
            sampler=optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=5),
        )
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=int(seed))
        if self._fixed_initialization is not None:
            for config in self._fixed_initialization.configurations:
                self._study.enqueue_trial(dict(config))
        self._pending_trials = {}
        self._ask_count = 0

    def ask(self) -> TrialSuggestion:
        study = self._require_study()
        search_space = self.require_search_space()
        trial = study.ask()
        config = {
            param.name: suggest_from_param(trial, param)
            for param in search_space
        }
        trial_number = int(trial.number)
        self._pending_trials[trial_number] = trial
        metadata = {
            "optuna_sampler": "tpe",
            "optuna_trial_number": trial_number,
            "optuna_startup_trials": 5,
        }
        if self._fixed_initialization is not None and self._ask_count < len(
            self._fixed_initialization.configurations
        ):
            protocol_metadata = self._fixed_initialization.suggestion(
                self._ask_count,
                algorithm=self.name,
            ).metadata
            metadata.update(protocol_metadata)
            metadata["optuna_phase"] = "benchmark_initialization"
        self._ask_count += 1
        return TrialSuggestion(
            config=search_space.coerce_config(config, use_defaults=False),
            metadata=metadata,
        )

    def tell(self, observation: TrialObservation) -> None:
        study = self._require_study()
        trial_state = self._require_trial_state()
        trial_number_raw = observation.suggestion.metadata.get("optuna_trial_number")
        if trial_number_raw is None:
            raise ValueError("Optuna TPE suggestions must preserve `optuna_trial_number` metadata.")

        trial_number = int(trial_number_raw)
        trial = self._pending_trials.pop(trial_number, None)
        if trial is None:
            raise ValueError(
                f"Optuna TPE could not find pending trial #{trial_number}. "
                "Replay history may not match the optimizer state."
            )

        assert self._primary_name is not None
        if observation.success:
            if self._primary_name not in observation.objectives:
                raise ValueError(
                    f"Successful observations must include the primary objective `{self._primary_name}`."
                )
            value = float(observation.objectives[self._primary_name])
            study.tell(trial, values=value, state=trial_state.COMPLETE)
        else:
            study.tell(trial, state=trial_state.FAIL)
        self.update_best_incumbent(observation)

    def seed(self, observation: TrialObservation) -> None:
        if self._fixed_initialization is not None and self._ask_count < len(
            self._fixed_initialization.configurations
        ):
            # Consume Optuna's enqueued fixed trial instead of adding a second
            # completed copy. This is the workflow path where the benchmark
            # evaluates the shared prefix outside the optimizer ask loop.
            expected = self.ask()
            self.assert_matching_config(expected.config, observation.suggestion.config)
            self.tell(self.make_replay_observation(expected, observation))
            return
        study = self._require_study()
        trial_state = self._require_trial_state()
        optuna = require_optuna()
        params = self.require_search_space().coerce_config(observation.suggestion.config, use_defaults=False)
        if observation.success:
            assert self._primary_name is not None
            if self._primary_name not in observation.objectives:
                raise ValueError(
                    f"Successful observations must include the primary objective `{self._primary_name}`."
                )
            trial = optuna.trial.create_trial(
                params=params,
                distributions=self._distributions,
                value=float(observation.objectives[self._primary_name]),
                state=trial_state.COMPLETE,
            )
        else:
            trial = optuna.trial.create_trial(
                params=params,
                distributions=self._distributions,
                state=trial_state.FAIL,
            )
        study.add_trial(trial)
        self.update_best_incumbent(observation)

    def _require_study(self) -> Any:
        if self._study is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called before ask/tell.")
        return self._study

    def _require_trial_state(self) -> Any:
        if self._trial_state_enum is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called before ask/tell.")
        return self._trial_state_enum


__all__ = ["OptunaTpeAlgorithm"]
