"""Phased reference GP optimizer: explore first, then exploit.

All the strong parts of the registered GP-EI baseline are kept untouched - the
BoTorch SingleTaskGP model, the running train-mean/std input geometry that lets
the candidate box widen and extrapolate toward distant basins, and the exact
expected-improvement machinery.

The single change is the acquisition schedule.  For the first few post-
initialization evaluations the strategy maximises an upper-confidence bound
(``beta=2``), which is more exploratory than expected improvement and improves
the chance of locating the best basin while the run is still far from any
optimum.  After that it reverts to the registered expected-improvement
acquisition for the remaining budget, so late refinement and final accuracy are
unchanged.

Only one engine and one model are used: the acquisition is switched on the same
:class:`GpEiAlgorithm` instance,
so no foreign candidates perturb the surrogate and the observation log fully
determines the schedule (replaying history reproduces identical proposals).
"""

from __future__ import annotations

from typing import Any

from solver_support import Algorithm, GpEiAlgorithm, TrialObservation, TrialSuggestion

_EXPLORATION_TRIALS = 12
_EXPLORATION_BETA = 2.0


class PhasedGpEi(Algorithm):
    """Reference GP-EI with a short UCB exploration prefix."""

    def __init__(self) -> None:
        self._engine: GpEiAlgorithm | None = None

    @property
    def name(self) -> str:
        return "gp_ei_phased"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        self._engine = GpEiAlgorithm(
            pool_size=None,
            startup_trials=2,
            xi=0.0,
            alpha=1e-06,
            n_restarts_optimizer=0,
            acqf_num_restarts=10,
            max_acqf_attempts=8,
            candidate_attempt_multiplier=20,
            acquisition="ei",
            acquisition_beta=_EXPLORATION_BETA,
            device="cpu",
            kernel="default",
            input_scaling="train_mean_std",
            parameter_transforms=None,
        )
        self._engine.setup(task_spec, seed, **kwargs)
        self._observations = 0
        self._initial = self._infer_initial_count(task_spec)

    @staticmethod
    def _infer_initial_count(task_spec: Any) -> int:
        try:
            protocol = task_spec.metadata.get("benchmark_protocol", {})
            initialization = protocol.get("initialization", {}) if isinstance(protocol, dict) else {}
            count = initialization.get("count") if isinstance(initialization, dict) else None
            if count is not None and int(count) > 0:
                return int(count)
        except Exception:  # noqa: BLE001 - metadata is best-effort only.
            pass
        return 20

    def ask(self) -> TrialSuggestion:
        assert self._engine is not None, "setup() must be called before ask()."
        exploring = self._observations < self._initial + _EXPLORATION_TRIALS
        self._engine.acquisition = "ucb" if exploring else "ei"
        suggestion = self._engine.ask()
        suggestion.metadata["gp_ei_phased_acquisition"] = self._engine.acquisition
        return suggestion

    def tell(self, observation: TrialObservation) -> None:
        assert self._engine is not None, "setup() must be called before tell()."
        self._engine.tell(observation)
        self._observations += 1

    def replay(self, history: list[TrialObservation]) -> None:
        for observation in history:
            self.tell(observation)

    def incumbents(self):
        assert self._engine is not None
        return self._engine.incumbents()


def create_optimizer() -> Algorithm:
    return PhasedGpEi()
