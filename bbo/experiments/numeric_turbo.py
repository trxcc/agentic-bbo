"""Isolated TuRBO restart/replay repair; the pinned acquisition core is unchanged."""
from bbo.algorithms.model_based.botorch_turbo import (
    BotorchTurboAlgorithm, _config_identity, require_botorch_turbo,
)


class RestartingTurbo(BotorchTurboAlgorithm):
    def _reset_runtime(self):
        super()._reset_runtime()
        self._epoch_start = 0

    def _successful_history(self):
        return [o for o in self._history[self._epoch_start:]
                if o.success and self._primary_name in o.objectives]

    def ask(self):
        fixed = self._fixed_initialization
        if fixed is not None and len(self._history) < len(fixed.configurations):
            suggestion = fixed.suggestion(len(self._history), algorithm=self.name)
            suggestion.metadata['turbo_phase'] = 'benchmark_initialization'
        else:
            if self._state is not None and self._state.restart_triggered:
                self._restart_count += 1
                self._epoch_start = len(self._history)
                self._state = None
            successful = self._successful_history()
            # The original shared prefix is used exactly once. Every later epoch
            # needs startup_trials successful, budgeted evaluations of its own.
            required = self.startup_trials if self._restart_count or fixed is None else 2
            if len(successful) < required:
                phase = ('restart_initialization' if self._restart_count else
                         'startup' if fixed is None else 'startup_recovery')
                suggestion = self._sobol_suggestion(phase=phase)
                if suggestion.metadata.get('turbo_duplicate_exhausted'):
                    raise RuntimeError('TuRBO exhausted fresh Sobol initialization candidates')
            else:
                # Numerical errors are surfaced; never relabel random fallback as BO.
                suggestion = self._turbo_suggestion(successful)
        suggestion.metadata.update(
            numeric_revision='numeric_v2', turbo_restart_count=self._restart_count,
            turbo_epoch_start=self._epoch_start,
            turbo_sobol_draws=int(self._require_startup_engine().num_generated),
        )
        self._ask_count += 1
        self._seen.add(_config_identity(suggestion.config))
        return suggestion

    def tell(self, observation):
        meta = observation.suggestion.metadata
        phase = meta.get('turbo_phase')
        if phase == 'restart':
            raise ValueError('Legacy broken restart history cannot be resumed as numeric_v2')
        epoch = int(meta.get('turbo_restart_count', self._restart_count))
        if epoch != self._restart_count:
            if (epoch != self._restart_count + 1 or phase != 'restart_initialization'
                    or self._state is None or not self._state.restart_triggered
                    or int(meta['turbo_epoch_start']) != len(self._history)):
                raise ValueError('Invalid TuRBO restart transition in replay history')
            self._restart_count = epoch
            self._epoch_start = len(self._history)
            self._state = None
        if int(meta.get('turbo_epoch_start', self._epoch_start)) != self._epoch_start:
            raise ValueError('TuRBO local epoch does not match replay history')
        # ask() initializes the trust state even if the subsequent evaluation fails.
        # Reconstruct that transition before tell(), including failure observations.
        if phase == 'acquisition' and self._state is None:
            previous = self._successful_history()
            if len(previous) < 2:
                raise ValueError('TuRBO acquisition requires at least two local observations')
            self._state = require_botorch_turbo()['vendor'].TurboState(
                dim=len(self._require_converter().feature_specs), batch_size=1,
                best_value=max(self._objective_to_maximization(o) for o in previous),
                success_tolerance=self.success_tolerance,
            )
        super().tell(observation)

    def replay(self, history):
        self._best = None
        self._reset_runtime()
        draws = 0
        for observation in history:
            meta = observation.suggestion.metadata
            if meta.get('turbo_phase') in {'startup', 'startup_recovery', 'restart_initialization'}:
                draws += 1 + int(meta.get('turbo_sobol_duplicate_attempt', 0))
            if 'turbo_sobol_draws' in meta and int(meta['turbo_sobol_draws']) != draws:
                raise ValueError('TuRBO Sobol draw count does not match replay history')
            if meta.get('turbo_phase') == 'sobol_fallback':
                raise ValueError('Numerical fallback history is not accepted by numeric_v2')
            self.tell(observation)
        if draws:
            self._require_startup_engine().fast_forward(draws)
        self._ask_count = len(self._history)
