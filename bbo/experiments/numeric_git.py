"""Frozen main-table GIT-BO float32 precision and replay settings."""
from dataclasses import replace
from bbo.algorithms.model_based.git_bo import GitBoAlgorithm
from bbo.core import ExternalOptimizerAdapter
from .numerical import NumericAlgorithmV2, transformed_overrides


class StableGitBoAlgorithm(GitBoAlgorithm):
    def _fit_tabpfn(self, *, x_train, y_train, deps):
        if self._regressor is None:
            self._regressor = deps['TabPFNRegressor'].create_default_for_version(
                deps['ModelVersion'].V2, device=self._device, n_estimators=self.n_estimators,
                ignore_pretraining_limits=True, fit_mode='fit_preprocessors', differentiable_input=True,
                random_state=self._seed, show_progress_bar=False, inference_precision=deps['torch'].float32)
        self._regressor.fit_with_differentiable_input(x_train, y_train)
        return self._regressor


class PaperGitBO(NumericAlgorithmV2):
    def setup(self, task_spec, seed=0, **kwargs):
        self.task_spec, self.seed, self.history = task_spec, seed, []
        protocol = dict(name='paper_shared_prefix', initialization=dict(strategy='fixed_configurations',
            seed=seed, count=len(self.initial), configurations=[o.suggestion.config for o in self.initial],
            source='frozen_paper_initialization'))
        spec = replace(task_spec, metadata={**task_spec.metadata, 'benchmark_protocol':protocol,
            'parameter_transforms':transformed_overrides(self.task_prior)['parameter_transforms']})
        self.backend = StableGitBoAlgorithm(n_candidates=2048, inference_batch_size=128, device='cuda:0')
        self.backend.setup(spec, seed=seed)

    def replay(self, history):
        history = list(history)
        n = self.row['initial']
        self.backend.replay(history[:n])
        self.history = history[:n]
        for observation in history[n:]:
            meta = observation.suggestion.metadata
            if meta.get('git_bo_phase') == 'sobol_fallback':
                expected = self.backend._sobol_suggestion(reason=meta['git_bo_fallback_reason'])
                ExternalOptimizerAdapter.assert_matching_config(expected.config, observation.suggestion.config)
            self.tell(observation)
