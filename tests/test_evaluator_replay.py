import pytest

from bbo.core import TrialSuggestion
from bbo.experiments.tasks import PaperTask


@pytest.mark.parametrize('name', ('hpo_bayesmark_breast_svm','hpo_bayesmark_diabetes_random_forest'))
def test_hpo_frozen_observation_matches_pinned_evaluator(name):
    pytest.importorskip('sklearn')
    task=PaperTask(name,suite='main')
    item=task.prefix[0]
    try:
        raw=task._evaluator()
        result=raw.evaluate(TrialSuggestion(config=item.suggestion.config,trial_id=0))
        assert result.success if hasattr(result,'success') else result.status.value=='success'
        loss=1-result.objectives['accuracy'] if 'accuracy' in result.objectives else result.objectives['mse']
        assert loss==pytest.approx(item.objectives['loss'],rel=1e-12,abs=1e-12)
    finally:
        task.cleanup()


def test_all_molecular_objectives_match_frozen_observations():
    pytest.importorskip('rdkit')
    from bbo.tasks import TASK_FAMILIES
    for name in TASK_FAMILIES['GuacaMol']:
        task=PaperTask(name,suite='main')
        key=task.spec.primary_objective.name
        item=min(task.prefix,key=lambda o:o.objectives[key])
        try:
            result=task._evaluator().evaluate(TrialSuggestion(config=item.suggestion.config,trial_id=0))
            assert result.status.value=='success'
            assert next(iter(result.objectives.values()))==pytest.approx(item.objectives[key],rel=1e-12,abs=1e-12)
        finally:
            task.cleanup()
