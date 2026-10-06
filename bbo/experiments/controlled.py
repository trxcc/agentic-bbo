"""Seven frozen prior treatments over three anonymous sparse objectives."""
from pathlib import Path
from bbo.core import (TaskSpec, TaskDescriptionRef, SearchSpace, FloatParam,
    ObjectiveSpec, ObjectiveDirection, TrialObservation, TrialSuggestion, EvaluationResult)
from .tasks import ASSETS, PaperTask
from .io import read
from .controlled_evaluator import FunctionDefinition

PRIORS = ('none','count','geometry','geometry_count','support','support_geometry','wrong_support_geometry')


class ControlledPriorTask(PaperTask):
    def __init__(self, name, *, seed=2, prior='none', journal=None):
        case = ASSETS / 'controlled' / f'{name}__s{seed}__{prior}.json'
        packet = read(case)
        self.row, self.documents = packet['row'], packet['documents']
        self.suite, self.anonymous, self.codec, self.raw = 'controlled', True, None, None
        self.instructions = None
        self.journal = None if journal is None else Path(journal)
        item = packet['definition']
        self.definition = FunctionDefinition(item['reviewer_id'], item['agent_task_id'], item['family'],
            tuple(item['active_variables']), tuple(item['decoy_variables']), tuple(item['truth']['center']), item['truth'])
        self._spec = TaskSpec(name=self.row['visible_task_id'],
            search_space=SearchSpace([FloatParam(f'x{i}', low=-5., high=5.) for i in range(1,13)]),
            objectives=(ObjectiveSpec('value', ObjectiveDirection.MINIMIZE),), max_evaluations=64,
            description_ref=TaskDescriptionRef(task_id=self.row['visible_task_id'], primary_path=case))
        self.prefix = [TrialObservation.from_evaluation(TrialSuggestion(config=o['config'],trial_id=o['trial_id']),
            EvaluationResult(objectives={'value':o['value']})) for o in packet['prefix']]

    def evaluate(self, suggestion):
        if suggestion.trial_id is None or not 16 <= suggestion.trial_id < 64:
            raise ValueError('Objective call outside the controlled-prior budget')
        self.spec.search_space.validate_config(suggestion.config)
        value = self._journal_call('evaluations', suggestion.trial_id,
            dict(trial_id=suggestion.trial_id, config=suggestion.config),
            lambda: EvaluationResult(objectives={'value':self.definition.evaluate(suggestion.config)}))
        return EvaluationResult(objectives=value['objectives'])

    def evaluate_final(self, suggestion):
        return None
