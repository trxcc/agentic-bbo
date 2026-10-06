"""Paper Direct control: persistent native sessions with every tool disabled."""
from pathlib import Path
from bbo.core import TrialSuggestion
from bbo.core.algo import Algorithm, Incumbent
from .io import read, save, digest, canonical, packet
from .direct_protocol import first_message, parse_candidate
from .tasks import ASSETS


class DirectAgent(Algorithm):
    name = 'direct'

    def __init__(self, task, out, profile, executable=None):
        self.task, self.out, self.profile = task, Path(out), profile
        self.executable = executable or 'codex'
        self.runtime = None

    def setup(self, task_spec, seed=0, **kwargs):
        self.task_spec, self.history = task_spec, []

    def replay(self, history):
        self.history = list(history)

    def tell(self, observation):
        self.history.append(observation)

    def incumbents(self):
        if not self.history:
            return []
        objective = self.task_spec.primary_objective
        best = (min if objective.direction.value == 'minimize' else max)(self.history, key=lambda o: o.objectives[objective.name])
        return [Incumbent(config=best.suggestion.config, score=best.objectives[objective.name],
            objectives=best.objectives, trial_id=best.suggestion.trial_id)]

    def ask(self):
        row = self.task.row
        index = len(self.history) - row['initial']
        folder = self.out / 'rounds' / f'{index+1:04d}'
        signature = digest([packet(o) for o in self.history])
        saved = folder / 'accepted.json'
        if saved.exists():
            accepted = read(saved)
            if accepted['history_sha256'] != signature:
                raise ValueError('Direct proposal history mismatch')
            return TrialSuggestion(config=accepted['candidate'], metadata=accepted['metadata'])
        if self.runtime is None:
            from .direct_runtime import NativeRuntime
            self.runtime = NativeRuntime(self.out, ASSETS / 'direct_instructions.md', [], None,
                                          profile=self.profile, executable=self.executable)
        if index == 0:
            message = first_message(dict(visible_task_id=self.task.spec.name,
                objective=dict(name=self.task.spec.primary_objective.name, direction=self.task.spec.primary_objective.direction.value),
                evaluation_budget=dict(initial_observations=row['initial'], new_evaluations=row['budget'], total_observations=row['total']),
                task_facts=self.task.documents['sections'], parameter_definitions=self.task.documents['parameters'],
                initial_history=[packet(o) for o in self.history]))
        else:
            message = dict(latest_host_observation=packet(self.history[-1]), round=index+1,
                remaining_evaluations=row['budget']-index, submission='Return exactly one complete {"config":{...}} JSON object.')
        prior = {digest(o.suggestion.config) for o in self.history}
        for attempt in range(4):
            location = folder if attempt == 0 else folder / f'correction_{attempt}'
            content = self.runtime.run(canonical(message) if isinstance(message, dict) else message, location)
            try:
                candidate = parse_candidate(content, self.task.spec.search_space)
                if digest(candidate) in prior:
                    raise ValueError('Duplicate previously evaluated candidate; return a different complete legal configuration.')
                break
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 3:
                    raise RuntimeError('No legal new candidate after three corrections') from exc
                message = 'Candidate rejected: '+str(exc)+'. No objective evaluation was performed. Return one complete legal {"config":{...}} JSON object.'
        metadata = dict(engine='codex', condition='STRICT_NO_TOOLS', proposal_history_sha256=signature)
        save(saved, dict(history_sha256=signature, candidate=candidate, metadata=metadata))
        return TrialSuggestion(config=candidate, metadata=metadata)

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
