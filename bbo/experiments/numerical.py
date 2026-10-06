"""Registered numerical baselines, shared initialization and deterministic replay."""
import copy
from dataclasses import replace
from pathlib import Path
from .io import save as atomic_json, read as read_json, digest as sha, packet
from .numeric_geometry import WarpedNative,UnitCubeCma
from .numeric_turbo import RestartingTurbo
from .reproducibility import gp_rng
from bbo.algorithms.baseline_factory import create_comparable_baseline
from bbo.core import TrialSuggestion,ExternalOptimizerAdapter
from bbo.core.algo import Algorithm

BACKENDS={'GP':'gp_ei','TURBO':'turbo','TPE':'tpe','CMA_ES':'cma_es','RANDOM':'random','SOBOL':'sobol'}


def transformed_overrides(prior):
    return dict(kernel='matern52', input_scaling='unit_cube',
        parameter_transforms={p['name']:p.get('transform','log' if p.get('log') else 'linear')
                              for p in (prior or {}).get('parameters',[])})


def history_payload(history):
    return [packet(o) for o in history]

class NumericAlgorithmV2(Algorithm):
    def __init__(self,out,row,task_prior,initial):
        self.directory=Path(out);self.row=row;self.task_prior=task_prior;self.initial=initial
    @property
    def name(self):return self.row['condition'].lower()
    def setup(self,task_spec,seed=0,**kwargs):
        self.task_spec=task_spec;self.seed=seed;self.history=[]
        group=self.row['condition'];transforms=transformed_overrides(self.task_prior)['parameter_transforms']
        protocol=dict(name='wf130_shared_prefix',initialization=dict(strategy='fixed_configurations',
            seed=seed,count=len(self.initial),configurations=[o.suggestion.config for o in self.initial],source='verified_shared_prefix'))
        spec=replace(task_spec,metadata={**task_spec.metadata,'benchmark_protocol':protocol,'parameter_transforms':transforms})
        overrides={**transformed_overrides(self.task_prior),'acquisition':'noisy_logei'} if group=='GP' else {'success_tolerance':3} if group=='TURBO' else {}
        if group == 'TURBO':
            self.backend=RestartingTurbo(success_tolerance=3, startup_trials=5)
        elif group == 'CMA_ES':
            self.backend=UnitCubeCma()
        elif group in {'RANDOM','TPE'}:
            self.backend=WarpedNative(create_comparable_baseline(BACKENDS[group]))
        else:
            self.backend=create_comparable_baseline(BACKENDS[group],overrides=overrides)
        self.backend.setup(spec,seed=seed)
    def tell(self,observation):
        self.backend.tell(observation);self.history.append(observation)
    def incumbents(self):return self.backend.incumbents()
    def replay(self,history):
        history=list(history);n=self.row['initial'];self.history=[]
        if self.row['condition'] in ['GP','TURBO']:
            self.backend.replay(history);self.history=history
        else:
            self.backend.replay(history[:n]);self.history=history[:n]
            for observation in history[n:]:
                expected=self.ask()
                ExternalOptimizerAdapter.assert_matching_config(expected.config,observation.suggestion.config)
                replayed=replace(observation,suggestion=replace(observation.suggestion,metadata=expected.metadata))
                self.tell(replayed)
    def ask(self):
        n=self.row['initial'];assert len(self.history)>=n
        path=self.directory/'rounds'/f'{len(self.history)-n+1:03d}'/'accepted.json'
        signature=sha(history_payload(self.history))
        with gp_rng(self.seed,len(self.history)):
            for _ in range(10000):
                suggestion=self.backend.ask()
                suggestion.metadata['numeric_revision']='numeric_v2'
                if self.row['condition'] not in ['RANDOM','SOBOL'] or all(suggestion.config!=o.suggestion.config for o in self.history):break
            else:raise RuntimeError('Random/Sobol failed to generate a fresh configuration')
        if self.row['condition']=='GP' and suggestion.metadata.get('gp_ei_phase')!='acquisition':
            atomic_json(path.with_name('rejected.json'),dict(config=suggestion.config,metadata=suggestion.metadata))
            raise RuntimeError('GP failed to produce an acquisition candidate; no silent fallback')
        if self.row['condition']=='TURBO' and suggestion.metadata.get('turbo_phase') not in ['acquisition','restart_initialization','startup_recovery']:
            atomic_json(path.with_name('rejected.json'),dict(config=suggestion.config,metadata=suggestion.metadata))
            raise RuntimeError('TuRBO numerical fallback; inspect instead of relabeling as a normal proposal')
        self.task_spec.search_space.validate_config(suggestion.config)
        value=dict(history_sha256=signature,candidate=suggestion.config,metadata=suggestion.metadata,engine='numeric')
        if path.exists():
            old=read_json(path);assert old['history_sha256']==signature
            ExternalOptimizerAdapter.assert_matching_config(old['candidate'],suggestion.config)
        else:atomic_json(path,value)
        return TrialSuggestion(config=copy.deepcopy(suggestion.config),metadata={**suggestion.metadata,'engine':'numeric'})
