"""Declared geometry is shared by GP fitting, predictions and bounded acquisition."""
import numpy as np
import pytest
from bbo.core import SearchSpace, FloatParam, IntParam, TaskSpec, ObjectiveSpec, ObjectiveDirection, TrialObservation, TrialSuggestion, EvaluationResult
from bbo.core.conversion import UnitCubeSearchSpaceConverter
from bbo.algorithms.model_based.gp_ei import GpEiAlgorithm, _candidate_feature_bounds

def spec():
    return TaskSpec(name='transforms',search_space=SearchSpace([
        FloatParam('rate',low=1e-6,high=1.),FloatParam('p',low=.01,high=.99),IntParam('n',low=1,high=100)]),
        objectives=(ObjectiveSpec('loss',ObjectiveDirection.MINIMIZE),),max_evaluations=12)

def history():
    return [TrialObservation.from_evaluation(TrialSuggestion(dict(rate=r,p=p,n=n),trial_id=i),
        EvaluationResult(objectives={'loss':float(i+1)})) for i,(r,p,n) in enumerate([(1e-6,.01,1),(.001,.5,10),(.1,.9,50),(1.,.99,100)])]

TRANSFORMS=dict(rate='log',p='logit',n='log')

def test_transformed_box_uses_original_warp_including_fixed_axes():
    converter=UnitCubeSearchSpaceConverter(spec().search_space,transforms=TRANSFORMS)
    candidate=SearchSpace([FloatParam('rate',low=1e-4,high=.01),FloatParam('p',low=.5,high=.5),IntParam('n',low=10,high=100)])
    bounds=_candidate_feature_bounds(converter,candidate)
    assert np.allclose(bounds,[[1/3,2/3],[.5,.5],[.5,1.]])
    assert np.allclose(_candidate_feature_bounds(converter,spec().search_space),[[0,1]]*3)
    for endpoint in [0,1]:
        config=converter.decode_vector(bounds[:,endpoint],clip=True)
        from bbo.algorithms.model_based.gp_ei import _repair_candidate_roundoff
        candidate.validate_config(_repair_candidate_roundoff(config,candidate))

def test_no_transforms_preserves_legacy_numeric_encoding():
    algorithm=GpEiAlgorithm(input_scaling='unit_cube');algorithm.setup(spec())
    vector=algorithm._converter.encode_vector(dict(rate=.001,p=.5,n=10))
    assert np.array_equal(vector,[.001,.5,10])

def test_transformed_training_scaling_is_identical_before_and_after_restriction():
    algorithm=GpEiAlgorithm(input_scaling='unit_cube',parameter_transforms=TRANSFORMS)
    algorithm.setup(spec());algorithm.replay(history())
    x,y=algorithm._diagnostic_training_arrays(history())
    transformed,offset,scale=algorithm._scale_training_features(x)
    assert np.allclose(x,transformed) and np.allclose(offset,0) and np.allclose(scale,1)
    assert np.allclose(x[1],[.5,.5,.5])
    algorithm.set_candidate_search_space(SearchSpace([FloatParam('rate',low=1e-4,high=.01),FloatParam('p',low=.5,high=.9),IntParam('n',low=10,high=100)]))
    assert np.array_equal(algorithm._scale_training_features(x)[0],transformed)

def test_transformed_tool_region_full_history_prediction_and_global_escape(tmp_path):
    pytest.importorskip('botorch')
    from bbo.algorithms.agentic.optimizer_backend import StatefulOptimizerBackend
    settings=dict(kernel='matern52',input_scaling='unit_cube',parameter_transforms=TRANSFORMS,pool_size=32,acqf_num_restarts=2)
    backend=StatefulOptimizerBackend(allowlist=['gp_ei'],state_path=tmp_path/'state.json',gp_overrides=settings)
    backend.restore(dict(backend='gp_ei',bounds={},acquisition=dict(name='noisy_logei',parameters={})))
    def call(name,args):return backend.execute(name,task_spec=spec(),history=history(),seed=2,incumbent=history()[0].suggestion.config,arguments=args)
    direct=GpEiAlgorithm(**settings,acquisition='noisy_logei');direct.setup(spec(),seed=2);direct.replay(history())
    proposal=call('optimizer_suggest',{})
    assert direct.ask().config==proposal['candidate']
    call('optimizer_set_bounds',{'bounds':{'rate':[1e-4,.01]}})
    state=backend.snapshot()
    region=call('optimizer_suggest',{'around':{'center':dict(rate=.001,p=.5,n=10),'vary':['rate'],'radius':.01}})
    assert region['candidate']['n']==10 and region['candidate']['p']==.5
    assert .000901-1e-12 <= region['candidate']['rate'] <= .001099+1e-12
    assert region['backend_history_size']==4
    assert region['suggestion_metadata']['gp_ei_phase']=='acquisition'
    assert backend.snapshot()==state
    assert call('optimizer_suggest',{'bounds':{}})['candidate']==proposal['candidate']
    configs=[dict(rate=.1,p=.9,n=50),region['candidate']]
    expected=direct.evaluate_virtual_configs(configs,include_acquisition=True)
    actual=call('optimizer_score',{'configs':configs})
    for a,b in zip(expected,actual['predictions']):
        assert a==b
    assert actual['budget_consumed'] is False and actual['evaluator_called'] is False
