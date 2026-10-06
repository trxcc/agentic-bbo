import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
import urllib.error
import urllib.request

import numpy as np
import pytest

from bbo.experiments.tasks import ASSETS, PaperTask, task_rows
from bbo.experiments.scoring import score_losses, reference_spec, quality
from bbo.experiments.agent import build_agent, IO_TOOLS, MENUS
from bbo.algorithms.agentic import MockAgentEngine
from bbo.core import TrialSuggestion


def test_optional_codex_catalog_is_used_only_when_selected(tmp_path, monkeypatch):
    task = PaperTask('bbob_f15_d10', suite='frontier')
    profile = dict(model='fixture-model', api_base='http://127.0.0.1:1/v1',
                   api_key_env='BBO_TEST_KEY', reasoning_effort='high')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'models': [{'slug': 'fixture-model'}]}))
    monkeypatch.setenv('BBO_CODEX_MODEL_CATALOG', str(catalog))
    agent = build_agent(task, tmp_path / 'with-catalog', profile, engine=MockAgentEngine())
    agent.setup(task.spec, seed=2, task_description=task.get_description())
    copied = agent._state_dir / 'isolated_codex' / 'model_catalog.json'
    assert json.loads(copied.read_text()) == json.loads(catalog.read_text())
    monkeypatch.delenv('BBO_CODEX_MODEL_CATALOG')
    agent = build_agent(task, tmp_path / 'without-catalog', profile, engine=MockAgentEngine())
    agent.setup(task.spec, seed=2, task_description=task.get_description())
    assert not (agent._state_dir / 'isolated_codex' / 'model_catalog.json').exists()


def test_resume_rejects_changed_codex_catalog(tmp_path, monkeypatch):
    from bbo.experiments.run import main
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'models': []}))
    monkeypatch.setenv('BBO_CODEX_MODEL_CATALOG', str(catalog))
    output = tmp_path / 'run'
    command = ['--task', 'bbob_f15_d10', '--dry-run', '--output', str(output)]
    assert main(command) == 0
    catalog.write_text(json.dumps({'models': [{'slug': 'different-model'}]}))
    with pytest.raises(ValueError, match='Resume settings or frozen input mismatch'):
        main(command + ['--resume'])


def test_exact_paper_inventory_and_initializations():
    from bbo.tasks import TASK_FAMILIES
    assert {k: len(v) for k,v in TASK_FAMILIES.items()} == dict(BBOB=24, HPO=25, DBTune=6, BBOPlace=12, GuacaMol=10)
    assert len(task_rows('main')) == 308
    assert len(task_rows('diagnostic')) == 32
    assert len(task_rows('frontier')) == 5
    for suite in ('main','diagnostic','frontier'):
        for row in task_rows(suite):
            task = PaperTask(row['task'], suite=suite, seed=row['seed'])
            assert len(task.prefix) == row['initial']
            assert task.spec.max_evaluations == row['initial'] + row['budget']
            assert task.sanity_check().ok
            assert task.raw is None


@pytest.mark.parametrize('row', task_rows('frontier'), ids=lambda r:r['task'])
def test_frontier_context_and_runtime_are_frozen(tmp_path, row):
    task = PaperTask(row['task'])
    agent = build_agent(task, tmp_path, dict(model='mock', api_base='http://127.0.0.1:1/v1',
        api_key_env='BBO_TEST_KEY', reasoning_effort='max'), engine=MockAgentEngine())
    agent.setup(task.spec, seed=2, task_description=task.get_description())
    agent.replay(task.prefix)
    snapshot = ASSETS / 'frontier' / row['id']
    for name in ('task.md','instructions.md','space.json','objective.json','task_details.json','parameter_catalog.json'):
        actual = (agent._workspace_dir / name).read_text()
        expected = (snapshot / name).read_text()
        assert (json.loads(actual) == json.loads(expected)) if name.endswith('.json') else actual == expected
    assert {t['function']['name'] for t in agent._agent_tool_specs()} == set(IO_TOOLS)
    assert agent.config.execution_backend == 'isolated_docker'
    assert agent.config.docker_cpus == 32
    assert agent.config.max_tool_calls == 0 and agent.config.timeout_seconds is None
    assert not agent.config.enable_memory and not agent.config.allow_fallback
    assert not agent._state_dir.is_relative_to(agent._workspace_dir)


@pytest.mark.parametrize('menu', ('T0','T1','T4'))
def test_tool_menus(tmp_path, menu):
    task=PaperTask('bbob_f02_d10',suite='diagnostic')
    agent=build_agent(task,tmp_path,dict(model='mock',api_base='http://127.0.0.1:1/v1',
        api_key_env='BBO_TEST_KEY',reasoning_effort='max'),tools=menu,engine=MockAgentEngine())
    agent.setup(task.spec,seed=2,task_description=task.get_description())
    assert {t['function']['name'] for t in agent._agent_tool_specs()} == set(IO_TOOLS+MENUS[menu])


def test_scoring_gp_anchor_tail_and_checkpoint_order():
    spec=reference_spec([10,12,14],gp_loss=5,family='HPO')
    assert quality(-10,spec)==0
    assert quality(-5,spec)==pytest.approx(.6)
    assert quality(0,spec)==pytest.approx(1-.4*np.exp(-1.5))
    result,q,_=score_losses([10,12,5,0],2,2,gp_loss=5,family='HPO')
    assert result['composite_score']==pytest.approx(.7*np.mean(q)+.3*q[-1])
    assert q[0]==pytest.approx(.6)
    assert q[-1]<1


def test_scoring_molecular_no_gp_anchor_and_early_end():
    a=score_losses([.8,.6,.4],2,2,upper_loss=0,gp_loss=.5,family='GuacaMol',early_end=True)
    b=score_losses([.8,.6,.4],2,2,upper_loss=0,gp_loss=.1,family='GuacaMol',early_end=True)
    assert a[1]==b[1]==pytest.approx([1/3,1/3])
    assert a[0]['filled_checkpoints']==1
    assert not a[0]['gp_anchor_used']
    with pytest.raises(ValueError,match='Incomplete'):
        score_losses([.8,.6,.4],2,2,upper_loss=0,family='GuacaMol')


def test_evaluator_journal_refuses_unresolved_or_changed_requests(tmp_path):
    from bbo.experiments.io import save
    task=PaperTask('bbob_f15_d10',journal=tmp_path)
    config=task.prefix[0].suggestion.config
    save(tmp_path/'evaluations/request_20.json',dict(trial_id=20,config=config))
    with pytest.raises(RuntimeError,match='Unresolved'):
        task.evaluate(TrialSuggestion(config=config,trial_id=20))
    with pytest.raises(ValueError,match='outside'):
        task.evaluate(TrialSuggestion(config=config,trial_id=120))
    task.cleanup()


def test_all_placement_bundles_replay_without_mgo():
    from bbo.tasks.bboplace.repair_backend import RepairEvaluator, PlacementData
    from bbo.tasks.bboplace.local_service import BUNDLE_ROOT
    files=list(BUNDLE_ROOT.glob('*.json'))
    assert len(files)==48
    assert not hasattr(PlacementData,'mgo')
    assert not hasattr(RepairEvaluator,'suggest')
    for path in files:
        packet=json.loads(path.read_text())
        evaluator=RepairEvaluator(packet)
        item=packet['initializations'][0]
        actual=evaluator.evaluate([item['config'][k] for k in evaluator.data.config_names])
        assert actual['hpwl']==item['hpwl']
        assert actual['repair_fallback']==item['repair_fallback']
        assert len(evaluator.data.macros)==32
        assert all(m.width>0 and m.height>0 for m in evaluator.data.macros)


def test_placement_service_has_no_mgo_endpoint():
    from bbo.tasks.bboplace.local_service import BBOPlaceLocalBridge, _Handler
    bridge=BBOPlaceLocalBridge()
    server=ThreadingHTTPServer(('127.0.0.1',0),type('Handler',(_Handler,),{'bridge':bridge}))
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    try:
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/mgo_suggest',b'{}')
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code==404
    finally:
        server.shutdown();server.server_close();worker.join()


def test_seven_controlled_priors_share_the_same_objectives_and_prefix():
    from bbo.experiments.controlled import ControlledPriorTask
    rows=task_rows('controlled')
    assert len(rows)==84
    groups={}
    for row in rows:
        task=ControlledPriorTask(row['task'], seed=row['seed'], prior=row['prior'])
        prefix=[(o.suggestion.config,o.objectives) for o in task.prefix]
        key=(row['task'],row['seed'])
        assert groups.setdefault(key,prefix)==prefix
        for config,objectives in prefix:
            assert task.definition.evaluate(config)==pytest.approx(objectives['value'],rel=1e-12)
        visible=json.dumps(task.documents)
        assert 'active_variables' not in visible and 'quadratic_coefficients' not in visible
    geometry=ControlledPriorTask('coarse_geometry_task_001',prior='geometry').documents['sections']['prior_information']
    assert 'exactly four' not in geometry


def test_six_selected_programs_load_without_source_changes():
    from bbo.experiments.developed import load_selected_optimizer
    for name in ('bbob_1_v5','bbob_2_v5','bbob_3_v4','hpo_1_v4','hpo_2_v5','hpo_3_v6'):
        optimizer=load_selected_optimizer(name)
        assert all(callable(getattr(optimizer,method)) for method in ('setup','ask','tell','replay','incumbents'))
