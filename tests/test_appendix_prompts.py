"""Prompt regressions against the manuscript's archived evidence, without model calls."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from bbo.algorithms.agentic import AgentResult, CodexEngine, MockAgentEngine
from bbo.core import EvaluationResult, TrialObservation, TrialSuggestion
from bbo.experiments.agent import IO_TOOLS, MENUS, build_agent
from bbo.experiments.controlled import ControlledPriorTask
from bbo.experiments.io import packet
from bbo.experiments.tasks import ASSETS, DIAGNOSTIC_TASKS, PaperTask

REFERENCE = json.loads((Path(__file__).parent / 'fixtures/appendix_prompts.json').read_text())
PROFILE = dict(model='PROMPT_AUDIT_ONLY', api_base='http://127.0.0.1:1/v1',
               api_key_env='BBO_PROMPT_AUDIT_KEY', reasoning_effort='max')
REAL_TASKS = tuple(name for name in DIAGNOSTIC_TASKS if not name.startswith('bbob_'))


def build(task, directory, tools='T0', engine=None):
    agent = build_agent(task, directory, PROFILE, tools=tools, engine=engine or MockAgentEngine())
    agent.setup(task.spec, seed=task.row['seed'], task_description=task.get_description())
    agent.replay(task.prefix)
    return agent


@pytest.mark.parametrize('relative,expected', REFERENCE['frontier_files_sha256'].items())
def test_frontier_files_match_appendix_source(relative, expected):
    assert hashlib.sha256((ASSETS / 'frontier' / relative).read_bytes()).hexdigest() == expected


def test_first_and_second_outgoing_messages_match_recorded_appendix(tmp_path):
    class CaptureEngine(CodexEngine):
        async def run_agent(self, session_id, message, work_copy, **kwargs):
            self.message = message
            return AgentResult(status='success', answer='', llm_log={'sessionId':'appendix-audit'})

    async def forbid_tool_execution(*args, **kwargs):
        raise AssertionError('Prompt capture must not invoke benchmark tools')

    task = PaperTask('hpo_bayesmark_digits_mlp_sgd')
    engine = CaptureEngine()
    agent = build(task, tmp_path, engine=engine)
    for index, session_id in enumerate(('', 'appendix-audit')):
        if index:
            recorded_feedback = TrialObservation.from_evaluation(
                TrialSuggestion(config=task.spec.search_space.defaults(), trial_id=5),
                EvaluationResult(objectives={'accuracy':0.9512921022067363}))
            agent.replay([*task.prefix, recorded_feedback])
            agent._campaign_session_id = session_id
        prompt = agent._build_agent_prompt(call_id=f'agent_call_{index:05d}', attempt_index=0)
        asyncio.run(engine._run_with_host_tools(session_id, prompt, agent._work_copy,
            agent_id=None, timeout=None, extra_env=None, tools=agent._agent_tool_specs(),
            tool_executor=forbid_tool_execution, max_tool_calls=0, final_instruction=None))
        expected = REFERENCE['first_user_message' if index == 0 else 'second_user_message']
        assert engine.message == expected


@pytest.mark.parametrize('name', DIAGNOSTIC_TASKS)
def test_tool_ablation_keeps_task_files_and_selection_prompt_fixed(tmp_path, name):
    captures = []
    for menu in MENUS:
        task = PaperTask(name, suite='diagnostic')
        agent = build(task, tmp_path / menu, tools=menu)
        files = {n:(agent._workspace_dir / n).read_bytes() for n in
                 ('task.md','instructions.md','task_details.json','parameter_catalog.json','space.json','objective.json')}
        prompt = agent._build_agent_prompt(call_id='agent_call_00000', attempt_index=0)
        assert {t['function']['name'] for t in agent._agent_tool_specs()} == set(IO_TOOLS + MENUS[menu])
        captures.append((files, prompt))
    assert captures[0] == captures[1] == captures[2]
    if name == 'hpo_bayesmark_breast_svm':
        assert captures[0][1].split('\n\nUse only supplied')[0] == REFERENCE['tool_task_prompt']


@pytest.mark.parametrize('name', REAL_TASKS)
@pytest.mark.parametrize('seed', (2,3,4,5))
def test_i1_changes_only_prior_sections_and_task_card_index(name, seed):
    full = PaperTask(name, suite='diagnostic', seed=seed, information='full')
    semantic = PaperTask(name, suite='diagnostic', seed=seed, information='semantic')
    expected = {k:v for k,v in full.documents['sections'].items() if k not in {'mechanisms','domain_knowledge'}}
    assert semantic.documents['sections'] == expected
    assert semantic.documents['parameters'] == full.documents['parameters']
    assert semantic.spec.primary_objective == full.spec.primary_objective
    assert [packet(o) for o in semantic.prefix] == [packet(o) for o in full.prefix]
    old_index = 'get_task_context sections: '+', '.join(full.documents['sections'])+'.'
    new_index = 'get_task_context sections: '+', '.join(expected)+'.'
    assert semantic.documents['short_task'] == full.documents['short_task'].replace(old_index,new_index)
    assert f"{full.row['initial']} shared initial observations and {full.row['budget']} new evaluations" in semantic.documents['short_task']


def test_i1_i2_workspace_protocol_and_parameter_files_are_identical(tmp_path):
    agents = [build(PaperTask('hpo_bayesmark_breast_svm', suite='diagnostic', information=level), tmp_path / level)
              for level in ('semantic','full')]
    for name in ('instructions.md','parameter_catalog.json','space.json','objective.json','context_tools.json'):
        assert (agents[0]._workspace_dir / name).read_bytes() == (agents[1]._workspace_dir / name).read_bytes()


def test_i0_breast_card_and_hidden_decoding_match_appendix():
    task = PaperTask('hpo_bayesmark_breast_svm', suite='diagnostic', information='anonymous')
    assert set(task.documents['sections']) == {'overview','submission'}
    assert task.documents['sections']['overview'] == (
        '# Anonymous optimization task\n\n'
        'Optimize an unknown scalar objective over 3 numerical inputs in [0,1].\n\n'
        'Objective: minimize `value`.\n\n'
        'Budget: 5 shared initial observations and 25 new evaluations (30 total).')
    assert [p['name'] for p in task.documents['parameters']] == ['x1','x2','x3']
    assert all(p['type']=='float' and p['low']==0 and p['high']==1 and p['default']==.5
               for p in task.documents['parameters'])
    assert task.codec.decode({'x1':.2,'x2':.4,'x3':.6}) == pytest.approx(
        {'C':10**.6,'gamma':10**(-3.6),'tol':10**(-2.6)})


@pytest.mark.parametrize('condition,expected', REFERENCE['controlled_task_a'].items())
def test_controlled_prior_text_matches_appendix(condition, expected):
    task = ControlledPriorTask('coarse_geometry_task_001', prior=condition)
    assert task.documents['sections']['prior_information'] == expected
    assert expected in task.documents['short_task']


def test_development_artifacts_are_the_approved_version_not_the_review_draft():
    folder = ASSETS / 'development_protocol'
    for relative, expected in REFERENCE['development_files_sha256'].items():
        assert hashlib.sha256((folder / relative).read_bytes()).hexdigest() == expected
    assert (folder / 'instructions.en.md').read_text().strip() == REFERENCE['development_role'].strip()
    assert (folder / 'later_prompt_template.md').read_text().strip() == REFERENCE['development_later'].strip()
