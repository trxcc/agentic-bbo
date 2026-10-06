"""The paper's isolated persistent workspace and tool-ablation menus."""
import copy
import json
import os
import shutil
from pathlib import Path

from bbo.algorithms.agentic.raw_agentic_bbo import create_raw_agentic_bbo
from bbo.core.context_policy import context_fingerprint
from .io import read, save

IO_TOOLS = ('get_task_context', 'get_search_space', 'get_trial_history',
            'get_incumbent', 'write_candidate', 'submit_candidate')
GP_TOOLS = ('optimizer_suggest', 'optimizer_predict', 'optimizer_score', 'optimizer_diagnostics',
            'optimizer_status', 'optimizer_set_bounds', 'optimizer_set_acquisition')
MENUS = {'T0': (), 'T1': ('optimizer_suggest',), 'T4': GP_TOOLS}
BOUNDARY = '''Use only supplied task context and host-evaluated observations. Do not inspect
hidden evaluators, task datasets, netlists, surrogate weights, other runs, parent
directories, repository files, external services or host credentials. Do not
perform unbudgeted objective evaluations. Surrogate modeling fitted only to
visible observations and scratch work below scratch/ are allowed. Only the host
evaluates a submitted candidate.
Internet search, downloads and remote tools are unavailable. Do not recreate
the objective or calculate its target-specific scoring components locally.
General numerical analysis and chemistry syntax/structure checks are allowed;
only host-returned evaluations may supply target scores. Do not install packages.'''


def build_agent(task, out, profile, *, tools='T0', resume=False, engine=None,
                executable=None, docker_image='agentic-bbo-frontier-agent:v2', api_base=None):
    agent = create_raw_agentic_bbo(model=profile['model'], provider='chat_completions',
        api_base=api_base or profile['api_base'], api_key_env=profile['api_key_env'],
        run_dir=out, resume=resume, engine=engine, executable=executable,
        history_limit=task.row['total'], initial_random=task.row['initial'], max_retries=3,
        docker_image=docker_image, enabled_tool_names=IO_TOOLS + MENUS[tools],
        optimizer_backend_allowlist=('gp_ei',) if tools != 'T0' else (),
        optimizer_max_calls_per_round=64,
        context_profile='a0_p0' if task.anonymous else 'readable_domain_prior',
        experiment_condition=f'{task.suite}_{tools}')

    def context(spec):
        agent._rendered_agent_context = task.documents['short_task']
        agent._context_io_documents = copy.deepcopy(task.documents)
        agent._agent_context_fingerprint = context_fingerprint(json.dumps(task.documents, sort_keys=True), agent.config.context_policy)
        if task.anonymous:
            agent._agent_task_alias = task.spec.name

    original_prompt, original_workspace = agent._build_agent_prompt, agent._write_workspace_context
    original_config = agent._build_framework_config
    def workspace():
        original_workspace()
        path = agent._workspace_dir / 'instructions.md'
        path.write_text(task.instructions if task.instructions is not None else path.read_text() + '\n\n' + BOUNDARY + '\n')
        path = agent._workspace_dir / 'manifest.json'
        manifest = read(path)
        manifest['memory_policy']['enabled'] = False
        manifest['research_policy']['allow_external_research'] = False
        save(path, manifest)
        agent._sync_workspace_snapshot()

    def config(log_dir):
        path = original_config(log_dir)
        text = 'project_doc_max_bytes = 0\n' + path.read_text()
        text = text.replace('[features]\n', '[features]\nskip_host_skill_discovery = true\nmemories = false\nhooks = false\n', 1)
        catalog_path = os.environ.get('BBO_CODEX_MODEL_CATALOG')
        if catalog_path:
            catalog = read(Path(catalog_path))
            if not isinstance(catalog, dict) or not isinstance(catalog.get('models'), list):
                raise ValueError('BBO_CODEX_MODEL_CATALOG must contain a models array')
        else:
            catalog = None
        if catalog is not None and profile['model'] in {m.get('slug') for m in catalog['models']}:
            state = agent._state_dir / 'isolated_codex'
            state.mkdir(exist_ok=True)
            shutil.copyfile(catalog_path, state / 'model_catalog.json')
            text = 'model_catalog_json="/state/model_catalog.json"\n' + text
        path.write_text(text)
        return path

    agent._prepare_agent_context = context
    agent._write_workspace_context = workspace
    agent._build_framework_config = config
    agent._build_agent_prompt = lambda **kw: original_prompt(**kw) + '\n\n' + BOUNDARY
    return agent
