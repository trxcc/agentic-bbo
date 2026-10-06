"""Run one frozen paper case, with explicit resume and no host fallback."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from .agent import build_agent, IO_TOOLS, MENUS
from .io import read, save, packet, digest
from .tasks import ASSETS, PaperTask, task_rows


def parser(default_suite="frontier"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('main', 'diagnostic', 'frontier', 'controlled'), default=default_suite)
    p.add_argument('--task')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--algorithm', choices=('agentic', 'direct', 'random', 'sobol', 'gp_ei', 'turbo',
        'tpe', 'cma_es', 'local_perturbation', 'git_bo', 'graph_ga', 'gpbo', 'developed'), default='agentic')
    p.add_argument('--tools', choices=tuple(MENUS), default='T0')
    p.add_argument('--information', choices=('full', 'semantic', 'anonymous'), default='full')
    from .controlled import PRIORS
    p.add_argument('--prior', choices=PRIORS, default='none')
    p.add_argument('--handoff-history', type=Path, help='Role study: take the shared prefix plus 16 BBOB or 8 HPO online evaluations')
    p.add_argument('--program', choices=('bbob_1_v5','bbob_2_v5','bbob_3_v4','hpo_1_v4','hpo_2_v5','hpo_3_v6'))
    p.add_argument('--model-profile', type=Path, help='JSON containing model, api_base, api_key_env and explicit reasoning_effort')
    p.add_argument('--output', type=Path, default=Path('results/run'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Prepare and audit context without model, service or objective calls')
    p.add_argument('--list-tasks', action='store_true')
    p.add_argument('--dbtune-url')
    p.add_argument('--placement-url')
    p.add_argument('--codex-executable')
    p.add_argument('--docker-image', default='agentic-bbo-frontier-agent:v2')
    return p


def load_profile(path, dry_run=False):
    if path is None:
        if not dry_run:
            raise ValueError('Agentic and Direct require --model-profile; no model is selected implicitly')
        return dict(model='PREPARATION_ONLY', api_base='https://model.invalid/v1',
                    api_key_env='BBO_MODEL_API_KEY', reasoning_effort='max')
    profile = read(path)
    required = {'model', 'api_base', 'api_key_env', 'reasoning_effort'}
    if not required <= profile.keys() or not all(profile[k] for k in required):
        raise ValueError('Incomplete model profile: ' + ', '.join(sorted(required)))
    if any(k in profile for k in ('api_key', 'token', 'credential_file')):
        raise ValueError('Pass only an environment variable name, not credentials or credential files')
    allowed = required | {'explicit_thinking_enabled', 'approved_response_model_ids', 'transport'}
    if profile.keys() - allowed:
        raise ValueError('Unknown model-profile keys: ' + ', '.join(sorted(profile.keys() - allowed)))
    target = urlsplit(profile['api_base'])
    if target.scheme not in {'http','https'} or not target.hostname or target.username or target.password or target.query:
        raise ValueError('api_base must be an HTTP(S) URL without embedded credentials or query parameters')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', profile['api_key_env']):
        raise ValueError('api_key_env must be an environment variable name')
    return profile


def build_numerical(task, out, name):
    if task.suite in {'diagnostic', 'controlled'}:
        from bbo.algorithms.baseline_factory import create_comparable_baseline
        return create_comparable_baseline(name)
    from .numerical import NumericAlgorithmV2
    methods = {'random':'RANDOM', 'sobol':'SOBOL', 'gp_ei':'GP', 'turbo':'TURBO', 'tpe':'TPE', 'cma_es':'CMA_ES'}
    if name == 'git_bo':
        from .numeric_git import PaperGitBO
        return PaperGitBO(out, dict(task.row, condition='GIT_BO'), dict(parameters=task.documents['parameters']), task.prefix)
    if name in methods:
        return NumericAlgorithmV2(out, dict(task.row, condition=methods[name]),
                                  dict(parameters=task.documents['parameters']), task.prefix)
    from bbo.algorithms import create_algorithm
    if name in {'graph_ga', 'gpbo'}:
        if task.row['family'] != 'GuacaMol':
            raise ValueError('Molecular algorithms require a SMILES task')
        return create_algorithm(name, initial_smiles=[o.suggestion.config['smiles'] for o in task.prefix])
    return create_algorithm(name)


def run(args):
    from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger
    from bbo.algorithms.agentic import MockAgentEngine
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((out / 'run.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = None if args.dry_run else out / 'host_evaluations'
        if args.suite == 'controlled':
            from .controlled import ControlledPriorTask
            task = ControlledPriorTask(args.task, seed=args.seed, prior=args.prior, journal=journal)
        else:
            task = PaperTask(args.task, suite=args.suite, seed=args.seed, journal=journal,
                             dbtune_url=args.dbtune_url, placement_url=args.placement_url, information=args.information)
        stack.callback(task.cleanup)
        is_llm = args.algorithm in {'agentic', 'direct'}
        if task.row['family']=='GuacaMol' and args.algorithm not in {'agentic','direct','graph_ga','gpbo'}:
            raise ValueError('This numerical method does not support the paper SMILES representation')
        profile = load_profile(args.model_profile, args.dry_run) if is_llm else None
        catalog_path = os.environ.get('BBO_CODEX_MODEL_CATALOG') if args.algorithm == 'agentic' else None
        catalog_sha256 = hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() if catalog_path else None
        prefix = task.prefix
        if args.handoff_history:
            allowed_tasks = {'bbob_f02_d10','bbob_f15_d10','hpo_bayesmark_breast_svm','hpo_bayesmark_diabetes_random_forest'}
            if args.suite != 'diagnostic' or args.task not in allowed_tasks or args.algorithm not in {'gp_ei','turbo','local_perturbation'}:
                raise ValueError('Handoff is restricted to the four role tasks and GP/TuRBO/local search')
            count = task.row['initial'] + (16 if task.row['family']=='BBOB' else 8)
            prefix = JsonlMetricLogger(args.handoff_history).load_history()[:count]
            if len(prefix) != count or [packet(o) for o in prefix[:task.row['initial']]] != [packet(o) for o in task.prefix]:
                raise ValueError('Handoff requires the identical shared initialization and a complete online prefix')
            if not all(o.success for o in prefix) or [o.suggestion.trial_id for o in prefix] != list(range(count)):
                raise ValueError('Invalid handoff prefix')
        contract = dict(row=task.row, suite=args.suite, algorithm=args.algorithm, tools=args.tools,
            information=args.information, profile=profile, docker_image=args.docker_image,
            docker_cpus=32, memory_gib=4, history_limit=task.row['total'],
            initialization_sha256=digest([packet(o) for o in task.prefix]),
            effective_prefix_sha256=digest([packet(o) for o in prefix]), program=args.program,
            context_sha256=digest(task.documents), scoring='rsi_unified_exponential_v2_20260923',
            dbtune_url=args.dbtune_url, placement_url=args.placement_url,
            codex_model_catalog_sha256=catalog_sha256)
        settings_path = out / 'settings.json'
        if settings_path.exists():
            if not args.resume:
                raise ValueError('Output already exists; use a fresh directory or --resume')
            if read(settings_path) != contract:
                raise ValueError('Resume settings or frozen input mismatch')
        else:
            if args.resume:
                raise ValueError('Cannot resume a directory without settings.json')
            save(settings_path, contract)
        if args.algorithm != 'agentic' and args.tools != 'T0':
            raise ValueError('Tool ablations require the Agentic algorithm')
        if not args.dry_run and is_llm and not os.environ.get(profile['api_key_env']):
            raise ValueError('Missing credential environment variable ' + profile['api_key_env'])
        if not args.dry_run:
            try:
                import torch
            except ImportError:
                pass
            else:
                torch.set_num_threads(1)
        api_base = None
        if not args.dry_run and args.algorithm == 'agentic':
            from .transport import audited_transport
            stack.enter_context(audited_transport(out, profile))
            if profile.get('transport') == 'gemini_native':
                from .gemini_native import Gateway
                gateway = Gateway(profile, os.environ[profile['api_key_env']], out)
                gateway.start()
                stack.callback(gateway.close)
                api_base = gateway.base_url
        if args.algorithm == 'agentic':
            agent = build_agent(task, out, profile, tools=args.tools, resume=args.resume,
                engine=MockAgentEngine() if args.dry_run else None, executable=args.codex_executable,
                docker_image=args.docker_image, api_base=api_base)
        elif args.algorithm == 'direct':
            from .direct import DirectAgent
            agent = DirectAgent(task, out, profile, executable=args.codex_executable)
            stack.callback(agent.close)
        elif args.algorithm == 'developed':
            from .developed import load_selected_optimizer
            if not args.program or args.suite != 'diagnostic':
                raise ValueError('Frozen optimizer deployment requires --suite diagnostic and --program')
            if args.task not in {'bbob_f02_d10','bbob_f15_d10','hpo_bayesmark_breast_svm','hpo_bayesmark_diabetes_random_forest'}:
                raise ValueError('Use the four disjoint held-out tasks for frozen deployment')
            family = 'bbob' if task.row['family']=='BBOB' else 'hpo'
            if not args.program.startswith(family+'_'):
                raise ValueError('Frozen program belongs to a different task family')
            agent = load_selected_optimizer(args.program)
        else:
            agent = build_numerical(task, out, args.algorithm)
        if args.dry_run:
            save(out / 'shared_initial.json', [packet(o) for o in prefix])
            if args.algorithm == 'agentic':
                agent.setup(task.spec, seed=args.seed, task_description=task.get_description())
                agent.replay(task.prefix)
                names = {t['function']['name'] for t in agent._agent_tool_specs()}
                if names != set(IO_TOOLS + MENUS[args.tools]):
                    raise ValueError('Incorrect exposed tool menu')
                save(out / 'runtime_audit.json', dict(passed=True, benchmark_tools=sorted(names),
                    execution_backend=agent.config.execution_backend, cpu=agent.config.docker_cpus,
                    context_access=agent.config.context_access, persistent=agent.config.persist_session_across_rounds,
                    timeout=agent.config.timeout_seconds, native_tool_limit=agent.config.max_tool_calls))
                (out / 'first_prompt.txt').write_text(agent._build_agent_prompt(call_id='agent_call_00000', attempt_index=0))
            return dict(status='prepared', task=args.task, initial=task.row['initial'], budget=task.row['budget'],
                        model_calls=0, objective_calls=0, output=str(out))
        logger = JsonlMetricLogger(out / 'trials.jsonl')
        history = logger.load_history()
        if not history:
            logger.bind_run(task_spec=task.spec, algorithm_name=agent.name, seed=args.seed, description_bundle=task.get_description())
            for o in prefix:
                logger.log(o)
        else:
            n = len(prefix)
            if [packet(o) for o in history[:n]] != [packet(o) for o in prefix]:
                raise ValueError('Logged initialization differs from frozen prefix')
            if [o.suggestion.trial_id for o in history] != list(range(len(history))) or not all(o.success for o in history):
                raise ValueError('Invalid or failed logged history; inspect before resuming')
        if len(history) == task.row['total'] and (out / 'summary.json').exists():
            return read(out / 'summary.json')
        summary = Experimenter(task, agent, logger, ExperimentConfig(seed=args.seed, resume=True,
            fail_fast_on_sanity=True, fail_fast_on_evaluation_failure=True)).run()
        result = asdict(summary)
        save(out / 'summary.json', result)
        return dict(status='completed', observations=summary.n_completed, output=str(out))


def main(argv=None, *, default_suite='frontier'):
    p = parser(default_suite)
    args = p.parse_args(argv)
    if args.list_tasks:
        print(json.dumps(task_rows(args.suite), indent=2))
        return 0
    if not args.task:
        p.error('--task is required unless --list-tasks is used')
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
