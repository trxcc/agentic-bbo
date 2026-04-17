# Collaborator Demo: How to Package a New Benchmark Task

Chinese version: `docs/collaborator_demo.zh.md`

This document shows the intended workflow for a collaborator who wants to add a new benchmark problem for future LLM agents.
The target is not only to make the code runnable, but also to make the task package inspectable, standardized, and replay-friendly.

## 1. Start from the task-description schema

Create a new directory:

```text
bbo/task_descriptions/<task_name>/
```

Copy the scaffold from `bbo/task_descriptions/_template/` and fill in at least:

- `background.md`
- `goal.md`
- `constraints.md`
- `prior_knowledge.md`

Recommended optional files:

- `evaluation.md`
- `submission.md`
- `environment.md`
- `notes.md`
- `history.md`

For documentation purposes, you may add localized companions such as `background.zh.md`; the loader ignores these localized files during benchmark execution.

Every task package must also provide at least one reproducible environment path:

- a task-local Docker workflow committed with the task package
- or explicit setup instructions in `environment.md`

## 2. Define the search space explicitly

Use `bbo.core.SearchSpace` and typed parameter specs instead of passing around free-form dicts.

```python
from bbo.core import FloatParam, ObjectiveDirection, ObjectiveSpec, SearchSpace, TaskSpec

space = SearchSpace(
    [
        FloatParam("temperature", low=20.0, high=120.0, default=60.0),
        FloatParam("flow_rate", low=0.1, high=2.0, default=1.0),
    ]
)

spec = TaskSpec(
    name="lab_pipeline_demo",
    search_space=space,
    objectives=(ObjectiveSpec("quality_loss", ObjectiveDirection.MINIMIZE),),
    max_evaluations=40,
)
```

Why this matters:

- algorithms can validate suggestions before evaluation
- tasks can compute deterministic defaults
- numeric optimizers can convert configs to vectors safely
- assertions fail early when a collaborator defines an inconsistent benchmark

## 3. Implement a concrete `Task`

Concrete benchmark logic should normally stay outside `bbo/core/`.
The task owns only task-specific behavior: how to evaluate one suggestion and how to expose static metadata.

```python
from bbo.core import EvaluationResult, ObjectiveDirection, ObjectiveSpec, Task, TaskDescriptionRef, TaskSpec, TrialStatus

class MyTask(Task):
    def __init__(self):
        self._spec = TaskSpec(
            name="lab_pipeline_demo",
            search_space=space,
            objectives=(ObjectiveSpec("quality_loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=40,
            description_ref=TaskDescriptionRef.from_directory(
                "lab_pipeline_demo",
                "bbo/task_descriptions/lab_pipeline_demo",
            ),
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion):
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        loss = expensive_or_simulated_evaluator(config)
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"quality_loss": float(loss)},
            metrics={"flow_rate": float(config["flow_rate"])},
        )
```

## 4. Add sanity checks that fail early

Useful sanity checks include:

- required markdown sections exist
- defaults fall inside bounds
- objective names are unique
- known optima or reference points match the search-space dimensionality
- success results always include the primary objective
- logged values are finite

The repo already performs many of these checks in `Task.sanity_check()` and `Experimenter._validate_result()`.
If a new task has domain-specific invariants, add them in the task subclass.

## 5. Keep logging append-only

Use `JsonlMetricLogger` as the source of truth.
Do not hide optimizer state in opaque checkpoint files if replay can reconstruct it.
This is especially important for agentic work because collaborators need to inspect runs after the fact.

Each JSONL line stores:

- the evaluated config
- suggestion metadata
- objective values
- auxiliary metrics
- timing
- task-description fingerprint

## 6. Reuse the plotters

If the task is visualizable, reuse the plotters in `bbo.core.plotting`.
The current repository provides:

- `OptimizationTracePlotter`
- `ObjectiveDistributionPlotter`
- `Landscape2DPlotter`
- `OptimizerComparisonPlotter`

If a new problem needs a custom visualization, follow the same design:

- make it an object
- save files directly to disk
- keep the style restrained and scientific
- make the plot reusable by more than one demo script if possible

## 7. Provide one lightweight runnable example

A collaborator-facing task should come with a command that runs in seconds.
That demo should:

- create the task
- create one or two algorithms
- execute the experiment loop
- save JSONL logs
- save plots
- write a summary JSON file

The current reference example is `examples/run_branin_suite.py`.

## 8. Validation checklist before merging

Run all of these:

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm suite --task branin_demo
```

If you add a new task, also run its dedicated smoke-test command and confirm:

- task sanity checks pass
- the task provides either Docker assets or `environment.md`
- JSONL is written
- resume works
- plots are generated
- task descriptions are complete

## 9. What belongs in `bbo/core/` versus outside it

Put code in `bbo/core/` only when it is reusable across many tasks.
Examples that belong in `bbo/core/`:

- search-space primitives
- logging protocol
- plotting utilities that are benchmark-agnostic
- ask/tell orchestration

Examples that should stay outside `bbo/core/`:

- domain-specific evaluators
- benchmark-specific heuristics
- prompt wording tied to one task
- any simulator or subprocess runner that only one benchmark uses
