# `bbo.core` Implementation Notes

Chinese version: `bbo/core/IMPLEMENTATION_PLAN.zh.md`

## Scope

`bbo.core` is the reusable kernel for this repository.
It is intentionally narrow and currently targets serial black-box optimization with replay-friendly logging.

## Included building blocks

- `space.py`: typed search spaces with validation, sampling, and numeric-vector conversion
- `task.py`: task contract, objectives, and sanity checks
- `trial.py`: standardized runtime records
- `logger.py`: append-only JSONL logging and resume state
- `experimenter.py`: serial ask/evaluate/tell orchestration
- `description.py`: standardized markdown task packages for agentic benchmarks
- `plotting.py`: reusable scientific-style plotters
- `adapters.py`: adapter base classes and interoperability helpers, centered on `ExternalOptimizerAdapter`

## Current reference tasks and algorithms

Reference tasks live outside `bbo/core/` in `bbo/tasks/`:

- `branin_demo`
- `sphere_demo`

Reference algorithms live outside `bbo/core/` in `bbo/algorithms/`:

- `RandomSearchAlgorithm`
- `PyCmaAlgorithm` in `bbo/algorithms/traditional/pycma.py`, implemented on top of `ExternalOptimizerAdapter`

## Design constraints

- keep the logger append-only
- make replay the default resume mechanism
- keep task descriptions structured and machine-checkable
- require each task to provide either task-local Docker assets or an `environment.md` file
- allow localized documentation companions without polluting runtime context
- avoid coupling `bbo/core/` to one benchmark's evaluator details

## Intended future direction

This core is designed so that future LLM-based optimizers can reuse:

- the structured task-description bundle
- deterministic run histories
- plot generation hooks
- a clean task/algorithm boundary
