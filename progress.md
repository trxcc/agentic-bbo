# Progress

Language versions:
- English: `progress.md`
- 中文：`progress.zh.md`

## Status

Completed.

## Delivered items

1. Converted the repository into a standard package layout under `bbo/`.
2. Split algorithms into family-based modules under `bbo/algorithms/`, with one file per algorithm in `bbo/algorithms/traditional/`.
3. Split tasks into family-based modules under `bbo/tasks/`, with synthetic benchmark definitions separated into `bbo/tasks/synthetic/`.
4. Preserved and strengthened the core protocol in `bbo/core/`, including validation, replay-friendly logging, task-description schema checks, plotters, and a generic external-optimizer adapter base class.
5. Added bilingual collaborator-facing documentation, including root guides, task-description guides, and implementation notes.
6. Added an explicit task-environment requirement: each task must provide either task-local Docker assets or an `environment.md` with setup instructions.
7. Added bilingual benchmark task-description companions while ensuring localized files are ignored at runtime.
8. Verified the repository with automated tests and runnable demos after the structural refactor.
9. Updated `manifest.json` and this progress report after final verification.

## Primary implementation paths

- `bbo/algorithms/registry.py`
- `bbo/algorithms/traditional/random_search.py`
- `bbo/algorithms/traditional/pycma.py`
- `bbo/core/adapters.py`
- `bbo/tasks/registry.py`
- `bbo/tasks/synthetic/base.py`
- `bbo/tasks/synthetic/branin.py`
- `bbo/tasks/synthetic/sphere.py`
- `bbo/run.py`
- `bbo/core/description.py`
- `bbo/core/plotting.py`
- `README.md`
- `README.zh.md`
- `docs/collaborator_demo.md`
- `docs/collaborator_demo.zh.md`

## Validation completed

- `uv sync --extra dev`
- `uv run python -m compileall -q bbo examples tests`
- `uv run pytest` -> `7 passed`
- `uv run python -m bbo.run --algorithm suite --task branin_demo --results-root artifacts/final_demo_v3`
- `uv run python -m bbo.run --algorithm random_search --task sphere_demo --max-evaluations 5 --results-root artifacts/smoke_cli_v3`
- `uv run python examples/run_branin_suite.py`

## Final reference outputs

### Branin suite

- suite summary: `artifacts/final_demo_v3/branin_demo/suite/seed_7/suite_summary.json`
- comparison plot: `artifacts/final_demo_v3/branin_demo/suite/seed_7/plots/comparison.png`

### Random search

- summary: `artifacts/final_demo_v3/branin_demo/random_search/seed_7/summary.json`
- best observed loss: `1.665515031871971`

### pycma

- summary: `artifacts/final_demo_v3/branin_demo/pycma/seed_7/summary.json`
- best observed loss: `0.6141187323445294`

## Notes

- The current benchmark families are intentionally modest in scope so collaborators can iterate on protocol and packaging before adding larger tasks.
