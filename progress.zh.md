# 进度说明

语言版本：
- English: `progress.md`
- 中文：`progress.zh.md`

## 状态

已完成。

## 已交付内容

1. 将仓库整理为标准的 `bbo/` 包结构。
2. 将算法按家族拆分到 `bbo/algorithms/`，并在 `bbo/algorithms/traditional/` 中做到“一个算法一个文件”。
3. 将任务按家族拆分到 `bbo/tasks/`，并把 synthetic benchmark 定义拆分到 `bbo/tasks/synthetic/`。
4. 保留并增强了 `bbo/core/` 中的核心协议，包括校验、可 replay 的 logging、task-description schema 检查、plotter，以及通用的外部优化器 adapter 基类。
5. 补齐了双语协作文档，包括根 README、task-description 指南和实现说明。
6. 明确增加了任务环境要求：每个 task 必须提供 task-local Docker 资产，或提供写在 `environment.md` 中的配置说明。
7. 为 benchmark task description 增加了双语对照文档，同时保证本地化副本不会在运行时被 loader 读入。
8. 在结构重构完成后，重新通过了自动化测试和可运行 demo 验证。
9. 在最终验证后更新了 `manifest.json` 和本进度文档。

## 主要实现路径

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

## 已完成验证

- `uv sync --extra dev`
- `uv run python -m compileall -q bbo examples tests`
- `uv run pytest` -> `7 passed`
- `uv run python -m bbo.run --algorithm suite --task branin_demo --results-root artifacts/final_demo_v3`
- `uv run python -m bbo.run --algorithm random_search --task sphere_demo --max-evaluations 5 --results-root artifacts/smoke_cli_v3`
- `uv run python examples/run_branin_suite.py`

## 最终参考产物

### Branin suite

- suite summary: `artifacts/final_demo_v3/branin_demo/suite/seed_7/suite_summary.json`
- comparison plot: `artifacts/final_demo_v3/branin_demo/suite/seed_7/plots/comparison.png`

### Random search

- summary: `artifacts/final_demo_v3/branin_demo/random_search/seed_7/summary.json`
- best observed loss: `1.665515031871971`

### pycma

- summary: `artifacts/final_demo_v3/branin_demo/pycma/seed_7/summary.json`
- best observed loss: `0.6141187323445294`

## 备注

- 当前 benchmark family 的规模保持在相对紧凑的范围内，便于协作者先围绕协议和封装方式迭代，再逐步扩展到更大的任务。
