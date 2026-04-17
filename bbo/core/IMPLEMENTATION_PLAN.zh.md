# `bbo.core` 实现说明

English version: `bbo/core/IMPLEMENTATION_PLAN.md`

## 范围

`bbo.core` 是当前仓库的可复用内核。
它刻意保持克制，目前主要面向串行 black-box optimization，并强调可 replay 的 logging / resume 语义。

## 当前包含的模块

- `space.py`：类型化搜索空间，支持校验、采样和数值向量互转
- `task.py`：任务协议、objective 定义和 sanity check
- `trial.py`：标准运行时记录结构
- `logger.py`：append-only JSONL logging 与 resume state
- `experimenter.py`：串行 ask/evaluate/tell 编排
- `description.py`：面向 agentic benchmark 的标准 markdown task package
- `plotting.py`：可复用的科研风格 plotter
- `adapters.py`：adapter 基类与互操作辅助工具，核心是 `ExternalOptimizerAdapter`

## 当前参考任务与算法

参考任务放在 `bbo/core/` 之外的 `bbo/tasks/`：

- `branin_demo`
- `sphere_demo`

参考算法放在 `bbo/core/` 之外的 `bbo/algorithms/`：

- `RandomSearchAlgorithm`
- 位于 `bbo/algorithms/traditional/pycma.py` 中、基于 `ExternalOptimizerAdapter` 实现的 `PyCmaAlgorithm`

## 设计约束

- logger 必须保持 append-only
- resume 默认通过 replay 恢复
- task description 必须结构化、可程序校验
- 每个 task 必须提供 task-local Docker 资产，或提供一个 `environment.md`
- 允许存在本地化文档副本，但不能污染运行时任务上下文
- 不要让 `bbo/core/` 和某一个 benchmark 的 evaluator 细节强耦合

## 未来方向

这个 core 的设计目标，是让未来的 LLM-based optimizer 能直接复用：

- 结构化 task-description bundle
- 可确定重放的 run history
- plot 生成钩子
- 清晰的 task / algorithm 边界
