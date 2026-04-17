# `bbo.core` 协作开发指南

English counterpart for the implementation notes: `bbo/core/IMPLEMENTATION_PLAN.md`

这份文档面向协作者，重点回答两个问题：

- `core/` 这一层到底负责什么
- 新功能应该放进 `core/`，还是放在外层 benchmark/task 层

## 1. 设计定位

`bbo.core` 是一个面向 benchmark 的最小公共层。
它负责提供稳定的协议，让不同任务、不同优化器、不同记录方式能够通过统一接口协同工作。

它不负责：

- 某个具体 benchmark 的 evaluator 细节
- 某个具体 agent 的 prompt 模板
- CLI 业务逻辑
- 任务私有的目录约定之外的运行时 hack

一句话概括：

> `core/` 只放跨任务可复用的协议、数据结构和通用机制。

## 2. 当前模块职责

### `space.py`

负责搜索空间的结构化表达：

- `FloatParam`
- `IntParam`
- `CategoricalParam`
- `SearchSpace`

以及：

- config 校验与 coercion
- 随机采样
- 数值向量互转
- 可选的 ConfigSpace 互操作

### `task.py`

负责任务协议：

- `TaskSpec`
- `ObjectiveSpec`
- `Task`
- `SanityCheckReport`

任务层只应该暴露：

- 静态 spec
- `evaluate()`
- 必要的 task-specific sanity check

### `trial.py`

定义运行时标准数据结构：

- `TrialSuggestion`
- `EvaluationResult`
- `TrialObservation`
- `TrialRecord`

协作开发时，优先扩展这些结构，而不是到处传裸 dict。

### `logger.py`

负责 append-only JSONL logging、resume state、history replay。

设计原则：

- JSONL 是真相来源
- resume 依赖 replay
- 不依赖 optimizer 私有 checkpoint

### `experimenter.py`

负责串行编排：

`sanity_check -> load description -> setup -> replay -> ask -> evaluate -> tell -> log`

### `description.py`

负责标准化任务描述：

- markdown schema
- 章节排序
- fingerprint
- 模板生成

这层不负责 method-specific prompt engineering。

补充约束：

- 每个 task package 必须提供 task-local Docker 工作流，或提供一个 `environment.md`

### `plotting.py`

负责通用可视化对象。

当前包含：

- `OptimizationTracePlotter`
- `ObjectiveDistributionPlotter`
- `Landscape2DPlotter`
- `OptimizerComparisonPlotter`

### `adapters.py`

负责外部优化器适配。

当前这里提供的是 `ExternalOptimizerAdapter` 这样的基础适配器基类，用来承载：

- task 绑定
- replay 辅助逻辑
- incumbent 维护
- 外部优化器常用的通用 helper

具体某个优化器的适配逻辑应放在算法目录中。
当前参考实现是 `bbo/algorithms/traditional/pycma.py` 里的 `PyCmaAlgorithm`，它展示了如何基于这个 base class 把第三方 ask/tell optimizer 接进 core 协议。

## 3. 什么应该放进 `core/`

通常满足以下任一条件，就应该考虑放进 `core/`：

- 对多个 task 都成立
- 对多个 optimizer 都成立
- 属于 ask/tell/replay/logging 的共性协议
- 是 agentic benchmark 通用的 markdown/task schema
- 是通用 plotter 或通用校验逻辑

例子：

- 新的 `TrialStatus`
- 通用 logger
- 通用 plotter
- 通用 search-space 参数类型

## 4. 什么不应该放进 `core/`

通常满足以下任一条件，就不要放进 `core/`：

- 只对某一个 benchmark task 有意义
- 依赖某个外部系统的私有运行方式
- 依赖某个 agent 的 prompt 模板
- 依赖某个任务特有的文件格式或实验逻辑

例子：

- 某个领域专用 evaluator
- 某个任务私有的 metrics 聚合脚本
- 某个 agent 的 prompt 拼接模板
- 只适用于一个 benchmark 的 surface renderer

## 5. 当前核心不变量

协作者修改 `core/` 时，应尽量保持这些不变量：

- `ask()` 只提议，不执行评估
- `evaluate()` 只属于 task
- `tell()` 消费标准化 observation
- logger 保持 append-only
- resume 通过 replay 恢复，而不是隐藏 checkpoint
- task description 必须结构化，而不是单一随手写的 prompt 文件

## 6. 推荐开发流程

1. 先写 task description
2. 再定义 search space 和 objective
3. 再实现 evaluator
4. 再接入 optimizer
5. 最后补 plot 和测试

## 7. 合并前检查

建议至少运行：

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm suite --task branin_demo
```

如果你加了新的 task，请额外确认：

- markdown schema 完整
- 本地化文档副本不会污染运行时 task context
- 提供了 Docker 资产或 `environment.md`
- sanity check 通过
- JSONL 正常写出
- resume 可复现
- plot 正常生成
