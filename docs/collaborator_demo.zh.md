# 协作者示例：如何封装一个新的 Benchmark Task

English version: `docs/collaborator_demo.md`

这份文档展示的是：如果协作者想给未来的 LLM agent 增加一个新的 benchmark 任务，推荐应该怎么做。
目标不只是“代码能跑”，而是要让任务包本身可读、规范、可 replay、可协作。

## 1. 先从 task-description schema 开始

先创建一个新目录：

```text
bbo/task_descriptions/<task_name>/
```

从 `bbo/task_descriptions/_template/` 复制模板，并至少填写：

- `background.md`
- `goal.md`
- `constraints.md`
- `prior_knowledge.md`

推荐可选文件：

- `evaluation.md`
- `submission.md`
- `environment.md`
- `notes.md`
- `history.md`

如果为了协作需要，也可以额外放诸如 `background.zh.md` 这样的中文对照文件；loader 在 benchmark 真正运行时会自动忽略这些本地化副本。

此外，每个 task package 都必须至少提供一种可复现的环境路径：

- 与任务一起提交的 task-local Docker 工作流
- 或者写在 `environment.md` 中的显式环境配置说明

## 2. 显式定义搜索空间

使用 `bbo.core.SearchSpace` 和类型化参数，而不是到处传裸 dict。

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

这样做的好处：

- 算法可以在评估前校验 suggestion
- task 可以有确定的默认值
- 数值优化器可以安全地把 config 转成向量
- 协作者一旦定义了不一致的 benchmark，可以尽早触发 assert 或 validation error

## 3. 实现一个具体 `Task`

具体 benchmark 逻辑通常不要放进 `bbo/core/`。
`Task` 自己只负责任务私有行为：如何评估一个 suggestion，以及如何暴露静态 spec。

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

## 4. 添加尽早失败的 sanity checks

比较有价值的 sanity check 包括：

- 必需 markdown 章节是否存在
- 默认值是否在合法范围内
- objective 名称是否唯一
- 已知 optimum 或参考点的维度是否与搜索空间匹配
- 成功 trial 是否一定返回主目标
- 日志中的数值是否有限

仓库里已经在 `Task.sanity_check()` 和 `Experimenter._validate_result()` 中做了很多通用检查。
如果新任务还有领域特定的不变量，就在具体 task 子类里继续补充。

## 5. 保持 logging 是 append-only

请把 `JsonlMetricLogger` 作为事实来源。
如果 replay 能恢复状态，就不要再额外依赖 opaque checkpoint 文件。
这对 agentic benchmark 特别重要，因为协作者需要在事后检查整个运行过程。

每条 JSONL 至少记录：

- 当前评估的 config
- suggestion metadata
- objective 值
- 辅助 metrics
- timing
- task-description fingerprint

## 6. 尽量复用 plotter

如果任务适合可视化，就优先复用 `bbo.core.plotting` 里的 plotter。
当前仓库提供：

- `OptimizationTracePlotter`
- `ObjectiveDistributionPlotter`
- `Landscape2DPlotter`
- `OptimizerComparisonPlotter`

如果新问题确实需要自定义图，也建议保持同样设计风格：

- 做成对象
- 直接保存到磁盘
- 风格克制、偏科研展示
- 如果可能，尽量让它能被多个 demo 复用

## 7. 至少提供一个轻量且可运行的例子

一个面向协作者的 task，最好要有一个几秒内能跑通的命令。
这个 demo 至少应当：

- 创建 task
- 创建一个或两个算法
- 执行 experiment loop
- 保存 JSONL 日志
- 保存 plots
- 输出 summary JSON

当前仓库的参考实现是 `examples/run_branin_suite.py`。

## 8. 合并前检查清单

建议至少执行：

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm suite --task branin_demo
```

如果你增加了新的 task，还应额外确认：

- task sanity checks 通过
- 任务提供了 Docker 资产或 `environment.md`
- JSONL 正常写出
- resume 正常工作
- plots 正常生成
- task description 完整可读

## 9. 什么应该放在 `bbo/core/`，什么不应该

只有在一个功能对多个 task 都可复用时，才建议放进 `bbo/core/`。
适合放进 `bbo/core/` 的内容：

- 搜索空间原语
- logging 协议
- benchmark-agnostic 的 plotting 工具
- ask/tell 编排逻辑

不适合放进 `bbo/core/` 的内容：

- 领域专用 evaluator
- 某个 benchmark 私有 heuristic
- 只服务一个任务的 prompt 模板
- 只属于某个 benchmark 的 simulator 或 subprocess runner
