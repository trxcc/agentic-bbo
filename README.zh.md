# Agentic BBO Benchmark Core（中文版）

语言版本：
- English: `README.md`
- 中文：`README.zh.md`

## 概述

这个仓库提供了一个紧凑但结构化的 agentic black-box optimization benchmark 框架。
仓库已经整理成标准 Python 包，主体位于 `bbo/`，并明确区分了可复用 core、算法族、任务族、文档资产以及可运行示例。

当前仓库的主要目标有三个：

- 为未来的 agent-based optimization 方法提供一个小而规范的 benchmark core
- 提供可以直接运行的传统优化 baseline，用于验证和比较
- 提供一套标准化 task-description 格式，方便协作者继续扩展新的 benchmark 问题

## 仓库结构

```text
.
├── AGENTS.md
├── README.md
├── README.zh.md
├── bbo/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── model_based/
│   │   ├── registry.py
│   │   └── traditional/
│   ├── core/
│   ├── run.py
│   ├── task_descriptions/
│   └── tasks/
├── docs/
├── examples/
├── tests/
└── pyproject.toml
```

### `bbo/core/`

这里放的是可复用 benchmark 抽象：

- 搜索空间定义
- 任务规范与 sanity check
- trial 记录结构
- logging 与 replay
- experiment 编排
- task-description 加载
- plotting 工具
- 外部优化器 adapter

### `bbo/algorithms/`

算法实现按家族组织。
当前家族为：

- `bbo/algorithms/agentic/`
  - `llambo.py`
  - `opro.py`
- `bbo/algorithms/model_based/`
  - `optuna_tpe.py`
- `bbo/algorithms/traditional/`
  - `random_search.py`
  - `pycma.py`

### `bbo/tasks/`

任务实现同样按家族组织。
当前家族为：

- `bbo/tasks/synthetic/`
  - `branin.py`
  - `sphere.py`
  - `base.py`

### `bbo/task_descriptions/`

这里存放 benchmark 上下文所需的标准化任务文档。
当前仓库包含：

- `branin_demo` 和 `sphere_demo` 的可执行 benchmark 描述
- 一个面向协作者的任务封装示例
- 一个可复用模板
- 中英文双语文档副本

## 安装

使用 `uv` 创建并同步环境：

```bash
uv sync --extra dev
```

如果需要 ConfigSpace 互操作辅助功能，可额外安装：

```bash
uv sync --extra dev --extra interop
```

如果需要使用 Optuna TPE，可按 optional extra 安装：

```bash
uv sync --extra dev --extra optuna
```

如果还要跑 `her_demo`、`oer_demo`、`molecule_qed_demo` 这些 scientific smoke，请再加上：

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
```

## 运行 demo

### 完整对比 suite

```bash
uv run python -m bbo.run --algorithm suite --task branin_demo
```

等价示例脚本：

```bash
uv run python examples/run_branin_suite.py
```

### Random-search baseline

```bash
uv run python examples/run_random_search_demo.py
```

### CMA-ES baseline

```bash
uv run python examples/run_pycma_demo.py
```

### Optuna TPE baseline

```bash
uv run python examples/run_optuna_tpe_demo.py
```

### LLAMBO baseline

```bash
uv run python examples/run_llambo_demo.py
```

### OPRO baseline

```bash
uv run python examples/run_opro_demo.py
```

### 直接使用 CLI

```bash
uv run python -m bbo.run \
  --algorithm pycma \
  --task branin_demo \
  --max-evaluations 36 \
  --sigma-fraction 0.18 \
  --popsize 6
```

Optuna TPE 的公开算法名是 `optuna_tpe`，并且支持 mixed/categorical search space：

```bash
uv run python -m bbo.run \
  --algorithm optuna_tpe \
  --task oer_demo \
  --max-evaluations 6
```

`suite` 仍然只包含传统算法 `random_search` 与 `pycma`，不会把 `optuna_tpe` 混进去。

LLAMBO 的公开算法名是 `llambo`。默认的 `heuristic` backend 是一个离线 smoke-test 路径，在不依赖网络的情况下保留了 LLAMBO 的 acquisition/surrogate 两阶段流程：

```bash
uv run python -m bbo.run \
  --algorithm llambo \
  --task branin_demo \
  --max-evaluations 12 \
  --llambo-backend heuristic
```

如果要走在线 OpenAI 路径，建议把凭证和 endpoint 选择放在用户可见的 runner / CLI 层，而不是写死在底层算法里。API key 放环境变量，模型、base-url、timeout 等通过 CLI 指定：

```bash
export OPENAI_API_KEY=your_key_here
uv run python -m bbo.run \
  --algorithm llambo \
  --task branin_demo \
  --max-evaluations 12 \
  --llambo-backend openai \
  --llambo-model gpt-4o-mini \
  --llambo-openai-api-key-env OPENAI_API_KEY \
  --llambo-openai-timeout-seconds 30
```

如果需要自定义 endpoint，也可以在 runner 层通过 `--llambo-openai-base-url`、`--llambo-openai-organization`、`--llambo-openai-project` 指定。

其他调参开关：
- `--llambo-openai-max-retries`（默认 3）—— 网络出错时指数退避重试。
- `--no-llambo-openai-use-structured-outputs`—— 关闭 ``json_schema`` 结构化输出，用于不支持该特性的兼容 API；backend 会自动降级为普通文本补全 + 正则解析。

在线 backend 的现成示例脚本位于 `examples/run_llambo_openai_demo.py`。

OPRO 的公开算法名是 `opro`。默认的 `heuristic` backend 是一个离线 smoke-test 路径，它把原始 OPRO 的“历史配置/分数元提示词”模式适配到了本仓库的统一搜索空间接口上：

```bash
uv run python -m bbo.run \
  --algorithm opro \
  --task branin_demo \
  --max-evaluations 12 \
  --opro-backend heuristic
```

如果要走在线 OpenAI 路径，凭证和 endpoint 配置与 LLAMBO 保持一致，都放在 runner / CLI 层：

```bash
export OPENAI_API_KEY=your_key_here
uv run python -m bbo.run \
  --algorithm opro \
  --task branin_demo \
  --max-evaluations 12 \
  --opro-backend openai \
  --opro-model gpt-4o-mini \
  --opro-openai-api-key-env OPENAI_API_KEY \
  --opro-openai-timeout-seconds 30
```

如果需要自定义 endpoint，也可以在 runner 层通过 `--opro-openai-base-url`、`--opro-openai-organization`、`--opro-openai-project` 指定。

其他调参开关：
- `--opro-openai-max-retries`（默认 3）—— 网络出错时指数退避重试。

在线 backend 的现成示例脚本位于 `examples/run_opro_openai_demo.py`。

## 输出结果

运行结果会把 JSONL 历史、summary 和 plots 写到 `artifacts/` 下。
当前已经验证过的一组参考输出位于 `artifacts/final_demo_v3/`。

当前会生成的可视化包括：

- 优化轨迹图
- 目标值分布图
- 适用于 2D 任务的 landscape 叠加图
- 不同优化器之间的对比图

## Task-description 规范

每个 benchmark task 应放在 `bbo/task_descriptions/<task_name>/` 下。
必需文件：

```text
background.md
goal.md
constraints.md
prior_knowledge.md
```

推荐可选文件：

```text
evaluation.md
submission.md
environment.md
notes.md
history.md
```

像 `background.zh.md` 这样的本地化对照文件是允许的，便于协作阅读。
为了保证运行时 benchmark 上下文保持确定性，loader 会自动忽略这些本地化副本。

此外，每个 task 还必须至少提供一种环境供给方式：

- task-local Docker 工作流
- 或者写在 `environment.md` 中的显式配置说明

相关文档：

- `bbo/task_descriptions/README.md`
- `bbo/task_descriptions/README.zh.md`
- `docs/collaborator_demo.md`
- `docs/collaborator_demo.zh.md`
- `bbo/core/DEVELOPER_GUIDE_zh.md`
- `bbo/core/IMPLEMENTATION_PLAN.md`
- `bbo/core/IMPLEMENTATION_PLAN.zh.md`

## 如何新增一个任务

1. 把 `bbo/task_descriptions/_template/` 复制到 `bbo/task_descriptions/<task_name>/`
2. 在 `bbo/tasks/` 下新增或扩展一个任务家族
3. 用 `SearchSpace` 和类型化参数显式定义搜索空间
4. 从 evaluator 返回规范化的 `EvaluationResult`
5. 补充测试，并执行下面的验证命令

## 验证命令

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm suite --task branin_demo --results-root artifacts/final_demo
```

Optuna smoke 示例：

```bash
uv run python -m bbo.run --algorithm optuna_tpe --task branin_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task her_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task oer_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm opro --task branin_demo --max-evaluations 6 --results-root artifacts/opro_smoke
```

## 当前参考 benchmark

- `branin_demo`：二维 synthetic benchmark，适合可视化和优化器比较
- `sphere_demo`：凸型 synthetic benchmark，适合 smoke test 与 replay/resume 验证
- `collaborator_problem_demo`：偏文档化的示例，用来展示如何封装一个更真实的 benchmark 问题
