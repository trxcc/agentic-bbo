# Task Description 规范

English version: `bbo/task_descriptions/README.md`

这个仓库把 task description 当成 benchmark 的一等公民。
每个 benchmark task 都应该有自己独立的目录，位于 `bbo/task_descriptions/<task_name>/`。
core loader 会校验统一 schema，这样 agentic 方法拿到的是结构化上下文，而不是一段随手写的 prompt。

## 必需文件

```text
bbo/task_descriptions/<task_name>/
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

## 各章节含义

- `background.md`：真实系统背景、工作负载，以及为什么这个问题重要
- `goal.md`：优化目标、评估约定和成功标准
- `constraints.md`：硬约束、禁止操作、预算和安全要求
- `prior_knowledge.md`：领域先验、经验规则、不变量和有价值的起点
- `evaluation.md`：指标、聚合规则、噪声模型、seed 和 tie-breaking
- `submission.md`：参数接口、I/O 约定，以及 benchmark 期望的产物布局
- `environment.md`：当没有 task-local Docker 工作流时，提供手动环境配置说明

## 环境提供要求

每个 task package 都必须至少提供以下两种方式中的一种：

- task-local Docker 工作流，例如 `Dockerfile`、`docker-compose.yml` 或 `docker/` 目录
- 一个写明明确配置步骤的 `environment.md`

任务的 sanity check 会强制检查这两种环境提供路径至少存在一种。

## 本地化对照文件

如果协作者需要，也可以添加 `background.zh.md`、`goal.zh.md` 这样的本地化对照文档。
这些文件只服务文档协作。
为了保证 benchmark 运行时上下文是确定的，loader 会自动忽略 `*.zh.md` 和 `*.en.md` 文件。

## 仓库内示例

- `bbo/task_descriptions/branin_demo/`：README 和测试中实际使用的 synthetic-function demo
- `bbo/task_descriptions/sphere_demo/`：轻量的 sanity-check 任务
- `bbo/task_descriptions/collaborator_problem_demo/`：更完整的协作者封装示例
- `bbo/task_descriptions/_template/`：可直接复制的新任务模板

像 `bbo/task_descriptions/autoresearch_train/` 这样的遗留目录目前只作为历史材料保留，不再是推荐 schema。
