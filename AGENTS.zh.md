# 仓库协作指南

## 项目结构与模块组织
现在这个仓库已经整理成一个真正的 Python 包，代码位于 `bbo/`。
顶层 benchmark 入口在 `bbo/run.py`。
算法按家族组织在 `bbo/algorithms/`，任务按家族组织在 `bbo/tasks/`，可复用、与具体 benchmark 无关的抽象放在 `bbo/core/`。
任务说明文档放在 `bbo/task_descriptions/<task_name>/`。
每个任务应遵循统一的 task-description 规范，至少包含 `background.md`、`goal.md`、`constraints.md` 和 `prior_knowledge.md`。

## 构建、测试与开发命令
- `uv sync --extra dev`：创建并同步环境，以 editable 模式安装当前包。
- `uv run pytest`：运行当前自动化测试。
- `uv run python -m compileall -q bbo examples tests`：快速语法 smoke test。
- `uv run python -m bbo.run --algorithm suite --task branin_demo`：运行标准端到端 demo。

## 代码风格与命名规范
使用 4 空格缩进、类型标注，以及简洁的公开 API docstring。
命名规则保持一致：函数和模块使用 `snake_case`，类和 dataclass 使用 `PascalCase`，注册表使用 `UPPER_SNAKE_CASE`。
`bbo/core/` 必须保持 benchmark-agnostic。
任何任务特定的评估逻辑、任务打包逻辑和 synthetic/real task wrapper 都应放在 `bbo/core/` 之外。
每个 task package 都应提供 task-local Docker 工作流，或提供一个写明配置步骤的 `environment.md`。
继续保持 append-only JSONL logging 和基于 replay 的 resume 语义。

## 测试规范
凡是修改 ask/tell 流程、adapter、logging、plotting 或 resume 逻辑，都应补充或更新测试。
优先使用轻量 synthetic task，而不是昂贵 evaluator。
测试名建议描述行为，例如 `test_resume_replays_trials_in_order`。
如果改动 task-description 逻辑，也要确认诸如 `*.zh.md` 这样的本地化文档不会被 loader 当作主 benchmark 上下文读入。

## Commit 与 PR 规范
当前 checkout 不包含 `.git`，因此无法查看本地历史风格。
commit message 建议使用简短祈使句，例如 `Add optimizer comparison plotter`。
在 PR 中请说明改动是否涉及 `bbo/core/` 还是只影响 task/demo 层，说明是否影响 JSONL schema 或 task-description schema，关联相关 issue，并列出实际跑过的验证命令。
只有在行为变化时再附上 CLI 输出或可视化产物路径。
