# Benchmark PR Notes

这份文档用于简要记录当前新增的 benchmark task / algorithm，以及后续 PR 需要满足的最低规范。

## 本轮新增的任务

### Scientific tasks

- `her_demo`
  - 基于 `HER_virtual_data.csv` 的 10 维连续优化任务。
  - 评估器使用随机森林 surrogate，主目标是 `regret`。
- `hea_demo`
  - 基于 `oracle_data.xlsx` 的高熵合金组成优化任务。
  - 对外暴露 `x1..x4` 设计变量，内部通过可行 simplex 变换恢复为 5 元组成，再用随机森林评估 `regret`。
- `oer_demo`
  - 混合搜索空间任务，包含类别、整数和连续变量。
  - 数据先清洗、one-hot 对齐，再由随机森林预测 `overpotential_mv`。
- `bh_demo`
  - 基于 Buchwald-Hartwig 数据集的连续优化任务。
  - 先做 feature selection，再在筛出的连续特征空间上用随机森林预测 `regret`。
- `molecule_qed_demo`
  - 以固定 SMILES 候选池为搜索空间的分子任务。
  - 评估器直接调用 RDKit 计算 `QED`，主目标是 `qed_loss = 1 - qed`。

## 本轮新增的算法

### Model-based algorithms

- `optuna_tpe`
  - 使用 Optuna TPE sampler。
  - 通过统一 ask/tell 接口接入 benchmark，并支持基于日志历史的 replay。
- `pfns4bo`
  - 提供统一的 `pfns4bo` 算法入口。
  - 对纯数值任务走 continuous backend；对 `oer_demo` 和 `molecule_qed_demo` 走固定候选池 backend。

## PR 代码需要满足的规范

### 1. 通用 benchmark 规范

- 必须遵守统一接口：
  - task 实现 `Task.spec` 与 `Task.evaluate()`
  - algorithm 实现 `setup()`、`ask()`、`tell()`、`incumbents()`
- 必须兼容 append-only JSONL logging。
- 必须兼容 replay-based resume，不能依赖隐藏 checkpoint 才能恢复运行。
- 固定 seed 后，应保证行为尽量可复现；如果不能完全确定性，必须在文档中明确写清噪声或随机性来源。
- 默认行为不能依赖工作区外路径，也不能默认联网下载数据或模型。

### 2. Task 代码规范

- 必须定义清晰的 `TaskSpec`：
  - `name`
  - `search_space`
  - `objectives`
  - `max_evaluations`
- `evaluate()` 返回值必须是合法的 `EvaluationResult`：
  - 成功时必须包含主目标
  - objective 必须是有限数
  - 建议补充 `metrics` 和 `metadata` 方便调试与复现
- 必须提供 `sanity_check()` 所需的最小可验证能力：
  - 数据存在
  - 关键列存在
  - 默认配置可以被评估
- task description 必须符合 `bbo/task_descriptions/_template` 的 schema：
  - 必需：`background.md`、`goal.md`、`constraints.md`、`prior_knowledge.md`
  - 推荐：`evaluation.md`、`submission.md`、`environment.md`
- 数据集应优先 vendoring 到工作区内，或至少提供 workspace-local override 机制；不能把 benchmark 设计成默认依赖个人机器路径。
- 如果 task 是 mixed-space、pool-based、surrogate-based，必须在文档里明确写清楚，不要伪装成真实在线实验。

### 3. Algorithm 代码规范

- 必须严格遵守 `bbo/core/algo.py` 定义的协议。
- 必须正确处理 task 的主目标方向：`minimize` / `maximize` 不能混淆。
- 必须兼容 logger history replay；历史 observation 重新灌入后，内部状态应与原运行语义一致。
- 如果算法只支持 numeric search space，必须显式声明或检查，而不是静默接受 mixed task。
- 如果算法只对少数任务提供特化路径，必须：
  - 在代码中明确限制范围
  - 在文档中明确说明原因
  - 避免把 task-specific hack 包装成通用能力
- 可选依赖必须做 graceful failure：缺包时给出清晰错误，而不是隐式崩溃。

### 4. 文档与数据规范

- 新增 task 时，必须同步补 task description。
- 新增 algorithm 时，至少要在 README 或对应文档中说明：
  - 它支持哪些任务类型
  - 它的依赖是什么
  - 最小 smoke 命令是什么
- vendored 数据应保持稳定目录结构，便于打包、上传和替换。
- 如果允许通过环境变量覆盖数据根目录，需要明确写清楚相对路径约定。

### 5. 最低验证要求

PR 合入前，至少应完成与改动范围对应的最小验证：

```bash
uv run pytest tests/test_scientific_tasks.py -q
```

如果改动了 PFNs4BO：

```bash
uv run pytest tests/test_pfns4bo.py -q
```

如果改动了 LLAMBO 或 OPRO：

```bash
uv run pytest tests/test_llambo.py -q
uv run pytest tests/test_opro.py -q
```

如果改动了 task / algorithm 通用运行链路，建议再补一轮 smoke：

```bash
uv run python -m bbo.run --algorithm random_search --task her_demo --max-evaluations 3
uv run python -m bbo.run --algorithm random_search --task hea_demo --max-evaluations 3
uv run python -m bbo.run --algorithm random_search --task oer_demo --max-evaluations 3
uv run python -m bbo.run --algorithm random_search --task bh_demo --max-evaluations 3
uv run python -m bbo.run --algorithm random_search --task molecule_qed_demo --max-evaluations 3
uv run python -m bbo.run --algorithm llambo --task branin_demo --max-evaluations 6 --llambo-backend heuristic
uv run python -m bbo.run --algorithm opro --task branin_demo --max-evaluations 6 --opro-backend heuristic
```

### 6. 一句话标准

如果一段新增代码不能被稳定描述、稳定测试、稳定 replay、稳定打包，那它就还不够像 benchmark 代码。
