# `run_all_registered_tasks.py` 程序说明

本文档描述 `examples/run_all_registered_tasks.py` 的设计、行为与产物，便于复现与改进。

---

## 1. 目的

在 **Agentbbo** 仓库内，对 **`bbo` 已注册的任务名**（`ALL_TASK_NAMES`）做一次**顺序批跑**：每个任务调用 `bbo.run.run_single_experiment`，使用**同一套**优化器（默认 `optuna_tpe`），并把 **BBOPlace** 按独立矩阵（`BBOPLACE_CASES`）再跑多组（benchmark × `n_macro`）。

适合：一次性覆盖「合成 / 科学任务 / 需 HTTP 的 DB 与代理 / BBOPlace」的回归或长时评估（需自行准备评估器与算力）。

---

## 2. 执行流程概览

1. **参数解析**：算法、随机种子、各任务 `max_evaluations`、是否作图、是否写汇总表等。
2. **端口探测**（`probe_evaluator_ports`）：对 `--http-host`（默认 `127.0.0.1`）做 TCP 连接测试：
   - **8070**：BBOPlace evaluator（宿主机端口；容器内通常为 8080）。
   - **8080**：MariaDB HTTP 评估器（`knob_http_mariadb_sysbench_*`）。
   - **8090**：Surrogate HTTP 评估器（`knob_http_surrogate_*`）。
3. **任务列表**（`_non_bboplace_tasks`）：
   - 始终包含非 HTTP 的注册任务（如 `branin_demo`、`her_demo` 等）。
   - **`knob_http_*`**：在 **auto** 模式下，仅当对应端口可达时加入；`--include-http` 强制全部加入；`--skip-http` 全部排除。
4. **BBOPlace**（`_should_run_bboplace`）：auto 模式下仅当 8070 可达时排入矩阵；`--include-http` 强制；`--skip-bboplace` 排除。
5. **构造计划** `planned`：`(label, run_single_experiment 的 kwargs)` 列表。
6. **顺序执行**：对每个计划调用 `_run_one` → `run_single_experiment`；失败捕获并记录，批跑可继续（最后非零退出码若存在失败）。
7. **汇总表**（可选）：`write_batch_objectives_table` 写 **CSV + JSON**（见下文）。

---

## 3. 命令行要点

| 选项 | 含义 |
|------|------|
| `--algorithm` | 所有子实验共用，默认 `optuna_tpe`。 |
| `--seed` | 各 run 的随机种子。 |
| `--max-evaluations` | 非 BBOPlace 任务的评估预算。 |
| `--bboplace-max-evaluations` | BBOPlace 每条矩阵的评估预算。 |
| `--include-http` / `--skip-http` | 与 `knob_http_*` 的强制包含/排除互斥。 |
| `--bboplace-only` / `--skip-bboplace` | 只跑 BBOPlace 矩阵 / 不跑 BBOPlace。 |
| `--no-plots` | 不写各 `run_dir/plots/`。 |
| `--no-table` | 不写 `batch_objectives_table.csv` / `.json`。 |
| `--results-subdir` | 结果子目录，默认 `batch_all_tasks`，根为 `runs/demo/`。 |
| `--list` / `--dry-run` | 仅列出计划 / 仅打印计划不执行。 |

完整说明以脚本内 `argparse` 与模块 docstring 为准。

---

## 4. 与 Optuna 日志

每个子实验若使用 `optuna_tpe`，会 **独立创建一个 in-memory Optuna Study**，因此终端可能出现多行：

`A new study created in memory with name: no-name-<uuid>`

这是**预期行为**（一任务一 study），不是重复建同一优化问题。若需降噪，可降低 Optuna 日志级别（如环境变量或 logging 配置）。

---

## 5. 输出目录结构

- 每次 `run_single_experiment` 在 `results_root` 下按 **任务名 / 算法名 / `seed_*`** 分配 `run_dir`，并写 `trials.jsonl`、`summary.json` 等（与单任务 demo 一致）。
- 默认 `results_root = <Agentbbo>/runs/demo/<results-subdir>/`。
- 未加 `--no-plots` 且存在有效 trial 时，`run_dir/plots/` 下会有 trace、distribution、用时等图（依任务维度可能还有 landscape / regret）。

---

## 6. 批跑结束：汇总表（`batch_objectives_table`）

在 `results_base` 目录下生成：

### 6.1 CSV（`batch_objectives_table.csv`）

- **行**：每种 `algorithm` 一行（当前批跑通常全局同一算法，故常为一行）。
- **列**：第一列 `algorithm`，其后**每列对应一个任务（实验）**，列名即 **`experiment_id`**（`task_id` 或 `bboplace:{benchmark}:n{n_macro}` 等）。
- **单元格**：优先 **`best_primary_objective`**；仅一个目标则为标量；多目标则为**单行 JSON**（`objectives` 字典串）。失败记空。**详情见 JSON `runs`。**

### 6.2 JSON（`batch_objectives_table.json`）

- `csv_columns`：与 CSV 表头一致。
- `algorithm_cells`：与 CSV 等价的数据结构。
- `runs`：逐条明细，含完整 `objectives`、`ok`、`error`、`run_dir` 等。

---

## 7. 修改矩阵与扩展

- **BBOPlace 组合**：编辑源码中常量 **`BBOPLACE_CASES`**（`tuple` of `(benchmark, n_macro)`）。
- **注册新任务**：在 `bbo.tasks.registry` 中注册后，会自动进入 `ALL_TASK_NAMES`（除非被端口/开关过滤）。

---

## 8. 依赖与运行方式

- Python **≥ 3.11**（见 `pyproject.toml`）。
- 推荐统一宿主环境：`uv sync --extra dev --extra task-host`
- 若你的 Docker 安装带有 Compose，可在仓库根目录执行 `docker compose -f docker-compose.task-services.yml up -d` 一次拉起三个任务服务
- 在仓库根目录：

  ```bash
  uv run python examples/run_all_registered_tasks.py
  ```

- 需 HTTP 的实验要本机或 `--http-host` 指向的主机上，对应 Docker/服务已监听相应端口。

---

## 9. 相关文件

**新机器从 0 准备运行环境**（安装依赖、起三个 Docker、验证端口）：见 **`docs/SETUP_RUN_ALL_BATCH.md`**。

| 路径 | 说明 |
|------|------|
| `examples/run_all_registered_tasks.py` | 本程序 |
| `bbo/run.py` | `run_single_experiment`、作图逻辑 |
| `bbo/tasks/registry.py` | `ALL_TASK_NAMES` 与任务工厂 |
| `examples/run_bboplace_demo.py` | 单跑 BBOPlace 的等价路径（自定义 `definition`） |
| `database.md`（仓库根或文档约定） | HTTP 端口与任务类型对照 |
