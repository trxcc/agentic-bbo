## Agentbbo 中的 knob（参数）surrogate 实验

本文介绍如何在本仓库中运行 **knob（数据库参数调优）surrogate 实验**。

这些任务是 **离线（offline）** benchmark：底层是序列化的 sklearn 模型（`*.joblib`）。默认运行方式是 HTTP sidecar：模型和旧版 sklearn 运行时已经封在 `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1` 镜像内，宿主机只发送归一化 knob 向量并接收预测指标。**不需要真实数据库实例，也不需要在宿主机 assets 目录放 `.joblib`。**

### 你会用到的入口

- **Task family**: surrogate knob tasks under `bbo/tasks/dbtune/`
- **Examples**: `examples/run_knob_surrogate_demo.py`
- **统一运行入口**：`python -m bbo.run`（推荐）或 `bbo.run.run_single_experiment()`
- **输出位置**：默认写到 `runs/demo/` 下的 JSONL trial 日志（`trials.jsonl`）和汇总（`summary.json`）

### 前置条件

- **Python 环境**：推荐用仓库管理的环境（`uv`）
- **Docker sidecar**：启动 `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`，默认监听 `127.0.0.1:8090`

用 `uv` 安装（推荐）：

```bash
uv sync --extra dev --extra task-host
```

### 可用的 surrogate knob 任务

用 Python 列出 **catalog / Docker canonical** id（`knob_surrogate_*`，与 `SURROGATE_BENCHMARKS` 一致）：

```bash
uv run python -c "from bbo.tasks import SURROGATE_TASK_IDS; print('\\n'.join(SURROGATE_TASK_IDS))"
```

**`python -m bbo.run` 与 `ALL_TASK_NAMES` 只注册 HTTP 型任务**（`knob_http_surrogate_*`）。本机直接加载 `.joblib` 请用
`from bbo.tasks.dbtune import create_surrogate_knob_task` 或脚本封装，不通过 CLI。

列出可供 `bbo.run` 的 HTTP surrogate task id：

```bash
uv run python -c "from bbo.tasks import HTTP_SURROGATE_TASK_IDS; print(*HTTP_SURROGATE_TASK_IDS, sep='\\n')"
```

常见 canonical 名（与 `assets/README.md`、Docker `GET /task/<id>` 一致）：`knob_surrogate_sysbench_5`、
`knob_surrogate_sysbench_all`、`knob_surrogate_job_5`、`knob_surrogate_job_all`、
`knob_surrogate_pg_5`、`knob_surrogate_pg_20`；CLI 上对应 `knob_http_surrogate_...`（多 `http_` 前缀）。

### 启动 surrogate sidecar

默认使用已经封装好 checkpoint 的 Docker Hub 镜像：

```bash
docker pull fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
docker rm -f agentbbo_surrogate_http 2>/dev/null
docker run -d --name agentbbo_surrogate_http -p 8090:8090 \
  fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

健康检查：

```bash
curl -sS http://127.0.0.1:8090/health
curl -sS http://127.0.0.1:8090/task/knob_surrogate_sysbench_5
```

只有在你要重新构建/发布 surrogate 镜像，或直接调用未注册的 in-process `create_surrogate_knob_task(...)` 时，才需要本地 `.joblib` 文件。

### 运行 knob 实验（推荐用 `bbo.run`）

跑 random-search baseline：

```bash
uv run python -m bbo.run \
  --task knob_http_surrogate_sysbench_5 \
  --algorithm random_search \
  --seed 1 \
  --max-evaluations 60
```

跑 CMA-ES（需要你环境里额外安装 `cma` / `pycma` 相关依赖）：

```bash
uv run python -m bbo.run \
  --task knob_http_surrogate_sysbench_5 \
  --algorithm pycma \
  --seed 1 \
  --max-evaluations 60 \
  --sigma-fraction 0.18 \
  --popsize 6
```

进程内覆盖 `*.joblib` / `knobs_*.json` 路径时，在代码里调
`create_surrogate_knob_task("knob_surrogate_sysbench_5", ..., surrogate_path=..., knobs_json_path=...)`（见
`bbo.run` 的 `run_single_experiment` 对 surrogate 的 kwargs）。**HTTP** 型（`--task knob_http_surrogate_*`）在
容器内加载模型，一般不在宿主机传 `--surrogate-path`。

### 运行示例脚本

`examples/run_knob_surrogate_demo.py` 本质上只是对 `run_single_experiment()` 的轻量封装：

```bash
uv run python examples/run_knob_surrogate_demo.py \
  --task knob_http_surrogate_sysbench_5 \
  --algorithm random_search \
  --seed 1 \
  --max-evaluations 60
```

### 输出：结果写到哪里

默认输出目录结构如下：

```text
runs/demo/<task>/<algorithm>/seed_<seed>/
  trials.jsonl
  summary.json
```

- **`trials.jsonl`**：每次评估（trial）一行 JSON 记录
- **`summary.json`**：聚合后的最优值、incumbents、以及 logger 汇总

### 进程内（Python 3.11）与 HTTP + Docker（Python 3.7）两种跑法

| 方式 | `task` 命名 | 说明 |
|------|------------|------|
| 进程内 | `create_surrogate_knob_task("knob_surrogate_sysbench_5", ...)` | 本机 `joblib` + 本机 `predict`；只用于开发/维护，**不在** `bbo.run` / `ALL_TASK_NAMES` 注册。 |
| 侧车 HTTP | `knob_http_surrogate_sysbench_5` 等 | 与**真实数据库任务同一思路**：BBO 只产生归一化点，**`POST` 发一个 `x`（`[0,1]^d` 列表）**，**容器里解码 knobs + 代理模型，返回一个标量 `y`**。模型与 sklearn 3.7 环境只在镜像里。 |

**HTTP 合约（与「真实库：发配置、回吞吐」平行）**：

- `POST /evaluate` 推荐请求体：`{"task_id": "knob_surrogate_sysbench_5", "x": [0.0, …, 1.0]}`，长度 `d` 与元数据一致。容器内用自带 `assets/knobs_*.json` 做 `[0,1]→` 物理量，再 `predict`；响应 `status: success` 与 `y`。
- 另支持旧字段 `features`（已是物理量、不解码）以便调试，新代码路径不必用。

**运行**（默认 `http://127.0.0.1:8090`，与数据库评估器 8080 错开）：

- 起容器：优先使用可复用镜像 `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`，详见 `bbo/tasks/dbtune/docker_surrogate/README.md`。
- 环境变量（宿主机）：`AGENTBBO_HTTP_SURROGATE_BASE_URL`、`AGENTBBO_HTTP_SURROGATE_TIMEOUT_SEC`（默认 120）
- 列出 HTTP 型 task id：

```bash
uv run python -c "from bbo.tasks.registry import HTTP_SURROGATE_TASK_IDS; print(*HTTP_SURROGATE_TASK_IDS, sep='\n')"
```

```bash
export AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090
uv run python -m bbo.run --task knob_http_surrogate_sysbench_5 --algorithm random_search --max-evaluations 20 --seed 1
```

**注意**：默认发布镜像已经包含 `.joblib` 与 `knobs_*.json`。宿主机 BBO 仅通过 `GET /task/...` 取维度/名字，**不**在 3.11 上反序列化模型。

### 常见问题排查

- **`GET /task/...` 返回 503 或 `joblib.load` 错误**
  - 检查你拉取的是完整发布镜像 `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`；若使用自建镜像，则重新用完整 checkpoint 构建。
- **HTTP 连接失败**
  - 确认容器已启动，且 `AGENTBBO_HTTP_SURROGATE_BASE_URL` 指向 `http://127.0.0.1:8090` 或实际服务地址。
- **使用 `--algorithm pycma` 时提示 `ModuleNotFoundError: cma`**
  - 你需要先在环境里安装 `cma` 依赖，然后再使用 `pycma`。
