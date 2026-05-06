# ruff: noqa: INP001
"""One-off: generate ``bbo/task_descriptions/<task_id>/`` for eight database tasks.

Run from AgentBBO repository root::

    python bbo/tasks/dbtune/gen_task_markdown.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bbo.tasks.dbtune.http_mariadb_specs import (  # noqa: E402
    DBTUNE_MARIADB_TASK_IDS,
    DATABASE_TASK_SPECS,
    HTTP_DATABASE_TASK_IDS,
    SYSBENCH_TEST_BY_WORKLOAD,
)

_TASK_DESC = _REPO / "bbo" / "task_descriptions"
_MARIADB_IMAGE = "fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1"

_WORKLOAD_COPY: dict[str, tuple[str, str, str]] = {
    "read_only": (
        "read-only (OLTP read)",
        "This benchmark uses the **sysbench** test ``oltp_read_only``.",
        "本任务使用 **sysbench** 的 ``oltp_read_only`` 只读工作负载。",
    ),
    "write_only": (
        "write-only (OLTP write)",
        "This benchmark uses the **sysbench** test ``oltp_write_only``.",
        "本任务使用 **sysbench** 的 ``oltp_write_only`` 只写工作负载。",
    ),
    "read_write": (
        "read/write (mixed OLTP)",
        "This benchmark uses the **sysbench** test ``oltp_read_write`` (classic mixed workload).",
        "本任务使用 **sysbench** 的 ``oltp_read_write`` 混合事务负载。",
    ),
    "point_select": (
        "point-select (primary-key lookups)",
        "This benchmark uses the **sysbench** test ``oltp_point_select``.",
        "本任务使用 **sysbench** 的 ``oltp_point_select`` 点查询负载。",
    ),
}


def _knob_paragraphs(spec) -> tuple[str, str, str]:
    """English long, Chinese, path under assets."""
    if spec.knob_asset_filename == "knobs_SYSBENCH_top5.json":
        return (
            "five knobs from the top-5 SHAP-ranked subset (`knobs_SYSBENCH_top5.json`).",
            "5 个旋钮（`knobs_SYSBENCH_top5.json`，与离线 `knob_surrogate_sysbench_5` 语义一致）。",
            "`bbo/tasks/dbtune/assets/knobs_SYSBENCH_top5.json`",
        )
    if spec.knob_asset_filename == "knobs_mysql_all_197.json":
        return (
            "the full **~197-dimensional** knob list (`knobs_mysql_all_197.json`), matching the offline `knob_surrogate_sysbench_all` space.",
            "约 197 维全量 MySQL 旋钮（`knobs_mysql_all_197.json`），与离线全量 surrogate 任务维度一致。",
            "`bbo/tasks/dbtune/assets/knobs_mysql_all_197.json`",
        )
    raise ValueError(spec.knob_asset_filename)


def _write_one(spec_id: str) -> None:
    spec = DATABASE_TASK_SPECS[spec_id]
    wl_en, wl_en_body, wl_zh_body = _WORKLOAD_COPY[spec.workload_key]
    knob_en, knob_zh, knob_path = _knob_paragraphs(spec)
    oltp = SYSBENCH_TEST_BY_WORKLOAD[spec.workload_key]
    out = _TASK_DESC / spec.task_id
    out.mkdir(parents=True, exist_ok=True)

    background = f"""# Background

`{spec.task_id}` is a **real** MariaDB benchmark in *AgentBBO*. The optimizer proposes a point in the unit hypercube; the packaged evaluator service (Flask inside the reusable image `{_MARIADB_IMAGE}`) writes `mysqld` knobs, restarts MariaDB, and runs **sysbench**, returning a scalar **throughput** score.

This packaging combines: **{wl_en}** with **{knob_en}**

{wl_en_body}

The measurement is **not** a surrogate: it is the container’s live database and sysbench output. What is “simulated” is only the synthetic `sbtest` dataset and fixed script parameters in `server.py`.

A Chinese companion is in `background.zh.md` (informational only; loaders use the English files for canonical context).
"""
    (out / "background.md").write_text(textwrap.dedent(background).strip() + "\n", encoding="utf-8")

    background_zh = f"""# 背景

`{spec.task_id}` 是 *AgentBBO* 中的 **真实 MariaDB** 黑盒调参：优化器给出 `[0,1]` 超立方体中的一点，Python 任务解码为 ``{knob_path}`` 中的物理旋钮，再发给评估容器服务；容器内写配置、重启、跑 **sysbench**，返回标量 TPS/吞吐类指标。

本任务在口径上固定为：**{spec.short_label_zh}**

{wl_zh_body}

与 HER/synthetic 不同，**数值来自容器内压测**；可重复性受硬件负载、冷/热缓存等影响。协作者可阅读本文件，运行时仍以 `background.md` 为准。

"""
    (out / "background.zh.md").write_text(textwrap.dedent(background_zh).strip() + "\n", encoding="utf-8")

    goal = f"""# Goal

**Maximize** the primary objective `throughput` (larger is better), implemented as the parsed **per-second throughput** in the JSON response field `y` (alias `tps`).

- **Search space:** one float in `[0,1]` per exposed knob, decoded to physical values with the same rules as the surrogate `KnobSpaceFromJson` helper.
- **One evaluation** = one successful `POST /evaluate` with `{{"knobs":{{...}},"workload":"{spec.workload_key}"}}` that runs **``{oltp}``** under the fixed `server.py` parameters.
- **Valid run:** evaluator `status` is `success` and the returned objective is finite.

Comparative benchmarks should keep **image version, `server.py` timing, and hardware** fixed.
"""
    (out / "goal.md").write_text(textwrap.dedent(goal).strip() + "\n", encoding="utf-8")

    goal_zh = f"""# 目标

**最大化**主目标 `throughput`（即评估服务返回的 TPS/吞吐，字段 `y` 或 `tps`）。

- 搜索域：与 knob JSON 中每个键对应的一维 `[0,1]` 连续坐标，再解码到物理整型/枚举值。
- 一次有效评估 = 一次成功的 `POST /evaluate`，`workload` 固定为 **``{spec.workload_key}``**，服务端映射到 **``{oltp}``**。
- 若 `status` 非 `success` 或出现超时，该 trial 记为失败（见 `trials.jsonl`）。

同组对比不同算法时，应固定**镜像、`server.py`、压测参数与 seed**；否则绝对 TPS 不可比。

**Knob 表：**{knob_zh}
"""
    (out / "goal.zh.md").write_text(textwrap.dedent(goal_zh).strip() + "\n", encoding="utf-8")

    constraints = f"""# Constraints

- **No concurrent in-flight `POST /evaluate` calls** against the same server instance: the container serializes work with a lock.
- **Per-evaluation** wall time can be minutes: set `AGENTBBO_HTTP_EVAL_TIMEOUT_SEC` (or a larger client timeout) accordingly.
- **Fixed workload** for this `task_id`: the sysbench command is always **``{oltp}``**; you cannot change it from the BBO search space.
- **Fixed knob schema** for this `task_id` unless the task is constructed with an explicit `knobs_json_path` override (not exposed in `bbo.run` by default).
- The evaluator is **not** a security sandbox: do not point it at production databases; use an isolated host or VM.
"""
    (out / "constraints.md").write_text(textwrap.dedent(constraints).strip() + "\n", encoding="utf-8")

    constraints_zh = f"""# 约束

- 同一评估器实例上**不要并发**发多个重评估；服务端已加锁，但仍建议工作流上串行化。
- 单次评估时间可达数分钟，客户端超时（默认 300s）可能不够，应按需调大。
- 本 `task_id` 下 **sysbench 子命令固定为 ``{oltp}``**，搜索空间**不可**在运行时改 workload。
- 旋钮表默认来自 {knob_path}；`bbo.run` 默认不暴露自定义 JSON 路径。
- 本基准面向**实验/调参**环境，请勿连接生产库。

与 `constraints.md` 配套；以英文为权威说明。
"""
    (out / "constraints.zh.md").write_text(textwrap.dedent(constraints_zh).strip() + "\n", encoding="utf-8")

    prior = f"""# Domain Prior Knowledge

- For **5-knob** tasks, the set was chosen for interpretability; `innodb_doublewrite` is an **enum (ON/OFF)**, the others are integers with wide ranges in the JSON.
- For **~197D** tasks, each evaluation is expensive: favor algorithms that make strong use of a **small** trial budget, or do staged dimensionality reduction outside this core if your research allows.
- Workload **``{oltp}``** stresses different subsystems: read-heavy vs write-heavy optima need not match.
- Expect **noise** between trials even for repeated configurations unless you add explicit replication in your experiment driver.

These notes are *hints*, not formal invariants of the search space.
"""
    (out / "prior_knowledge.md").write_text(textwrap.dedent(prior).strip() + "\n", encoding="utf-8")

    prior_zh = f"""# 先验与经验

- **5 维** 任务更便于作图与人工解释；其中 `innodb_doublewrite` 为 ON/OFF，其余在 JSON 内为带上下界的整数型旋钮。
- **全量 ~197 维** 更贴近“全表调参”场景，但单次评估与算法收敛成本都高，适合高维黑盒方法研究。
- 不同 **sysbench** 子命令（`{oltp}`）对 CPU/IO/锁竞争的需求不同，**跨 workload 的旋钮最优解不可直接外推**。
- 同配置重复评估仍可能有波动，除非你在实验驱动层加多次取均值。

与 `prior_knowledge.md` 一致方向；以英文为权威说明。
"""
    (out / "prior_knowledge.zh.md").write_text(textwrap.dedent(prior_zh).strip() + "\n", encoding="utf-8")

    eval_md = f"""# Evaluation Protocol

- **Data source (inside the container):** a sysbench `prepare` run seeds `sbtest` with fixed `--tables` / `--table-size` in `server.py` (the entrypoint may call `oltp_read_write prepare` once, which is compatible with the other bundled **oltp_*** `run` tests in typical sysbench builds).
- **Metrics:** primary = `throughput` == parsed **per-second** figure from the sysbench text report for **``{oltp}``**; if `transactions` is missing for a script, the server falls back to a `queries` line when present.
- **Request contract:** the Python `HttpDatabaseKnobTask` sends:
  - `workload: "{spec.workload_key}"` (server maps to **``{oltp}``**)
  - `knobs: {{name: "value"}}` (strings acceptable to MariaDB for those variables)
- **Reproducibility:** fix seed for the *optimizer* (BBO `seed`) and keep machine load low; the DB is still a noisy environment.

When reporting results, list **image digest / git commit**, this `task_id`, and the knob JSON filename.
"""
    (out / "evaluation.md").write_text(textwrap.dedent(eval_md).strip() + "\n", encoding="utf-8")

    env_md = f"""# Environment

## Shared reusable Docker image (all eight dbtune MariaDB/sysbench tasks)

```bash
docker pull {_MARIADB_IMAGE}
docker rm -f agentbbo_http_mariadb_eval 2>/dev/null
docker run -d --name agentbbo_http_mariadb_eval -p 8080:8080 {_MARIADB_IMAGE}
```

If the evaluator implementation changes, rebuild and export the two dbtune images with `scripts/package_dbtune_images.sh`, then publish the new tag before using it for comparisons.

`docker-compose.task-services.yml` uses the same image by default. Override it with `AGENTBBO_DBTUNE_MARIADB_IMAGE` if you publish under another Docker Hub namespace or tag.

## Client-side environment (Python)

| Variable | Role |
|----------|------|
| `AGENTBBO_HTTP_EVAL_BASE_URL` | Base URL, default `http://127.0.0.1:8080` |
| `AGENTBBO_HTTP_EVAL_TIMEOUT_SEC` | **Per-POST** timeout (seconds), default `300` |

## This task

| Field | Value |
|------|--------|
| `task_id` | `{spec.task_id}` |
| `workload` (JSON) | `{spec.workload_key}` -> `{oltp}` |
| Knob JSON (default) | {knob_path} |

Health check: `GET /health` on the same base URL.
"""
    (out / "environment.md").write_text(textwrap.dedent(env_md).strip() + "\n", encoding="utf-8")


def main() -> None:
    for tid in DBTUNE_MARIADB_TASK_IDS:
        _write_one(tid)
    print("OK:", len(DBTUNE_MARIADB_TASK_IDS), "tasks ->", _TASK_DESC)


if __name__ == "__main__":
    main()
