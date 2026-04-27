# 环境配置

推荐使用仓库受管环境，并安装 scientific task、Optuna 和开发依赖：

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
```

这是一个推荐的已知可行环境，不是唯一可行环境。
这个任务的硬性依赖是 RDKit 和 `selfies`，Optuna smoke 还需要安装 `optuna` extra。

最小 smoke test：

```bash
uv run python -m bbo.run --algorithm optuna_tpe --task qed_selfies_demo --max-evaluations 6
```

也可以先用随机搜索验证任务闭环：

```bash
uv run python -m bbo.run --algorithm random_search --task qed_selfies_demo --max-evaluations 3
```

