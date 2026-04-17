# 环境配置

这个 synthetic task 直接使用仓库统一的 Python 环境。
协作者可以通过以下命令完成配置：

```bash
uv sync --extra dev
```

最小 smoke test：

```bash
uv run python -m bbo.run --algorithm random_search --task branin_demo --max-evaluations 3
```

这个 benchmark 当前不需要单独的 task-specific Docker 镜像。
