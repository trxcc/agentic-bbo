# 环境配置

这个任务与其他 synthetic benchmark 共用仓库级环境。
可通过以下命令完成配置：

```bash
uv sync --extra dev
```

最小 smoke test：

```bash
uv run python -m bbo.run --algorithm random_search --task sphere_demo --max-evaluations 3
```
