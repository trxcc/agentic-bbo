# 环境配置

这个任务直接使用仓库统一的 Python 环境，不要求 GPU，也不需要 task-local Docker 镜像。
推荐安装方式：

```bash
uv sync --extra dev --extra bo-tutorial
```

默认情况下，这个任务直接从工作区内的 `bbo/tasks/scientific/data/examples/` 读取随仓库提供的数据。
如果你想用自己上传的数据包覆盖默认数据，可以把 `BBO_BO_TUTORIAL_SOURCE_ROOT=/path/to/source_root` 指向新的根目录；像 `examples/Molecule/zinc.txt.gz` 这样的路径都会相对于那个根目录解释。
如有需要，也可以通过 `BBO_BO_TUTORIAL_CACHE_ROOT=/path/to/cache` 重定向缓存后的数据资产。

最小 smoke test：

```bash
uv run python -m bbo.run --algorithm random_search --task molecule_qed_demo --max-evaluations 3
```

除了标准 tutorial 依赖之外，这个任务还要求安装 RDKit。
如果目标环境无法安装 RDKit，请明确记录这个 blocker，而不是用假的 QED evaluator 代替。
