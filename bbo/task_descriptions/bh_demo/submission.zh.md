# 提交接口

参数 schema：

- 当前任务实例中的有效参数，就是由 `task.spec.search_space` 暴露出来的那组被选中的 descriptor 列，它们取决于当前任务 seed。
- 每个有效参数都是 `float`，并受对应列在分发数据中的最小值与最大值约束。
- 精确的有效特征集合记录在 `task.sanity_check().metadata["selected_features"]` 中，对应边界记录在 `task.sanity_check().metadata["selected_feature_bounds"]` 中。

每个 trial 预期写出的内容：

- `config`：在当前选中特征集合上提出的 descriptor 向量
- `objectives.regret`：surrogate 预测得到的 regret
- `metrics.predicted_yield` 和 `metrics.raw_yield_max`
- `metadata`：分发数据集的溯源信息，例如 `relative_path`、`source_root`、`source_path`、`cache_path`、`source_ref`、`sha256`、`size_bytes`

通过 `bbo.run` 执行时，标准产物包括 append-only 的 `trials.jsonl`、`summary.json`、`plots/trace.png` 和 `plots/distribution.png`。
