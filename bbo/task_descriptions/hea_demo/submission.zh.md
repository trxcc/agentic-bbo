# 提交接口

参数 schema：

- `x1`、`x2`、`x3`、`x4`：全部为区间 `[0.0, 1.0]` 上的 `float`

每个 trial 预期写出的内容：

- `config`：提出的设计空间坐标 `x1..x4`
- `objectives.regret`：surrogate 预测得到的 regret
- `metrics.predicted_target`，以及解码后的组成指标 `composition::Co`、`composition::Fe`、`composition::Mn`、`composition::V`、`composition::Cu`
- `metadata`：分发数据集的溯源信息，例如 `relative_path`、`source_root`、`source_path`、`cache_path`、`source_ref`、`sha256`、`size_bytes`

通过 `bbo.run` 执行时，标准产物包括 append-only 的 `trials.jsonl`、`summary.json`、`plots/trace.png` 和 `plots/distribution.png`。
