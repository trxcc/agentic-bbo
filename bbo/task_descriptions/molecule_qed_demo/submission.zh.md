# 提交接口

参数 schema：

- `SMILES`：`categorical`，从压缩包成员 `zinc.txt` 载入的 249455 条缓存字符串中选择

每个 trial 预期写出的内容：

- `config.SMILES`：提出的分子字符串
- `objectives.qed_loss`：报告的主目标
- `metrics.qed`：原始 RDKit QED 分数
- `metadata.smiles`、`metadata.valid_smiles`，以及分发数据集的溯源信息，例如 `relative_path`、`source_root`、`source_path`、`cache_path`、`source_ref`、`sha256`、`size_bytes`

通过 `bbo.run` 执行时，标准产物包括 append-only 的 `trials.jsonl`、`summary.json`、`plots/trace.png` 和 `plots/distribution.png`。
