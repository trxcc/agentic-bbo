# 提交接口

参数 schema：

- `AcidRed871_0gL`、`L-Cysteine-50gL`、`MethyleneB_250mgL`、`NaCl-3M`、`NaOH-1M`、`P10-MIX1`、`PVP-1wt`、`RhodamineB1_0gL`、`SDS-1wt`、`Sodiumsilicate-1wt`：全部为区间 `[0.0, 5.0]` 上的 `float`

每个 trial 预期写出的内容：

- `config`：提出的 10 维配方
- `objectives.regret`：surrogate 预测得到的 regret
- `metrics.predicted_target`、`metrics.raw_target_max`，以及 `metrics.coord::<feature>`
- `metadata`：分发数据集的溯源信息，例如 `relative_path`、`source_root`、`source_path`、`cache_path`、`source_ref`、`sha256`、`size_bytes`

通过 `bbo.run` 执行时，标准产物包括 append-only 的 `trials.jsonl`、`summary.json`、`plots/trace.png` 和 `plots/distribution.png`。
