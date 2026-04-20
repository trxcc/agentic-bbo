# 提交接口

参数 schema：

- `Metal_1`、`Metal_2`、`Metal_3`：`categorical`，其精确候选集合来自清洗后的缓存数据集
- `Metal_1_Proportion`、`Metal_2_Proportion`：区间 `[0.0, 100.0]` 上的 `float`
- `Metal_3_Proportion`：区间 `[0.0, 33.33333333]` 上的 `float`
- `Hydrothermal Temp degree`：区间 `[-77, 320]` 上的 `int`
- `Hydrothermal Time min`：区间 `[0, 2790]` 上的 `int`
- `Annealing Temp degree`：区间 `[25, 1400]` 上的 `int`
- `Annealing Time min`：区间 `[0, 943]` 上的 `int`
- `Proton Concentration M`：区间 `[0.1, 3.7]` 上的 `float`
- `Catalyst_Loading mg cm -2`：区间 `[0.0, 1.266]` 上的 `float`

每个 trial 预期写出的内容：

- `config`：提出的 mixed candidate
- `objectives.overpotential_mv`：预测过电位
- `metrics.predicted_overpotential_mv`，以及 `metrics.choice::Metal_1|Metal_2|Metal_3`
- `metadata`：分发数据集的溯源信息，例如 `relative_path`、`source_root`、`source_path`、`cache_path`、`source_ref`、`sha256`、`size_bytes`

通过 `bbo.run` 执行时，标准产物包括 append-only 的 `trials.jsonl`、`summary.json`、`plots/trace.png` 和 `plots/distribution.png`。
