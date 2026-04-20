# 约束

- 暴露给优化器的控制变量 `AcidRed871_0gL`、`L-Cysteine-50gL`、`MethyleneB_250mgL`、`NaCl-3M`、`NaOH-1M`、`P10-MIX1`、`PVP-1wt`、`RhodamineB1_0gL`、`SDS-1wt`、`Sodiumsilicate-1wt` 全部都是区间 `[0.0, 5.0]` 上的 float。
- 只能使用分发后的 tutorial 数据集 `examples/HER/HER_virtual_data.csv`；不要替换成 synthetic fallback 数据或外部标签。
- 必须保留仓库的 append-only 日志与基于 replay 的 resume 语义；一旦 trial 被写入 JSONL，就不应再修改。
- 这个任务是 surrogate benchmark，不应被表述为脱离当前数据支持范围的真实湿实验结论。
