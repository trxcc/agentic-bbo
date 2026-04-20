# 评估协议

- 数据来源：缓存后的 `examples/HER/HER_virtual_data.csv`，共 812 行，包含 10 个暴露特征列和 `Target`。
- 预处理：保留这 10 个 HER 控制量，计算 `Target.max() - Target`，并在该 regret 目标上拟合 `RandomForestRegressor(n_estimators=100, random_state=<task seed>)`。
- 主目标记为 `regret`；同时记录 `predicted_target`、`raw_target_max` 以及 `coord::<feature>` 形式的逐坐标诊断指标。
- 一旦任务 seed 固定，评估就是确定性的。改变任务 seed 会重新拟合 surrogate，并可能改变分数，因此做 benchmark 对比时应保持 seed 不变。
- 如果两个 trial 的主目标完全相同，则优先采用更早写入日志的那个 trial，因为运行是按 append-only 顺序 replay 的。
