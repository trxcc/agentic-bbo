# 评估协议

- 数据来源：缓存后的 `examples/BH/BH_dataset.csv`，共 1728 行。
- 预处理：先计算 `yield.max() - yield`，再从候选特征中删除 `cost` 和 `new_index`，然后按 tutorial 风格执行特征筛选：`extractor=random_forest`、`max_n=20`、`max_cum_imp=0.8`、`min_imp=0.01`。
- Oracle：在选出的 descriptor 子集上拟合 `RandomForestRegressor(n_estimators=100, random_state=<task seed>)`，并预测 regret。
- 主目标记为 `regret`；同时记录辅助指标 `predicted_yield` 和 `raw_yield_max`。
- 一旦任务 seed 固定，评估就是确定性的。改变任务 seed 既可能改变搜索空间坐标，也可能改变目标值，因此做 benchmark 对比时必须固定 seed；完全相同的分数按更早记录的 trial 处理。
