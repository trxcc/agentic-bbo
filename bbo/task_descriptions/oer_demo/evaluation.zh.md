# 评估协议

- 数据来源：缓存后的 `examples/OER/OER.csv`；benchmark 会把它确定性地清洗成 1319 行的表格，并将 `examples/OER/OER_clean.csv` 保留为参考产物。
- 清洗流程：删除没有目标值的行、去重、规范化类别文本、把数值列强制转换、裁剪部分数值异常点，并只保留目标值位于 5%-95% 分位带内的样本。
- 编码方式：对 `Metal_1`、`Metal_2`、`Metal_3` 使用 `pandas.get_dummies` 做 one-hot，然后把候选样本的列对齐到训练矩阵。
- Oracle：`RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)`，预测 `overpotential_mv`。
- 评估是确定性的，而且在任务侧基本与 seed 无关，因为 oracle 使用固定随机状态。主目标记为 `overpotential_mv`；同时记录 `predicted_overpotential_mv` 和 `choice::<metal_slot>` 指标；若主目标完全相同，则按更早写入日志的 trial 处理。
