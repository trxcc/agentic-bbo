# 评估协议

- 数据来源：缓存后的 `examples/HEA/data/oracle_data.xlsx`，共 286 行。
- 训练视角：在原始合金组成 `Co`、`Fe`、`Mn`、`V`、`Cu` 上拟合 `RandomForestRegressor(n_estimators=100, random_state=<task seed>)` 来预测 `target`。
- 查询路径：先用 tutorial 的 `_phi_inv` 逻辑把提出的 `x1..x4` 解码，再预测 `target`，最后返回 `regret = target.max() - predicted_target`。
- 主目标记为 `regret`；同时记录解码后的组成指标 `composition::Co`、`composition::Fe`、`composition::Mn`、`composition::V`、`composition::Cu`。
- 一旦任务 seed 固定，评估就是确定性的。改变任务 seed 会重新拟合随机森林，并可能改变 regret 数值，因此做对比时应保持 seed 不变；分数相同的情况按更早记录的 trial 处理。
