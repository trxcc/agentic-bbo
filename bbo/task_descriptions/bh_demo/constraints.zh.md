# 约束

- 面向优化器的控制变量，是任务构造时被选出来的连续 descriptor 列；每个被选中的坐标都受其分发数据最小值与最大值约束。
- benchmark 并不直接优化 `yield`。它会先把 `yield` 变换成 regret，而原始列 `cost` 和 `new_index` 永远不是合法优化变量。
- 特征筛选流程被固定为 tutorial 风格的配方：`extractor=random_forest`、`max_n=20`、`max_cum_imp=0.8`、`min_imp=0.01`。
- 为了公平比较，应保持任务 seed 固定，因为改变 seed 可能同时改变被选中的特征子集和拟合后的 surrogate；同时必须保留 append-only 日志和基于 replay 的 resume 语义。
