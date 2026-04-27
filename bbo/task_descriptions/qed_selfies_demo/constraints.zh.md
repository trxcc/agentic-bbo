# 约束

- 暴露出的搜索空间是固定长度 SELFIES token 序列，每个位置都是一个 `CategoricalParam`。
- `__EOS__` 和 `__PAD__` 是任务内部使用的特殊 token，不会传给 SELFIES 解码器。
- 评估必须安装 RDKit 和 `selfies` 包。
- 空序列、无法解码的 SELFIES、或 RDKit 无法解析的 SMILES 都会得到最差分数：`qed = 0.0`，`qed_loss = 1.0`。
- 这个任务是 token 空间上的生成式搜索，不限制优化器只能选择源归档中已有的 SMILES。
- 必须保留 append-only 日志和基于 replay 的 resume 行为。

