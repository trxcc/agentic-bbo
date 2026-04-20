# 约束

- 暴露参数 `SMILES` 是类别型变量，它的取值只能来自缓存压缩包成员 `zinc.txt`。
- 评估必须依赖 RDKit；用 mock score 或学习到的 surrogate 来替换 QED，都不算这个任务的合法实现。
- 无效 SMILES 不会在接口层被直接拒绝，但会通过 `qed = 0.0` 和 `qed_loss = 1.0` 被赋予最差分数。
- 必须保留 append-only 日志和基于 replay 的 resume 语义，并把这个任务视为固定候选池选择问题，而不是开放式分子生成问题。
