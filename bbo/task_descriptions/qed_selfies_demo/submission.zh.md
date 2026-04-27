# 提交接口

- 优化器必须提交从 `selfies_token_00` 到最后一个 token 槽位的完整配置。
- 每个 token 值都必须来自任务声明的类别候选。
- `__EOS__` 表示分子序列结束，`__PAD__` 会被忽略。
- 日志中的标准配置仍然是 token 字典，解码后的 SELFIES 和 SMILES 会记录在 metadata 中。
- 默认评估预算是 `40`，smoke test 通常使用更小预算，例如 `3` 或 `6`。

