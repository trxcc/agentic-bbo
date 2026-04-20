# 约束

- 暴露给优化器的设计变量 `x1`、`x2`、`x3`、`x4` 全部都是区间 `[0.0, 1.0]` 上的 float。
- 内部解码后得到的 `Co`、`Fe`、`Mn`、`V`、`Cu` 组成必须始终落在 `[0.05, 0.35]` 内，并且五个组分之和约等于 `1.0`。
- 只能使用缓存后的 tutorial 工作簿 `examples/HEA/data/oracle_data.xlsx`；不要随意替换 decoder、可行域几何结构或 surrogate 目标。
- 必须保留 append-only 日志和基于 replay 的 resume 语义，确保 benchmark 运行可审计、可复现。
