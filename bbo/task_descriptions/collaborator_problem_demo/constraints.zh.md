# 约束

- 参数 proposal 必须落在声明的边界或枚举集合内。
- evaluator 必须是确定性的，或者清楚说明其噪声模型。
- 运行应能仅依靠 append-only JSONL 日志恢复，而不依赖隐藏 optimizer checkpoint。
- 如果真实任务存在安全关键区域，必须在这里明确写出，而不是假设优化器会自行推断。
