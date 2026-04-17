# 约束

- 只能在声明的边界内提出参数值。
- 这是一个纯 black-box benchmark：只能使用暴露出的搜索空间接口和历史 trial 记录。
- 目标评估按串行、append-only 的方式执行；不要修改历史记录，也不要依赖隐藏 checkpoint。
