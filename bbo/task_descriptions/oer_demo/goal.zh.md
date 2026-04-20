# 目标

最小化主目标 `overpotential_mv`。
在这个封装 benchmark 中，预测过电位越低，就代表候选催化剂与工艺方案越好。

一个合法提交只能修改任务暴露出来的混合搜索空间变量：3 个类别型金属身份、3 个组成比例、4 个热处理整型参数、质子浓度，以及催化剂载量。
一次 evaluation 计作对一个候选样本完成清洗、编码并送入拟合 oracle 的一次打分。

如果没有额外覆盖，任务默认在 40 次 evaluation 后停止。
由于接口是 mixed space 而非纯数值空间，仓库中的 smoke 检查使用 `random_search` 并设置 `--max-evaluations 3`。
