# 目标

最小化主目标 `regret`，其中 `regret = target.max() - predicted_target`。
regret 越小，就表示在 tutorial 目标下预测的合金表现越好。

一个合法提交只能修改 4 个暴露设计变量 `x1`、`x2`、`x3`、`x4`。
benchmark 会在内部把它们映射为满足约束的五元组成；一次 evaluation 计作一次“解码 + surrogate 查询”。

如果没有额外覆盖，任务默认在 40 次 evaluation 后停止。
仓库中的 smoke 检查使用同一任务接口，但预算缩短为 `--max-evaluations 3`。
