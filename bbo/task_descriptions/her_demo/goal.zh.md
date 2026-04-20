# 目标

最小化主目标 `regret`，其中 `regret = Target.max() - predicted_Target`。
因此，regret 越小，就等价于找到预测 HER 表现越高的配置。

一个合法提交只能修改任务搜索空间中暴露出来的 10 个连续控制变量。
一次 evaluation 计作对一个候选配置进行一次 surrogate 预测。

如果调用方没有显式覆盖 `max_evaluations`，这个任务默认在 40 次 evaluation 后停止。
本仓库中的 smoke workflow 使用同一接口，但把预算缩短为 `--max-evaluations 3`。
