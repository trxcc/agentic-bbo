# 目标

最小化主目标 `regret`，其中 `regret = yield.max() - predicted_yield`。
因此 regret 越小，就意味着预测的 Buchwald-Hartwig 收率越高。

一个合法提交只能修改任务实例在特征筛选之后暴露出来的那些连续 descriptor 坐标。
一次 evaluation 计作对一个候选 descriptor 向量进行一次 surrogate 预测。

如果没有额外覆盖，任务默认在 40 次 evaluation 后停止。
仓库中的 smoke 检查使用同一接口，但把预算缩短为 `--max-evaluations 3`。
