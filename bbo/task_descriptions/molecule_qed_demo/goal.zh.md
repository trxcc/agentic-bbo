# 目标

最小化主目标 `qed_loss`，其中 `qed_loss = 1.0 - qed`。
因此 loss 越低，就等价于选择到 RDKit QED 更高的分子。

一个合法提交只能修改唯一的类别参数 `SMILES`，而且它的取值必须来自缓存压缩包成员 `zinc.txt`。
一次 evaluation 计作对一个候选字符串执行一次 RDKit 解析和 QED 计算。

如果没有额外覆盖，任务默认在 40 次 evaluation 后停止。
仓库中的 smoke 检查使用同一接口，但把预算缩短为 `--max-evaluations 3`。
