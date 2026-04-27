# 目标

最小化主目标 `qed_loss`，其中：

- `qed_loss = 1.0 - qed`
- `qed` 是 RDKit 对解码后分子计算得到的 QED 值

因此，更低的 loss 等价于提出 QED 更高的有效分子。

一次有效提交需要给每个 `selfies_token_XX` 类别型参数赋值。
任务会从左到右读取 token，遇到 `__EOS__` 后停止，忽略 `__PAD__`，并对拼出的 SELFIES 分子进行评分。

默认评估预算是 `40`。

