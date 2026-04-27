# 评估协议

- 数据来源：本地 BO tutorial ZINC 归档只用于构建 SELFIES token 词表和默认有效分子。
- 每次评估会先规范化 token config，移除 `__PAD__`，遇到 `__EOS__` 停止，然后把 SELFIES 解码成 SMILES 并计算 RDKit QED。
- 任务会报告：
  - 主目标：`qed_loss = 1.0 - qed`
  - 指标：原始 `qed` 和实际使用的 SELFIES token 数量
  - 元数据：解码后的 `selfies`、`smiles`、token 列表、有效性标记和解码错误
- 无效或空序列会得到 `qed = 0.0` 和 `qed_loss = 1.0`。
- 对固定 token 配置，评估器是确定性的。

