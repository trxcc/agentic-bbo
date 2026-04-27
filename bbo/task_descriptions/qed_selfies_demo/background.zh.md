# 背景

`qed_selfies_demo` 是一个使用 SELFIES token 表示的生成式 QED 分子优化任务。
它复用 BO tutorial 的 ZINC 归档数据来构建 SELFIES token 词表和默认有效分子，但暴露给优化器的不是固定 SMILES 候选池。

与 `molecule_qed_demo` 不同，这个任务的搜索空间是固定长度的 SELFIES token 序列。
Optuna TPE 这类只支持结构化参数的优化器可以逐个位置选择 token，从而组合出不一定存在于原始 ZINC 池中的新分子。

评估器会把 token 序列拼成 SELFIES，解码为 SMILES，然后直接用 RDKit 计算 QED。

