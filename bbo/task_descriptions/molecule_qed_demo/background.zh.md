# 背景

`molecule_qed_demo` 封装了 BO tutorial 仓库中的分子选择案例。
它的源压缩包默认直接随工作区提供，位置是 `bbo/tasks/scientific/data/examples/Molecule/zinc.txt.gz`；benchmark 在运行时仍按相对路径 `examples/Molecule/zinc.txt.gz` 使用它。这个 benchmark 的搜索空间是一个固定的候选 SMILES 池，来自压缩包内部的 `zinc.txt` 成员文件。

这个任务对应的科学决策问题是：当需要从大型候选库中筛选更有药物相性质的分子时，下一条应该优先打分哪个候选分子。
与其他表格型 scientific task 不同，这个 benchmark 保留了一个真实且确定性的打分函数，而不是用学习到的 surrogate 替代目标。

真实部分是对每个候选 SMILES 使用 RDKit 计算 QED。
benchmark 的简化之处在于搜索接口本身：候选必须来自固定的缓存压缩包，evaluator 中不存在生成式化学模型、合成环路或真实湿实验测定。
