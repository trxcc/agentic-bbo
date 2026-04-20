# 背景

`bh_demo` 封装了 BO tutorial 仓库中的 Buchwald-Hartwig 反应案例。
它的源数据表默认直接随工作区提供，位置是 `bbo/tasks/scientific/data/examples/BH/BH_dataset.csv`；benchmark 在运行时仍按相对路径 `examples/BH/BH_dataset.csv` 使用它。原始表格包含反应收率标签，以及大量连续的 ligand、solvent 和 process descriptor。

这个任务对应的科学决策问题是：在不穷举所有条件组合的情况下，搜索能够提升收率的反应设置。
在当前封装 benchmark 里，这个决策问题被投影到了 descriptor space 中，使优化器可以在一个较紧凑的连续接口上工作。

真实部分是缓存后的反应数据集及其 descriptor 列。
模拟部分是 evaluator：它先把 `yield` 转成 regret，再执行随机森林特征筛选，并用拟合的随机森林 surrogate 代替真实化学实验打分。
