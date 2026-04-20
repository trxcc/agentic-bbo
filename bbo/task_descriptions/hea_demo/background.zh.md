# 背景

`hea_demo` 封装了 BO tutorial 仓库中的高熵合金（HEA）组成优化案例。
它的源数据工作簿默认直接随工作区提供，位置是 `bbo/tasks/scientific/data/examples/HEA/data/oracle_data.xlsx`；benchmark 在运行时仍按相对路径 `examples/HEA/data/oracle_data.xlsx` 使用它。其中每一行都记录了一个满足约束的五元合金组成 `Co`、`Fe`、`Mn`、`V`、`Cu`，以及相应的 tutorial 目标分数。

这个任务对应的科学决策问题是：在 simplex 约束和单元素上下界同时存在时，如何分配合金组分，而每次新的材料实验又都很昂贵。
贝叶斯优化在这里的价值，是在不穷举所有可行配比的情况下，提出下一组值得测试的成分。

真实部分是缓存后的合金数据集及其受约束的组成几何结构。
模拟部分是 benchmark evaluator：它先把面向优化器的设计变量解码成真实合金组分，再查询一个拟合后的随机森林 surrogate，而不是真正执行合成与表征流程。
