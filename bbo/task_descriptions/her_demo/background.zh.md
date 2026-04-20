# 背景

`her_demo` 封装了论文 *Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial* 中的析氢反应（HER）案例。
它的源数据文件默认直接随工作区提供，位置是 `bbo/tasks/scientific/data/examples/HER/HER_virtual_data.csv`；benchmark 在运行时仍按相对路径 `examples/HER/HER_virtual_data.csv` 对它做 staging。这个文件包含 10 个非负配方控制变量以及一个测得的 `Target` 指标。

这个任务对应的科学决策问题是：当在 10 个控制维度上做完整湿实验扫描代价过高时，下一组催化剂/添加剂配方应该优先尝试什么。
在本 benchmark 中，这个决策过程被近似成一个数据集驱动的 oracle，以便快速、可复现地比较优化方法。

真实部分是分发后的 tutorial 数据集及其变量语义。
模拟部分是 benchmark evaluator：它会在缓存后的表格上重新拟合一个随机森林 surrogate，并返回 surrogate 预测，而不是真正执行 HER 实验。
