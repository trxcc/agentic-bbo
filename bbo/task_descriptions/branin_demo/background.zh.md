# 背景

Branin-Hoo 函数是一个经典的低维 synthetic benchmark，具有三个等价的全局最优点。
在这个仓库里，它主要扮演紧凑型参考 benchmark：评估成本较低，但结构足够丰富，能够展示优化器在 exploration 和 exploitation 之间的行为差异。

这个任务刻意设计成 2D，这样协作者可以直接看完整响应面，并通过可视化确认优化器是否真的覆盖到了正确区域。
