# 约束

- `Metal_1`、`Metal_2`、`Metal_3` 是类别型参数，它们的合法标签来自清洗后的缓存数据集。
- 数值边界是硬性的搜索空间限制：`Metal_1_Proportion in [0, 100]`、`Metal_2_Proportion in [0, 100]`、`Metal_3_Proportion in [0, 33.33333333]`、`Hydrothermal Temp degree in [-77, 320]`、`Hydrothermal Time min in [0, 2790]`、`Annealing Temp degree in [25, 1400]`、`Annealing Time min in [0, 943]`、`Proton Concentration M in [0.1, 3.7]`、`Catalyst_Loading mg cm -2 in [0.0, 1.266]`。
- 必须沿用 tutorial 的数据清洗和 dummy-column 对齐逻辑；不要悄悄引入别的编码方式、额外类别水平或隐藏的归一化规则。
- 这个 benchmark 不会额外施加“组成比例必须求和”或“金属必须互异”之类的约束；这是当前封装任务接口的属性，而不是化学结论。
- 每次运行都必须保留 append-only 日志和基于 replay 的 resume 语义。
