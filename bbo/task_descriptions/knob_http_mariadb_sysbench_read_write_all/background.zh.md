# 背景

`knob_http_mariadb_sysbench_read_write_all` 是 *AgentBBO* 中的 **真实 MariaDB** 黑盒调参：优化器给出 `[0,1]` 超立方体中的一点，Python 任务解码为 ```bbo/tasks/dbtune/assets/knobs_mysql_all_197.json``` 中的物理旋钮，再发给评估容器服务；容器内写配置、重启、跑 **sysbench**，返回标量 TPS/吞吐类指标。

本任务在口径上固定为：**读写、全量旋钮**

本任务使用 **sysbench** 的 ``oltp_read_write`` 混合事务负载。

与 HER/synthetic 不同，**数值来自容器内压测**；可重复性受硬件负载、冷/热缓存等影响。协作者可阅读本文件，运行时仍以 `background.md` 为准。
