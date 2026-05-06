# Background

`knob_http_mariadb_sysbench_read_write_all` is a **real** MariaDB benchmark in *AgentBBO*. The optimizer proposes a point in the unit hypercube; the packaged evaluator service (Flask inside the reusable image `fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1`) writes `mysqld` knobs, restarts MariaDB, and runs **sysbench**, returning a scalar **throughput** score.

This packaging combines: **read/write (mixed OLTP)** with **the full **~197-dimensional** knob list (`knobs_mysql_all_197.json`), matching the offline `knob_surrogate_sysbench_all` space.**

This benchmark uses the **sysbench** test ``oltp_read_write`` (classic mixed workload).

The measurement is **not** a surrogate: it is the container’s live database and sysbench output. What is “simulated” is only the synthetic `sbtest` dataset and fixed script parameters in `server.py`.

A Chinese companion is in `background.zh.md` (informational only; loaders use the English files for canonical context).
