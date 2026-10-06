# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

Every submitted parameter is a float in [0,1]. Physical definitions below are decoded using the rules above; they are not direct submission values.

| Parameter | Meaning | Physical definition | Default |
| --- | --- | --- | --- |
| `innodb_adaptive_hash_index_parts` | Number of adaptive hash-index partitions. | Integer: `1` to `512` | `8` |
| `innodb_compression_failure_threshold_pct` | Compression failure percentage that triggers adjustment of reserved page space. | Integer: `0` to `100` | `5` |
| `innodb_stats_method` | How index statistics treat NULL values. | Enumeration: `["nulls_equal","nulls_unequal","nulls_ignored"]` | `"nulls_equal"` |
| `innodb_stats_persistent` | Controls persistent storage of optimizer statistics. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_thread_concurrency` | Maximum threads concurrently inside InnoDB; physical value 0 means no such cap. | Integer: `0` to `1000` | `0` |

## Fixed task definitions

**Workload (`workload`)**

JOB

**Database (`database`)**

MySQL

**Units (`units_policy`)**

Use the units in this task's parameter definitions. Do not invent unit conversions for entries without an explicit unit.
