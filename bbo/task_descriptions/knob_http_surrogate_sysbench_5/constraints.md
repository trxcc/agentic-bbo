# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

Every submitted parameter is a float in [0,1]. Physical definitions below are decoded using the rules above; they are not direct submission values.

| Parameter | Meaning | Physical definition | Default |
| --- | --- | --- | --- |
| `innodb_doublewrite` | Controls doublewrite-buffer protection against incomplete data-page writes. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_thread_concurrency` | Maximum threads concurrently inside InnoDB; physical value 0 means no such cap. | Integer: `0` to `1000` | `0` |
| `max_heap_table_size` | Size limit for user-created MEMORY tables; also constrains in-memory temporary tables. | Integer: `16384` to `1073741824` | `16777216` |
| `query_prealloc_size` | Preallocated memory size for statement parsing and execution. | Integer: `8192` to `134217728` | `8192` |
| `tmp_table_size` | Size limit for internal in-memory temporary tables, jointly constrained by max_heap_table_size. | Integer: `1024` to `1073741824` | `16777216` |

## Fixed task definitions

**Workload (`workload`)**

SYSBENCH

**Database (`database`)**

MySQL

**Units (`units_policy`)**

Use the units in this task's parameter definitions. Do not invent unit conversions for entries without an explicit unit.
