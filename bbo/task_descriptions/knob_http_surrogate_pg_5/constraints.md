# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

Every submitted parameter is a float in [0,1]. Physical definitions below are decoded using the rules above; they are not direct submission values.

| Parameter | Meaning | Physical definition | Default |
| --- | --- | --- | --- |
| `backend_flush_after` | Amount written by one backend before requesting operating-system writeback. | Integer: `0` to `256` | `0` |
| `checkpoint_completion_target` | Fraction of the checkpoint interval over which checkpoint writes are spread. | Float: `0.0` to `1.0` | `0.5` |
| `max_worker_processes` | Maximum background worker processes supported by the server. | Integer: `0` to `262143` | `8` |
| `shared_buffers` | Memory allocated to shared database-page buffers. | Integer: `16` to `1000000` | `1024` |
| `wal_buffers` | Shared buffer space for WAL data; physical value -1 selects automatic sizing. | Integer: `-1` to `262143` | `-1` |

## Fixed task definitions

**Workload (`workload`)**

JOB

**Database (`database`)**

PostgreSQL

**Units (`units_policy`)**

Use the units in this task's parameter definitions. Do not invent unit conversions for entries without an explicit unit.
