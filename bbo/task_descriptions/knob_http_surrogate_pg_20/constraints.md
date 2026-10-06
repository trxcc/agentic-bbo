# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

Every submitted parameter is a float in [0,1]. Physical definitions below are decoded using the rules above; they are not direct submission values.

| Parameter | Meaning | Physical definition | Default |
| --- | --- | --- | --- |
| `bgwriter_delay` | Delay between background-writer rounds. | Integer: `10` to `10000` | `200` |
| `bgwriter_flush_after` | Amount written by the background writer before requesting operating-system writeback. | Integer: `0` to `256` | `64` |
| `bgwriter_lru_maxpages` | Maximum buffers written per background-writer round. | Integer: `0` to `1073741823` | `100` |
| `bgwriter_lru_multiplier` | Multiplier on recent buffer demand used to estimate background cleaning needs. | Float: `0.0` to `10.0` | `2.0` |
| `checkpoint_completion_target` | Fraction of the checkpoint interval over which checkpoint writes are spread. | Float: `0.0` to `1.0` | `0.5` |
| `checkpoint_flush_after` | Amount written during a checkpoint before requesting operating-system writeback. | Integer: `0` to `256` | `32` |
| `checkpoint_timeout` | Maximum interval between automatic WAL checkpoints. | Integer: `30` to `86400` | `300` |
| `commit_siblings` | Minimum concurrent open transactions considered before applying commit_delay. | Integer: `0` to `1000` | `5` |
| `deadlock_timeout` | Lock-wait interval before checking for a deadlock. | Integer: `1` to `2147483647` | `1000` |
| `default_statistics_target` | Default target controlling the detail of collected column statistics. | Integer: `1` to `10000` | `100` |
| `effective_cache_size` | Planner estimate of cache available to a query; does not allocate memory. | Integer: `1` to `1000000` | `524288` |
| `effective_io_concurrency` | Estimated concurrent disk I/O operations supported by the storage system. | Integer: `0` to `1000` | `1` |
| `min_wal_size` | Lower WAL disk-usage threshold below which old WAL files are recycled for reuse. | Integer: `2` to `8000` | `80` |
| `random_page_cost` | Planner cost estimate for a nonsequential disk-page read. | Float: `0.0` to `100` | `4.0` |
| `seq_page_cost` | Planner cost estimate for a sequential disk-page read. | Float: `0.0` to `100` | `1.0` |
| `temp_buffers` | Per-session buffer limit for access to temporary tables. | Integer: `100` to `10000` | `1024` |
| `wal_buffers` | Shared buffer space for WAL data; physical value -1 selects automatic sizing. | Integer: `-1` to `262143` | `-1` |
| `wal_sync_method` | Operating-system method used to force WAL updates to disk. | Enumeration: `["fsync","fdatasync","open_sync","open_datasync"]` | `"fdatasync"` |
| `wal_writer_delay` | Delay between WAL-writer activity rounds. | Integer: `1` to `10000` | `200` |
| `work_mem` | Base memory limit per sort or hash operation before spilling to temporary disk files. | Integer: `64` to `1000000` | `4096` |

## Fixed task definitions

**Workload (`workload`)**

JOB

**Database (`database`)**

PostgreSQL

**Units (`units_policy`)**

Use the units in this task's parameter definitions. Do not invent unit conversions for entries without an explicit unit.
