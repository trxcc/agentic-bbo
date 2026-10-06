# Domain Prior Knowledge

Settings cover memory and caching, I/O, concurrency, query optimization, logging, replication, and authentication. Increasing every resource setting is not a general improvement rule.

Some settings apply only to particular storage engines or enabled features; some are legacy settings. A parameter's real database meaning does not establish its importance in this surrogate.

The available materials do not specify the full hardware and database-version configuration used to train the surrogate. Do not assume every described mechanism is active in this workload.

## Domain experience and conditions

effective_cache_size and page-cost settings influence planner estimates; they do not allocate cache or directly change disk speed. A changed estimate can select a different query plan.

More work_mem may reduce sort or hash spills, but this is a per-operation budget. Several operations and sessions can consume it concurrently.

WAL buffers and checkpoint pacing concern logging and background I/O. Their benefit depends on workload and storage conditions; the surrogate response must test any proposed direction.

JOB contains analytical joins. Statistics, memory, caching, and plan selection can interact, so parameter changes need not produce smooth latency changes.
