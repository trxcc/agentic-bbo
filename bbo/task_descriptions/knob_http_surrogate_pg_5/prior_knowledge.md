# Domain Prior Knowledge

Settings cover memory and caching, I/O, concurrency, query optimization, logging, replication, and authentication. Increasing every resource setting is not a general improvement rule.

Some settings apply only to particular storage engines or enabled features; some are legacy settings. A parameter's real database meaning does not establish its importance in this surrogate.

The available materials do not specify the full hardware and database-version configuration used to train the surrogate. Do not assume every described mechanism is active in this workload.

## Domain experience and conditions

shared_buffers allocates database-page cache. More cache may reduce reads, but it competes with other database memory and the operating-system cache.

WAL buffers and checkpoint pacing concern logging and background I/O. Their benefit depends on workload and storage conditions; the surrogate response must test any proposed direction.

JOB contains analytical joins. Statistics, memory, caching, and plan selection can interact, so parameter changes need not produce smooth latency changes.
