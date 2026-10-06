# Domain Prior Knowledge

Settings cover memory and caching, I/O, concurrency, query optimization, logging, replication, and authentication. Increasing every resource setting is not a general improvement rule.

Some settings apply only to particular storage engines or enabled features; some are legacy settings. A parameter's real database meaning does not establish its importance in this surrogate.

The available materials do not specify the full hardware and database-version configuration used to train the surrogate. Do not assume every described mechanism is active in this workload.

## Domain experience and conditions

Client connection limits and internal InnoDB concurrency limits are different. More concurrency may increase throughput or intensify lock and resource contention. A physical innodb_thread_concurrency value of 0 means no such concurrency cap.

Statistics and cache settings jointly affect query plans, temporary tables, and sorting. Directions suggested by these mechanisms still need to be checked against this task's evaluation feedback.
