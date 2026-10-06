# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

Every submitted parameter is a float in [0,1]. Physical definitions below are decoded using the rules above; they are not direct submission values.

| Parameter | Meaning | Physical definition | Default |
| --- | --- | --- | --- |
| `autocommit` | Controls whether ordinary statements automatically commit transactions. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `automatic_sp_privileges` | Controls automatic execution and alteration privileges for stored-routine creators. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `back_log` | Limits the queue of connection requests waiting to be accepted. | Integer: `1` to `65535` | `900` |
| `binlog_cache_size` | Memory allocated per transaction to cache binary-log entries. | Integer: `4096` to `4294967296` | `32768` |
| `binlog_direct_non_transactional_updates` | Controls whether nontransactional table updates bypass the transaction cache and go directly to the binary log. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `binlog_error_action` | Chooses whether the server continues or aborts after a binary-log write error. | Enumeration: `["IGNORE_ERROR","ABORT_SERVER"]` | `"ABORT_SERVER"` |
| `binlog_format` | Selects statement-based, row-based, or mixed binary logging. | Enumeration: `["ROW","STATEMENT","MIXED"]` | `"ROW"` |
| `binlog_group_commit_sync_delay` | Delay before synchronizing a binary-log commit group. | Integer: `0` to `1000000` | `0` |
| `binlog_group_commit_sync_no_delay_count` | Number of grouped transactions that ends the synchronization delay early. | Integer: `0` to `100000` | `0` |
| `binlog_order_commits` | Controls whether transaction commit order follows binary-log order. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `binlog_row_image` | Chooses whether row events include all columns, necessary columns, or omit unnecessary large-object columns. | Enumeration: `["full","minimal","noblob"]` | `"full"` |
| `binlog_rows_query_log_events` | Controls inclusion of original query events in row-based logs. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `binlog_stmt_cache_size` | Binary-log cache allocated to nontransactional statements. | Integer: `4096` to `4294967296` | `32768` |
| `bulk_insert_buffer_size` | Size of the tree cache used for MyISAM bulk inserts. | Integer: `0` to `83886080` | `8388608` |
| `check_proxy_users` | Controls whether built-in authentication plugins check and apply proxy-user mappings. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `concurrent_insert` | Conditions under which MyISAM tables allow concurrent inserts. | Enumeration: `["NEVER","AUTO","ALWAYS"]` | `"AUTO"` |
| `connect_timeout` | Timeout for receiving the client connection handshake. | Integer: `2` to `31536000` | `10` |
| `default_week_format` | Default week-numbering mode for WEEK() when no mode is supplied. | Integer: `0` to `7` | `0` |
| `delay_key_write` | Controls delayed writeback of MyISAM index updates. | Enumeration: `["ON","OFF","ALL"]` | `"ON"` |
| `div_precision_increment` | Additional decimal precision of a division result relative to its dividend. | Integer: `0` to `30` | `4` |
| `end_markers_in_json` | Controls end markers in optimizer JSON output. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `eq_range_index_dive_limit` | Threshold for switching multiple equality-range estimates from index dives to statistics. | Integer: `0` to `4294967295` | `200` |
| `expire_logs_days` | Automatic binary-log retention period in days. | Integer: `0` to `99` | `0` |
| `explicit_defaults_for_timestamp` | Controls explicit default-value rules for TIMESTAMP columns. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `flush` | Controls synchronization of changes to disk after each SQL statement. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `flush_time` | Interval for periodically closing tables and synchronizing data; ignored when flush is enabled. | Integer: `0` to `10` | `0` |
| `ft_min_word_len` | Minimum word length indexed by MyISAM full-text indexes. | Integer: `1` to `8` | `4` |
| `ft_query_expansion_limit` | Maximum number of highly relevant matches used for full-text query expansion. | Integer: `0` to `1000` | `20` |
| `general_log` | Controls the general query log. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `group_concat_max_len` | Maximum length of GROUP_CONCAT() results. | Integer: `4` to `18446700000000000000` | `1024` |
| `host_cache_size` | Capacity of the client host-information cache. | Integer: `0` to `65536` | `0` |
| `innodb_adaptive_flushing` | Controls workload-adaptive dirty-page flushing. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_adaptive_flushing_lwm` | Redo-log occupancy low-water mark for adaptive flushing. | Integer: `0` to `70` | `10` |
| `innodb_adaptive_hash_index_parts` | Number of adaptive hash-index partitions. | Integer: `1` to `512` | `8` |
| `innodb_adaptive_max_sleep_delay` | Upper bound for automatically adjusted InnoDB thread sleep delays; 0 disables adjustment. | Integer: `0` to `1000000` | `150000` |
| `innodb_api_bk_commit_interval` | Background automatic commit interval for the InnoDB memcached interface. | Integer: `1` to `1073741824` | `5` |
| `innodb_api_disable_rowlock` | Controls disabling of row locks in the InnoDB memcached interface. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_api_enable_binlog` | Controls binary logging for the InnoDB memcached interface. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_api_enable_mdl` | Controls table locking against DDL changes for tables used by the InnoDB memcached interface. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_autoextend_increment` | Growth increment for the auto-extending InnoDB system tablespace. | Integer: `1` to `1000` | `64` |
| `innodb_buffer_pool_size` | Size of the InnoDB memory pool caching data and index pages. | Integer: `10307921510` to `15618062894` | `13958643712` |
| `innodb_change_buffer_max_size` | Maximum share of the buffer pool used by the change buffer. | Integer: `0` to `50` | `25` |
| `innodb_change_buffering` | Selects which nonunique secondary-index modifications may be buffered. | Enumeration: `["none","inserts","deletes","changes","purges","all"]` | `"all"` |
| `innodb_cmp_per_index_enabled` | Controls collection of per-index compression statistics. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_commit_concurrency` | Maximum number of threads committing concurrently; 0 means unlimited. | Integer: `0` to `1000` | `0` |
| `innodb_compression_failure_threshold_pct` | Compression failure percentage that triggers adjustment of reserved page space. | Integer: `0` to `100` | `5` |
| `innodb_compression_level` | Compression level, affecting computation cost and compression effectiveness. | Integer: `0` to `9` | `6` |
| `innodb_compression_pad_pct_max` | Maximum reserved free-space percentage in compressed pages to reduce compression failures. | Integer: `0` to `75` | `50` |
| `innodb_concurrency_tickets` | Operation allowance for a thread after it enters InnoDB. | Integer: `1` to `4294967295` | `5000` |
| `innodb_deadlock_detect` | Controls active deadlock detection. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_default_row_format` | Default row format for new InnoDB tables. | Enumeration: `["DYNAMIC","COMPACT","REDUNDANT"]` | `"DYNAMIC"` |
| `innodb_disable_sort_file_cache` | Controls bypassing the operating-system cache for temporary index-build files. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_doublewrite` | Controls doublewrite-buffer protection against incomplete data-page writes. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_file_per_table` | Controls separate tablespace files for newly created tables. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_fill_factor` | Target page fill for sorted B-tree index builds, reserving space for growth. | Integer: `10` to `100` | `100` |
| `innodb_flush_log_at_timeout` | Interval for periodic redo-log flushing. | Integer: `1` to `2700` | `1` |
| `innodb_flush_log_at_trx_commit` | Redo-log write and disk-flush policy at transaction commit. | Enumeration: `["0","1","2"]` | `1` |
| `innodb_flush_neighbors` | Controls whether dirty neighbors are flushed along with a dirty page. | Enumeration: `["0","1","2"]` | `1` |
| `innodb_flush_sync` | Controls whether checkpoint bursts can exceed configured I/O rate limits. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_flushing_avg_loops` | Number of flushing-state snapshots retained, affecting adaptation to workload changes. | Integer: `1` to `1000` | `30` |
| `innodb_ft_cache_size` | Per-table full-text index memory-cache size. | Integer: `1600000` to `80000000` | `8000000` |
| `innodb_ft_enable_diag_print` | Controls additional full-text search diagnostics. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_ft_enable_stopword` | Controls stopword filtering in full-text indexes. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_ft_max_token_size` | Maximum InnoDB full-text token length; the ngram parser uses a separate setting. | Integer: `10` to `84` | `84` |
| `innodb_ft_min_token_size` | Minimum InnoDB full-text token length; the ngram parser uses a separate setting. | Integer: `0` to `16` | `3` |
| `innodb_ft_num_word_optimize` | Maximum words processed per full-text index optimization. | Integer: `1000` to `10000` | `2000` |
| `innodb_ft_result_cache_limit` | Result-cache size limit for one full-text query. | Integer: `1000000` to `4294967295` | `2000000000` |
| `innodb_ft_sort_pll_degree` | Parallel threads for tokenization and indexing during full-text index construction. | Integer: `1` to `16` | `2` |
| `innodb_ft_total_cache_size` | Combined full-text index cache limit across all tables. | Integer: `32000000` to `1600000000` | `640000000` |
| `innodb_io_capacity` | Normal I/O capacity estimate for InnoDB background tasks. | Integer: `100` to `2000000` | `200` |
| `innodb_io_capacity_max` | Maximum I/O capacity for accelerated background flushing. | Integer: `100` to `40000` | `400` |
| `innodb_log_buffer_size` | Memory buffer size for redo entries before writing redo-log files. | Integer: `262144` to `4294967295` | `16777216` |
| `innodb_log_compressed_pages` | Controls logging of recompressed page images in the redo log. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_log_file_size` | Size of one redo-log file. | Integer: `4194304` to `1073741824` | `50331648` |
| `innodb_log_files_in_group` | Number of files in the redo-log group. | Integer: `2` to `10` | `2` |
| `innodb_log_write_ahead_size` | Redo-log write-ahead block size. | Integer: `512` to `16384` | `8192` |
| `innodb_lru_scan_depth` | Depth of each buffer-pool LRU scan by page-cleaning threads. | Integer: `100` to `10240` | `1024` |
| `innodb_max_dirty_pages_pct` | Target upper percentage of dirty pages in the buffer pool. | Integer: `0` to `99` | `75` |
| `innodb_max_dirty_pages_pct_lwm` | Dirty-page low-water mark for starting flushing early. | Integer: `0` to `99` | `0` |
| `innodb_max_purge_lag` | Purge backlog threshold that triggers delays to foreground operations. | Integer: `0` to `4294967295` | `0` |
| `innodb_max_purge_lag_delay` | Maximum foreground-operation delay caused by purge backlog. | Integer: `0` to `10000000` | `0` |
| `innodb_max_undo_log_size` | Undo-tablespace size threshold for marking a tablespace for truncation. | Integer: `10485760` to `18446700000000000000` | `1073741824` |
| `innodb_old_blocks_time` | Protection interval after first access before an old page can enter the new-page region. | Integer: `0` to `4294967295` | `1000` |
| `innodb_online_alter_log_max_size` | Maximum temporary-log size for concurrent changes during online DDL. | Integer: `65536` to `18446700000000000000` | `134217728` |
| `innodb_open_files` | Maximum tablespace files InnoDB may keep open simultaneously. | Integer: `10` to `655350` | `2000` |
| `innodb_optimize_fulltext_only` | Controls whether OPTIMIZE TABLE optimizes only full-text indexes. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_page_cleaners` | Number of background dirty-page cleaner threads. | Integer: `1` to `8` | `4` |
| `innodb_print_all_deadlocks` | Controls logging of every deadlock to the error log. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_purge_batch_size` | Number of undo-log pages processed per purge batch. | Integer: `1` to `5000` | `300` |
| `innodb_purge_rseg_truncate_frequency` | Frequency of purge checks for freeing rollback segments. | Integer: `1` to `128` | `128` |
| `innodb_purge_threads` | Number of background threads purging old record versions. | Integer: `1` to `32` | `4` |
| `innodb_random_read_ahead` | Controls random read-ahead. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_read_ahead_threshold` | Sequential-page-access threshold for linear read-ahead. | Integer: `0` to `64` | `56` |
| `innodb_read_io_threads` | Number of background read I/O threads. | Integer: `1` to `64` | `4` |
| `innodb_replication_delay` | Replication-thread delay when the InnoDB concurrency limit is reached. | Integer: `0` to `10000` | `0` |
| `innodb_rollback_on_timeout` | Chooses whole-transaction or current-statement rollback after a lock-wait timeout. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_rollback_segments` | Number of rollback segments available to transactions generating undo records. | Integer: `1` to `128` | `128` |
| `innodb_sort_buffer_size` | Sort-buffer size for building InnoDB indexes. | Integer: `65536` to `67108864` | `1048576` |
| `innodb_spin_wait_delay` | Delay between mutex spin checks. | Integer: `0` to `6000` | `6` |
| `innodb_stats_auto_recalc` | Controls automatic persistent-statistics recalculation after substantial table changes. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_stats_include_delete_marked` | Controls inclusion of delete-marked, unpurged records in persistent statistics. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_stats_method` | How index statistics treat NULL values. | Enumeration: `["nulls_equal","nulls_unequal","nulls_ignored"]` | `"nulls_equal"` |
| `innodb_stats_on_metadata` | Controls updating nonpersistent statistics during metadata queries. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_stats_persistent` | Controls persistent storage of optimizer statistics. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_stats_transient_sample_pages` | Sample-page count for estimating nonpersistent index statistics. | Integer: `1` to `100` | `8` |
| `innodb_strict_mode` | Controls treating certain invalid InnoDB table-definition options as errors. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_sync_array_size` | Number of internal wait arrays coordinating threads waiting for locks. | Integer: `1` to `1024` | `1` |
| `innodb_sync_spin_loops` | Mutex spin-loop count before a waiting thread sleeps. | Integer: `0` to `30000` | `30` |
| `innodb_table_locks` | Controls InnoDB recognition and handling of explicit MySQL table locks. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_thread_concurrency` | Maximum threads concurrently inside InnoDB; physical value 0 means no such cap. | Integer: `0` to `1000` | `0` |
| `innodb_thread_sleep_delay` | Initial sleep delay for threads waiting to enter InnoDB. | Integer: `0` to `1000000` | `10000` |
| `innodb_undo_log_truncate` | Controls automatic undo-tablespace truncation and reclamation. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `innodb_use_native_aio` | Controls operating-system native asynchronous I/O on supported platforms. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `innodb_write_io_threads` | Number of background write I/O threads. | Integer: `1` to `64` | `4` |
| `join_buffer_size` | Buffer size for joins that cannot directly use indexes and related operations. | Integer: `128` to `1073741824` | `262144` |
| `keep_files_on_create` | Controls preserving existing MyISAM data or index files and reporting an error on name conflicts at table creation. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `key_buffer_size` | MyISAM index-block cache size. | Integer: `8` to `17179869184` | `8388608` |
| `key_cache_age_threshold` | How long an unused hot MyISAM key-cache block waits before moving to the warm region. | Integer: `100` to `30000` | `300` |
| `key_cache_block_size` | MyISAM key-cache block size. | Integer: `512` to `16384` | `1024` |
| `key_cache_division_limit` | Minimum fraction of the MyISAM key cache assigned to the warm region. | Integer: `1` to `100` | `100` |
| `local_infile` | Controls client-side file reads through LOAD DATA LOCAL INFILE. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `log_bin_trust_function_creators` | Controls relaxed stored-function safety checks when binary logging is enabled. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `log_bin_use_v1_row_events` | Controls use of the older row-event format in binary logs. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `log_builtin_as_identified_by_password` | Controls legacy-compatible binary-log representation of account-management statements. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `log_output` | Selects file, table, or no output for general and slow query logs. | Enumeration: `["TABLE","FILE","NONE"]` | `"FILE"` |
| `log_queries_not_using_indexes` | Controls slow logging of queries that do not use indexes. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `log_slave_updates` | Controls writing replicated updates to the replica's own binary log. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `log_slow_admin_statements` | Controls slow logging of long-running administrative statements. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `log_statements_unsafe_for_binlog` | Controls warnings for statements unsafe for statement-based replication. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `log_syslog_include_pid` | Controls process identifiers in error messages sent to the system log. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `log_timestamps` | Selects UTC or system time zone for log timestamps. | Enumeration: `["UTC","SYSTEM"]` | `"UTC"` |
| `long_query_time` | Execution-time threshold for treating a query as slow. | Integer: `0` to `20` | `10` |
| `low_priority_updates` | Controls read priority over updates in engines using table-level locks. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `lower_case_table_names` | Table-name letter casing on storage and name-comparison rules. | Integer: `0` to `2` | `0` |
| `master_verify_checksum` | Controls source-server checksum verification when sending binary-log events. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `max_allowed_packet` | Maximum communication packet or generated intermediate-string size. | Integer: `1024` to `1073741824` | `4194304` |
| `max_binlog_cache_size` | Maximum total size of a transaction's binary-log cache. | Integer: `4096` to `18446744073709551615` | `18446744073709500416` |
| `max_binlog_size` | Rotation size threshold for one binary-log file. | Integer: `4096` to `1073741824` | `1073741824` |
| `max_binlog_stmt_cache_size` | Maximum total binary-log cache size for nontransactional statements. | Integer: `4096` to `18446744073709500416` | `18446744073709500416` |
| `max_connections` | Maximum simultaneous client connections accepted by the server. | Integer: `1` to `100000` | `151` |
| `max_delayed_threads` | Legacy delayed-insert thread limit; MySQL 5.7 no longer supports DELAYED inserts. | Integer: `0` to `16384` | `20` |
| `max_digest_length` | Parser memory limit for statement-digest computation. | Integer: `0` to `1048576` | `1024` |
| `max_error_count` | Maximum retained diagnostic errors, warnings, and notes. | Integer: `0` to `65535` | `64` |
| `max_heap_table_size` | Size limit for user-created MEMORY tables; also constrains in-memory temporary tables. | Integer: `16384` to `1073741824` | `16777216` |
| `max_join_size` | Limits joins expected to examine a large number of rows. | Integer: `1` to `18446744073709551615` | `18446744073709551615` |
| `max_length_for_sort_data` | Threshold for additional row data carried during sorting. | Integer: `4` to `8388608` | `1024` |
| `max_points_in_geometry` | Upper bound on the points-per-circle argument of ST_Buffer_Strategy(). | Integer: `3` to `1048576` | `65536` |
| `max_prepared_stmt_count` | Maximum simultaneous prepared statements on the server. | Integer: `0` to `1048576` | `16382` |
| `max_seeks_for_key` | Upper bound on assumed index seeks when the optimizer estimates lookup costs. | Integer: `1` to `18446744073709551615` | `18446744073709500416` |
| `max_sort_length` | Maximum variable-length value prefix used for sorting comparisons. | Integer: `4` to `8388608` | `1024` |
| `max_user_connections` | Maximum concurrent connections per user; 0 disables this global cap. | Integer: `0` to `4294967295` | `0` |
| `max_write_lock_count` | Number of successive write-lock requests before waiting readers receive a turn. | Integer: `1` to `18446744073709551615` | `18446744073709500416` |
| `multi_range_count` | Legacy setting documented as having no effect in MySQL 5.7. | Integer: `1` to `4294967295` | `256` |
| `mysql_native_password_proxy_users` | Controls proxy-user mapping support in the mysql_native_password plugin. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `net_buffer_length` | Initial connection communication-buffer size. | Integer: `1024` to `1048576` | `16384` |
| `net_read_timeout` | Timeout while waiting to read more client data. | Integer: `1` to `60` | `30` |
| `net_write_timeout` | Timeout while waiting to write data to a client. | Integer: `1` to `120` | `60` |
| `ngram_token_size` | Token length for the ngram full-text parser. | Integer: `1` to `10` | `2` |
| `offline_mode` | Controls rejection or disconnection of ordinary clients while preserving administrative access. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `old_alter_table` | Controls preference for the older table-copy implementation of ALTER TABLE. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `old_passwords` | Password-hashing method used by PASSWORD(); a legacy authentication setting. | Enumeration: `["0","2"]` | `0` |
| `open_files_limit` | Open-file limit requested by the server from the operating system. | Integer: `0` to `655350` | `50000` |
| `optimizer_prune_level` | Controls pruning of the optimizer's join-order search. | Integer: `0` to `1` | `1` |
| `optimizer_search_depth` | Depth of the optimizer's join-order search. | Integer: `0` to `62` | `62` |
| `preload_buffer_size` | Buffer size for preloading MyISAM indexes. | Integer: `1024` to `1073741824` | `32768` |
| `query_alloc_block_size` | Memory allocation block size for statement parsing and execution. | Integer: `1024` to `134217728` | `8192` |
| `query_cache_limit` | Maximum individual result size admitted to the query cache. | Integer: `0` to `134217728` | `1048576` |
| `query_cache_min_res_unit` | Minimum allocation unit for cached query-result storage. | Integer: `512` to `65536` | `4096` |
| `query_cache_size` | Memory size of the legacy query-result cache. | Integer: `0` to `2147483648` | `1048576` |
| `query_cache_wlock_invalidate` | Controls whether a table's write lock prevents use of its cached query results. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `query_prealloc_size` | Preallocated memory size for statement parsing and execution. | Integer: `8192` to `134217728` | `8192` |
| `range_alloc_block_size` | Allocation block size during range-access optimization. | Integer: `4096` to `65536` | `4096` |
| `read_buffer_size` | Read-buffer size for MyISAM sequential scans and related operations. | Integer: `8192` to `2147479552` | `131072` |
| `read_rnd_buffer_size` | Buffer size for reading MyISAM rows in sorted order and related operations. | Integer: `1` to `134217728` | `262144` |
| `require_secure_transport` | Controls rejection of insecure client transports. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `session_track_gtids` | Selects which transactions' GTIDs are reported to clients. | Enumeration: `["OFF","OWN_GTID","ALL_GTIDS"]` | `"OFF"` |
| `session_track_schema` | Controls reporting of default-database changes to clients. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `session_track_state_change` | Controls reporting of session-state-change flags to clients. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `session_track_transaction_info` | Selects transaction-state or transaction-characteristic information reported to clients. | Enumeration: `["OFF","STATE","CHARACTERISTICS"]` | `"OFF"` |
| `show_compatibility_56` | Controls MySQL 5.6 compatibility in system-variable and status interfaces. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `skip_external_locking` | Controls skipping operating-system external file locks for MyISAM. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `skip_name_resolve` | Controls skipping DNS hostname resolution during client authentication. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `skip_networking` | Controls disabling TCP/IP client connections. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `slow_query_log` | Controls the slow query log. | Enumeration: `["ON","OFF"]` | `"OFF"` |
| `sort_buffer_size` | Per-session buffer size for operations requiring sorting. | Integer: `32768` to `134217728` | `262144` |
| `stored_program_cache` | Soft limit on stored routines cached per connection. | Integer: `16` to `524288` | `256` |
| `sync_binlog` | Commit-group interval for synchronizing the binary log to disk; 0 delegates flushing to the operating system. | Integer: `0` to `4294967295` | `1` |
| `sync_frm` | Controls synchronization of legacy table-definition files after writes. | Enumeration: `["ON","OFF"]` | `"ON"` |
| `table_definition_cache` | Number of tables accommodated by the table-definition cache. | Integer: `400` to `524288` | `1400` |
| `table_open_cache` | Open-table cache capacity across threads. | Integer: `1` to `250000` | `2000` |
| `table_open_cache_instances` | Number of open-table cache instances used to distribute lock contention. | Integer: `1` to `64` | `16` |
| `thread_cache_size` | Number of reusable client-connection threads cached. | Integer: `0` to `16384` | `0` |
| `tmp_table_size` | Size limit for internal in-memory temporary tables, jointly constrained by max_heap_table_size. | Integer: `1024` to `1073741824` | `16777216` |
| `transaction_alloc_block_size` | On-demand allocation block size for the transaction memory pool. | Integer: `1024` to `131072` | `8192` |
| `transaction_prealloc_size` | Initial preallocated size of the transaction memory pool. | Integer: `1024` to `131072` | `4096` |
| `updatable_views_with_limit` | Controls updates through views with LIMIT when a complete unique key is absent. | Enumeration: `["YES","NO"]` | `"YES"` |

## Fixed task definitions

**Workload (`workload`)**

SYSBENCH

**Database (`database`)**

MySQL

**Units (`units_policy`)**

Use the units in this task's parameter definitions. Do not invent unit conversions for entries without an explicit unit.
