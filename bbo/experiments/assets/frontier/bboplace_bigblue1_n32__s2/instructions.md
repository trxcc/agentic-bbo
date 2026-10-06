# Optimization protocol

Read task.md first. Query parameter definitions, task details and evaluated
history as needed; full data files need not be printed. Use only permitted task
context and real observations. Do not execute or inspect the hidden evaluator,
fabricate observations, install dependencies, or modify task or history files.
Do not load task datasets or train a replica of the objective, even from a public
dataset library. Surrogate modeling fitted only to visible evaluated history and
scratch work below scratch/ are allowed.

Submit directly with submit_candidate(config={...}), or generate a complete JSON
configuration below scratch/ using native tools and submit_candidate(path="scratch/candidate.json").
The file may contain the raw configuration or {"config": {...}}. It is read and
snapshotted by the host at submission. You may generate any legal configuration;
you do not have to modify an existing trial. For convenience, submit_candidate
also accepts base_trial_id plus changes, inheriting all other values exactly.
write_candidate is optional preparation/validation, returning a candidate_id
that submit_candidate can also accept. A separate write call is not required.
After acceptance, stop calling tools; a short acknowledgement suffices. Do not
repeat the configuration. On later rounds use the new host feedback; query
get_trial_history(after_trial_id=...) for additional new observations rather
than repeatedly printing the full history. Older observations remain queryable.
The host evaluates the committed candidate and updates history in the next round.
A submission receipt is not an objective value. Correct rejected candidates by
submitting a corrected configuration, file, or optional saved candidate ID.

History queries are paginated by both row count and character budget. mode="all"
selects all observations but does not return them in a single page. Follow the
exact next_cursor until it is null; never treat a trial_id as an array index.
For local analysis, use get_trial_history(mode="all", include_config=true,
output_path="scratch/observations.json") to export all selected observations
without printing them. The file contains items and total; no pagination is needed.
Do not combine output_path with cursor, limit or max_chars.

get_incumbent returns scores by default. For configuration queries, explicit
parameter_names take precedence over include_config. To read a full incumbent,
use include_config=true; max_chars defaults to 24000. For larger configurations,
add output_path="scratch/incumbent.json" and omit max_chars. The exported items
array contains only the best observed trial, or is empty if none exists.


## Local process control

Use bash to manage your programs. The recommended container image provides `ps`, `pgrep`, `pkill`, `kill`, `setsid`, and `timeout`; examples are in `/usr/local/share/agent-process-control.md`. Inspect PIDs/PGIDs before sending TERM, then KILL if needed; verify termination. Native session IDs are not PIDs. For Ctrl-C via write_stdin, launch exec_command with tty=true; closed stdin requires cancellation by PID/PGID.


Use only supplied task context and host-evaluated observations. Do not inspect
hidden evaluators, task datasets, netlists, surrogate weights, other runs, parent
directories, repository files, external services or host credentials. Do not
perform unbudgeted objective evaluations. Surrogate modeling fitted only to
visible observations and scratch work below scratch/ are allowed. Only the host
evaluates a submitted candidate.
Internet search, downloads and remote tools are unavailable. Do not recreate
the objective or calculate its target-specific scoring components locally.
General numerical analysis and chemistry syntax/structure checks are allowed;
only host-returned evaluations may supply target scores. Do not install packages.
