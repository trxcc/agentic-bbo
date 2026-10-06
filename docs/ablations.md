# Ablation Contracts

The diagnostic task pool is BBOB f02/f15, Breast/SVM, Diabetes/RF, PostgreSQL-5,
Sysbench-5, Adaptec1/n32 and Bigblue1/n32, seeds 2--5. Molecular tasks are excluded.
The new-evaluation budgets are 50, 25, 50 and 50 respectively. Initialization is
the same frozen prefix as the matching broad-benchmark task.

## Tools

`T0`: get_task_context, get_search_space, get_trial_history, get_incumbent,
write_candidate, submit_candidate. `T1` additionally enables optimizer_suggest.
`T4` adds optimizer_predict, optimizer_score, optimizer_diagnostics,
optimizer_status, optimizer_set_bounds and optimizer_set_acquisition.
The tool backend cannot call the evaluator or consume evaluation budget.
The GP tool acquisition defaults to EI, as in the diagnostic study; broad-table
numerical baselines retain their separately frozen numerical revision.

## Task Information

`full` retains semantic definitions and domain priors. `semantic` removes the
rendered `mechanisms` and `domain_knowledge` sections and updates their task-card
index, preserving the other text, budget wording and parameter catalog.
`anonymous` exposes a unit-cube codec and opaque names on the six real
diagnostic tasks. BBOB retains the reviewed anonymous bounded numerical domain.
Information treatments alter only the agent-visible task documents and encoding,
not the host objective or raw shared initialization.

## Controlled Priors

The three anonymous 12-dimensional deterministic objectives each have four
active variables. The seven conditions are none, count, geometry, geometry plus
count, correct support, correct support plus geometry, and incorrect support
plus correct geometry. The frozen task documents preserve the actual condition
wording. Every condition uses the same 16 initial and 48 new evaluations.
Definitions and initializations remain host-only; the agent sees no coefficients
or optimum locations.

## LLM Role

Handoff takes the actual online agent prefix after 16 new BBOB evaluations or
eight new HPO evaluations. GP, TuRBO or local search then consumes the remaining
budget with zero further model calls. Full-budget controls use only the common
initial prefix. The launcher validates prefix identity instead of sampling a new
initialization or relabeling a different run as a handoff.

The separate GP-policy control role is not implemented. Its `gp_status`,
`gp_diagnostics` and `commit_gp_policy` interfaces are distinct from the T4
candidate-generation tool study.

## Optimizer Development

`assets/developed_optimizers` contains the six selected, unedited programs and
their original support API entry point. Program selection was made using the
development feedback, not held-out performance. Three independent sessions per
family developed optimizers on four tasks and deployed them on two disjoint tasks.
The original development feedback used initialization-range normalization; the
paper tables subsequently rescored trajectories with the unified RSI rule. The
CSV files retain both quantities so they cannot be confused.

The release provides frozen-program deployment and the original development
prompt/contract artifacts. It does not replace selected programs with newly
generated ones or select a session using test performance. Recreating a development
session requires implementing the supplied development feedback loop in the chosen
model service; this release does not expose the historical campaign scheduler.
