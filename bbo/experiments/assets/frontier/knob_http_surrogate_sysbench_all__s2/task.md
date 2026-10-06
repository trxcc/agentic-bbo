# MySQL SYSBENCH 196 knobs

Task ID: `knob_http_surrogate_sysbench_all`.

Choose 196 MySQL configuration parameters for a SYSBENCH transactional workload to maximize throughput predicted by a fixed surrogate model. The surrogate is a previously built performance predictor; optimization does not execute a live database workload.

Objective: maximize `throughput`.

Budget: 50 shared initial observations and 200 new evaluations (250 total).

## Submission

1. Submit all 196 parameters. Their names identify database settings, but every submitted value must be a floating-point number u in [0,1]. Do not submit the physical numbers or enumeration labels from the parameter table directly.
2. Integer settings decode as round(min + u*(max-min)). An input of 0 selects the minimum and 1 selects the maximum; intermediate inputs are linearly interpolated and rounded.
3. For K enumeration options, divide [0,1] into K equal-width intervals. Select enum_values[min(floor(u*K),K-1)] using zero-based indexing; u=1 selects the last option. Option order does not imply a performance ranking.
4. Encoding examples: innodb_thread_concurrency has physical range 0...1000, so input 0.25 decodes to 250. autocommit has options ["ON","OFF"], so 0.25 selects ON and 0.75 selects OFF. These illustrate encoding, not recommended configurations.

## Read details as needed

There are 196 active parameters. get_search_space supports names, query, optional annotated groups and paged index/details views.

get_task_context sections: overview, submission, scoring, mechanisms, domain_knowledge, additional. Read scoring rules and relevant mechanisms before choosing a candidate.

get_trial_history and get_incumbent return scores first; request parameter_names to inspect selected values.

Follow instructions.md: submit_candidate accepts a full config or workspace JSON file directly; write_candidate is optional.
