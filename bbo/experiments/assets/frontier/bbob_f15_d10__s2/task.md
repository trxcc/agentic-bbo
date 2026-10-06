# Anonymous optimization task

Optimize an unknown scalar objective over 10 bounded numerical parameters. Evaluations are deterministic for this fixed task.

Objective: minimize `value`.

Budget: 20 shared initial observations and 100 new evaluations (120 total).

## Submission

Submit one complete finite configuration within the declared parameter types and bounds. Use the exact coordinates in the search space.

## Read details as needed

There are 10 active parameters. get_search_space supports names, query, optional annotated groups and paged index/details views.

get_task_context sections: overview, submission. Read available details as needed before choosing a candidate.

get_trial_history and get_incumbent return scores first; request parameter_names to inspect selected values.

Follow instructions.md: submit_candidate accepts a full config or workspace JSON file directly; write_candidate is optional.
