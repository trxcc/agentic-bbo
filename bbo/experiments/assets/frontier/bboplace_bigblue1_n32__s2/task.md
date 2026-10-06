# Macro placement: bboplace_bigblue1_n32

Minimize HPWL for a fixed 32-macro placement task. The 64 coordinates x_0 through x_31 and y_0 through y_31 refer to the same ordered macro list. Coordinates lie in [0,224] on a 224 by 224 grid.

Budget: 50 shared initial observations and 200 new evaluations (250 total).

## Submission

Submit one complete configuration using the declared coordinates. Only host submissions produce objective observations.

## Read details as needed

There are 64 active parameters. get_search_space supports names, query, optional annotated groups and paged index/details views.

get_task_context sections: overview, scoring, domain_knowledge, submission. Read scoring rules and relevant mechanisms before choosing a candidate.

get_trial_history and get_incumbent return scores first; request parameter_names to inspect selected values.

Follow instructions.md: submit_candidate accepts a full config or workspace JSON file directly; write_candidate is optional.
