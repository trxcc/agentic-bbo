Develop a reusable Python black-box optimizer in strategy.py. An editable GP-EI
implementation is provided as a starting point; you may change its search logic.
The program interface and available libraries are described in solver_contract.md.

Your goal is to maximize development_score: 70% normalized anytime improvement
and 30% normalized final improvement, averaged over the development tasks and
seeds. Aim for an optimizer that also works on unseen tasks. Read the exact
score definition with get_development_context(section="scoring", task_alias=...).

Use these tools:
- get_development_context: inspect a development task and its search space.
- get_development_feedback: inspect a submitted version's score, observations
  and errors; version 0 is the starting GP-EI program.
- check_program(path="strategy.py"): run syntax and synthetic interface checks.
- submit_program(path="strategy.py"): submit the current code for host evaluation.
  This uses one version slot and ends the round; stop tool use after acceptance.
  Results and remaining slots arrive in the next message.

Use native Bash/Python to edit strategy.py and analyze the supplied feedback.
Only the host evaluates real objectives. The host automatically selects and
freezes the best eligible submitted version when development finishes.
