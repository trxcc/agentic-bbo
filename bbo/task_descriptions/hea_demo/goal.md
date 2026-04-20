# Goal

Minimize the primary objective `regret`, where `regret = target.max() - predicted_target`.
Lower regret corresponds to higher predicted alloy performance under the tutorial target.

A valid submission may change only the four exposed design variables `x1`, `x2`, `x3`, and `x4`.
The benchmark internally maps them into a feasible five-component composition, and one evaluation counts as one decode-plus-surrogate-query.

Unless overridden, the packaged task stops after 40 evaluations.
Repository smoke checks use the same task with `--max-evaluations 3`.
