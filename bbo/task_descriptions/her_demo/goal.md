# Goal

Minimize the primary objective `regret`, where `regret = Target.max() - predicted_Target`.
Lower regret is therefore equivalent to finding configurations with higher predicted HER performance.

A valid submission may change only the ten exposed continuous controls in the task search space.
One evaluation counts as one surrogate prediction for one proposed configuration.

Unless the caller overrides `max_evaluations`, the packaged task stops after 40 evaluations.
The smoke workflow in this repository uses the same interface with `--max-evaluations 3`.
