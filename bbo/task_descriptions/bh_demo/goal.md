# Goal

Minimize the primary objective `regret`, where `regret = yield.max() - predicted_yield`.
Lower regret therefore means higher predicted Buchwald-Hartwig yield.

A valid submission may change only the active continuous descriptor coordinates exposed by the task instance after feature selection.
One evaluation counts as one surrogate prediction on one proposed descriptor vector.

Unless overridden, the packaged task stops after 40 evaluations.
Repository smoke checks use the same interface with `--max-evaluations 3`.
