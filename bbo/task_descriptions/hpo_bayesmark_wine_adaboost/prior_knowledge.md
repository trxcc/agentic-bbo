# Domain Prior Knowledge

n_estimators is an upper bound on boosting rounds; fitting can stop early after a perfect fit.

learning_rate scales learner contributions and interacts with the number of rounds.

With estimator=None, this classifier uses a default decision tree of maximum depth 1 as its weak learner.

random_state is fixed at 0. Other fixed settings are listed below.

## Domain experience and conditions

Learning rate and round count jointly determine how the ensemble fits the data. A larger value for either is not automatically an improvement.
