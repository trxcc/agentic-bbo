# Evaluation Protocol

The 150 samples are split into 120 training samples and 30 held-out samples, with four features per sample. Optimization feedback uses five-fold stratified cross-validation on the training split, without shuffling (shuffle=False).

Cross-validation error is 1 - mean fold accuracy. Minimizing this error is equivalent to maximizing mean accuracy. The run objective below specifies which metric the interface returns; held-out results are not optimization feedback.

The random forest uses 10 trees and random_state=0. Other fixed settings appear under fixed_estimator_parameters. No additional data preprocessing is applied.
