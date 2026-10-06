# Evaluation Protocol

The fixed split has 1437 training and 360 held-out samples, with 64 features per sample. Feedback uses five-fold StratifiedKFold cross-validation on the training split, with shuffle=False.

Cross-validation error is 1 - mean fold accuracy. Minimizing this error is equivalent to maximizing mean accuracy. The run objective below specifies which metric the interface returns; held-out results are not optimization feedback.

Held-out results are not optimization feedback. No extra input scaling or preprocessing is added to the released arrays. Fixed estimator settings are listed below.
