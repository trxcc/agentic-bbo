# Evaluation Protocol

The fixed split has 353 training and 89 held-out samples, with 10 features per sample. Feedback uses five-fold KFold cross-validation on the training split, with shuffle=False.

The objective is mean cross-validation squared prediction error on the released standardized regression target; lower is better.

Held-out results are not optimization feedback. No extra input scaling or preprocessing is added to the released arrays. Fixed estimator settings are listed below.
