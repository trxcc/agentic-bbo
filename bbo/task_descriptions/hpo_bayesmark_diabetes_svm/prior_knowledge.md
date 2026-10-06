# Domain Prior Knowledge

The estimator is SVR with an RBF kernel. Only C, gamma, and tol are tuned.

Regularization strength decreases as C increases. gamma controls the distance scale of the RBF kernel.

tol controls solver stopping tolerance; reducing it is not itself a guarantee of better cross-validation performance.

This is SVR regression with fixed epsilon=0.1.

## Domain experience and conditions

C and gamma jointly affect model flexibility. Their effects depend on the dataset feature scales, which are fixed in this task.
