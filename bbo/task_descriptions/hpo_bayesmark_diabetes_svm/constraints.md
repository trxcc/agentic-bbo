# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

| Parameter | Meaning | Type | Allowed values | Search transform |
| --- | --- | --- | --- | --- |
| `C` | Inverse regularization strength of the support vector model. | float | [1.0, 1000.0] | log |
| `gamma` | RBF kernel coefficient controlling its distance scale. | float | [0.0001, 0.001] | log |
| `tol` | Solver stopping tolerance. | float | [1e-05, 0.1] | log |

## Fixed task definitions

**Dataset (`dataset`)**

| Field | Value |
| --- | --- |
| `key` | `"diabetes"` |
| `display_name` | `"Diabetes Progression"` |
| `problem_type` | `"regression"` |
| `total_samples` | `442` |
| `train_samples` | `353` |
| `test_samples` | `89` |
| `feature_count` | `10` |
| `class_counts_train` | `[]` |

**Estimator (`estimator`)**

SVR

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `cache_size` | `200` |
| `coef0` | `0.0` |
| `degree` | `3` |
| `epsilon` | `0.1` |
| `kernel` | `"rbf"` |
| `max_iter` | `-1` |
| `shrinking` | `true` |
| `verbose` | `false` |
