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
| `key` | `"iris"` |
| `display_name` | `"Iris"` |
| `problem_type` | `"classification"` |
| `total_samples` | `150` |
| `train_samples` | `120` |
| `test_samples` | `30` |
| `feature_count` | `4` |
| `class_counts_train` | `[39,37,44]` |

**Estimator (`estimator`)**

SVC

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `break_ties` | `false` |
| `cache_size` | `200` |
| `class_weight` | `null` |
| `coef0` | `0.0` |
| `decision_function_shape` | `"ovr"` |
| `degree` | `3` |
| `kernel` | `"rbf"` |
| `max_iter` | `-1` |
| `probability` | `true` |
| `random_state` | `0` |
| `shrinking` | `true` |
| `verbose` | `false` |
