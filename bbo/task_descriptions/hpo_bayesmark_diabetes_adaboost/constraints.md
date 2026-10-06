# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

| Parameter | Meaning | Type | Allowed values | Search transform |
| --- | --- | --- | --- | --- |
| `n_estimators` | Maximum number of boosting rounds. | integer | [10, 100] | linear |
| `learning_rate` | Weight scaling for each boosting learner. | float | [0.0001, 10.0] | log |

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

AdaBoostRegressor

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `estimator` | `null` |
| `loss` | `"linear"` |
| `random_state` | `0` |
