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
| `key` | `"iris"` |
| `display_name` | `"Iris"` |
| `problem_type` | `"classification"` |
| `total_samples` | `150` |
| `train_samples` | `120` |
| `test_samples` | `30` |
| `feature_count` | `4` |
| `class_counts_train` | `[39,37,44]` |

**Estimator (`estimator`)**

AdaBoostClassifier

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `estimator` | `null` |
| `random_state` | `0` |
