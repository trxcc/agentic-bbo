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
| `key` | `"wine"` |
| `display_name` | `"Wine Recognition"` |
| `problem_type` | `"classification"` |
| `total_samples` | `178` |
| `train_samples` | `142` |
| `test_samples` | `36` |
| `feature_count` | `13` |
| `class_counts_train` | `[45,55,42]` |

**Estimator (`estimator`)**

AdaBoostClassifier

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `estimator` | `null` |
| `random_state` | `0` |
