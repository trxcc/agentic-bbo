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
| `key` | `"breast"` |
| `display_name` | `"Breast Cancer Wisconsin"` |
| `problem_type` | `"classification"` |
| `total_samples` | `569` |
| `train_samples` | `455` |
| `test_samples` | `114` |
| `feature_count` | `30` |
| `class_counts_train` | `[165,290]` |

**Estimator (`estimator`)**

AdaBoostClassifier

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `estimator` | `null` |
| `random_state` | `0` |
