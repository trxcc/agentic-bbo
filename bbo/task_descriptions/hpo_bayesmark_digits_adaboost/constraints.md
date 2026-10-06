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
| `key` | `"digits"` |
| `display_name` | `"Optical Digits"` |
| `problem_type` | `"classification"` |
| `total_samples` | `1797` |
| `train_samples` | `1437` |
| `test_samples` | `360` |
| `feature_count` | `64` |
| `class_counts_train` | `[151,147,141,154,151,142,137,140,135,139]` |

**Estimator (`estimator`)**

AdaBoostClassifier

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `estimator` | `null` |
| `random_state` | `0` |
