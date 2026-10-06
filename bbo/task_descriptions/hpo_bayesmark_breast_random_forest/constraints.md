# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

| Parameter | Meaning | Type | Allowed values | Search transform |
| --- | --- | --- | --- | --- |
| `max_depth` | Maximum depth of each tree. | integer | [1, 15] | linear |
| `min_samples_split` | Minimum fraction of fitting samples required to split an internal node. | float | [0.01, 0.99] | logit |
| `min_samples_leaf` | Minimum fraction of fitting samples required in each leaf after a split. | float | [0.01, 0.49] | logit |
| `min_weight_fraction_leaf` | Minimum fraction of total fitting-sample weight required in a leaf. | float | [0.01, 0.49] | logit |
| `max_features` | Fraction of input features considered at each split. | float | [0.01, 0.99] | logit |
| `min_impurity_decrease` | Minimum weighted impurity decrease required for a split. | float | [0.0, 0.5] | linear |

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

RandomForestClassifier

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `bootstrap` | `true` |
| `ccp_alpha` | `0.0` |
| `class_weight` | `null` |
| `criterion` | `"gini"` |
| `max_leaf_nodes` | `null` |
| `max_samples` | `null` |
| `monotonic_cst` | `null` |
| `n_estimators` | `10` |
| `n_jobs` | `null` |
| `oob_score` | `false` |
| `random_state` | `0` |
| `verbose` | `0` |
| `warm_start` | `false` |
