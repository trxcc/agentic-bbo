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
| `key` | `"iris"` |
| `display_name` | `"Iris"` |
| `problem_type` | `"classification"` |
| `total_samples` | `150` |
| `train_samples` | `120` |
| `test_samples` | `30` |
| `feature_count` | `4` |
| `class_counts_train` | `[39,37,44]` |

**Estimator (`estimator`)**

DecisionTreeClassifier

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `ccp_alpha` | `0.0` |
| `class_weight` | `null` |
| `criterion` | `"gini"` |
| `max_leaf_nodes` | `null` |
| `monotonic_cst` | `null` |
| `random_state` | `0` |
| `splitter` | `"best"` |
