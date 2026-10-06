# Constraints

Change only the declared parameters, using their exact names, types, and bounds.

| Parameter | Meaning | Type | Allowed values | Search transform |
| --- | --- | --- | --- | --- |
| `hidden_layer_sizes` | Number of neurons in the single hidden layer. | integer | [50, 200] | linear |
| `alpha` | L2 regularization strength. | float | [1e-05, 10.0] | log |
| `batch_size` | Mini-batch size for stochastic gradient descent. | integer | [10, 250] | linear |
| `learning_rate_init` | Initial SGD learning rate before inverse scaling. | float | [1e-05, 0.1] | log |
| `power_t` | Exponent controlling the inverse-scaling learning-rate schedule. | float | [0.1, 0.9] | logit |
| `tol` | Minimum improvement in the internal validation score for early stopping. | float | [1e-05, 0.1] | log |
| `momentum` | Momentum coefficient in SGD updates. | float | [0.001, 0.999] | logit |
| `validation_fraction` | Fraction held out from each fitting fold for internal early stopping. | float | [0.1, 0.9] | logit |

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

MLPRegressor

**Fixed estimator settings (`fixed_estimator_parameters`)**

| Field | Value |
| --- | --- |
| `activation` | `"tanh"` |
| `beta_1` | `0.9` |
| `beta_2` | `0.999` |
| `early_stopping` | `true` |
| `epsilon` | `1e-08` |
| `learning_rate` | `"invscaling"` |
| `loss` | `"squared_error"` |
| `max_fun` | `15000` |
| `max_iter` | `40` |
| `n_iter_no_change` | `10` |
| `nesterovs_momentum` | `true` |
| `random_state` | `0` |
| `shuffle` | `true` |
| `solver` | `"sgd"` |
| `verbose` | `false` |
| `warm_start` | `false` |
