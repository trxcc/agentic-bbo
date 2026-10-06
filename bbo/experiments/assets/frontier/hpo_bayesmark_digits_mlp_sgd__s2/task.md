# Optical Digits / MLPClassifier

Task ID: `hpo_bayesmark_digits_mlp_sgd`.

Tune 8 hyperparameters of MLPClassifier on Optical Digits to reduce cross-validation classification error.

Objective: maximize `accuracy`.

Budget: 5 shared initial observations and 25 new evaluations (30 total).

## Submission

1. Submit every parameter listed below, using its declared integer or floating-point type and inclusive bounds.
2. Submit original hyperparameter values. The transform column describes search coordinates, not a transformation to apply before submission: for example, submit 0.1 itself rather than log(0.1) or logit(0.1).

## Parameters

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

## Read details as needed

There are 8 active parameters. get_search_space supports names, query, optional annotated groups and paged index/details views.

get_task_context sections: overview, submission, scoring, mechanisms, domain_knowledge, additional. Read scoring rules and relevant mechanisms before choosing a candidate.

get_trial_history and get_incumbent return scores first; request parameter_names to inspect selected values.

Follow instructions.md: submit_candidate accepts a full config or workspace JSON file directly; write_candidate is optional.
