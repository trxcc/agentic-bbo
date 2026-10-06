# Domain Prior Knowledge

Floating-point min_samples_split and min_samples_leaf become sample-count thresholds through ceil(fraction * number of samples fitted in the fold).

Without sample_weight, samples start with equal weights. min_weight_fraction_leaf constrains the fraction of total sample weight in each leaf.

Floating-point max_features becomes max(1, int(max_features * n_features_in_)) features per split. Different inputs can select the same number of features. Finding a valid split may require inspecting more features than this number.

Depth, leaf size, minimum split size, and minimum impurity decrease jointly constrain tree growth.

## Domain experience and conditions

Deeper trees can fit finer patterns but can also overfit. Larger leaf and split thresholds restrict growth; evaluate their interactions rather than assuming maximum depth always helps.
