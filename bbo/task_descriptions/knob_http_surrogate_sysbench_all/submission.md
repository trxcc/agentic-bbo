# Submission Interface

Submit all 196 parameters. Their names identify database settings, but every submitted value must be a floating-point number u in [0,1]. Do not submit the physical numbers or enumeration labels from the parameter table directly.

Integer settings decode as round(min + u*(max-min)). An input of 0 selects the minimum and 1 selects the maximum; intermediate inputs are linearly interpolated and rounded.

For K enumeration options, divide [0,1] into K equal-width intervals. Select enum_values[min(floor(u*K),K-1)] using zero-based indexing; u=1 selects the last option. Option order does not imply a performance ranking.

Encoding examples: innodb_thread_concurrency has physical range 0...1000, so input 0.25 decodes to 250. autocommit has options ["ON","OFF"], so 0.25 selects ON and 0.75 selects OFF. These illustrate encoding, not recommended configurations.
