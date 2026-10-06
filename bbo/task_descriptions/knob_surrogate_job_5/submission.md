# Submission Interface

Submit all 5 settings as floating-point numbers u in [0,1]. Names identify real database settings, but physical values and enumeration labels from the table must not be submitted directly.

For integer settings, decode as round(min + u*(max-min)). For physical floating-point settings, decode as min + u*(max-min) without rounding.

For K enumeration options, divide [0,1] into K equal-width intervals. Select enum_values[min(floor(u*K),K-1)] using zero-based indexing; u=1 selects the last option. Option order does not imply a performance ranking.

Encoding example: innodb_adaptive_hash_index_parts has physical range 1...512; input 0.25 decodes to 129. This illustrates encoding, not a recommended configuration.

Enumeration example: for innodb_stats_method, input 0 selects "nulls_equal", and 1 selects "nulls_ignored". The order is part of the encoding, not a performance ranking.
