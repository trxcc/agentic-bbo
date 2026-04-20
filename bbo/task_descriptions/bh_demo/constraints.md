# Constraints

- The optimizer-facing controls are the continuous descriptor columns selected at task construction time; each selected coordinate is bounded by its staged-data minimum and maximum.
- `yield` is not optimized directly. The benchmark first transforms it into regret, and the raw columns `cost` and `new_index` are never valid optimization variables.
- Feature selection is fixed by the tutorial-style recipe `extractor=random_forest`, `max_n=20`, `max_cum_imp=0.8`, and `min_imp=0.01`.
- Keep the task seed fixed for fair comparisons because changing the seed can change both the selected feature subset and the fitted surrogate. Preserve append-only logs and replay-based resume semantics.
