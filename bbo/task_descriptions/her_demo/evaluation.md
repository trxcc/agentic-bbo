# Evaluation Protocol

- Data source: staged copy of `examples/HER/HER_virtual_data.csv` with 812 rows and 10 exposed feature columns plus `Target`.
- Preprocessing: keep the ten HER controls, compute `Target.max() - Target`, and fit `RandomForestRegressor(n_estimators=100, random_state=<task seed>)` on that regret target.
- Report `regret` as the primary objective; also log `predicted_target`, `raw_target_max`, and per-coordinate diagnostics `coord::<feature>`.
- Once the task seed is fixed, evaluation is deterministic. Changing the task seed refits the surrogate and may change scores, so benchmark comparisons should hold the seed fixed.
- When two trials obtain the same primary objective, prefer the earlier logged trial because runs are replayed in append-only order.
