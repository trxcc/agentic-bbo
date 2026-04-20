# Evaluation Protocol

- Data source: staged copy of `examples/BH/BH_dataset.csv` with 1728 rows.
- Preprocessing: compute `yield.max() - yield`, drop `cost` and `new_index` from candidate features, and run tutorial-style feature filtering with `extractor=random_forest`, `max_n=20`, `max_cum_imp=0.8`, and `min_imp=0.01`.
- Oracle: fit `RandomForestRegressor(n_estimators=100, random_state=<task seed>)` on the selected descriptor subset and predict regret.
- Report `regret` as the primary objective and log auxiliary metrics `predicted_yield` and `raw_yield_max`.
- Once the task seed is fixed, evaluation is deterministic. Changing the task seed can change both search-space coordinates and objective values, so benchmark comparisons must keep the seed fixed. Exact ties are broken by earlier logged trial.
