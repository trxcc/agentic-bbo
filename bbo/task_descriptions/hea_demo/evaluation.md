# Evaluation Protocol

- Data source: staged copy of `examples/HEA/data/oracle_data.xlsx` with 286 rows.
- Training view: fit `RandomForestRegressor(n_estimators=100, random_state=<task seed>)` on raw alloy fractions `Co`, `Fe`, `Mn`, `V`, `Cu` to predict `target`.
- Query path: decode proposed `x1..x4` with the tutorial `_phi_inv` logic, predict `target`, and return `regret = target.max() - predicted_target`.
- Log `regret` as the primary objective and record decoded composition metrics `composition::Co`, `composition::Fe`, `composition::Mn`, `composition::V`, and `composition::Cu`.
- Once the task seed is fixed, evaluation is deterministic. Changing the task seed refits the random forest and may change regret values, so comparisons should keep the seed fixed. Ties are resolved by earlier logged trial.
