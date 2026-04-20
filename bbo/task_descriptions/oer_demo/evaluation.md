# Evaluation Protocol

- Data source: staged copy of `examples/OER/OER.csv`; the benchmark cleans it down to a deterministic 1319-row table and keeps `examples/OER/OER_clean.csv` as a reference artifact.
- Cleaning: drop rows without target, remove duplicates, normalize categorical text, coerce numeric columns, clip selected numerical outliers, and keep the central 5%-95% target band.
- Encoding: one-hot encode `Metal_1`, `Metal_2`, and `Metal_3` with `pandas.get_dummies`, then align proposal columns to the training matrix.
- Oracle: `RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)` predicting `overpotential_mv`.
- Evaluation is deterministic and effectively seed-independent on the task side because the oracle uses a fixed random state. Report `overpotential_mv` as the primary objective, log `predicted_overpotential_mv` and `choice::<metal_slot>` metrics, and break exact ties by earlier logged trial.
