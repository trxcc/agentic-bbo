# Constraints

- `Metal_1`, `Metal_2`, and `Metal_3` are categorical parameters whose valid labels come from the cleaned staged dataset.
- Numeric bounds are hard search-space limits: `Metal_1_Proportion in [0, 100]`, `Metal_2_Proportion in [0, 100]`, `Metal_3_Proportion in [0, 33.33333333]`, `Hydrothermal Temp degree in [-77, 320]`, `Hydrothermal Time min in [0, 2790]`, `Annealing Temp degree in [25, 1400]`, `Annealing Time min in [0, 943]`, `Proton Concentration M in [0.1, 3.7]`, and `Catalyst_Loading mg cm -2 in [0.0, 1.266]`.
- Use the tutorial cleaning and dummy-column alignment logic; do not silently introduce alternative encodings, extra categorical levels, or hidden normalization rules.
- This benchmark does not enforce composition-sum or distinct-metal constraints beyond the exposed interface; treat that as a property of the packaged task, not as a chemistry claim.
- Preserve append-only logs and replay-based resume semantics for every run.
