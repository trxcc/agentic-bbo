# Submission Interface

Parameter schema:

- `Metal_1`, `Metal_2`, `Metal_3`: `categorical`, with exact choice sets taken from the cleaned staged dataset
- `Metal_1_Proportion`, `Metal_2_Proportion`: `float` in `[0.0, 100.0]`
- `Metal_3_Proportion`: `float` in `[0.0, 33.33333333]`
- `Hydrothermal Temp degree`: `int` in `[-77, 320]`
- `Hydrothermal Time min`: `int` in `[0, 2790]`
- `Annealing Temp degree`: `int` in `[25, 1400]`
- `Annealing Time min`: `int` in `[0, 943]`
- `Proton Concentration M`: `float` in `[0.1, 3.7]`
- `Catalyst_Loading mg cm -2`: `float` in `[0.0, 1.266]`

Expected per-trial logging:

- `config`: the mixed candidate proposal
- `objectives.overpotential_mv`: predicted overpotential
- `metrics.predicted_overpotential_mv` and `metrics.choice::Metal_1|Metal_2|Metal_3`
- `metadata`: staged dataset provenance such as `relative_path`, `source_root`, `source_path`, `cache_path`, `source_ref`, `sha256`, and `size_bytes`

Standard run artifacts produced by `bbo.run` are append-only `trials.jsonl`, `summary.json`, `plots/trace.png`, and `plots/distribution.png`.
