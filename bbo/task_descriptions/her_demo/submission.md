# Submission Interface

Parameter schema:

- `AcidRed871_0gL`, `L-Cysteine-50gL`, `MethyleneB_250mgL`, `NaCl-3M`, `NaOH-1M`, `P10-MIX1`, `PVP-1wt`, `RhodamineB1_0gL`, `SDS-1wt`, `Sodiumsilicate-1wt`: all `float` in `[0.0, 5.0]`

Expected per-trial logging:

- `config`: the proposed 10D formulation
- `objectives.regret`: surrogate-predicted regret
- `metrics.predicted_target`, `metrics.raw_target_max`, and `metrics.coord::<feature>`
- `metadata`: staged dataset provenance such as `relative_path`, `source_root`, `source_path`, `cache_path`, `source_ref`, `sha256`, and `size_bytes`

Standard run artifacts produced by `bbo.run` are append-only `trials.jsonl`, `summary.json`, `plots/trace.png`, and `plots/distribution.png`.
