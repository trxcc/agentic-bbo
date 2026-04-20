# Submission Interface

Parameter schema:

- `x1`, `x2`, `x3`, `x4`: all `float` in `[0.0, 1.0]`

Expected per-trial logging:

- `config`: the proposed design-space coordinates `x1..x4`
- `objectives.regret`: surrogate-predicted regret
- `metrics.predicted_target` plus decoded composition metrics `composition::Co`, `composition::Fe`, `composition::Mn`, `composition::V`, and `composition::Cu`
- `metadata`: staged dataset provenance such as `relative_path`, `source_root`, `source_path`, `cache_path`, `source_ref`, `sha256`, and `size_bytes`

Standard run artifacts produced by `bbo.run` are append-only `trials.jsonl`, `summary.json`, `plots/trace.png`, and `plots/distribution.png`.
