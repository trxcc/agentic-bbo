# Submission Interface

Parameter schema:

- The active parameters are the selected descriptor columns exposed by `task.spec.search_space` for the current task seed.
- Every active parameter is a `float` bounded by the selected column's staged-data minimum and maximum.
- The exact active set is recorded in `task.sanity_check().metadata["selected_features"]` and the corresponding bounds are recorded in `task.sanity_check().metadata["selected_feature_bounds"]`.

Expected per-trial logging:

- `config`: the proposed descriptor vector over the active selected features
- `objectives.regret`: surrogate-predicted regret
- `metrics.predicted_yield` and `metrics.raw_yield_max`
- `metadata`: staged dataset provenance such as `relative_path`, `source_root`, `source_path`, `cache_path`, `source_ref`, `sha256`, and `size_bytes`

Standard run artifacts produced by `bbo.run` are append-only `trials.jsonl`, `summary.json`, `plots/trace.png`, and `plots/distribution.png`.
