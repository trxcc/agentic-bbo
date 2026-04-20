# Submission Interface

Parameter schema:

- `SMILES`: `categorical`, chosen from the 249455 staged strings loaded from archive member `zinc.txt`

Expected per-trial logging:

- `config.SMILES`: the proposed molecule string
- `objectives.qed_loss`: the reported primary objective
- `metrics.qed`: raw RDKit QED score
- `metadata.smiles`, `metadata.valid_smiles`, and staged dataset provenance such as `relative_path`, `source_root`, `source_path`, `cache_path`, `source_ref`, `sha256`, and `size_bytes`

Standard run artifacts produced by `bbo.run` are append-only `trials.jsonl`, `summary.json`, `plots/trace.png`, and `plots/distribution.png`.
