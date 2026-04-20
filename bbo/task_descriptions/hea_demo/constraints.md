# Constraints

- The exposed design variables `x1`, `x2`, `x3`, and `x4` are all floats in `[0.0, 1.0]`.
- Internal decoding must always produce `Co`, `Fe`, `Mn`, `V`, and `Cu` fractions inside `[0.05, 0.35]`, with the five fractions summing to approximately `1.0`.
- Use only the staged tutorial workbook `examples/HEA/data/oracle_data.xlsx`; do not replace the decoder, feasible-set geometry, or surrogate target with ad hoc alternatives.
- Preserve append-only logs and replay-based resume semantics so that benchmark runs remain auditable and reproducible.
