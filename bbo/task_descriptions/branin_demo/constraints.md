# Constraints

- Only propose values inside the declared bounds.
- The benchmark is purely black-box: use only the exposed search-space API and observed trial history.
- Treat the objective as serial and append-only; do not mutate prior records or depend on hidden checkpoints.
