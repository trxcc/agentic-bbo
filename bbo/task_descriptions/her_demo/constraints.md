# Constraints

- The exposed controls `AcidRed871_0gL`, `L-Cysteine-50gL`, `MethyleneB_250mgL`, `NaCl-3M`, `NaOH-1M`, `P10-MIX1`, `PVP-1wt`, `RhodamineB1_0gL`, `SDS-1wt`, and `Sodiumsilicate-1wt` are all floats in `[0.0, 5.0]`.
- Use only the staged tutorial dataset `examples/HER/HER_virtual_data.csv`; do not swap in synthetic fallback data or external labels.
- Preserve the repository's append-only logging and replay-based resume semantics; prior trials must remain immutable once written to JSONL.
- Treat this as a surrogate benchmark, not a claim of wet-lab validity outside the support of the staged dataset.
