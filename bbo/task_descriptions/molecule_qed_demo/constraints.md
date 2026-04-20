# Constraints

- The exposed parameter `SMILES` is categorical and may take values only from the staged archive member `zinc.txt`.
- RDKit is mandatory for evaluation; replacing QED with a mock score or a learned surrogate is not a valid implementation of this task.
- Invalid SMILES are not rejected by the interface, but they receive the worst score through `qed = 0.0` and `qed_loss = 1.0`.
- Preserve append-only logs and replay-based resume semantics, and treat this task as fixed-pool selection rather than open-ended molecular generation.
