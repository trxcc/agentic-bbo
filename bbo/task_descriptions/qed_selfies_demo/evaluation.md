# Evaluation Protocol

- Data source: the local BO tutorial ZINC archive is used only to seed a SELFIES token vocabulary and a valid default molecule.
- Each evaluation normalizes the submitted token config, removes `__PAD__`, stops at `__EOS__`, decodes SELFIES to SMILES, and computes RDKit QED.
- The task reports:
  - primary objective: `qed_loss = 1.0 - qed`
  - metrics: raw `qed` and the number of emitted SELFIES tokens
  - metadata: decoded `selfies`, decoded `smiles`, token list, validity flags, and decode errors
- Invalid or empty sequences receive `qed = 0.0` and `qed_loss = 1.0`.
- The evaluator is deterministic for a fixed token configuration.

