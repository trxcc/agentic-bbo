# Evaluation Protocol

- Data source: staged copy of `examples/Molecule/zinc.txt.gz`; the evaluator reads the archive member `zinc.txt` and exposes its 249455 SMILES strings as a categorical pool.
- For each evaluation, parse the proposed string with `rdkit.Chem.MolFromSmiles`, compute `rdkit.Chem.QED.qed`, and report `qed_loss = 1.0 - qed`.
- Invalid molecules receive `qed = 0.0`, so their primary objective is the worst possible `qed_loss = 1.0`.
- The evaluator is deterministic and seed-independent on the task side. Report `qed_loss` as the primary objective, log raw `qed`, and record the chosen `smiles` plus `valid_smiles` in metadata.
- Exact ties are broken by earlier logged trial because the benchmark replays runs in append-only order.
