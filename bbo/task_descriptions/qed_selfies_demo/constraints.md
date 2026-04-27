# Constraints

- The exposed search space is a fixed-length sequence of categorical SELFIES token slots.
- The special tokens `__EOS__` and `__PAD__` control sequence length and are not passed to the SELFIES decoder.
- RDKit and the `selfies` package are mandatory for evaluation.
- Empty, undecodable, or RDKit-invalid sequences receive the worst score through `qed = 0.0` and `qed_loss = 1.0`.
- The task is generative over the token space; it is not restricted to selecting a SMILES string from the source archive.

