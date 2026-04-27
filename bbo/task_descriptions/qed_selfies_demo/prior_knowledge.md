# Domain Prior Knowledge

- SELFIES is a robust molecular string representation, so token sequences are a better categorical interface for generative molecular search than raw SMILES characters.
- QED is deterministic and bounded in `[0, 1]`, so every increase in raw QED maps linearly to lower loss.
- Short valid sequences can already represent simple molecules; longer sequences expand the search space quickly.
- The vocabulary is seeded from local ZINC examples, but successful candidates may decode to molecules not present in the original archive.

