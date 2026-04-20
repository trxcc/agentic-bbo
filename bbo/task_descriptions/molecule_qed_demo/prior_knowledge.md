# Domain Prior Knowledge

- `qed` is deterministic and bounded in `[0, 1]`, so every improvement in raw QED maps linearly to a lower reported loss.
- The benchmark search space is a fixed candidate library, not a molecular editing or generation policy; diversity must come from choosing different archive entries.
- Invalid strings are allowed but strongly dominated because they collapse to the worst possible QED.
- No medicinal-chemistry prior beyond the staged ZINC pool and RDKit's `MolFromSmiles` plus `QED.qed` should be assumed.
