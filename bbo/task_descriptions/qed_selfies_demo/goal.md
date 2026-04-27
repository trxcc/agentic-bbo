# Goal

Minimize the primary objective `qed_loss`, where `qed_loss = 1.0 - qed`.
Lower loss is equivalent to proposing molecules with higher RDKit QED.

A valid benchmark configuration assigns every `selfies_token_XX` categorical parameter. The task concatenates tokens until `__EOS__`, ignores `__PAD__`, decodes the resulting SELFIES string to SMILES, and scores that molecule.

Unless overridden, the packaged task stops after 40 evaluations.

