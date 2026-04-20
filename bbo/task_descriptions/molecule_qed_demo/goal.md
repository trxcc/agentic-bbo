# Goal

Minimize the primary objective `qed_loss`, where `qed_loss = 1.0 - qed`.
Lower loss is therefore equivalent to selecting molecules with higher RDKit QED.

A valid submission may change only the single categorical parameter `SMILES`, and its value must come from the staged archive member `zinc.txt`.
One evaluation counts as one RDKit parse-and-QED computation for one proposed string.

Unless overridden, the packaged task stops after 40 evaluations.
Repository smoke checks use the same interface with `--max-evaluations 3`.
