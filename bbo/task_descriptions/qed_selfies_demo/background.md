# Background

`qed_selfies_demo` is a generative molecule benchmark that uses the BO tutorial ZINC archive only to seed a SELFIES token vocabulary and a valid default molecule.

Unlike `molecule_qed_demo`, this task does not expose one categorical choice over archived SMILES strings. It exposes a fixed-length sequence of SELFIES token parameters, allowing optimizers such as Optuna TPE to search over token combinations that may decode to molecules outside the original pool.

The evaluator decodes the proposed SELFIES sequence to SMILES and computes RDKit QED directly.

