# Optimization interface

Submit one non-empty RDKit-parseable SMILES string in the `smiles` field. Each valid molecule receives one bounded scalar objective; invalid molecules receive the least favorable value. The scoring implementation and unevaluated molecule scores are unavailable.
