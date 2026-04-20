# Domain Prior Knowledge

- This benchmark operates in descriptor space rather than raw reagent identity space, so geometrically nearby points need not correspond to obviously similar chemistry.
- The feature selector usually keeps a mix of reaction-condition variables (for example temperature or concentration) and ligand/solvent descriptors; both kinds of coordinates matter in the packaged task.
- Minimizing regret is equivalent to maximizing predicted yield, but the regret transform makes the task direction consistent with the repository's minimize-first conventions.
- Because feature selection depends on a random-forest seed, fixing the task seed is part of the domain protocol, not just an implementation detail.
