# Background

`molecule_qed_demo` packages the molecule-selection case study from the BO tutorial repository.
Its bundled source archive is stored in this workspace at `bbo/tasks/scientific/data/examples/Molecule/zinc.txt.gz`. The benchmark still refers to it by the relative path `examples/Molecule/zinc.txt.gz`, and the search space is a fixed pool of candidate SMILES strings drawn from the `zinc.txt` member inside that archive.

The scientific decision problem is which candidate molecule to score next when screening a large library for drug-likeness proxies.
Unlike the tabular scientific tasks, this benchmark keeps a real deterministic scoring function instead of replacing the objective with a learned surrogate.

The real component is RDKit's QED computation on each proposed SMILES string.
The benchmark simplification is the search interface itself: candidates must come from a fixed staged archive, and there is no generative chemistry model, synthesis loop, or wet-lab assay in the evaluator.
