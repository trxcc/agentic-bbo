# Background

`bh_demo` packages the Buchwald-Hartwig reaction case study from the BO tutorial repository.
Its bundled source table is stored in this workspace at `bbo/tasks/scientific/data/examples/BH/BH_dataset.csv`. The benchmark still refers to it by the relative path `examples/BH/BH_dataset.csv`, and it contains reaction-yield labels together with a large set of continuous ligand, solvent, and process descriptors.

The scientific decision problem is to search reaction settings that improve yield without exhaustively testing every condition combination.
In the packaged benchmark, that decision is framed in descriptor space so optimizers can work on a compact continuous interface.

The real component is the staged reaction dataset and its descriptor columns.
The simulated component is the evaluator: it converts `yield` into regret, runs random-forest feature selection, and queries a fitted random-forest surrogate instead of executing a real chemistry experiment.
