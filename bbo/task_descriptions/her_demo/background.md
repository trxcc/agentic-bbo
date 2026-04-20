# Background

`her_demo` packages the hydrogen-evolution-reaction (HER) case study from *Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial*.
The bundled source dataset is stored in this workspace at `bbo/tasks/scientific/data/examples/HER/HER_virtual_data.csv`; the benchmark still refers to it by the relative path `examples/HER/HER_virtual_data.csv` when staging it at runtime. It describes ten nonnegative formulation controls for a photocatalytic system plus a measured `Target` score.

The scientific decision problem is which catalyst or additive recipe to try next when a full wet-lab sweep over all ten controls would be expensive.
In this benchmark, that decision is approximated with a dataset-backed oracle so optimization methods can be compared quickly and reproducibly.

The real component is the staged tutorial dataset and its domain-specific variable semantics.
The simulated component is the benchmark evaluator: it refits a random-forest surrogate on the staged table and returns surrogate predictions instead of running a physical HER experiment.
