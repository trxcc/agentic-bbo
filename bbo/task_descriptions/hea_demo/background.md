# Background

`hea_demo` packages the high-entropy-alloy (HEA) composition case study from the BO tutorial repository.
Its bundled source workbook is stored in this workspace at `bbo/tasks/scientific/data/examples/HEA/data/oracle_data.xlsx`. The benchmark still refers to it by the relative path `examples/HEA/data/oracle_data.xlsx`, and each row records a feasible five-component alloy composition over `Co`, `Fe`, `Mn`, `V`, and `Cu` together with a tutorial target score.

The scientific decision problem is how to allocate alloy fractions under simplex and per-component bounds when each new materials experiment is expensive.
Bayesian optimization is helpful here because it proposes the next composition without exhaustively enumerating the feasible composition grid.

The real component is the staged alloy dataset and its constrained composition geometry.
The simulated component is the benchmark evaluator: it decodes optimizer-facing design variables into alloy fractions, then queries a fitted random-forest surrogate instead of a real synthesis-and-characterization workflow.
