# Background

`oer_demo` packages the oxygen-evolution-reaction (OER) catalyst-design case study from the BO tutorial repository.
Its bundled source table is stored in this workspace at `bbo/tasks/scientific/data/examples/OER/OER.csv`. The benchmark still refers to it by the relative path `examples/OER/OER.csv`, and the data mix catalyst identities, composition ratios, hydrothermal and annealing conditions, proton concentration, catalyst loading, and measured overpotential.

The scientific decision problem is to choose both composition and process settings that reduce overpotential at `10 mA cm^-2` without exhaustively screening a combinatorial synthesis space.
This is exactly the kind of mixed categorical-plus-numerical design problem for which Bayesian optimization is attractive.

The real component is the staged OER dataset and the cleaning recipe carried over from the tutorial workflow.
The simulated component is the evaluator: it cleans the table, aligns one-hot encodings, and uses a fitted random-forest oracle instead of running an electrochemistry experiment.
