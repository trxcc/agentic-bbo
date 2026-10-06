# Constraints

- Submit exactly one value for each of x1, x2, ..., x10.
- Each value must be finite and lie in [-5, 5].
- Evaluations have fixed cost; per-trial fidelity or budget values are unsupported.
- The default run has 120 evaluations: a shared 20-point scrambled Sobol design followed by 100 optimizer-selected points.
- Comparable GP-EI and BoTorch TuRBO runs use 2048 internal candidates per model-based selection step.
- Optimization is sequential with batch size q=1.
