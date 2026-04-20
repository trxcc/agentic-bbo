# Goal

Minimize the primary objective `overpotential_mv`.
Lower predicted overpotential indicates a better catalyst-and-process proposal in this packaged benchmark.

A valid submission may change only the mixed search-space variables exposed by the task: three categorical metal identities, three composition ratios, four thermal-process integers, proton concentration, and catalyst loading.
One evaluation counts as one cleaned, encoded candidate passed through the fitted oracle.

Unless overridden, the packaged task stops after 40 evaluations.
Repository smoke checks use `random_search` with `--max-evaluations 3` because the interface is mixed rather than fully numeric.
