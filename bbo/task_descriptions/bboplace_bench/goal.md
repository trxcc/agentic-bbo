# Goal

- **Primary metric**: minimize **HPWL** returned by the evaluator (lower is better).
- **Search vector**: a continuous vector of length `2 * n_macro`. The first `n_macro` entries are macro **x** coordinates; the second `n_macro` entries are macro **y** coordinates.
- **Grid bounds**: conceptually `x ∈ [0, n_grid_x)` and `y ∈ [0, n_grid_y)` on the placement grid (defaults: `n_grid_x = n_grid_y = 224`). The packaged `SearchSpace` uses inclusive floating-point upper bounds at `n_grid_x` / `n_grid_y` to match the reference HTTP API (`xu`/`yu` reported by the server).
- **Optimum**: the true minimum HPWL and minimizing configuration are **unknown** to the agent; do not assume a closed-form optimum.
