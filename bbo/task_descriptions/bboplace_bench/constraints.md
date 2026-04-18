# Constraints

- Bound constraints: Each coordinate must stay within [0, n_grid_x) for x or [0, n_grid_y) for y; anything outside is invalid.
- No invalid placements: Avoid coordinates that cause macros to exceed grid boundaries or overlap with each other/fixed blocks. Such configurations yield poor HPWL but are not forbidden by the search space itself.
- No closed-form optimum assumption: Do not assume a known global minimum. Treat the evaluator as a black box and respect implicit budget limits (e.g., API latency, rate limits, memory).