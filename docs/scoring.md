# Scoring

Convert every objective to loss orientation first (lower is better). For maximizing
objectives use its negative. An additive shift, e.g. `1-accuracy`, is equivalent
only when the initialization, GP and theoretical references receive the same shift.

For each of the B new evaluations, take the best observation so far, including the
shared initialization. Normalize that incumbent before averaging. The final score is

`S = 0.7/B * sum(q_j, j=1..B) + 0.3*q_B`.

In quality orientation (higher is better), let b be the best initial value and r
the matched GP final value.

- HPO, DBTune and BBOPlace: for `b <= y <= r`, `q(y)=0.6*(y-b)/(r-b)`.
  Above r, `q(y)=1-0.4*exp(-1.5*(y-r)/(r-b))`. The exponential tail approaches 1;
  an empirical search maximum is not a 1 anchor.
- If GP does not improve on b, use `q(y)=1-exp(-(y-b)/scale)`. Scale is the shared
  initialization's IQR, or its range if the IQR vanishes. Flat initializations fail
  explicitly; no method-dependent denominator is invented.
- BBOB: use the theoretical optimum u and linear segments through `(b,0)`,
  `(r,0.6)`, `(u,1)`. If r coincides with an endpoint, use the documented linear
  fallback instead of imposing contradictory anchors.
- GuacaMol: `q(y)=min(1,(y-b)/(u-b))`, with no GP anchor. The paper uses a raw
  score upper reference of 0.58 for Median1 (equivalently, loss 0.42 when loss is
  `1 - score`); most other molecular objectives use a raw-score reference of 1.

The implementation in `bbo/experiments/scoring.py` follows the
RSI-exponential scoring protocol. It rejects incomplete trajectories unless
carry-forward is explicitly enabled. An early-ended run then retains the last
incumbent for every unused checkpoint; those checkpoints still count in B.

Use a task/seed/budget-matched GP reference. Do not recalibrate per compared method,
reuse a different task's GP, or normalize a mean raw objective. Average scored seeds
within each task, then tasks within a family; family means receive equal weights
when computing a five-family mean. For paper ranks, rank methods using their
seed-averaged task scores before averaging ranks within the reported cohort; do
not average per-seed ranks. Missing runs and incomplete runs are distinct.
