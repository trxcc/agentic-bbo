"""Pinned TuRBO-1 core from the BoTorch v0.18.1 official tutorial.

Source notebook:
https://github.com/meta-pytorch/botorch/blob/v0.18.1/tutorials/turbo_1/turbo_1.ipynb

Only the tutorial's state and candidate-generation cells are extracted.  The
algorithm logic below is intentionally kept aligned with that source; the
benchmark-specific ask/tell integration lives in ``botorch_turbo.py`` one
directory above this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine


SOURCE_TAG = "v0.18.1"
SOURCE_COMMIT = "cd0249c60b2a81af0d91b5e7d462ef6f574fceec"
SOURCE_NOTEBOOK_BLOB_SHA = "5d1239c15353e6764bc7ca4955b61c2e0600f472"
SOURCE_URL = "https://github.com/meta-pytorch/botorch/blob/v0.18.1/tutorials/turbo_1/turbo_1.ipynb"


@dataclass
class TurboState:
    """Turbo state used to track the recent history of the trust region."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # type: ignore[assignment]
    success_counter: int = 0
    success_tolerance: int = 10  # The original paper uses 3; this is the official tutorial value.
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """Update trust-region counters and length from new objective values."""

    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    n_candidates: Optional[int] = None,
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: str = "ts",
) -> torch.Tensor:
    """Generate one TuRBO batch in the normalized unit cube."""

    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=X.dtype, device=X.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=X.dtype, device=X.device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        # This mirrors the official tutorial cell (including its upper bound).
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=X.device)] = 1

        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)
    else:
        ei = qExpectedImprovement(model, Y.max())
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


__all__ = [
    "SOURCE_COMMIT",
    "SOURCE_NOTEBOOK_BLOB_SHA",
    "SOURCE_TAG",
    "SOURCE_URL",
    "TurboState",
    "generate_batch",
    "update_state",
]
