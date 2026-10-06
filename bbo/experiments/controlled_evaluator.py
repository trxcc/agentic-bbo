"""Host-only sparse objectives for the controlled-prior study."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import numpy as np

@dataclass(frozen=True)
class FunctionDefinition:
    reviewer_id: str
    agent_task_id: str
    family: str
    active_variables: tuple[str, ...]
    decoy_variables: tuple[str, ...]
    center: tuple[float, ...]
    truth: dict[str, Any]

    def evaluate(self, config: dict[str, Any]) -> float:
        active = np.asarray(
            [float(config[name]) for name in self.active_variables], dtype=float
        )
        center = np.asarray(self.center, dtype=float)
        shifted = active - center
        if self.family == "custom_separable_single_basin":
            quadratic = np.asarray(
                self.truth["quadratic_coefficients"], dtype=float
            )
            quartic = np.asarray(self.truth["quartic_coefficients"], dtype=float)
            return float(
                np.sum(quadratic * np.square(shifted) + quartic * shifted**4)
            )
        if self.family == "custom_sparse_curved_interactions":
            direct = np.asarray(self.truth["direct_coefficients"], dtype=float)
            residual_weights = np.asarray(
                self.truth["residual_coefficients"], dtype=float
            )
            z1, z2, z3, z4 = shifted
            residuals = np.asarray(
                (
                    z2 - 0.60 * z1 - 0.15 * z1**2,
                    z3 - 0.50 * z2 + 0.12 * z2**2,
                    z4 + 0.40 * z3 - 0.10 * z3**2,
                ),
                dtype=float,
            )
            return float(
                np.sum(direct * np.square(shifted))
                + np.sum(residual_weights * np.square(residuals))
            )
        if self.family == "custom_separable_irregular_multibasin":
            envelope = np.asarray(
                self.truth["envelope_coefficients"], dtype=float
            )
            first = np.asarray(self.truth["first_amplitudes"], dtype=float)
            second = np.asarray(self.truth["second_amplitudes"], dtype=float)
            angular = np.asarray(self.truth["angular_scales"], dtype=float)
            return float(
                np.sum(
                    envelope * np.square(shifted)
                    + first * (1.0 - np.cos(angular * shifted))
                    + second
                    * (1.0 - np.cos(math.sqrt(2.0) * angular * shifted))
                )
            )
        raise ValueError(f"Unknown function family: {self.family}")
