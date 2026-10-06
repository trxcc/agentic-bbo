"""Reusable optimizer-compute boundary behind agent-facing tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol

from ...core import TaskSpec, TrialObservation


class OptimizationBackend(Protocol):
    """Compute-only backend: implementations must never call the evaluator."""

    def execute(
        self,
        action: str,
        *,
        task_spec: TaskSpec,
        history: Iterable[TrialObservation],
        seed: int,
        incumbent: dict[str, Any] | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


class StatefulOptimizerBackend:
    """Adapter over the shared baseline registry and persistent policy state.

    The local import deliberately keeps the backend independent from tool
    schemas and avoids an import cycle with ``optimizer_tools``.
    """

    def __init__(self, *, allowlist: Iterable[str], state_path: Path | str,
                 gp_overrides: dict[str, Any] | None = None) -> None:
        self.allowlist = tuple(allowlist)
        self.state_path = Path(state_path)
        # Host-owned numerical configuration, never supplied through tool arguments.
        self.gp_overrides = dict(gp_overrides or {})

    def execute(
        self,
        action: str,
        *,
        task_spec: TaskSpec,
        history: Iterable[TrialObservation],
        seed: int,
        incumbent: dict[str, Any] | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        from .tools.optimizer_tools import execute_optimizer_action

        result = execute_optimizer_action(
            action=action,
            task_spec=task_spec,
            history=history,
            allowlist=self.allowlist,
            state_path=self.state_path,
            seed=seed,
            incumbent=incumbent,
            arguments=arguments,
            gp_overrides=self.gp_overrides,
        )
        if result.get("evaluator_called") is not False:
            raise RuntimeError("OptimizationBackend violated evaluator isolation")
        if result.get("budget_consumed") is not False:
            raise RuntimeError("OptimizationBackend consumed objective budget")
        return result

    def snapshot(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        import json

        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def restore(self, state: dict[str, Any]) -> None:
        import json

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
