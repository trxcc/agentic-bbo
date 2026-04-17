"""Algorithm protocol for benchmark-oriented BBO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .task import TaskSpec
from .trial import TrialObservation, TrialSuggestion


@dataclass(frozen=True)
class Incumbent:
    """Best-known configuration reported by an algorithm."""

    config: dict[str, Any]
    score: float | None = None
    objectives: dict[str, float] = field(default_factory=dict)
    trial_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Algorithm(ABC):
    """Abstract optimization algorithm with an ask/tell interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm identifier."""

    @abstractmethod
    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        """Initialize the algorithm for one task."""

    @abstractmethod
    def ask(self) -> TrialSuggestion:
        """Suggest the next configuration to evaluate."""

    @abstractmethod
    def tell(self, observation: TrialObservation) -> None:
        """Report a completed trial observation."""

    @abstractmethod
    def incumbents(self) -> list[Incumbent]:
        """Return the best-known configuration(s)."""

    def seed(self, observation: TrialObservation) -> None:
        self.tell(observation)

    def replay(self, history: list[TrialObservation]) -> None:
        for observation in history:
            self.tell(observation)


__all__ = ["Algorithm", "Incumbent"]
