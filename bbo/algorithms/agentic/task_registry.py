"""Planner task registry for Pablo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class TaskCard:
    name: str
    text: str
    attempts: int = 0
    successes: int = 0
    is_default: bool = False

    @property
    def success_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return float(self.successes) / float(self.attempts)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "text": self.text,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "is_default": self.is_default,
        }


class TaskRegistry:
    """Bounded task registry with trim-on-overflow semantics."""

    def __init__(self, *, default_tasks: Mapping[str, str], max_tasks: int = 20):
        if max_tasks < len(default_tasks):
            raise ValueError("max_tasks must be at least the number of default tasks.")
        self.max_tasks = int(max_tasks)
        self._cards: dict[str, TaskCard] = {
            name: TaskCard(name=name, text=text, is_default=True)
            for name, text in default_tasks.items()
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object], *, default_tasks: Mapping[str, str], max_tasks: int) -> "TaskRegistry":
        registry = cls(default_tasks=default_tasks, max_tasks=max_tasks)
        cards = snapshot.get("cards", []) if isinstance(snapshot, dict) else []
        for item in cards:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            text = str(item.get("text", "")).strip()
            if not name or not text:
                continue
            registry._cards[name] = TaskCard(
                name=name,
                text=text,
                attempts=int(item.get("attempts", 0)),
                successes=int(item.get("successes", 0)),
                is_default=bool(item.get("is_default", False)),
            )
        registry._trim_to_capacity()
        return registry

    def update_from_planner(self, tasks: Mapping[str, str]) -> list[str]:
        inserted: list[str] = []
        for raw_name, raw_text in tasks.items():
            name = str(raw_name).strip()
            text = str(raw_text).strip()
            if not name or not text:
                continue
            card = self._cards.get(name)
            if card is None:
                self._cards[name] = TaskCard(name=name, text=text, is_default=False)
                inserted.append(name)
            else:
                card.text = text
        self._trim_to_capacity()
        return inserted

    def record_attempt(self, task_name: str, *, success: bool) -> None:
        if task_name not in self._cards:
            return
        card = self._cards[task_name]
        card.attempts += 1
        if success:
            card.successes += 1

    def summary(self, *, limit: int | None = None) -> list[dict[str, object]]:
        cards = sorted(
            self._cards.values(),
            key=lambda card: (-card.success_rate, -card.successes, -card.attempts, card.name),
        )
        if limit is not None:
            cards = cards[:limit]
        return [card.to_dict() for card in cards]

    def active_tasks(self, *, limit: int | None = None) -> list[TaskCard]:
        cards = sorted(
            self._cards.values(),
            key=lambda card: (
                0 if card.is_default else 1,
                -card.success_rate,
                -card.successes,
                -card.attempts,
                card.name,
            ),
        )
        if limit is not None:
            cards = cards[:limit]
        return cards

    def snapshot(self) -> dict[str, object]:
        return {
            "max_tasks": self.max_tasks,
            "cards": [card.to_dict() for card in self.active_tasks()],
        }

    def _trim_to_capacity(self) -> None:
        while len(self._cards) > self.max_tasks:
            removable = [card for card in self._cards.values() if not card.is_default]
            if not removable:
                break
            removable.sort(key=lambda card: (card.success_rate, -card.attempts, card.name))
            self._cards.pop(removable[0].name, None)


__all__ = ["TaskCard", "TaskRegistry"]
