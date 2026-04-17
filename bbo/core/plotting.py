"""Scientific-style plotting helpers for benchmark runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .task import ObjectiveDirection
from .trial import TrialRecord


@dataclass(frozen=True)
class PlotArtifact:
    """Metadata for one generated plot."""

    name: str
    path: Path


class ScientificPlotter:
    """Base class that applies a restrained scientific plotting style."""

    def __init__(self, *, dpi: int = 180):
        self.dpi = dpi

    @contextmanager
    def style(self):
        with plt.rc_context(
            {
                "figure.dpi": self.dpi,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "--",
                "font.family": "DejaVu Serif",
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.facecolor": "#fbfbfb",
                "figure.facecolor": "white",
                "savefig.bbox": "tight",
            }
        ):
            yield

    @staticmethod
    def _ensure_parent(path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _primary_series(records: Iterable[TrialRecord], objective_name: str) -> np.ndarray:
        return np.asarray([float(record.objectives[objective_name]) for record in records if objective_name in record.objectives])

    @staticmethod
    def _running_best(values: np.ndarray, direction: ObjectiveDirection) -> np.ndarray:
        if values.size == 0:
            return values
        running: list[float] = []
        best = values[0]
        for value in values:
            if direction == ObjectiveDirection.MINIMIZE:
                best = min(best, value)
            else:
                best = max(best, value)
            running.append(best)
        return np.asarray(running, dtype=float)


class OptimizationTracePlotter(ScientificPlotter):
    """Plot the raw objective trace and incumbent trajectory."""

    def plot(
        self,
        records: list[TrialRecord],
        *,
        objective_name: str,
        direction: ObjectiveDirection,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        values = self._primary_series(records, objective_name)
        evaluations = np.arange(1, len(values) + 1)
        running = self._running_best(values, direction)

        with self.style():
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            ax.plot(evaluations, values, color="#4c72b0", marker="o", linewidth=1.2, markersize=3.5, label="observed")
            ax.plot(evaluations, running, color="#c44e52", linewidth=2.0, label="incumbent")
            ax.set_xlabel("Evaluation")
            ax.set_ylabel(objective_name)
            ax.set_title(title)
            ax.legend(loc="best")
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="optimization_trace", path=output)


class ObjectiveDistributionPlotter(ScientificPlotter):
    """Plot the empirical distribution of objective values."""

    def plot(
        self,
        records: list[TrialRecord],
        *,
        objective_name: str,
        output_path: Path | str,
        title: str,
        bins: int = 16,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        values = self._primary_series(records, objective_name)

        with self.style():
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            ax.hist(values, bins=bins, color="#55a868", edgecolor="white", alpha=0.9)
            ax.axvline(float(np.median(values)), color="#8172b2", linestyle="--", linewidth=1.5, label="median")
            ax.set_xlabel(objective_name)
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.legend(loc="best")
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="objective_distribution", path=output)


class Landscape2DPlotter(ScientificPlotter):
    """Plot a 2D objective landscape with evaluated points overlaid."""

    def plot(
        self,
        task: object,
        records: list[TrialRecord],
        *,
        objective_name: str,
        output_path: Path | str,
        title: str,
        resolution: int = 180,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        if not hasattr(task, "surface_grid"):
            raise TypeError("Landscape2DPlotter requires a task with a `surface_grid()` method.")
        xx, yy, zz = task.surface_grid(resolution=resolution)
        evaluated = np.asarray([[record.config[name] for name in task.spec.search_space.names()] for record in records], dtype=float)
        values = self._primary_series(records, objective_name)
        best_index = int(np.argmin(values)) if values.size else None

        with self.style():
            fig, ax = plt.subplots(figsize=(6.8, 5.4))
            contour = ax.contourf(xx, yy, zz, levels=28, cmap="viridis")
            fig.colorbar(contour, ax=ax, label=objective_name)
            ax.scatter(evaluated[:, 0], evaluated[:, 1], c="white", s=20, edgecolors="black", linewidths=0.5, label="samples")
            if best_index is not None:
                ax.scatter(
                    evaluated[best_index, 0],
                    evaluated[best_index, 1],
                    c="#c44e52",
                    s=55,
                    edgecolors="black",
                    linewidths=0.7,
                    label="best observed",
                )
            ax.set_xlabel(task.spec.search_space.names()[0])
            ax.set_ylabel(task.spec.search_space.names()[1])
            ax.set_title(title)
            ax.legend(loc="upper right")
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="landscape_2d", path=output)


class OptimizerComparisonPlotter(ScientificPlotter):
    """Compare multiple optimizers on their running-best trajectories."""

    def plot(
        self,
        histories: dict[str, list[TrialRecord]],
        *,
        objective_name: str,
        direction: ObjectiveDirection,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        with self.style():
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            palette = ["#4c72b0", "#c44e52", "#55a868", "#8172b2"]
            for color, (label, records) in zip(palette, histories.items(), strict=False):
                values = self._primary_series(records, objective_name)
                evaluations = np.arange(1, len(values) + 1)
                running = self._running_best(values, direction)
                ax.plot(evaluations, running, linewidth=2.1, color=color, label=label)
            ax.set_xlabel("Evaluation")
            ax.set_ylabel(f"best {objective_name}")
            ax.set_title(title)
            ax.legend(loc="best")
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="optimizer_comparison", path=output)


__all__ = [
    "Landscape2DPlotter",
    "ObjectiveDistributionPlotter",
    "OptimizationTracePlotter",
    "OptimizerComparisonPlotter",
    "PlotArtifact",
    "ScientificPlotter",
]
