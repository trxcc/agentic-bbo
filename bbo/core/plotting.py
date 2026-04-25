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
            palette = ["#4c72b0", "#c44e52", "#55a868", "#8172b2", "#ccaa7a"]
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


def _per_trial_elapsed_sec(records: list[TrialRecord]) -> np.ndarray:
    return np.asarray(
        [float(r.elapsed_seconds) if r.elapsed_seconds is not None else 0.0 for r in records],
        dtype=float,
    )


class RegretTracePlotter(ScientificPlotter):
    """When ``known_optimum`` is in task metadata, plot incumbent regret vs global optimum (one objective)."""

    def plot(
        self,
        records: list[TrialRecord],
        *,
        objective_name: str,
        direction: ObjectiveDirection,
        known_optimum: float,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        values = self._primary_series(records, objective_name)
        if values.size == 0:
            with self.style():
                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                ax.text(0.5, 0.5, "no trial records", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                fig.savefig(output)
                plt.close(fig)
            return PlotArtifact(name="regret_empty", path=output)
        running = self._running_best(values, direction)
        if direction == ObjectiveDirection.MINIMIZE:
            regret = running - float(known_optimum)
        else:
            regret = float(known_optimum) - running
        regret = np.maximum(regret, 0.0)
        x = np.arange(1, len(regret) + 1)
        with self.style():
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.plot(x, regret, color="#c44e52", linewidth=2.0, marker="o", markersize=3.0, label="incumbent regret")
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Regret w.r.t. known optimum")
            ax.set_title(title)
            ax.legend(loc="best")
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="regret_trace", path=output)


class PerTrialEvalTimePlotter(ScientificPlotter):
    """Per-evaluation wall time (``elapsed_seconds``) as a bar chart."""

    def plot(
        self,
        records: list[TrialRecord],
        *,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        times = _per_trial_elapsed_sec(records)
        if not records:
            with self.style():
                fig, ax = plt.subplots(figsize=(7.0, 3.8))
                ax.text(0.5, 0.5, "no trial records", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                fig.savefig(output)
                plt.close(fig)
            return PlotArtifact(name="per_trial_eval_time_empty", path=output)
        n = len(times)
        x = np.arange(1, n + 1)
        with self.style():
            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            ax.bar(x, times, color="#4c72b0", edgecolor="white", alpha=0.9, width=0.88)
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Elapsed time (s)")
            ax.set_title(title)
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="per_trial_eval_time", path=output)


class CumulativeEvalTimePlotter(ScientificPlotter):
    """Cumulative sum of per-evaluation times (one algorithm)."""

    def plot(
        self,
        records: list[TrialRecord],
        *,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        times = _per_trial_elapsed_sec(records)
        if not records:
            with self.style():
                fig, ax = plt.subplots(figsize=(7.0, 3.8))
                ax.text(0.5, 0.5, "no trial records", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                fig.savefig(output)
                plt.close(fig)
            return PlotArtifact(name="cumulative_eval_time_empty", path=output)
        cum = np.cumsum(times)
        x = np.arange(1, len(cum) + 1)
        with self.style():
            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            ax.fill_between(x, 0, cum, color="#4c72b0", alpha=0.15)
            ax.plot(x, cum, color="#4c72b0", linewidth=2.0)
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Cumulative eval time (s)")
            ax.set_title(title)
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="cumulative_eval_time", path=output)


class CumulativeEvalTimeComparisonPlotter(ScientificPlotter):
    """Compare cumulative evaluation wall time across optimizers (one metric: wall time)."""

    def plot(
        self,
        histories: dict[str, list[TrialRecord]],
        *,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        with self.style():
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            palette = ["#4c72b0", "#c44e52", "#55a868", "#8172b2", "#ccaa7a"]
            for color, (label, recs) in zip(palette, histories.items(), strict=False):
                times = _per_trial_elapsed_sec(list(recs))
                if not np.any(times > 0):
                    continue
                cum = np.cumsum(times)
                x = np.arange(1, len(cum) + 1)
                ax.plot(x, cum, linewidth=2.0, color=color, label=label)
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Cumulative eval time (s)")
            ax.set_title(title)
            ax.legend(loc="best")
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="cumulative_eval_time_comparison", path=output)


class ScalarBarPlotter(ScientificPlotter):
    """One figure, one metric: bar chart for numeric scalars (e.g. best objective, total time)."""

    def plot(
        self,
        series: dict[str, float],
        *,
        ylabel: str,
        output_path: Path | str,
        title: str,
    ) -> PlotArtifact:
        output = self._ensure_parent(output_path)
        if not series:
            with self.style():
                fig, ax = plt.subplots(figsize=(5.0, 3.0))
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                fig.savefig(output)
                plt.close(fig)
            return PlotArtifact(name="scalar_bar_empty", path=output)
        labels = list(series.keys())
        values = [float(v) for v in series.values()]
        x = np.arange(len(labels))
        with self.style():
            fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(labels)), 4.2))
            colors = ["#4c72b0", "#c44e52", "#55a868", "#8172b2", "#ccaa7a", "#8c8c8c"] * 2
            ax.bar(x, values, color=colors[: len(values)], edgecolor="white", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            for i, v in enumerate(values):
                ax.text(x[i], v, f"{v:.4g}", ha="center", va="bottom", fontsize=8, rotation=0)
            fig.savefig(output)
            plt.close(fig)
        return PlotArtifact(name="scalar_bars", path=output)


__all__ = [
    "CumulativeEvalTimeComparisonPlotter",
    "CumulativeEvalTimePlotter",
    "Landscape2DPlotter",
    "ObjectiveDistributionPlotter",
    "OptimizationTracePlotter",
    "OptimizerComparisonPlotter",
    "PerTrialEvalTimePlotter",
    "PlotArtifact",
    "RegretTracePlotter",
    "ScalarBarPlotter",
    "ScientificPlotter",
]
