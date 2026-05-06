"""Load sklearn surrogate stored as joblib dict: model + X-name (bundle format: model, X-name)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _require_surrogate_deps() -> None:
    try:
        import joblib  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Surrogate tasks require optional dependencies. Install with: uv sync --extra surrogate"
        ) from exc


class JoblibSurrogate:
    """Wraps sklearn predictor trained on physical knob features (column order = X-name)."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self._model = model
        self.feature_names = tuple(feature_names)

    @classmethod
    def from_path(cls, path: str | Path) -> JoblibSurrogate:
        _require_surrogate_deps()
        import joblib

        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Surrogate file not found: {path.resolve()}")
        try:
            payload = joblib.load(path)
        except ValueError as exc:
            msg = str(exc)
            if "EOF" in msg or "array data" in msg:
                raise RuntimeError(
                    "Failed to load surrogate joblib (file appears truncated or incomplete). "
                    f"Path: {path.resolve()}\n"
                    "Re-stage the full local `.joblib` described in `bbo/tasks/dbtune/assets/README.md` "
                    "or use `AGENTIC_BBO_*` to point this in-process task to a full file. "
                    "If the file is tracked with Git LFS, run `git lfs pull` first."
                ) from exc
            raise
        if not isinstance(payload, dict):
            raise ValueError("Expected joblib dict with keys 'model' and 'X-name'")
        model = payload.get("model")
        names = payload.get("X-name")
        if model is None or names is None:
            raise ValueError("joblib payload must contain 'model' and 'X-name'")
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return cls(model=model, feature_names=list(names))

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def predict(self, features_row: NDArray[np.float64]) -> float:
        x = np.asarray(features_row, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")
        y = self._model.predict(x)
        return float(np.asarray(y).ravel()[0])


__all__ = ["JoblibSurrogate"]
