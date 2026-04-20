"""Dataset staging and validation helpers for BO tutorial scientific tasks."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REPO_URL = "https://github.com/zwyu-ai/BO-Tutorial-for-Sci"
SOURCE_ROOT_ENV = "BBO_BO_TUTORIAL_SOURCE_ROOT"
CACHE_ROOT_ENV = "BBO_BO_TUTORIAL_CACHE_ROOT"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "artifacts" / "dataset_cache" / "bo_tutorial"
VENDORED_SOURCE_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_SOURCE_ROOT_CANDIDATES = (
    VENDORED_SOURCE_ROOT,
    PROJECT_ROOT / "artifacts" / "sources" / "BO-Tutorial-for-Sci",
)


@dataclass(frozen=True)
class DatasetAsset:
    """Metadata for one source dataset staged into the local cache."""

    label: str
    task_name: str
    relative_path: str
    source_root: Path
    source_path: Path
    cache_root: Path
    cache_path: Path
    source_ref: str | None
    sha256: str
    size_bytes: int

    def as_metadata(self) -> dict[str, str | int | None]:
        return {
            "label": self.label,
            "task_name": self.task_name,
            "relative_path": self.relative_path,
            "source_root": str(self.source_root),
            "source_path": str(self.source_path),
            "cache_root": str(self.cache_root),
            "cache_path": str(self.cache_path),
            "source_ref": self.source_ref,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def resolve_cache_root(cache_root: Path | str | None = None) -> Path:
    """Resolve the writable dataset cache root and create it when needed."""

    configured = cache_root or os.environ.get(CACHE_ROOT_ENV) or DEFAULT_CACHE_ROOT
    path = Path(configured).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source_root(
    source_root: Path | str | None = None,
    *,
    required_paths: Iterable[str] = (),
) -> Path:
    """Resolve the bundled scientific dataset root or raise a clear error."""

    candidates: list[Path] = []
    configured = source_root or os.environ.get(SOURCE_ROOT_ENV)
    if configured is not None:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_SOURCE_ROOT_CANDIDATES)

    missing_messages: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists():
            missing_messages.append(f"- missing candidate: {resolved}")
            continue
        missing_required = [relative for relative in required_paths if not (resolved / relative).exists()]
        if missing_required:
            missing_messages.append(
                f"- candidate exists but is incomplete: {resolved} (missing {missing_required!r})"
            )
            continue
        return resolved

    hints = "\n".join(missing_messages)
    raise FileNotFoundError(
        "Could not locate the scientific task datasets in the workspace. "
        f"Expected bundled data under {VENDORED_SOURCE_ROOT} or set `{SOURCE_ROOT_ENV}` "
        "to a workspace-local dataset root that preserves the original `examples/...` layout.\n"
        f"Original tutorial repository for provenance: {SOURCE_REPO_URL}\n"
        f"Checked candidates:\n{hints}"
    )


def compute_sha256(path: Path) -> str:
    """Compute the sha256 checksum for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_ref(source_root: Path) -> str | None:
    """Return the git ref for the source checkout when available."""

    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    ref = result.stdout.strip()
    return ref or None


def stage_dataset_asset(
    relative_path: str,
    *,
    label: str,
    task_name: str,
    source_root: Path | str | None = None,
    cache_root: Path | str | None = None,
) -> DatasetAsset:
    """Copy one source dataset into the local cache and return its metadata."""

    resolved_source_root = resolve_source_root(source_root, required_paths=(relative_path,))
    resolved_cache_root = resolve_cache_root(cache_root)
    source_path = (resolved_source_root / relative_path).resolve()
    cache_path = (resolved_cache_root / relative_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    source_sha256 = compute_sha256(source_path)
    copy_required = True
    if cache_path.exists():
        copy_required = compute_sha256(cache_path) != source_sha256
    if copy_required:
        shutil.copy2(source_path, cache_path)

    return DatasetAsset(
        label=label,
        task_name=task_name,
        relative_path=relative_path,
        source_root=resolved_source_root,
        source_path=source_path,
        cache_root=resolved_cache_root,
        cache_path=cache_path,
        source_ref=resolve_source_ref(resolved_source_root),
        sha256=source_sha256,
        size_bytes=source_path.stat().st_size,
    )


__all__ = [
    "CACHE_ROOT_ENV",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_SOURCE_ROOT_CANDIDATES",
    "DatasetAsset",
    "PROJECT_ROOT",
    "SOURCE_REPO_URL",
    "SOURCE_ROOT_ENV",
    "VENDORED_SOURCE_ROOT",
    "compute_sha256",
    "resolve_cache_root",
    "resolve_source_ref",
    "resolve_source_root",
    "stage_dataset_asset",
]
