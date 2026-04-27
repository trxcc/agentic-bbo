"""Generative QED scientific benchmark task using fixed-length SELFIES tokens."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ...core import (
    CategoricalParam,
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)
from .data_assets import SOURCE_REPO_URL, DatasetAsset, stage_dataset_asset
from .molecule import MOLECULE_ARCHIVE_MEMBER, MOLECULE_DATASET_RELATIVE_PATH, load_zinc_smiles

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
QED_SELFIES_TASK_NAME = "qed_selfies_demo"
QED_SELFIES_DEFAULT_MAX_EVALUATIONS = 40
QED_SELFIES_DEFAULT_MAX_TOKENS = 16
QED_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT = 4096
QED_SELFIES_PAD_TOKEN = "__PAD__"
QED_SELFIES_EOS_TOKEN = "__EOS__"
QED_SELFIES_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / QED_SELFIES_TASK_NAME
QED_SELFIES_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
QED_SELFIES_FALLBACK_SMILES = (
    "C",
    "CC",
    "CCO",
    "CCN",
    "COC",
    "c1ccccc1",
    "CC(=O)O",
)


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The QED SELFIES task requires RDKit. Install it with "
            "`uv sync --extra dev --extra bo-tutorial` or provide a compatible conda environment."
        ) from exc
    return Chem, QED


def _require_selfies():
    try:
        import selfies as sf
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The QED SELFIES task requires the `selfies` package. Install it with "
            "`uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return sf


@dataclass(frozen=True)
class SelfiesDecodeResult:
    """Decoded molecule representation for one token configuration."""

    selfies: str
    smiles: str
    tokens: tuple[str, ...]
    valid_selfies: bool
    decode_error: str | None = None


@dataclass
class QedSelfiesTaskConfig:
    """Configuration for one fixed-length SELFIES/QED benchmark task."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    max_selfies_tokens: int = QED_SELFIES_DEFAULT_MAX_TOKENS
    vocabulary_source_limit: int = QED_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT
    metadata: dict[str, Any] = field(default_factory=dict)


class QedSelfiesTask(Task):
    """Open-ended QED task over a fixed-length SELFIES token search space."""

    def __init__(self, config: QedSelfiesTaskConfig | None = None):
        self.config = config or QedSelfiesTaskConfig()
        if self.config.max_selfies_tokens <= 0:
            raise ValueError("max_selfies_tokens must be positive.")
        if self.config.vocabulary_source_limit <= 0:
            raise ValueError("vocabulary_source_limit must be positive.")

        self._asset = stage_dataset_asset(
            MOLECULE_DATASET_RELATIVE_PATH,
            label="QED/SELFIES",
            task_name=QED_SELFIES_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        self._smiles_list = load_zinc_smiles(self._asset.cache_path)
        if not self._smiles_list:
            raise ValueError("The QED SELFIES task requires at least one source SMILES string.")

        Chem, QED = _require_rdkit()
        self._chem = Chem
        self._qed = QED
        self._selfies = _require_selfies()
        self._max_tokens = int(self.config.max_selfies_tokens)

        vocab, default_smiles, default_selfies, default_tokens, default_qed = self._build_vocabulary_and_default()
        choices = (QED_SELFIES_PAD_TOKEN, QED_SELFIES_EOS_TOKEN, *vocab)
        default_config = self._tokens_to_config(default_tokens)
        search_space = SearchSpace(
            [
                CategoricalParam(
                    self._token_param_name(index),
                    choices=choices,
                    default=default_config[self._token_param_name(index)],
                )
                for index in range(self._max_tokens)
            ]
        )
        description_dir = self.config.description_dir or QED_SELFIES_DESCRIPTION_DIR
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "source_item_count": int(len(self._smiles_list)),
            "vocabulary_source_limit": int(self.config.vocabulary_source_limit),
            "selfies_vocabulary_size": int(len(vocab)),
            "search_token_choices": int(len(choices)),
            "max_selfies_tokens": int(self._max_tokens),
            "archive_member": MOLECULE_ARCHIVE_MEMBER,
            "default_smiles": default_smiles,
            "default_selfies": default_selfies,
            "default_qed": float(default_qed),
        }
        self._spec = TaskSpec(
            name=QED_SELFIES_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("qed_loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or QED_SELFIES_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(QED_SELFIES_TASK_NAME, description_dir),
            metadata={
                "display_name": "QED SELFIES Demo",
                "source_paper": QED_SELFIES_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_cache_path": str(self._asset.cache_path),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "representation": "fixed_length_selfies_tokens",
                "selfies_pad_token": QED_SELFIES_PAD_TOKEN,
                "selfies_eos_token": QED_SELFIES_EOS_TOKEN,
                "dimension": self._max_tokens,
                **self.config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def dataset_asset(self) -> DatasetAsset:
        return self._asset

    @property
    def dataset_summary(self) -> dict[str, Any]:
        return dict(self._dataset_summary)

    @staticmethod
    def _token_param_name(index: int) -> str:
        return f"selfies_token_{index:02d}"

    def _build_vocabulary_and_default(self) -> tuple[tuple[str, ...], str, str, tuple[str, ...], float]:
        vocabulary: set[str] = set()
        scored_candidates: list[tuple[str, str, tuple[str, ...], float]] = []

        source_smiles = list(QED_SELFIES_FALLBACK_SMILES)
        source_smiles.extend(self._smiles_list[: self.config.vocabulary_source_limit])
        for smiles in self._unique_strings(source_smiles):
            encoded = self._encode_smiles(smiles)
            if encoded is None:
                continue
            selfies, tokens = encoded
            vocabulary.update(tokens)
            molecule = self._chem.MolFromSmiles(smiles)
            if molecule is None or len(tokens) > self._max_tokens:
                continue
            scored_candidates.append((smiles, selfies, tokens, float(self._qed.qed(molecule))))

        if not vocabulary:
            raise ValueError("Could not build a SELFIES token vocabulary from the source molecules.")
        if not scored_candidates:
            raise ValueError("Could not find a valid default molecule within max_selfies_tokens.")

        default_smiles, default_selfies, default_tokens, default_qed = max(scored_candidates, key=lambda item: item[3])
        return tuple(sorted(vocabulary)), default_smiles, default_selfies, default_tokens, default_qed

    @staticmethod
    def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            unique.append(text)
            seen.add(text)
        return tuple(unique)

    def _encode_smiles(self, smiles: str) -> tuple[str, tuple[str, ...]] | None:
        try:
            selfies = self._selfies.encoder(smiles)
        except Exception:
            return None
        if not selfies:
            return None
        tokens = tuple(self._selfies.split_selfies(selfies))
        if not tokens:
            return None
        return str(selfies), tokens

    def _tokens_to_config(self, tokens: tuple[str, ...]) -> dict[str, str]:
        values = list(tokens[: self._max_tokens])
        if len(values) < self._max_tokens:
            values.append(QED_SELFIES_EOS_TOKEN)
        while len(values) < self._max_tokens:
            values.append(QED_SELFIES_PAD_TOKEN)
        return {self._token_param_name(index): values[index] for index in range(self._max_tokens)}

    def config_from_smiles(self, smiles: str) -> dict[str, str]:
        """Convert one SMILES string into the canonical fixed-length SELFIES config."""

        encoded = self._encode_smiles(smiles)
        if encoded is None:
            raise ValueError(f"Could not encode SMILES as SELFIES: {smiles!r}")
        _, tokens = encoded
        return self.spec.search_space.coerce_config(self._tokens_to_config(tokens), use_defaults=False)

    def config_from_selfies(self, selfies: str) -> dict[str, str]:
        """Convert a SELFIES string into the canonical fixed-length SELFIES config."""

        tokens = tuple(self._selfies.split_selfies(str(selfies)))
        if not tokens:
            raise ValueError("SELFIES string must contain at least one token.")
        return self.spec.search_space.coerce_config(self._tokens_to_config(tokens), use_defaults=False)

    def _decode_config(self, config: dict[str, Any]) -> SelfiesDecodeResult:
        tokens: list[str] = []
        for index in range(self._max_tokens):
            token = str(config[self._token_param_name(index)])
            if token == QED_SELFIES_EOS_TOKEN:
                break
            if token == QED_SELFIES_PAD_TOKEN:
                continue
            tokens.append(token)
        selfies = "".join(tokens)
        if not selfies:
            return SelfiesDecodeResult(
                selfies="",
                smiles="",
                tokens=tuple(tokens),
                valid_selfies=False,
                decode_error="empty_selfies",
            )
        try:
            smiles = str(self._selfies.decoder(selfies))
        except Exception as exc:
            return SelfiesDecodeResult(
                selfies=selfies,
                smiles="",
                tokens=tuple(tokens),
                valid_selfies=False,
                decode_error=f"{type(exc).__name__}: {exc}",
            )
        return SelfiesDecodeResult(
            selfies=selfies,
            smiles=smiles,
            tokens=tuple(tokens),
            valid_selfies=bool(smiles),
            decode_error=None,
        )

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        decoded = self._decode_config(config)
        molecule = self._chem.MolFromSmiles(decoded.smiles) if decoded.valid_selfies else None
        qed = 0.0 if molecule is None else float(self._qed.qed(molecule))
        qed_loss = 1.0 - qed
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"qed_loss": qed_loss},
            metrics={
                "qed": qed,
                "selfies_token_count": len(decoded.tokens),
            },
            elapsed_seconds=elapsed,
            metadata={
                **self._asset.as_metadata(),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "selfies": decoded.selfies,
                "smiles": decoded.smiles,
                "selfies_tokens": list(decoded.tokens),
                "valid_selfies": decoded.valid_selfies,
                "valid_smiles": molecule is not None,
                "decode_error": decoded.decode_error,
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if self._dataset_summary["selfies_vocabulary_size"] <= 0:
            report.add_error("empty_selfies_vocabulary", "The QED SELFIES task has an empty token vocabulary.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["qed_loss"])):
                report.add_error("non_finite_prediction", "The QED SELFIES task produced a non-finite QED loss.")
            if not default_result.metadata.get("valid_smiles"):
                report.add_error("invalid_default_molecule", "The default SELFIES config did not decode to a valid molecule.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("qed_selfies_evaluation_failed", f"The QED SELFIES task could not score the default config: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def create_qed_selfies_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    max_selfies_tokens: int = QED_SELFIES_DEFAULT_MAX_TOKENS,
    vocabulary_source_limit: int = QED_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT,
    metadata: dict[str, Any] | None = None,
) -> QedSelfiesTask:
    return QedSelfiesTask(
        QedSelfiesTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            max_selfies_tokens=max_selfies_tokens,
            vocabulary_source_limit=vocabulary_source_limit,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "QED_SELFIES_DEFAULT_MAX_EVALUATIONS",
    "QED_SELFIES_DEFAULT_MAX_TOKENS",
    "QED_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT",
    "QED_SELFIES_DESCRIPTION_DIR",
    "QED_SELFIES_EOS_TOKEN",
    "QED_SELFIES_PAD_TOKEN",
    "QED_SELFIES_TASK_NAME",
    "QedSelfiesTask",
    "QedSelfiesTaskConfig",
    "SelfiesDecodeResult",
    "create_qed_selfies_task",
]
